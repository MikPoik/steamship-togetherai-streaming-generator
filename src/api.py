import json
import logging
import time
from typing import Any, Dict, List, Optional, Type
import os
import tiktoken
from pydantic import Field

from steamship import Steamship, Block, Tag, SteamshipError, MimeTypes, File
from steamship.data.tags.tag_constants import TagKind, RoleTag, TagValueKey, ChatTag
from steamship.invocable import Config, InvocableResponse, InvocationContext
from steamship.plugin.inputs.raw_block_and_tag_plugin_input import (
    RawBlockAndTagPluginInput, )
from steamship.plugin.inputs.raw_block_and_tag_plugin_input_with_preallocated_blocks import (
    RawBlockAndTagPluginInputWithPreallocatedBlocks, )
from steamship.plugin.outputs.block_type_plugin_output import BlockTypePluginOutput
from steamship.plugin.outputs.plugin_output import (
    UsageReport,
    OperationType,
    OperationUnit,
)
from steamship.plugin.outputs.stream_complete_plugin_output import (
    StreamCompletePluginOutput, )
from steamship.plugin.request import PluginRequest
from steamship.plugin.streaming_generator import StreamingGenerator
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    before_sleep_log,
    wait_exponential_jitter,
)

import requests
import sseclient


#logging.root.setLevel(logging.INFO)
class TogetherAIPlugin(StreamingGenerator):
    """
    Plugin for generating text using TogetherAI-chat model.
    """

    class TogetherAIPluginConfig(Config):
        api_key: str = Field(
            "",
            description=
            "An Together.ai key to use. If left default, will use Steamship's API key.",
        )
        max_tokens: int = Field(
            256,
            description=
            "The maximum number of tokens to generate per request. Can be overridden in runtime options.",
        )
        model: Optional[str] = Field(
            "NousResearch/Nous-Hermes-Llama2-13b",
            description=
            "The  model to use. Can be a pre-existing fine-tuned model.",  #for chatopenai compatibility
        )
        temperature: Optional[float] = Field(
            0.4,
            description=
            "Controls randomness. Lower values produce higher likelihood / more predictable results; "
            "higher values produce more variety. Values between 0-1.",
        )
        top_p: Optional[int] = Field(
            1,
            description=
            "Controls the nucleus sampling, where the model considers the results of the tokens with "
            "top_p probability mass. Values between 0-1.",
        )
        presence_penalty: Optional[int] = Field(
            0,
            description=
            "Control how likely the model will reuse words. Positive values penalize new tokens based on "
            "whether they appear in the text so far, increasing the model's likelihood to talk about new topics. Number between -2.0 and 2.0.",
        )
        max_retries: int = Field(
            8,
            description="Maximum number of retries to make when generating.")
        request_timeout: Optional[float] = Field(
            600,
            description=
            "Timeout for requests to TogetherAI completion API. Default is 600 seconds.",
        )
        stream_tokens: Optional[bool] = Field(True,
                                              description="Stream tokens")
        default_role: str = Field(
            RoleTag.USER.value,
            description=
            "The default role to use for a block that does not have a Tag of kind='role'",
        )
        default_system_prompt: str = Field(
            "",
            description=
            "System prompt that will be prepended before every request")

    @classmethod
    def config_cls(cls) -> Type[Config]:
        return cls.TogetherAIPluginConfig

    config: TogetherAIPluginConfig

    def __init__(
        self,
        client: Steamship = None,
        config: Dict[str, Any] = None,
        context: InvocationContext = None,
    ):

        super().__init__(client, config, context)
        self._api_base = "https://api.together.xyz/inference"

    def prepare_message(self, block: Block) -> Optional[Dict[str, str]]:
        role = None
        name = None
        function_selection = False

        for tag in block.tags:
            if tag.kind == TagKind.ROLE:
                # this is a legacy way of extracting roles
                role = tag.name

            if tag.kind == TagKind.CHAT and tag.name == ChatTag.ROLE:
                if values := tag.value:
                    if role_name := values.get(TagValueKey.STRING_VALUE, None):
                        role = role_name

            if tag.kind == TagKind.ROLE and tag.name == RoleTag.FUNCTION:
                if values := tag.value:
                    if fn_name := values.get(TagValueKey.STRING_VALUE, None):
                        name = fn_name

            if tag.kind == "function-selection":
                function_selection = True

            if tag.kind == "name":
                name = tag.name

        if role is None:
            role = self.config.default_role

        if role not in ["function", "system", "assistant", "user"]:
            logging.warning(
                f"unsupported role {role} found in message. skipping...")
            return None

        if role == "function" and not name:
            name = "unknown"  # protect against missing function names

        if function_selection:
            return {
                "role": RoleTag.ASSISTANT.value,
                "content": None,
                "function_call": json.loads(block.text)
            }

        if name:
            return {"role": role, "content": block.text, "name": name}

        return {"role": role, "content": block.text}

    def prepare_messages(self, blocks: List[Block]) -> List[Dict[str, str]]:
        messages = []
        if self.config.default_system_prompt != "":
            messages.append({
                "role": RoleTag.SYSTEM,
                "content": self.config.default_system_prompt
            })
        # TODO: remove is_text check here when can handle image etc. input
        messages.extend([
            msg for msg in (self.prepare_message(block) for block in blocks
                            if block.text is not None and block.text != "")
            if msg is not None
        ])
        return messages

    def generate_with_retry(
        self,
        user: str,
        messages: List[Dict[str, str]],
        options: Dict,
        output_blocks: List[Block],
    ) -> List[UsageReport]:
        """Call the API to generate the next section of text."""
        logging.info(
            f"Making Together-ai -chat completion call on behalf of user with id: {user}"
        )
        options = options or {}
        stopwords = options.get("stop", None)
        #logging.warning("options: " + str(options))

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential_jitter(jitter=5),
            before_sleep=before_sleep_log(logging.root, logging.INFO),
            retry=(
                retry_if_exception_type(
                    ConnectionError
                )  # handle 104s that manifest as ConnectionResetError
            ),
            after=after_log(logging.root, logging.INFO),
        )
        def _generate_with_retry() -> Any:
            nonlocal messages
            prompt = ""

            #format single prompt based on model
            if "zephyr-chat" in self.config.model.lower():
                if len(messages) == 1:
                    prompt = messages[0]["content"]
                elif len(messages) == 2:
                    prompt = f'<|{messages[0]["role"]}|>\n{messages[0]["content"]}</s>'
                    prompt += f'\n<|{messages[1]["role"]}|>\n{messages[1]["content"]}</s>'
                    prompt += "\n<|assistant|>\n"
                else:
                    for msg in messages:
                        prompt += f'\n<|{msg["role"]}|>\n{msg["content"]}</s>'
                    prompt += "\n<|assistant|>\n"

            elif "nous-hermes-llama" in self.config.model.lower(
            ) or "mythomax" in self.config.model.lower():
                if len(messages) == 1:
                    prompt = f'{messages[0]["content"]}'
                elif len(messages) == 2:
                    prompt = f'### Instruction:\n{messages[0]["content"]}\n'
                    prompt += f'\n### Input:\n{messages[1]["role"]}:\n{messages[1]["content"]}\n'
                    prompt += f'\n### Response:\nassistant:\n'
                    #print(prompt)
                else:
                    # Check if the first message has a role "system" and handle it
                    if messages and messages[0]['role'] == RoleTag.SYSTEM:
                        prompt = f'### Instruction:\n{messages[0]["content"]}\n'
                        messages = messages[
                            1:]  # Remove the first message from the list
                    else:
                        prompt = ""
                    # Loop through the remaining messages and add them to the 'prompt'
                    for msg_index, msg in enumerate(messages):
                        if msg_index == len(
                                messages
                        ) - 1:  # Check if it is the last message
                            prompt += f'\n\n### Input:\n{msg["role"]}:\n{msg["content"]}\n'
                        else:
                            prompt += f'\n    {msg["role"]}:\n    {msg["content"]}\n'
                    prompt += "\n### Response:\nassistant:\n"
                    #print(prompt)

            else:
                if len(messages) == 1:
                    prompt = messages[0]["content"]
                elif len(messages) == 2:
                    prompt = f'<|im_start|>{messages[0]["role"]}\n{messages[0]["content"]}<|im_end|>'
                    prompt += f'\n<|im_start|>{messages[1]["role"]}\n{messages[1]["content"]}<|im_end|>'
                    prompt += "\n<|assistant|>\n"
                else:
                    for msg in messages:
                        prompt += f'\n<|im_start|>{msg["role"]}\n{msg["content"]}<|im_end|>'
                    prompt += "\n<|im_start|>assistant\n"
                #print(prompt)
            payload = {
                "model": self.config.model,
                "prompt": prompt,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "repetition_penalty": self.config.presence_penalty,
                "stream_tokens": self.config.stream_tokens,
                "stop": stopwords,
            }
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            }

            togetherai_response = requests.post(self._api_base,
                                                json=payload,
                                                headers=headers,
                                                stream=True)
            togetherai_response.raise_for_status()
            return togetherai_response

        response = _generate_with_retry()
        output_texts = [""]
        tagged = False
        client = sseclient.SSEClient(response)
        for event in client.events():
            if event.data == "[DONE]":
                break

            partial_result = json.loads(event.data)
            token = partial_result
            output_block = output_blocks[0]
            if not tagged:
                Tag.create(
                    self.client,
                    file_id=output_block.file_id,
                    block_id=output_block.id,
                    kind=TagKind.ROLE,
                    name=RoleTag(RoleTag.ASSISTANT),
                )
                tagged = True
            if 'choices' in partial_result and partial_result[
                    'choices'] and 'text' in partial_result['choices'][0]:
                text_chunk = partial_result['choices'][0]['text']

                if stopwords is not None and text_chunk not in stopwords:
                    if len(text_chunk) > 0:
                        output_block.append_stream(
                            bytes(text_chunk, encoding="utf-8"))
                        output_texts[0] += text_chunk
                elif stopwords is None:
                    if len(text_chunk) > 0:
                        output_block.append_stream(
                            bytes(text_chunk, encoding="utf-8"))
                        output_texts[0] += text_chunk

        for output_block in output_blocks:
            output_block.finish_stream()
        usage_reports = []

        return usage_reports

    def run(
        self,
        request: PluginRequest[RawBlockAndTagPluginInputWithPreallocatedBlocks]
    ) -> InvocableResponse[StreamCompletePluginOutput]:
        """Run the text generator against all the text, combined"""

        self.config.extend_with_dict(request.data.options, overwrite=True)

        messages = self.prepare_messages(request.data.blocks)

        user_id = self.context.user_id if self.context is not None else "testing"
        usage_reports = self.generate_with_retry(
            messages=messages,
            user=user_id,
            options=request.data.options,
            output_blocks=request.data.output_blocks,
        )

        return InvocableResponse(data=StreamCompletePluginOutput(
            usage=usage_reports))

    def determine_output_block_types(
        self, request: PluginRequest[RawBlockAndTagPluginInput]
    ) -> InvocableResponse[BlockTypePluginOutput]:

        if request.data.options is not None and 'model' in request.data.options:
            raise SteamshipError("Model may not be overridden in options")

        self.config.extend_with_dict(request.data.options, overwrite=True)

        # We return one block per completion-choice we're configured for (config.n)
        # This expectation is coded into the generate_with_retry method
        block_types_to_create = [MimeTypes.TXT.value]
        return InvocableResponse(data=BlockTypePluginOutput(
            block_types_to_create=block_types_to_create, usage=[]))


if __name__ == "__main__":

    with Steamship.temporary_workspace() as client:
        llm = TogetherAIPlugin(client=client,
                               config={
                                   "api_key": os.getenv("TOGETHER_API_KEY"),
                                   "model":
                                   "NousResearch/Nous-Hermes-Llama2-70b"
                               })
        blocks = [
            Block(
                text="Role-play as repeater",
                tags=[Tag(kind=TagKind.ROLE, name=RoleTag.SYSTEM)],
                mime_type=MimeTypes.TXT,
            ),
            Block(
                text="Repeat after me: What's up?",
                tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)],
                mime_type=MimeTypes.TXT,
            )
        ]
        blocks_to_allocate = llm.determine_output_block_types(
            PluginRequest(
                data=RawBlockAndTagPluginInput(blocks=blocks, options={})))
        file = File.create(client, blocks=[])
        output_blocks = []
        for block_type_to_allocate in blocks_to_allocate.data.block_types_to_create:
            assert block_type_to_allocate == MimeTypes.TXT.value
            output_blocks.append(
                Block.create(
                    client,
                    file_id=file.id,
                    mime_type=MimeTypes.TXT.value,
                    streaming=True,
                ))
        response = llm.run(
            PluginRequest(data=RawBlockAndTagPluginInputWithPreallocatedBlocks(
                blocks=blocks,
                options={"stop": ['<|im_end|>', '</s>']},
                output_blocks=output_blocks)))
        print(response)
        result_blocks = [
            Block.get(client, _id=block.id) for block in output_blocks
        ]
        print(result_blocks)
