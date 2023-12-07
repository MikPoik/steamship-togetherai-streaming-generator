"""Test gpt-4 generation via integration tests."""
import json
from typing import Optional
import os
from steamship import Block, File, Steamship, MimeTypes, Tag
from steamship.data import TagKind
from steamship.data.tags.tag_constants import RoleTag, TagValueKey

GENERATOR_HANDLE = "together-ai-llm-test"

client = Steamship(workspace="zephyrstreamtest4")
#with Steamship.temporary_workspace() as steamship:
gpt4 = client.use_plugin(GENERATOR_HANDLE,
                         version="1.0.19",
                         config={
                             "model": "NousResearch/Nous-Hermes-Llama2-70b",
                             "api_key": os.getenv("TOGETHER_API_KEY"),
                             "default_system_prompt": "You are a assistant",
                         })

file = File.create(
    client,
    blocks=[
        Block(
            text="repeat after me: What's up?",
            tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)],
            mime_type=MimeTypes.TXT,
        ),
    ],
)

generate_task = gpt4.generate(input_file_id=file.id, options={"stop": ['<|im_end|>', '</s>']})
#print(generate_task)
generate_task.wait()
#print(generate_task)
output = generate_task.output
print(output)
assert len(output.blocks) == 1
