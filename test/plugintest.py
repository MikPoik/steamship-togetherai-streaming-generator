"""Test gpt-4 generation via integration tests."""
import json
from typing import Optional

from steamship import Block, File, Steamship, MimeTypes, Tag
from steamship.data import TagKind
from steamship.data.tags.tag_constants import RoleTag, TagValueKey

GENERATOR_HANDLE = "togetherai-llm-dev"

client = Steamship(workspace="zephyrstreamtest4")
#with Steamship.temporary_workspace() as steamship:
gpt4 = client.use_plugin(GENERATOR_HANDLE,
                         version="1.0.3",
                         config={
                             "model": "teknium/OpenHermes-2-Mistral-7B",
                             "api_key": "",
                             "default_system_prompt": "You are a assistant",
                         })

file = File.create(
    client,
    blocks=[
        Block(
            text="tell me a short story",
            tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)],
            mime_type=MimeTypes.TXT,
        ),
    ],
)

generate_task = gpt4.generate(input_file_id=file.id)
#print(generate_task)
generate_task.wait()
#print(generate_task)
output = generate_task.output
assert len(output.blocks) == 1
print(output)
