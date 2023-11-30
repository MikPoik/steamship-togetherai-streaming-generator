# Together.ai Plugin for Steamship

This plugin provides access to Together.ai language models for text generation.



### Examples

#### Basic

```python
llm = steamship.use_plugin(together-ai-llm")
task = llm.generate(text=prompt)
task.wait()
for block in task.output.blocks:
    print(block.text)
```

#### With Runtime Parameters

```python
llm = steamship.use_plugin("together-ai-llm")
task = llm.generate(text=prompt, options={"stop": ["6", "7"]})
task.wait()
for block in task.output.blocks:
    print(block.text)
```



