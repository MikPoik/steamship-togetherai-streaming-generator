import together
import os

together.api_key = os.getenv("TOGETHER_API_KEY")
prompt = "<|im_start|>user\n What are Isaac Asimov's Three Laws of Robotics?\n<|im_start|>assistant\n" 

output = together.Complete.create(
  prompt = "<|im_start|>user\n What are Isaac Asimov's Three Laws of Robotics?<|im_end|>\n<|im_start|>assistant\n", 
  model = "teknium/OpenHermes-2-Mistral-7B", 
  max_tokens = 256,
  temperature = 0.8,
  top_k = 60,
  top_p = 0.6,
  repetition_penalty = 1.1,
  stop = ['<|im_end|>', '\n\n']
)
print(output['output']['choices'][0]['text'])


for token in together.Complete.create_streaming(prompt=prompt,
                                                 model = "teknium/OpenHermes-2-Mistral-7B", 
                                                 max_tokens = 256,
                                                 temperature = 0.8,
                                                 top_k = 60,
                                                 top_p = 0.6,
                                                 repetition_penalty = 1.1,
                                                 stop = ['<|im_end|>', '\n\n']):
    print(token, end="", flush=True)
print("\n")

# print generated text
#print(output['output']['choices'][0]['text'])