import torch
from torch import Tensor
from vllm import LLM, SamplingParams
from prompts_gen import generate_random_prompt
import logging
import random

# 启用详细日志
logging.basicConfig(level=logging.INFO)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=128,
    ignore_eos=True,
)

print("###e")

llm = LLM(
    model="/datadisk/llama-7b",
    tensor_parallel_size=1,
    swap_space=4,
    gpu_memory_utilization=0.95,
)

print("###e")

prompts = [
    # "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
prompts_cached_block = []
for i in range(3):
    prompt = generate_random_prompt()
    prompts.append(prompt)
    prompts_cached_block.append(torch.ones([2, 4]) * i)
    
print(prompts, prompts_cached_block)

outputs = llm.generate(prompts, sampling_params, 
                       prompts_cached_block=prompts_cached_block)

for out in outputs:
    print(f"Prompt: {out.prompt}\nGenerated: {out.outputs[0].text}...\n")
