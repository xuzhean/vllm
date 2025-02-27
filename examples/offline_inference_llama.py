from vllm import LLM, SamplingParams
from prompts_gen import generate_random_prompt
import logging
import random

# 启用详细日志
logging.basicConfig(level=logging.INFO)

# 简单的 prompts.
prompts = [
    # "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]

# 随机 prompts.
for i in range(3):
    prompts.append(generate_random_prompt(2))
    
print(prompts)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=128,
    ignore_eos=True,
)

llm = LLM(
    model="/datadisk/llama-7b",
    tensor_parallel_size=1,
    swap_space=24,
    gpu_memory_utilization=0.95,
    enforce_eager=True,
    enable_prefix_caching=True,
)

outputs = llm.generate(prompts, sampling_params)

for out in outputs:
    print(f"Prompt: {out.prompt}\nGenerated: {out.outputs[0].text[:256]}...\n")
