import torch
from torch import Tensor
from vllm import LLM, SamplingParams
from prompts_gen import generate_random_prompt
import logging
import random
import os

# 环境变量记录常用值，避免大量传递 config
# os.environ['ENABLE_PROMPT_A_STORE'] = 'True'
os.environ['ENABLE_PROMPT_A_STORE'] = 'False'
os.environ['NUM_LAYERS'] = '32'
os.environ['BLOCK_SIZE'] = '16'
os.environ['HIDDEN_SIZE'] = '4096'

# 启用详细日志
logging.basicConfig(level=logging.INFO)
# 固定随机种子
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=128,
    ignore_eos=True,
)

llm = LLM(
    model="/datadisk/llama-7b",
    # tensor_parallel_size=2,
    # worker_use_ray=True,
    tensor_parallel_size=1,
    swap_space=4,
    gpu_memory_utilization=0.95,
)

prompts = [
    # "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
prompts_cached_block = []
num_requests = 3
for i in range(num_requests):
    prompt = generate_random_prompt(3)
    prompts.append(prompt)
    # cached_block = []
    # for layer in range(32):
    #     cached_block.append(torch.rand(5, 16, 4096, dtype=torch.half))
    # prompts_cached_block.append(cached_block)
    
# 开启 prompt a store 时，不使用 cache 加载
if os.getenv('ENABLE_PROMPT_A_STORE') == 'True':
    prompts_cached_block = None
else:
    prompts_cached_block = []
    loaded_tensors = torch.load('tmp.pt')
    num_layers = int(os.getenv('NUM_LAYERS'))
    for i in range(num_requests):
        cached_block = []
        for layer_idx in range(num_layers):
            cached_block.append(loaded_tensors[str(i) + '.' + str(layer_idx)])
        prompts_cached_block.append(cached_block)        

outputs, a_store_dict = llm.generate(prompts, sampling_params, 
                                     prompts_cached_block=prompts_cached_block)

for out in outputs:
    print(f"@@ Prompt: {out.prompt}\n## Generated: {out.outputs[0].text}...\n")

if a_store_dict is not None:
    num_layers = int(os.getenv('NUM_LAYERS'))
    save_tensors_dict = dict()
    for id, a_store in a_store_dict.items():
        print(f"{id=}: {a_store[0]}")
        for layer_idx in range(num_layers):
            save_tensors_dict[str(id) + '.' + str(layer_idx)] = a_store[layer_idx]
    torch.save(save_tensors_dict, 'tmp.pt')
            