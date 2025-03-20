"""A GPU worker class."""
from collections import defaultdict
import gc
import os
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor
import torch.distributed

from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, VisionLanguageConfig)
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.parallel_utils import pynccl_utils
from vllm.model_executor.parallel_utils.communication_op import (
    broadcast_tensor_dict)
from vllm.model_executor.parallel_utils.custom_all_reduce import init_custom_ar
from vllm.model_executor.parallel_utils.parallel_state import (
    ensure_model_parallel_initialized)
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.model_runner import ModelRunner


class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        vision_language_config: Optional[VisionLanguageConfig] = None,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        self.vision_language_config = vision_language_config
        if self.vision_language_config:
            assert not self.lora_config, (
                "To be tested: vision language model with LoRA settings.")

        self.model_runner = ModelRunner(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            lora_config=self.lora_config,
            kv_cache_dtype=kv_cache_dtype,
            is_driver_worker=is_driver_worker,
            vision_language_config=vision_language_config)
        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.cache_engine = None
        self.gpu_cache = None

    def init_device(self) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_distributed_environment(self.parallel_config, self.rank,
                                     self.distributed_init_method,
                                     self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
        cache_dtype: str,
    ) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model and returns the maximum
        number of GPU and CPU cache blocks that can be allocated.

        Args:
            block_size: The size of the cache block.
            gpu_memory_utilization: The fraction of the total GPU memory to use.
            cpu_swap_space: The size of the CPU swap space in bytes.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        peak_memory = self.init_gpu_memory - free_gpu_memory
        assert peak_memory > 0, (
            "Error in memory profiling. This happens when the GPU memory was "
            "not properly cleaned up before initializing the vLLM instance.")

        cache_block_size = self.get_cache_block_size_bytes(
            block_size, cache_dtype)
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_memory) //
            cache_block_size)
        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()
        gc.collect()
        torch.cuda.empty_cache()
        return num_gpu_blocks, num_cpu_blocks

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config)
        self.gpu_cache = self.cache_engine.gpu_cache
        self.model_runner.set_block_size(self.cache_engine.block_size)

    def warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def cache_swap(
        self,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        # Issue cache operations.
        # TODO(woosuk): Profile swapping overhead and optimize if needed.
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)

    ################ DEMO ################

    def move_prompt_cache(self, a_cache: Tensor, k_cache: Tensor, v_cache: Tensor, 
                          qkv_proj: QKVParallelLinear, rotary_emb: RotaryEmbedding,
                          kv_size: int, positions: Tensor):
        a_gpu = a_cache.to('cuda')
        qkv, _ = qkv_proj(a_gpu)
        q, k, v = qkv.split([kv_size, kv_size, kv_size], dim=-1)
        q, k = rotary_emb(positions, q, k)
        k_cache.copy_(k.reshape(-1))
        v_cache.copy_(v.reshape(-1))
        
    def demo_process_prompts_cache(self, seq_group_metadata_list: List[SequenceGroupMetadata]):
        for seq_group_metadata in seq_group_metadata_list:
            if not seq_group_metadata.is_prompt:
                continue
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]
            
            computed_block_nums = []
            seq_data = seq_group_metadata.seq_data[seq_id]
            block_table = seq_group_metadata.block_tables[seq_id]
            prompt_a_cached_block = seq_data.get_prompt_a_cached_block()
            if prompt_a_cached_block is None:
                continue

            assert prompt_a_cached_block[0].size(0) <= len(block_table)
            for i in range(prompt_a_cached_block[0].size(0)):
                computed_block_nums.append(block_table[i])

            model = self.model_runner.model.model
            block_size = prompt_a_cached_block[0].size(1)
            kv_size = self.gpu_cache[0].size(2) // block_size
            for layer in range(len(prompt_a_cached_block)):
                attn = model.layers[layer].self_attn
                for i in range(prompt_a_cached_block[layer].size(0)):
                    positions = torch.arange(i * block_size, (i + 1) * block_size,
                                             device=self.gpu_cache[0].device)
                    self.move_prompt_cache(prompt_a_cached_block[layer][i],
                                           self.gpu_cache[layer][0][block_table[i]],
                                           self.gpu_cache[layer][1][block_table[i]],
                                           attn.qkv_proj, attn.rotary_emb,
                                           kv_size, positions)
            seq_data.update_num_computed_tokens(block_size * prompt_a_cached_block[0].size(0))
            seq_data.remove_prompt_a_cached_block()

    # NOTE:          
    # cpu act cache: num_layers * [num_blocks, block_size, hidden_size]
    # gpu kv cache: num_layers * [2, num_blocks, block_size * num_kv_heads * head_size]


    def process_prompts_store_demo(self, seq_group_metadata_list: List[SequenceGroupMetadata],
                                   a_store_list: List[Tensor],
                                   a_store_dict: Optional[Dict[int, List[Tensor]]] = None):
        block_size = int(os.getenv("BLOCK_SIZE"))
        hidden_size = int(os.getenv("HIDDEN_SIZE"))
        assert a_store_dict is not None
        
        # a_store 从 GPU 移动到 CPU
        assert a_store_list[0].device.type == 'cuda'
        a_store_list_cpu = []
        for layer_idx in range(len(a_store_list)):
            a_store_list_cpu.append(a_store_list[layer_idx].cpu())
        
        cur_idx = 0
        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]
            
            seq_data = seq_group_metadata.seq_data[seq_id]
            block_num, unblocked_size = seq_data.get_pormpt_a_stored_info()
            print(f"@@ {block_num=}, {unblocked_size=}")
            prompt_a_stored_block = []
            
            for layer_idx in range(len(a_store_list_cpu)):
                blocks = a_store_list_cpu[layer_idx][cur_idx : cur_idx + block_num * block_size]
                blocks = blocks.reshape(block_num, block_size, hidden_size)
                prompt_a_stored_block.append(blocks)
                
            cur_idx += block_num * block_size + unblocked_size
            
            a_store_dict[int(seq_group_metadata.request_id)] = prompt_a_stored_block


    ################ FAST ################

    def process_prompts_cache(self, seq_group_metadata_list: List[SequenceGroupMetadata]):
        # 按层组织需要处理的数据
        layer_batches = defaultdict(lambda: {
            'a_cpu': [],
            'block_tables': [],
            'position_offsets': [],
            'seq_data_list': []
        })

        for seq_group_metadata in seq_group_metadata_list:
            if not seq_group_metadata.is_prompt:
                return False
            
            seq_ids = list(seq_group_metadata.seq_data.keys())
            if not seq_ids:
                continue
            seq_id = seq_ids[0]  # 每个组只有一个序列
            
            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_a_cached_block = seq_data.get_prompt_a_cached_block()
            if prompt_a_cached_block is None:
                continue
            
            block_table = seq_group_metadata.block_tables[seq_id]
            num_blocks = prompt_a_cached_block[0].size(0)
            if num_blocks == 0:
                continue
            
            # 记录全局位置偏移量（考虑已有计算量），一定为 0
            base_offset = seq_data.get_num_computed_tokens()
            block_size = prompt_a_cached_block[0].size(1)
            
            # 按层收集
            for layer_idx in range(len(prompt_a_cached_block)):
                layer_blocks = prompt_a_cached_block[layer_idx]
                layer_batches[layer_idx]['a_cpu'].append(layer_blocks)
                layer_batches[layer_idx]['block_tables'].extend(
                    [block_table[i] for i in range(num_blocks)])
                layer_batches[layer_idx]['position_offsets'].extend(
                    [base_offset + i * block_size for i in range(num_blocks)])
                layer_batches[layer_idx]['seq_data_list'].append(seq_data)
            
            # 更新序列
            seq_data.remove_prompt_a_cached_block()
            # NOTE: 暂时不更新 computed_tokens 是因为该版本的 vllm 不支持 chunked prefill
            #       目前仅考虑完整缓存部分的 prefill 时间对比：使用激活值缓存重算 kv / 直接缓存 kv / 完整算 kv
            # seq_data.update_num_computed_tokens(block_size * num_blocks)

        # 按层批量处理
        model = self.model_runner.model.model
        for layer_idx, batch_data in layer_batches.items():
            if not batch_data['a_cpu']:
                continue

            # 合并CPU激活值
            a_cpu = torch.cat([block for blocks in batch_data['a_cpu'] for block in blocks], dim=0)
            
            # 异步传输到GPU
            a_gpu = a_cpu.to('cuda', non_blocking=True)
            
            # 批量计算QKV投影
            attn = model.layers[layer_idx].self_attn
            qkv, _ = attn.qkv_proj(a_gpu)
            block_size = batch_data['a_cpu'][0].size(1)
            kv_size = self.gpu_cache[0].size(2) // block_size
            assert(block_size == 16)
            assert(kv_size == 4096)
            q, k, v = qkv.split([kv_size, kv_size, kv_size], dim=-1)
            
            # 批量生成位置编码
            positions = torch.cat([
                torch.arange(offset, offset + block_size, device=a_gpu.device)
                for offset in batch_data['position_offsets']
            ])
            q, k = attn.rotary_emb(positions, q, k)

            # 批量填充KV缓存
            total_blocks = len(batch_data['block_tables'])
            for i in range(total_blocks):
                block_start = i * block_size
                block_end = block_start + block_size
                
                k_block = k[block_start:block_end].contiguous().view(-1)
                v_block = v[block_start:block_end].contiguous().view(-1)
                
                target_block = batch_data['block_tables'][i]
                self.gpu_cache[layer_idx][0][target_block].copy_(k_block)
                self.gpu_cache[layer_idx][1][target_block].copy_(v_block)

        return True

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None,
        blocks_to_swap_in: Optional[Dict[int, int]] = None,
        blocks_to_swap_out: Optional[Dict[int, int]] = None,
        blocks_to_copy: Optional[Dict[int, List[int]]] = None,
        a_store_dict: Optional[Dict[int, List[Tensor]]] = None,
    ) -> Optional[SamplerOutput]:
        if self.is_driver_worker:
            assert seq_group_metadata_list is not None
            num_seq_groups = len(seq_group_metadata_list)
            assert blocks_to_swap_in is not None
            assert blocks_to_swap_out is not None
            assert blocks_to_copy is not None
            data = {
                "num_seq_groups": num_seq_groups,
                "blocks_to_swap_in": blocks_to_swap_in,
                "blocks_to_swap_out": blocks_to_swap_out,
                "blocks_to_copy": blocks_to_copy,
                "seq_group_metadata_list": seq_group_metadata_list,
            }
            broadcast_tensor_dict(data, src=0)
        else:
            data = broadcast_tensor_dict(src=0)
            num_seq_groups = data["num_seq_groups"]
            blocks_to_swap_in = data["blocks_to_swap_in"]
            blocks_to_swap_out = data["blocks_to_swap_out"]
            blocks_to_copy = data["blocks_to_copy"]
            seq_group_metadata_list = data["seq_group_metadata_list"]

        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return {}

        # 处理所有输入的 prompt a cache，同时返回是否是 prompt
        is_prompt = self.process_prompts_cache(seq_group_metadata_list)

        # TODO: 按 layer 返回 hook
        # TODO: 多卡间使用 nvlink 通信激活值
        
        # 开启 prompt a store
        num_layers = int(os.getenv('NUM_LAYERS'))
        # print(f"@ {is_prompt}, {os.getenv('ENABLE_PROMPT_A_STORE', 'False')=}")
        a_store_list = [torch.empty(()) for i in range(num_layers)] \
            if is_prompt and (os.getenv('ENABLE_PROMPT_A_STORE', 'False') == 'True') else None

        output, a_store_list = self.model_runner.execute_model(seq_group_metadata_list,
                                                 self.gpu_cache, a_store_list=a_store_list)

        # TODO: execute 结束，将激活值存下来（已实现 demo）
        # 实现方式：prompt_a_cached_block 并列开 prompt_a_stored_block
        # 开关控制是否写回；可以用来检查正确性，同时模拟多轮对话

        if a_store_list is not None:
            self.process_prompts_store_demo(seq_group_metadata_list, a_store_list, a_store_dict)

        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()

    @property
    def max_model_len(self) -> int:
        return self.model_config.max_model_len

    @property
    def vocab_size(self) -> int:
        return self.model_runner.vocab_size

    def get_cache_block_size_bytes(self, block_size: int,
                                   cache_dtype: str) -> int:
        """Get the size of the KV cache block size in bytes.
        """
        return CacheEngine.get_cache_block_size(block_size, cache_dtype,
                                                self.model_config,
                                                self.parallel_config)


def init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size}).")
    elif not distributed_init_method:
        raise ValueError(
            "distributed_init_method must be set if torch.distributed "
            "is not already initialized")
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    if pynccl_utils.is_initialized():
        pynccl_world_size = pynccl_utils.get_world_size()
        if pynccl_world_size != parallel_config.world_size:
            raise RuntimeError(
                "pynccl is already initialized but the pynccl world "
                "size does not match parallel_config.world_size "
                f"({pynccl_world_size} vs. {parallel_config.world_size}).")
    elif parallel_config.world_size > 1:
        # NOTE(woosuk): We don't initialize pynccl process group when world size
        # is 1.
        pynccl_utils.init_process_group(
            world_size=parallel_config.world_size,
            local_rank=local_rank,
            rank=rank,
            init_method=distributed_init_method,
        )

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    if pynccl_utils.is_initialized():
        pynccl_utils.all_reduce(torch.zeros(1).cuda())
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)

    # Initialize a custom fast all-reduce implementation.
    if not parallel_config.disable_custom_all_reduce:
        init_custom_ar()


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")
