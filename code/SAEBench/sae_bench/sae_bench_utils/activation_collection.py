import os
from typing import Any

import einops
import torch
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
)

# Relevant at ctx len 128
LLM_NAME_TO_BATCH_SIZE = {
    "pythia-70m-deduped": 512,
    "pythia-160m-deduped": 256,
    "gemma-2-2b": 32,
    "gemma-2-9b": 32,
    "gemma-2-2b-it": 32,
    "gemma-2-9b-it": 32,
    "openai-community/gpt2":32,
}

LLM_NAME_TO_DTYPE = {
    "pythia-70m-deduped": "float32",
    "pythia-160m-deduped": "float32",
    "gemma-2-2b": "bfloat16",
    "gemma-2-2b-it": "bfloat16",
    "gemma-2-9b": "bfloat16",
    "gemma-2-9b-it": "bfloat16",
    "openai-community/gpt2":"bfloat16",
}


def get_module(model: PreTrainedModel, layer_num: int) -> torch.nn.Module:
    """If missing, refer to sae_bench/sae_bench_utils/misc_notebooks/test_submodule.ipynb for an example of how to get the module for a given model."""
   
    if model.config.architectures[0] == "Gemma2ForCausalLM":
        return model.model.layers[layer_num]  # type: ignore
    elif model.config.architectures[0] == "GPTNeoXForCausalLM":
        return model.gpt_neox.layers[layer_num]  # type: ignore
    
    # code added   Liubo 
    # ------
    # gpt2
    elif model.config.architectures[0] == "GPT2LMHeadModel":
        return model.transformer.h[layer_num]  # type: ignore
    # ------
    else:
        raise ValueError(
            f"Model {model.config.architectures[0]} not supported, please add the appropriate module. See docstring for get_module()"
        )


@torch.no_grad()
def get_layer_activations(
    model: PreTrainedModel,
    target_layer: int,
    inputs: BatchEncoding,
    source_pos_B: torch.Tensor,
) -> torch.Tensor:
    acts_BLD = None

    def gather_target_act_hook(module, inputs, outputs):
        nonlocal acts_BLD
        acts_BLD = outputs[0]
        return outputs

    handle = get_module(model, target_layer).register_forward_hook(
        gather_target_act_hook
    )

    _ = model(
        input_ids=inputs["input_ids"].to(model.device),  # type: ignore
        attention_mask=inputs.get("attention_mask", None),
    )

    handle.remove()

    assert acts_BLD is not None

    acts_BD = acts_BLD[list(range(acts_BLD.shape[0])), source_pos_B, :].clone()

    return acts_BD


@jaxtyped(typechecker=beartype)
@torch.no_grad
def get_bos_pad_eos_mask(
    tokens: Int[torch.Tensor, "dataset_size seq_len"], tokenizer: AutoTokenizer | Any
) -> Bool[torch.Tensor, "dataset_size seq_len"]:
    mask = (
        (tokens == tokenizer.pad_token_id)  # type: ignore
        | (tokens == tokenizer.eos_token_id)  # type: ignore
        | (tokens == tokenizer.bos_token_id)  # type: ignore
    ).to(dtype=torch.bool)
    return ~mask


@jaxtyped(typechecker=beartype)
@torch.no_grad
def get_llm_activations(
    tokens: Int[torch.Tensor, "dataset_size seq_len"],
    model: HookedTransformer,
    batch_size: int,
    layer: int,
    hook_name: str,
    mask_bos_pad_eos_tokens: bool = False,
    show_progress: bool = True,
) -> Float[torch.Tensor, "dataset_size seq_len d_model"]:
    """Collects activations for an LLM model from a given layer for a given set of tokens.
    VERY IMPORTANT NOTE: If mask_bos_pad_eos_tokens is True, we zero out activations for BOS, PAD, and EOS tokens.
    Later, we ignore zeroed activations."""

    all_acts_BLD = []

    for i in tqdm(
        range(0, len(tokens), batch_size),
        desc="Collecting activations",
        disable=not show_progress,
    ):
        tokens_BL = tokens[i : i + batch_size]

        acts_BLD = None

        def activation_hook(resid_BLD: torch.Tensor, hook):
            nonlocal acts_BLD
            acts_BLD = resid_BLD

        model.run_with_hooks(
            tokens_BL, stop_at_layer=layer + 1, fwd_hooks=[(hook_name, activation_hook)]
        )

        if mask_bos_pad_eos_tokens:
            attn_mask_BL = get_bos_pad_eos_mask(tokens_BL, model.tokenizer)
            acts_BLD = acts_BLD * attn_mask_BL[:, :, None]  # type: ignore

        all_acts_BLD.append(acts_BLD)

    return torch.cat(all_acts_BLD, dim=0)


@jaxtyped(typechecker=beartype)
@torch.no_grad
def get_all_llm_activations(
    tokenized_inputs_dict: dict[
        str, dict[str, Int[torch.Tensor, "dataset_size seq_len"]]
    ],
    model: HookedTransformer,
    batch_size: int,
    layer: int,
    hook_name: str,
    mask_bos_pad_eos_tokens: bool = False,
) -> dict[str, Float[torch.Tensor, "dataset_size seq_len d_model"]]:
    """If we have a dictionary of tokenized inputs for different classes, this function collects activations for all classes.
    We assume that the tokenized inputs have both the input_ids and attention_mask keys.
    VERY IMPORTANT NOTE: We zero out masked token activations in this function. Later, we ignore zeroed activations."""
    all_classes_acts_BLD = {}
    #import ipdb;ipdb.set_trace()
    for class_name in tokenized_inputs_dict:
        tokens = tokenized_inputs_dict[class_name]["input_ids"]

        acts_BLD = get_llm_activations(
            tokens, model, batch_size, layer, hook_name, mask_bos_pad_eos_tokens
        )

        all_classes_acts_BLD[class_name] = acts_BLD

    return all_classes_acts_BLD


@jaxtyped(typechecker=beartype)
@torch.no_grad
def collect_sae_activations(
    tokens: Int[torch.Tensor, "dataset_size seq_len"],
    model: HookedTransformer,
    sae: SAE | Any,
    batch_size: int,
    layer: int,
    hook_name: str,
    mask_bos_pad_eos_tokens: bool = False,
    selected_latents: list[int] | None = None,
    activation_dtype: torch.dtype | None = None,
) -> Float[torch.Tensor, "dataset_size seq_len indexed_d_sae"]:
    """Collects SAE activations for a given set of tokens.
    Note: If evaluating many SAEs, it is more efficient to use save_activations() and encode_precomputed_activations()."""
    sae_acts = []

    for i in tqdm(range(0, tokens.shape[0], batch_size)):
        tokens_BL = tokens[i : i + batch_size]
        _, cache = model.run_with_cache(
            tokens_BL, stop_at_layer=layer + 1, names_filter=hook_name
        )
        resid_BLD: Float[torch.Tensor, "batch seq_len d_model"] = cache[hook_name]

        sae_act_BLF: Float[torch.Tensor, "batch seq_len d_sae"] = sae.encode(resid_BLD)

        if selected_latents is not None:
            sae_act_BLF = sae_act_BLF[:, :, selected_latents]

        if mask_bos_pad_eos_tokens:
            attn_mask_BL = get_bos_pad_eos_mask(tokens_BL, model.tokenizer)
        else:
            attn_mask_BL = torch.ones_like(tokens_BL, dtype=torch.bool)

        attn_mask_BL = attn_mask_BL.to(device=sae_act_BLF.device)

        sae_act_BLF = sae_act_BLF * attn_mask_BL[:, :, None]

        if activation_dtype is not None:
            sae_act_BLF = sae_act_BLF.to(dtype=activation_dtype)

        sae_acts.append(sae_act_BLF)

    all_sae_acts_BLF = torch.cat(sae_acts, dim=0)
    return all_sae_acts_BLF


# CODE ADDED
def robust_analyze_token_activations(
    token_string: str,
    tokens: Int[torch.Tensor, "dataset_size seq_len"],
    all_sae_acts_BLF: Float[torch.Tensor, "dataset_size seq_len d_sae"],
    model: HookedTransformer,
    selected_latents: list[int] = None,
    threshold: float = 0.1,
    max_examples: int = 10,
    case_sensitive: bool = False,
    debug: bool = True,
    output_contexts: bool = True  # 新增参数：是否输出上下文
) -> dict:
    """
    鲁棒的分析函数，可以处理各种tokenization情况
    
    参数:
        output_contexts: 是否在找到token时输出上下文
    """
    if debug:
        print(f"\n=== 鲁棒分析: '{token_string}' ===")
    
    # 尝试多种可能的tokenization变体
    variants = [
        token_string,
        f" {token_string}",  # 前面加空格
        token_string.capitalize(),
        token_string.upper(),
    ]
    
    if not case_sensitive:
        variants.append(token_string.lower())
    
    # 收集所有可能的token ID
    possible_token_ids = set()
    token_id_to_variant = {}
    
    for variant in variants:
        token_ids = model.tokenizer.encode(variant)
        for token_id in token_ids:
            decoded = model.tokenizer.decode([token_id])
            # 排除特殊标记
            if decoded not in ['<bos>', '<eos>', '<pad>', '<unk>']:
                possible_token_ids.add(token_id)
                token_id_to_variant[token_id] = variant
    
    if debug:
        print(f"可能的token ID: {possible_token_ids}")
        for token_id in possible_token_ids:
            print(f"  ID {token_id}: '{model.tokenizer.decode([token_id])}' (来自 '{token_id_to_variant[token_id]}')")
    
    # 查找所有匹配的token位置
    all_token_positions = []
    for token_id in possible_token_ids:
        token_positions = torch.nonzero(tokens == token_id)
        for pos in token_positions:
            all_token_positions.append((pos[0].item(), pos[1].item(), token_id))
    
    if not all_token_positions:
        print(f"错误: 数据集中未找到token '{token_string}' 的任何变体")
        return {}
    
    print(f"找到 {len(all_token_positions)} 个'{token_string}'的出现")
    
    # 限制分析的数量
    if len(all_token_positions) > max_examples:
        print(f"只分析前 {max_examples} 个出现")
        all_token_positions = all_token_positions[:max_examples]
    
    # 收集所有出现位置的激活情况
    all_activations = []
    activated_latents_sum = {}
    
    for i, (sample_idx, token_idx, token_id) in enumerate(all_token_positions):
        # 获取目标token文本和上下文
        target_token_text = model.tokenizer.decode([tokens[sample_idx, token_idx].item()])
        context_start = max(0, token_idx - 5)  # 增加上下文范围
        context_end = min(tokens.shape[1], token_idx + 6)  # 增加上下文范围
        context_ids = tokens[sample_idx, context_start:context_end]
        context_text = model.tokenizer.decode(context_ids)
        
        # 输出上下文（无论debug模式如何）
        if output_contexts:
            print(f"\n出现 {i+1}: 样本 {sample_idx}, token位置 {token_idx}")
            print(f"Token: '{target_token_text}' (ID: {token_id})")
            print(f"上下文: '{context_text}'")
        
        # 获取该位置的激活向量
        activations = all_sae_acts_BLF[sample_idx, token_idx, :]
        
        if debug:
            print(f"激活值总和: {activations.sum().item()}")
            print(f"最大激活值: {activations.max().item()}")
            print(f"非零激活数量: {(activations > 0).sum().item()}")
        
        # 找出超过阈值的激活
        active_mask = activations > threshold
        active_indices = torch.nonzero(active_mask).squeeze(-1)
        
        # 获取对应的latent编号和激活值
        activated_latents = {}
        for idx in active_indices:
            latent_idx = selected_latents[idx.item()] if selected_latents is not None else idx.item()
            activation_value = activations[idx].item()
            activated_latents[latent_idx] = activation_value
            
            # 累加激活值用于统计
            if latent_idx in activated_latents_sum:
                activated_latents_sum[latent_idx]["sum"] += activation_value
                activated_latents_sum[latent_idx]["count"] += 1
            else:
                activated_latents_sum[latent_idx] = {"sum": activation_value, "count": 1}
        
        all_activations.append({
            "sample_idx": sample_idx,
            "token_idx": token_idx,
            "token_id": token_id,
            "token_text": target_token_text,
            "context": context_text,
            "activated_latents": activated_latents,
            "activation_sum": activations.sum().item(),
            "max_activation": activations.max().item(),
            "non_zero_count": (activations > 0).sum().item()
        })
    
    # 计算平均激活
    avg_activations = {}
    for latent_idx, data in activated_latents_sum.items():
        avg_activations[latent_idx] = data["sum"] / data["count"]
    
    # 按平均激活值排序
    sorted_avg_activations = sorted(avg_activations.items(), key=lambda x: x[1], reverse=True)
    
    # 准备结果
    result = {
        "token": token_string,
        "possible_token_ids": list(possible_token_ids),
        "total_occurrences": len(all_token_positions),
        "analyzed_occurrences": len(all_activations),
        "activations_by_occurrence": all_activations,
        "avg_activations_by_latent": dict(sorted_avg_activations),
        "top_activated_latents": sorted_avg_activations[:10]
    }
    
    return result
def analyze_token_activations(
    token_string: str,
    tokens: Int[torch.Tensor, "dataset_size seq_len"],
    all_sae_acts_BLF: Float[torch.Tensor, "dataset_size seq_len d_sae"],
    model: HookedTransformer,
    selected_latents: list[int] = None,
    threshold: float = 0.1,
    max_examples: int = 10
) -> dict:
    """
    分析特定文本token在所有潜在维度上的激活情况
    
    参数:
        token_string: 要分析的文本token（如"teacher"）
        tokens: 整个数据集的token ID张量
        all_sae_acts_BLF: SAE激活张量
        model: 使用的模型（用于tokenizer）
        selected_latents: 选择的潜在维度列表，如果为None则使用所有维度
        threshold: 激活阈值
        max_examples: 最多分析多少个该token的出现
    
    返回:
        包含分析结果的字典
    """
    # 将文本token转换为token ID
    token_id = model.tokenizer.encode(token_string)
    if len(token_id) != 1:
        print(f"警告: '{token_string}' 对应多个token ID: {token_id}")
        # 使用第一个token ID
        token_id = token_id[-1]
    else:
        token_id = token_id[0]
    
    print(f"Token '{token_string}' 的ID: {token_id}")
    
    # 找到所有出现该token的位置
    token_positions = torch.nonzero(tokens == token_id)
    
    if len(token_positions) == 0:
        print(f"错误: 数据集中未找到token '{token_string}' (ID: {token_id})")
        return {}
    
    print(f"找到 {len(token_positions)} 个'{token_string}'的出现")
    
    # 限制分析的数量
    if len(token_positions) > max_examples:
        print(f"只分析前 {max_examples} 个出现")
        token_positions = token_positions[:max_examples]
    
    # 收集所有出现位置的激活情况
    all_activations = []
    activated_latents_sum = {}
    
    for i, pos in enumerate(token_positions):
        sample_idx, token_idx = pos[0].item(), pos[1].item()
        
        # 获取该位置的激活向量
        activations = all_sae_acts_BLF[sample_idx, token_idx, :]
        
        # 找出超过阈值的激活
        active_mask = activations > threshold
        active_indices = torch.nonzero(active_mask).squeeze(-1)
        
        # 获取对应的latent编号和激活值
        activated_latents = {}
        for idx in active_indices:
            latent_idx = selected_latents[idx.item()] if selected_latents is not None else idx.item()
            activation_value = activations[idx].item()
            activated_latents[latent_idx] = activation_value
            
            # 累加激活值用于统计
            if latent_idx in activated_latents_sum:
                activated_latents_sum[latent_idx]["sum"] += activation_value
                activated_latents_sum[latent_idx]["count"] += 1
            else:
                activated_latents_sum[latent_idx] = {"sum": activation_value, "count": 1}
        
        # 获取上下文
        context_start = max(0, token_idx - 3)
        context_end = min(tokens.shape[1], token_idx + 4)
        context_ids = tokens[sample_idx, context_start:context_end]
        context_text = model.tokenizer.decode(context_ids)
        
        all_activations.append({
            "sample_idx": sample_idx,
            "token_idx": token_idx,
            "context": context_text,
            "activated_latents": activated_latents,
            "activation_sum": activations.sum().item()
        })
    
    # 计算平均激活
    avg_activations = {}
    for latent_idx, data in activated_latents_sum.items():
        avg_activations[latent_idx] = data["sum"] / data["count"]
    
    # 按平均激活值排序
    sorted_avg_activations = sorted(avg_activations.items(), key=lambda x: x[1], reverse=True)
    
    # 准备结果
    result = {
        "token": token_string,
        "token_id": token_id,
        "total_occurrences": len(token_positions),
        "analyzed_occurrences": len(all_activations),
        "activations_by_occurrence": all_activations,
        "avg_activations_by_latent": dict(sorted_avg_activations),
        "top_activated_latents": sorted_avg_activations[:10]  # 前10个最常激活的latent
    }
    
    return result

def print_token_analysis(result: dict):
    """打印token分析结果的摘要"""
    if not result:
        print("没有分析结果")
        return
    
    print(f"\n=== Token '{result['token']}' (ID: {result['token_id']}) 分析结果 ===")
    print(f"总出现次数: {result['total_occurrences']}")
    print(f"分析的出现次数: {result['analyzed_occurrences']}")
    
    # 打印最常激活的latent
    print(f"\n最常激活的潜在维度 (阈值 > 0.1):")
    print("-" * 50)
    for latent_idx, avg_activation in result["top_activated_latents"]:
        print(f"Latent {latent_idx}: 平均激活值 = {avg_activation:.4f}")
    
    # 打印几个具体示例
    print(f"\n具体示例:")
    print("-" * 50)
    for i, activation in enumerate(result["activations_by_occurrence"][:3]):  # 只显示前3个
        print(f"示例 {i+1}:")
        print(f"  位置: 样本 {activation['sample_idx']}, token {activation['token_idx']}")
        print(f"  上下文: '{activation['context']}'")
        print(f"  激活的latent数量: {len(activation['activated_latents'])}")
        print(f"  激活值总和: {activation['activation_sum']:.4f}")
        
        if activation['activated_latents']:
            print("  激活的latent:")
            for latent_idx, activation_value in activation['activated_latents'].items():
                print(f"    Latent {latent_idx}: {activation_value:.4f}")
        print()
# -----------



@jaxtyped(typechecker=beartype)
@torch.no_grad
def get_feature_activation_sparsity(
    tokens: Int[torch.Tensor, "dataset_size seq_len"],
    model: HookedTransformer,
    sae: SAE | Any,
    batch_size: int,
    layer: int,
    hook_name: str,
    mask_bos_pad_eos_tokens: bool = False,
) -> Float[torch.Tensor, "d_sae"]:
    """Get the activation sparsity for each SAE feature.
    Note: If evaluating many SAEs, it is more efficient to use save_activations() and get the sparsity from the saved activations."""
    device = sae.device
    running_sum_F = torch.zeros(sae.W_dec.shape[0], dtype=torch.float32, device=device)
    total_tokens = 0

    for i in tqdm(range(0, tokens.shape[0], batch_size)):
        tokens_BL = tokens[i : i + batch_size]
        _, cache = model.run_with_cache(
            tokens_BL, stop_at_layer=layer + 1, names_filter=hook_name
        )
        resid_BLD: Float[torch.Tensor, "batch seq_len d_model"] = cache[hook_name]

        sae_act_BLF: Float[torch.Tensor, "batch seq_len d_sae"] = sae.encode(resid_BLD)
        # make act to zero or one
        sae_act_BLF = (sae_act_BLF > 0).to(dtype=torch.float32)

        if mask_bos_pad_eos_tokens:
            attn_mask_BL = get_bos_pad_eos_mask(tokens_BL, model.tokenizer)
        else:
            attn_mask_BL = torch.ones_like(tokens_BL, dtype=torch.bool)

        attn_mask_BL = attn_mask_BL.to(device=sae_act_BLF.device)

        sae_act_BLF = sae_act_BLF * attn_mask_BL[:, :, None]
        total_tokens += attn_mask_BL.sum().item()

        running_sum_F += einops.reduce(sae_act_BLF, "B L F -> F", "sum")

    return running_sum_F / total_tokens


@jaxtyped(typechecker=beartype)
@torch.no_grad
def create_meaned_model_activations(
    all_llm_activations_BLD: dict[
        str, Float[torch.Tensor, "batch_size seq_len d_model"]
    ],
) -> dict[str, Float[torch.Tensor, "batch_size d_model"]]:
    """Mean activations across the sequence length dimension for each class while ignoring padding tokens.
    VERY IMPORTANT NOTE: We assume that the activations have been zeroed out for masked tokens."""

    all_llm_activations_BD = {}
    for class_name in all_llm_activations_BLD:
        acts_BLD = all_llm_activations_BLD[class_name]
        dtype = acts_BLD.dtype

        activations_BL = einops.reduce(acts_BLD, "B L D -> B L", "sum")
        nonzero_acts_BL = (activations_BL != 0.0).to(dtype=dtype)
        nonzero_acts_B = einops.reduce(nonzero_acts_BL, "B L -> B", "sum")

        meaned_acts_BD = (
            einops.reduce(acts_BLD, "B L D -> B D", "sum") / nonzero_acts_B[:, None]
        )
        all_llm_activations_BD[class_name] = meaned_acts_BD

    return all_llm_activations_BD


@jaxtyped(typechecker=beartype)
@torch.no_grad
def get_sae_meaned_activations(
    all_llm_activations_BLD: dict[
        str, Float[torch.Tensor, "batch_size seq_len d_model"]
    ],
    sae: SAE | Any,
    sae_batch_size: int,
) -> dict[str, Float[torch.Tensor, "batch_size d_sae"]]:
    """Encode LLM activations with an SAE and mean across the sequence length dimension for each class while ignoring padding tokens.
    VERY IMPORTANT NOTE: We assume that the activations have been zeroed out for masked tokens."""

    dtype = sae.dtype

    all_sae_activations_BF = {}
    for class_name in all_llm_activations_BLD:
        all_acts_BLD = all_llm_activations_BLD[class_name]

        all_acts_BF = []

        for i in range(0, len(all_acts_BLD), sae_batch_size):
            acts_BLD = all_acts_BLD[i : i + sae_batch_size]
            acts_BLF = sae.encode(acts_BLD)

            activations_BL = einops.reduce(acts_BLD, "B L D -> B L", "sum")
            nonzero_acts_BL = (activations_BL != 0.0).to(dtype=dtype)
            nonzero_acts_B = einops.reduce(nonzero_acts_BL, "B L -> B", "sum")

            acts_BLF = acts_BLF * nonzero_acts_BL[:, :, None]
            acts_BF = (
                einops.reduce(acts_BLF, "B L F -> B F", "sum") / nonzero_acts_B[:, None]
            )
            acts_BF = acts_BF.to(dtype=dtype)

            all_acts_BF.append(acts_BF)

        all_acts_BF = torch.cat(all_acts_BF, dim=0)
        all_sae_activations_BF[class_name] = all_acts_BF

    return all_sae_activations_BF


@jaxtyped(typechecker=beartype)
@torch.no_grad()
def save_activations(
    tokens: Int[torch.Tensor, "dataset_size seq_len"],
    model: HookedTransformer,
    batch_size: int,
    layer: int,
    hook_name: str,
    num_chunks: int,
    save_size: int,
    artifacts_dir: str,
):
    """Save transformer activations to disk in chunks for later processing.

    Saves files named 'activations_XX_of_YY.pt' where XX is the chunk number (1-based)
    and YY is num_chunks. Each file contains a dict with 'activations' and 'tokens' keys."""
    dataset_size = tokens.shape[0]

    for save_idx in range(num_chunks):
        start_idx = save_idx * save_size
        end_idx = min((save_idx + 1) * save_size, dataset_size)
        tokens_SL = tokens[start_idx:end_idx]
        activations_list = []

        for i in tqdm(
            range(0, tokens_SL.shape[0], batch_size),
            desc=f"Saving chunk {save_idx + 1}/{num_chunks}",
        ):
            tokens_BL = tokens_SL[i : i + batch_size]
            _, cache = model.run_with_cache(
                tokens_BL, stop_at_layer=layer + 1, names_filter=hook_name
            )
            resid_BLD = cache[hook_name]

            activations_list.append(resid_BLD.cpu())

        activations_SLD = torch.cat(activations_list, dim=0)
        save_path = os.path.join(
            artifacts_dir, f"activations_{save_idx + 1}_of_{num_chunks}.pt"
        )

        file_contents = {"activations": activations_SLD, "tokens": tokens_SL.cpu()}

        torch.save(file_contents, save_path)
        print(f"Saved activations and tokens to {save_path}")


@jaxtyped(typechecker=beartype)
@torch.no_grad()
def encode_precomputed_activations(
    sae: SAE | Any,
    sae_batch_size: int,
    num_chunks: int,
    activation_dir: str,
    mask_bos_pad_eos_tokens: bool = False,
    selected_latents: list[int] | None = None,
    activation_dtype: torch.dtype | None = None,
) -> Float[torch.Tensor, "dataset_size seq_len d_sae"]:
    """Process saved activations through an SAE model, handling memory constraints through batching.

    This is the second stage of activation processing, meant to be run after save_activations().
    It loads the saved activation chunks, processes them through the SAE, and optionally:
    - Applies masking for special tokens
    - Selects specific SAE features
    - Converts to a specified dtype

    The batched processing allows handling large datasets that don't fit in memory.

    Returns:
        Tensor of encoded activations [dataset_size, seq_len, d_sae]
        If selected_latents is provided, d_sae will be len(selected_latents)
        Otherwise, d_sae will be the full SAE feature dimension"""

    all_sae_acts = []

    for save_idx in range(num_chunks):
        activation_file = os.path.join(
            activation_dir, f"activations_{save_idx + 1}_of_{num_chunks}.pt"
        )
        data = torch.load(activation_file)
        resid_SLD = data["activations"].to(device=sae.device)
        tokens_SL = data["tokens"]

        sae_act_batches = []
        num_samples = resid_SLD.shape[0]

        for batch_start in tqdm(
            range(0, num_samples, sae_batch_size),
            desc=f"Encoding chunk {save_idx + 1}/{num_chunks}",
        ):
            batch_end = min(batch_start + sae_batch_size, num_samples)
            resid_BLD = resid_SLD[batch_start:batch_end]
            tokens_BL = tokens_SL[batch_start:batch_end]

            sae_act_BLF = sae.encode(resid_BLD)

            if selected_latents is not None:
                sae_act_BLF = sae_act_BLF[:, :, selected_latents]

            if mask_bos_pad_eos_tokens:
                attn_mask_BL = get_bos_pad_eos_mask(tokens_BL, sae.model.tokenizer)  # type: ignore
            else:
                attn_mask_BL = torch.ones_like(tokens_BL, dtype=torch.bool)

            attn_mask_BL = attn_mask_BL.to(device=sae_act_BLF.device)
            sae_act_BLF = sae_act_BLF * attn_mask_BL[:, :, None]

            if activation_dtype is not None:
                sae_act_BLF = sae_act_BLF.to(dtype=activation_dtype)

            sae_act_batches.append(sae_act_BLF)

        sae_act_SLF = torch.cat(sae_act_batches, dim=0)
        all_sae_acts.append(sae_act_SLF)

    all_sae_acts = torch.cat(all_sae_acts, dim=0)
    return all_sae_acts
