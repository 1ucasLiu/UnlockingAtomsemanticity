from abc import ABC, abstractmethod

import einops
import torch
import torch.nn as nn
from transformer_lens import HookedTransformer

import sae_bench.custom_saes.custom_sae_config as sae_config


class BaseSAE(nn.Module, ABC):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
    ):
        super().__init__()

        # Required parameters
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_in))

        # b_enc and b_dec don't have to be used in the encode/decode methods
        # if your SAE doesn't use biases, leave them as zeros
        # NOTE: core/main.py checks for cosine similarity with b_enc, so it's nice to have the field available
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Required attributes
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype

        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        self.cfg = sae_config.CustomSAEConfig(
            model_name,
            d_in=d_in,
            d_sae=d_sae,
            hook_name=hook_name,
            hook_layer=hook_layer,
        )
        self.cfg.dtype = self.dtype.__str__().split(".")[1]
        self.to(dtype=self.dtype, device=self.device)

    @abstractmethod
    def encode(self, x: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError("Encode method must be implemented by child classes")

    @abstractmethod
    def decode(self, feature_acts: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError("Decode method must be implemented by child classes")

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError("Forward method must be implemented by child classes")

    def to(self, *args, **kwargs):
        """Handle device and dtype updates"""
        super().to(*args, **kwargs)
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)

        if device:
            self.device = device
        if dtype:
            self.dtype = dtype
        return self

    @torch.no_grad()
    def check_decoder_norms(self) -> bool:
        """
        It's important to check that the decoder weights are normalized.
        """
        norms = torch.norm(self.W_dec, dim=1).to(dtype=self.dtype, device=self.device)

        # In bfloat16, it's common to see errors of (1/256) in the norms
        tolerance = (
            1e-2 if self.W_dec.dtype in [torch.bfloat16, torch.float16] else 1e-5
        )

        if torch.allclose(norms, torch.ones_like(norms), atol=tolerance):
            return True
        else:
            #import ipdb;ipdb.set_trace()
            max_diff = torch.max(torch.abs(norms - torch.ones_like(norms)))
            print(f"Decoder weights are not normalized. Max diff: {max_diff.item()}")
           
            print("执行归一化操作...")
            
            # 添加归一化逻辑
            # 1. 避免除以零 - 将零范数替换为1
            safe_norms = torch.where(norms == 0, torch.ones_like(norms), norms)
            
            # 2. 执行归一化
            normalized_W_dec = self.W_dec / safe_norms[:, None]
            
            # 3. 检查归一化结果
            normalized_norms = torch.norm(normalized_W_dec, dim=1)
            normalized_diff = torch.max(torch.abs(normalized_norms - torch.ones_like(normalized_norms)))
            print(f"归一化后最大差异: {normalized_diff.item()}")
            
            # 4. 更新权重矩阵
            self.W_dec.data.copy_(normalized_W_dec)
            
            # 5. 验证归一化是否成功
            final_norms = torch.norm(self.W_dec, dim=1)
            if torch.allclose(final_norms, torch.ones_like(final_norms), atol=tolerance):
                print("归一化成功完成")
                return True
            else:
                print("警告: 归一化后仍未能满足精度要求")
                import ipdb; ipdb.set_trace()
                return False
            #return False

    @torch.no_grad()
    def test_sae(self, model_name: str):
        assert self.W_dec.shape == (self.cfg.d_sae, self.cfg.d_in)
        assert self.W_enc.shape == (self.cfg.d_in, self.cfg.d_sae)

        model = HookedTransformer.from_pretrained(model_name, device=self.device)

        test_input = "The scientist named the population, after their distinctive horn, Ovid’s Unicorn. These four-horned, silver-white unicorns were previously unknown to science"

        _, cache = model.run_with_cache(
            test_input,
            prepend_bos=True,
            names_filter=[self.cfg.hook_name],
            stop_at_layer=self.cfg.hook_layer + 1,
        )
        acts = cache[self.cfg.hook_name]

        encoded_acts = self.encode(acts)
        decoded_acts = self.decode(encoded_acts)

        flattened_acts = einops.rearrange(acts, "b l d -> (b l) d")
        reconstructed_acts = self(flattened_acts)
        # match flattened_acts with decoded_acts
        reconstructed_acts = reconstructed_acts.reshape(acts.shape)

        assert torch.allclose(reconstructed_acts, decoded_acts)

        l0 = (encoded_acts[:, 1:] > 0).float().sum(-1).detach()
        print(f"average l0: {l0.mean().item()}")
