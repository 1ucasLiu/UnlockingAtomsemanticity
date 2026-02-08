from pathlib import Path

import torch

from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
from sae_bench.evals.autointerp.main import run_eval

# with open("openai_api_key.txt") as f:
#     api_key = f.read().strip()
api_key="key"
base_url = ""
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

random_seed = 42
batch_size_prompts = 16
n_eval_reconstruction_batches = 20
n_eval_sparsity_variance_batches = 20
context_size = 128
filename = "resid_post_layer_8/trainer_0/ae.pt"
repo_id = "100fvu_hsic-512k-gemma-2-2b-2000-resid_post_layer_8-5.0e+08-20250812_2003"
exclude_special_tokens_from_reconstruction = True
import sae_bench.custom_saes.topk_sae as topk_sae
sae = topk_sae.load_dictionary_learning_topk_sae(
    repo_id=repo_id,
    filename=filename,
    model_name="google/gemma-2-2b",
    device="cuda",  # type: ignore
    dtype=torch.float32,
    layer=8,
    local_dir="hsic" #"transformed_sae"
)
filename = filename.replace("ae.pt","")
selected_saes = [(f"{repo_id}_{filename}", sae)] 
torch.set_grad_enabled(False)

# # ! Demo 1: just 4 specially chosen latents. Must specify n_latents=None explicitly. Also must specify llm_batch_size and llm_batch_size when not running from main.py.
# cfg = AutoInterpEvalConfig(
#     model_name="openai-community/gpt2",#"gpt2-small",
#     n_latents=None,
#     override_latents=[9, 11, 15],
#     llm_dtype="bfloat16",
#     llm_batch_size=32,
# )

output_path = "eval_results_gemma-final/autointerp"
#output_path.mkdir(exist_ok=True)
# results = run_eval(
#     cfg,
#     selected_saes,
#     str(device),
#     api_key,
#     base_url,
#     output_path=str(output_path),
#     save_logs_path=str(save_logs_path),
# )  # type: ignore
# print(results)

# ! Demo 2: 100 randomly chosen latents
cfg = AutoInterpEvalConfig(
    model_name="google/gemma-2-2b", n_latents=50, llm_dtype="bfloat16", llm_batch_size=32
)
save_logs_path =  "eval_results_gemma-final/autointerp_logs.txt"
#save_logs_path.unlink(missing_ok=True)
results = run_eval(
    cfg,
    selected_saes,
    str(device),
    api_key,
    base_url,
    output_path=str(output_path),
    save_logs_path=str(save_logs_path),
)  # type: ignore
print(results)

# python demo.py
