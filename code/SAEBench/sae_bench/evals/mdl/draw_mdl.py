import json
import os
import torch
import matplotlib.pyplot as plt
import sae_bench.custom_saes.topk_sae as topk_sae
from sae_bench.sae_bench_utils import (
    activation_collection,
    general_utils,
    get_eval_uuid,
    get_sae_bench_version,
    get_sae_lens_version,
)
from sae_bench.sae_bench_utils.graphing_utils import (
    plot_2var_graph,
    plot_2var_graph_dict_size,
    plot_3var_graph,
    plot_correlation_heatmap,
    plot_correlation_scatter,
    plot_interactive_3var_graph,
    plot_training_steps,
)
from sae_bench.sae_bench_utils.sae_selection_utils import select_saes_multiple_patterns


eval_path = "evals/mdl"
image_path = os.path.join(eval_path, "images")


if not os.path.exists(image_path):
    os.makedirs(image_path)

repo_id = ""
filename = "blocks.8.hook_resid_post/ae.pt"
layer = 8

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
llm_dtype = torch.bfloat16


model_name = "openai-community/gpt2"
hook_name = f"blocks.{layer}.hook_resid_post"
sae = topk_sae.load_dictionary_learning_topk_sae(
    repo_id,
    filename,
    model_name,
    device,  # type: ignore
    dtype,
    layer=layer,
    local_dir="transformed_sae"
)
sae_filename = filename.replace("ae.pt","")
selected_saes = [(f"{repo_id}_{sae_filename}", sae)] 
output_path ="eval_results/mdl"
eval_results = {}
for sae_release, sae_object_or_id in selected_saes:
    sae_id, sae, sparsity = general_utils.load_and_format_sae(
        sae_release, sae_object_or_id, device
    )  # type: ignore
    sae = sae.to(device=device, dtype=llm_dtype)

    sae_result_path = general_utils.get_results_filepath(
        output_path, sae_release, sae_id
    )

    with open(sae_result_path) as f:
        single_sae_results = json.load(f)

    eval_results[f"{sae_release}_{sae_id}"] = single_sae_results["eval_results"][-1]
    values = single_sae_results["eval_results"]
    num_bins = [entry["num_bins"] for entry in values]
    mse_loss = [entry["mse_loss"] for entry in values]

    # Plotting the line for the current sae_id
    plt.plot(num_bins, mse_loss, marker="o", label=sae_id)

# Customizing plot
plt.xlabel("Number of Bins (num_bins)")
plt.ylabel("MSE Loss")
plt.title("MSE Loss vs Number of Bins for Each SAE ID")
# plt.legend()
plt.grid(True)
plt.show()
plt.savefig(f"{image_path}/mdl.png")


exit()

k = 1
custom_metric = f"sae_top_{k}_test_accuracy"
custom_metric = "description_length"
# custom_metric = "mse_loss"
# custom_metric = "llm_top_1_test_accuracy"
custom_metric_name = f"description length, {num_bins} bins"
# custom_metric_name = f"k={k}-Sparse Probe Accuracy"
title_3var = f"L0 vs Loss Recovered vs {custom_metric_name}"
title_2var = f"L0 vs {custom_metric_name}"
image_base_name = os.path.join(image_path, custom_metric)

# plot_3var_graph(
#     plotting_results,
#     title_3var,
#     custom_metric,
#     colorbar_label="Custom Metric",
#     output_filename=f"{image_base_name}_3var.png",
# )
plotting_results = eval_results
plot_2var_graph_dict_size(
    eval_results,
    custom_metric,
    title=title_2var,
    output_filename=f"{image_base_name}_2var.png",
)
# plot_interactive_3var_graph(plotting_results, custom_metric)

# At this point, if there's any additional .json files located alongside the ae.pt and eval_results.json
# You can easily adapt them to be included in the plotting_results dictionary by using something similar to add_ae_config_results()
k = 1
custom_metric = f"sae_top_{k}_test_accuracy"
custom_metric = "description_length"
# custom_metric = "mse_loss"
# custom_metric = "llm_top_1_test_accuracy"
custom_metric_name = f"description length, {num_bins} bins"
# custom_metric_name = f"k={k}-Sparse Probe Accuracy"
title_3var = f"L0 vs Loss Recovered vs {custom_metric_name}"
title_2var = f"L0 vs {custom_metric_name}"
image_base_name = os.path.join(image_path, custom_metric)

plot_3var_graph(
    plotting_results,
    title_3var,
    custom_metric,
    colorbar_label="Custom Metric",
    output_filename=f"{image_base_name}_3var.png",
)
plot_2var_graph(
    plotting_results,
    custom_metric,
    title=title_2var,
    output_filename=f"{image_base_name}_2var.png",
)