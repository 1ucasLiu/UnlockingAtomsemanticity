import json
import os

import matplotlib.pyplot as plt


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
def plot_results(
    results_path: str,
    filename: str,
    custom_metric: str,
    custom_metric_name: str,
    layer: int,
):
    filepath = os.path.join(results_path, filename)

    with open(filepath) as f:
        eval_results = json.load(f)

    sae_releases = eval_results["custom_eval_config"]["sae_releases"]

    sae_data = {"basic_eval_results": {}, "sae_config_dictionary_learning": {}}

    for release_name in sae_releases:
        sae_data_filename = f"sae_bench_data/{release_name}_data.json"

        with open(sae_data_filename) as f:
            sae_release_data = json.load(f)

        sae_data["basic_eval_results"].update(sae_release_data["basic_eval_results"])
        sae_data["sae_config_dictionary_learning"].update(
            sae_release_data["sae_config_dictionary_learning"]
        )

    # Gather all values in one dict for plotting
    plotting_results = eval_results

    for sae_name in eval_results:
        plotting_results[sae_name]["l0"] = sae_data["basic_eval_results"][sae_name][
            "l0"
        ]
        # plotting_results[sae_name]["sparsity_penalty"] = get_sparsity_penalty(
        #     sae_data["sae_config_dictionary_learning"][sae_name]
        # )
        plotting_results[sae_name]["frac_recovered"] = sae_data["basic_eval_results"][
            sae_name
        ]["frac_recovered"]

        # Add all trainer info
        plotting_results[sae_name] = (
            plotting_results[sae_name]
            | sae_data["sae_config_dictionary_learning"][sae_name]["trainer"]
        )
        plotting_results[sae_name]["buffer"] = sae_data[
            "sae_config_dictionary_learning"
        ][sae_name]["buffer"]

    title_3var = f"L0 vs Loss Recovered vs {custom_metric_name}"
    title_2var = f"L0 vs {custom_metric_name}, Layer {layer}, Gemma-2-2B"
    image_base_name = os.path.join(image_path, custom_metric)

    # plot_3var_graph(
    #     plotting_results,
    #     title_3var,
    #     custom_metric,
    #     colorbar_label="Custom Metric",
    #     output_filename=f"{image_base_name}_3var.png",
    # )
    plot_2var_graph(
        plotting_results,
        custom_metric,
        title=title_2var,
        output_filename=f"{image_base_name}_2var.png",
        y_label=custom_metric_name,
    )

    if "checkpoints" in filename:
        plot_training_steps(
            plotting_results,
            custom_metric,
            y_label=custom_metric_name,
            title=f"Steps vs {custom_metric_name}",
            output_filename=f"{image_base_name}_steps_vs_diff.png",
        )


eval_path = "./eval_results/sparse_probing"
eval_path = "./eval_results/scr"
image_path = os.path.join(eval_path, "images")

if not os.path.exists(image_path):
    os.makedirs(image_path)


k = 10

if "sparse_probing" in eval_path:
    custom_metric = f"sae_top_{k}_test_accuracy"
    custom_metric_name = f"k={k}-Sparse Probe Accuracy"
elif "scr" in eval_path:
    custom_metric = f"scr_metric_threshold_{k}"
    custom_metric_name = f"SCR {k} latents"
else:
    raise ValueError("Unknown eval path")


layer = 8

filename = f"0.0-8ef-550k_blocks.{layer}.hook_resid_post__custom_sae_eval_results.json"

plot_results(eval_path, filename, custom_metric, custom_metric_name, layer)