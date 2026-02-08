
import os
import sae_bench.sae_bench_utils.graphing_utils as graphing_utils

import matplotlib.pyplot as plt
from PIL import Image
import io
from matplotlib.lines import Line2D

labels_show = graphing_utils.TRAINER_LABELS
trainer_markers = graphing_utils.TRAINER_MARKERS
trainer_colors = graphing_utils.TRAINER_COLORS



def merge_figs(fig_list, save_path="merged.png", dpi=300):
    """
    拼接图像并在底部中央显示统一图例
    """
    spacing = 20
    images = []
    
    legend_handles = []
    legend_labels = []

    for fig in fig_list:
        original_size = fig.get_size_inches()
        # #import ipdb;ipdb.set_trace()
        # scale = 4
        # fig.set_size_inches(original_size[0]*scale, original_size[1]*scale)
        for ax in fig.axes:
            if ax.get_legend() is not None:
                handles, labels = ax.get_legend_handles_labels()
                #import ipdb;ipdb.set_trace()
                for h, trainer in zip(handles, labels):
                    linestyle = "--" if "fvu_only" in trainer else "-"
                    # 使用单点坐标，让图例自动处理线段显示
                    legend_handle = Line2D([0], [0],  # 单点坐标
                                        color=trainer_colors[trainer],
                                        marker=trainer_markers[trainer],
                                        markersize=26,
                                        linestyle=linestyle,
                                        linewidth=4)

                    if trainer not in legend_labels:
                        legend_handles.append(legend_handle)
                        legend_labels.append(trainer)
                ax.get_legend().remove()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')#
        buf.seek(0)
        images.append(Image.open(buf))
    

    n = len(images)
    if n == 0:
        return
    
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    content_width = cols * max_width
    content_height = rows * max_height
    
    # 创建图例
    legend_img = None
    prefix_order = ["Top-K", "Batch Top-K"]  # 控制先后顺序
    sorted_pairs = []
    legend_labels = [labels_show[label] for label in legend_labels]
    legend_labels,legend_handles = zip(*sorted(zip(legend_labels,legend_handles)))

    for prefix in prefix_order:
        for label, handle in zip(legend_labels, legend_handles):
            if label.startswith(prefix + " ") or label==prefix:
                sorted_pairs.append((label, handle))

    # 拆开为两个列表
    legend_labels, legend_handles = zip(*sorted_pairs)
    
    #legend_labels,legend_handles = zip(*sorted(zip(legend_labels,legend_handles)))
    fig_legend = plt.figure(figsize=(10, 6), dpi=dpi)
    #import ipdb;ipdb.set_trace()
    leg = fig_legend.legend(legend_handles, legend_labels, 
                            loc='lower right', ncol=2,
                            handlelength=4,#1.8, 
                            fontsize=23, frameon=True)
    
    # 图例
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_edgecolor('black')
    #leg.get_frame().set_linewidth(0.5)
    
    # 渲染
    buf_legend = io.BytesIO()
    fig_legend.savefig(buf_legend, format='png', dpi=dpi, 
                        bbox_inches='tight', facecolor='white')
    buf_legend.seek(0)
    legend_img = Image.open(buf_legend)
    plt.close(fig_legend)
    

    legend_height = legend_img.height if legend_img else 0
  
    
    right_padding = 250 
    total_width = content_width+right_padding 
    total_height = content_height + legend_height + spacing 
    
    new_im = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    
    #添加图例
    if legend_img:
        legend_x = (total_width - legend_img.width) -160   #右下角
        legend_y = content_height + spacing +10

        new_im.paste(legend_img, (legend_x, legend_y))
    
    y_offset = 0
    
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if idx < n:
                img = images[idx]
                x_offset = c * max_width + (max_width - img.width) // 2
                y_pos = y_offset + r * max_height + (max_height - img.height) // 2
                new_im.paste(img, (x_offset, y_pos))
    #
    if ".png" in save_path:
        new_im.save(save_path, format='PNG', dpi=(dpi, dpi))
    elif ".pdf" in save_path:
        new_im.save(save_path, format='pdf', dpi=(dpi, dpi))
    print(f"图像已保存至: {save_path}")
    
    return new_im

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


image_path ="images/final_images/merge_topk_batchtopk"
#image_path ="images/final_images/topk"
#image_path = 'images/topk_losstype'

final_image = image_path+"/results.pdf"
#final_image = image_path+"/results.png"
if not os.path.exists(image_path):
    os.makedirs(image_path)

#results_folders = ["eval_results/Topk-new-test/Layer17-1003"]
results_folders =["eval_results"] # batchTopK
#results_folders=['eval_results/loss_type_500M']
#results_folders=["eval_results_gemma-final/500M"]   # TopK
eval_folders = []
core_folders = []
#output_folder = "images/relu" # "images/gemma_final_0805"    # 
#output_folder = "eval_results_gemma-final/test_unlearning_seed"  # 测试unlearning 不同seed结果
output_folder = image_path# "eval_results_gemma-final/tpp_subset_data"  # 测试使用不同子数据集所得到的tpp分数差异
eval_types=["core","absorption","ravel",'tpp','unlearning','scr','sparse_probing'] #, "autointerp"
#eval_types=["core"]    #,
#eval_types=["core","absorption","ravel",'autointerp']


# new_sae_key = "vanilla"
# trainer_markers = {
#     "standard": "o",
#     "jumprelu": "X",
#     "topk": "^",
#     "p_anneal": "*",
#     "gated": "d",
#     new_sae_key: "s",  # New SAE
# }

# trainer_colors = {
#     "standard": "blue",
#     "jumprelu": "orange",
#     "topk": "green",
#     "p_anneal": "red",
#     "gated": "purple",
#     new_sae_key: "black",  # New SAE
# }



figs =[]
for eval_type in eval_types:
    eval_folders = []
    if eval_type in ['tpp','scr',"sparse_probing"]:
        # 处于最后综合图考虑，TPP放在最后单独画
        continue
    for results_folder in results_folders:
        eval_folders.append(f"{results_folder}/{eval_type}")
        core_folders.append(f"{results_folder}/core")
    eval_filenames = graphing_utils.find_eval_results_files(eval_folders)
    core_filenames = graphing_utils.find_eval_results_files(core_folders)
  
    print("***",eval_type,"***")
    
    fig = graphing_utils.plot_results(
        eval_filenames,
        core_filenames,
        eval_type,
        f"{output_folder}/{eval_type}",
        k=10,
        trainer_markers=trainer_markers,
        trainer_colors=trainer_colors,
        return_fig=True
    )
    #import ipdb; ipdb.set_trace()
    figs.append(fig)


if "sparse_probing" in eval_types:
    eval_type="sparse_probing" #k=1
    for results_folder in results_folders:
        eval_folders.append(f"{results_folder}/{eval_type}")
        core_folders.append(f"{results_folder}/core")
    eval_filenames = graphing_utils.find_eval_results_files(eval_folders)
    core_filenames = graphing_utils.find_eval_results_files(core_folders)


    fig = graphing_utils.plot_results(
        eval_filenames,
        core_filenames,
        eval_type,
        f"{output_folder}/{eval_type}",
        k=1,
        trainer_markers=trainer_markers,
        trainer_colors=trainer_colors,
        return_fig=True,
    )
    figs.append(fig)
    
if 'scr' in eval_types and 'tpp' in eval_types:
    for eval_type in ['scr','tpp']:
        for results_folder in results_folders:
            eval_folders.append(f"{results_folder}/{eval_type}")
            core_folders.append(f"{results_folder}/core")
        eval_filenames = graphing_utils.find_eval_results_files(eval_folders)
        core_filenames = graphing_utils.find_eval_results_files(core_folders)

        print("***",eval_type,"***")

        fig = graphing_utils.plot_results(
            eval_filenames,
            core_filenames,
            eval_type,
            f"{output_folder}/{eval_type}",
            k=10,
            trainer_markers=trainer_markers,
            trainer_colors=trainer_colors,
            return_fig=True
        )
        #import ipdb; ipdb.set_trace()
        figs.append(fig)


print("Plot finished ")


#final_image = image_path+"/Topk.png"
merge_figs(figs,save_path=final_image)




