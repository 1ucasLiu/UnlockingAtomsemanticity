# 假设你的文本保存在 text.txt
import json
input_file = "eval_results/autointerp/100fvu_hsic-1024k-sample-gemma-2-9b-resid_post_layer_8-5.0e+08-20251003_2311_resid_post_layer_8_trainer_0__custom_sae_eval_results.json"
input_file = "/home/liubo/workspace/SAEBench-main/eval_results/autointerp/fvu_only-1024k-gemma-2-9b-resid_post_layer_8-5.0e+08-20251003_2309_resid_post_layer_8_trainer_0__custom_sae_eval_results.json"
input_file="/home/liubo/workspace/SAEBench-main/eval_results/autointerp/fvu_only-1024k-gemma-2-9b-resid_post_layer_8-5.0e+08-20251003_2309_resid_post_layer_8_trainer_0__custom_sae_eval_results.json"
input_file="/home/liubo/workspace/SAEBench-main/eval_results/autointerp/100fvu_hsic-1024k-sample-gemma-2-9b-resid_post_layer_8-5.0e+08-20251003_2311_resid_post_layer_8_trainer_0__custom_sae_eval_results.json"
latent_num=869
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

raw_text = data["eval_result_unstructured"][str(latent_num)]['logs']

# 将 \n 转换成换行，保留其它字符
decoded_text = raw_text.replace("\\n", "\n")  # 仅替换换行

# 保存或打印
print(decoded_text)
prefix = f"check_latent{latent_num}"
if "hsic" in input_file:
    prefix+="hsic"
output_file = prefix+".txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(decoded_text)
