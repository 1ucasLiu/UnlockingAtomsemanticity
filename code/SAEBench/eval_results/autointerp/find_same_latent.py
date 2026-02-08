import json

# 文件路径
file1 = "/home/liubo/workspace/SAEBench-main/eval_results/autointerp/fvu_only-1536k-gemma-2-9b-resid_post_layer_8-5.0e+08-20251022_1333_resid_post_layer_8_trainer_0__custom_sae_eval_results.json"
file2 = "/home/liubo/workspace/SAEBench-main/eval_results/autointerp/100fvu_hsic-1536k-sample-gemma-2-9b-resid_post_layer_8-5.0e+08-20251022_1331_resid_post_layer_8_trainer_0__custom_sae_eval_results.json"
#file1 ="/home/liubo/workspace/SAEBench-main/eval_results/autointerp/fvu_only-1024k-gemma-2-9b-resid_post_layer_8-5.0e+08-20251003_2309_resid_post_layer_8_trainer_0__custom_sae_eval_results.json"
#file2 = "/home/liubo/workspace/SAEBench-main/eval_results/autointerp/100fvu_hsic-1024k-sample-gemma-2-9b-resid_post_layer_8-5.0e+08-20251003_2311_resid_post_layer_8_trainer_0__custom_sae_eval_results.json"
# 你要比较的字段名
field_name = "eval_result_unstructured"

# 读取 JSON 文件
with open(file1, "r", encoding="utf-8") as f:
    data1 = json.load(f)

with open(file2, "r", encoding="utf-8") as f:
    data2 = json.load(f)

# 提取指定字段下的 key
keys1 = set(data1.get(field_name, {}).keys())
keys2 = set(data2.get(field_name, {}).keys())

# 找出重复的 key
common_keys = keys1 & keys2

print("eval latents num" ,len(keys1),len(keys2))

# 输出结果
# 这里是随机取得latent num  没有重复很正常
if common_keys:
    print("共有相同的 latent:")
    for k in common_keys:
        print(k)
else:
    print("没有相同latent")
