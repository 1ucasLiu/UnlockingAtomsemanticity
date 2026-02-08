import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict,Counter
import json
from datasets import load_dataset
import pickle as pkl
import string
import torch.nn.functional as F
import pandas as pd

def batch_iter(dataset, batch_size=64):
    batch = []
    for idx in range(len(dataset)):
        batch.append((idx, dataset[str(idx)]))
        if len(batch) >= batch_size:
            yield batch
            batch=[]
    if batch:
        yield batch

def preproc(batch):
    sent_ids = []
    text_data = []
    for idx, text in batch:
        sent_ids.append(int(idx))
        text_data.append(text)
    
    input_data = tokenizer(
        text_data,
        padding="longest",  # 补齐到最长
        max_length=256,
        truncation=True,
        return_tensors="pt",
        )
    input_data["sent_ids"] = sent_ids
    return input_data

def is_valid_token(token_str):
    """
    返回 True 表示 token 是有效的（不是数字，不是标点，不是特殊 token）。
    """
    # 转小写方便匹配特殊 token
    token_lower = token_str.lower()

    tokens = {"<bos>", "<eos>", "<pad>", "<unk>"," "}
    if token_lower in tokens:
        return False
    
    # 排除纯数字
    if token_str.isdigit():
        return False
    
    # 排除标点符号
    if all(c in string.punctuation for c in token_str):
        return False
    
    return True


device="cuda:0"
model_name = "gemma-2-2b"
if "9b" in model_name:
    local_model_path = "/home/liubo/.cache/huggingface/hub/models--google--gemma-2-9b/snapshots/33c193028431c2fde6c6e51f29e6f17b60cbfac6"
elif "2b" in model_name:
    local_model_path = "/home/liubo/.cache/huggingface/hub/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf"
elif 'gpt2' in model_name:
    local_model_path = '/home/liubo/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e'
elif "bert-base" in model_name:
    local_model_path='/home/liubo/.cache/huggingface/hub/models--google-bert--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594'
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16,
    #attn_implementation="flash_attention_2",
    ).to(device)



tokenizer = AutoTokenizer.from_pretrained(local_model_path)
tokenizer.pad_token = tokenizer.eos_token
dataset_path = 'analyze_sae/data_250_samples.json'

dataset_list = json.loads(open(dataset_path).read())

#print(dataset_list)
#print(dataset_list.keys())
total_tokens = 0
check_tokens = []
for sent in dataset_list:
    tokens = tokenizer.encode(dataset_list[sent], add_special_tokens=False)  # 加上 [CLS]/[SEP] 等特殊 token
    total_tokens += len(tokens)
    check_tokens.append(tokens)

print(f"token num: {total_tokens}")
#print(check_tokens)


# 目标目录下所有sae_name
from dictionary_learning.trainers.top_k import AutoEncoderTopK
#sae = AutoEncoder(activation_dim=2, dict_size=2)
base_dir = os.getcwd()
layer = 4
sae_dir = "/home/liubo/workspace/SAEBench-main/sae_2b/fvu_hsic_full_hsic/topk_layer4"
#sae_dir = "/home/liubo/workspace/SAEBench-main/sae_2b/fvu_hsic_full_hsic/topk_layer4"
#sae_dir = "../SAEBench-main/final_saes_dictionary_learning/500M/hsic/coff_100"
subdirs = [d for d in os.listdir(sae_dir) if os.path.isdir(os.path.join(sae_dir, d))]
cos_sim_dict={}
for sae_name in subdirs:
    print(sae_name)
    # 获取k值
    with open(f"{sae_dir}/{sae_name}/resid_post_layer_{layer}/trainer_0/config.json",'r',encoding="utf8") as f:
        cfg = json.load(f)
    k = cfg["trainer"]['k']
    sae_type = cfg["trainer"]["dict_class"]
    if "fvu_only" in sae_name:
        loss_type = "fvu_only"
    elif "hsic" in sae_name:
        loss_type = "hsic"
    else:
        loss_type=""    

    #print("k ",k)
    ae_path = os.path.join(
        base_dir,
        sae_dir,
        sae_name,
        f"resid_post_layer_{layer}/trainer_0/ae.pt"
    )
    sae_name4label = sae_name.split('-')[0]

    sae = AutoEncoderTopK.from_pretrained(ae_path,k, device=device)


    batch_size=32



    # 用于存储所有 token 样本的 latent 激活向量
    all_token_vectors = [] 
    # 用于存储每个 token 样本的唯一标识符
    # 格式: [ (sent_id_1, index_j_1), (sent_id_2, index_j_2), ... ]
    all_token_ids = []
    global_token_count = 0 
    token_nums={}

    # 将每个有效 token 视为一个样本

    for batch in tqdm(batch_iter(dataset_list, batch_size), total=len(dataset_list)//batch_size+1):
        input_data = preproc(batch)
        input_ids = input_data["input_ids"].to(device)

        sent_ids = input_data["sent_ids"] 
        mask = input_data["attention_mask"].to(device)
        
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
        
        hidden_states = output["hidden_states"][8] 
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # 遍历 batch 中的每个样本（句子）
        for i in range(batch_size):
            current_sent_id = sent_ids[i]
            
            # 遍历序列中的每个 token
            for j in range(seq_len):
                token_id = input_ids[i, j].item()
                
                # 排除特殊 token (如 [CLS], [SEP], [PAD])
                if (not is_valid_token(tokenizer.decode(token_id))) :
                    continue
                if token_id in token_nums:
                    token_nums[token_id]+=1
                else:
                    token_nums[token_id]=1
                token_emb = hidden_states[i, j].unsqueeze(0)
                
                with torch.no_grad():
                    #获得 latent 激活向量 (latent_dim,)
                    encoded = sae.encode(token_emb).squeeze(0)  
                
                all_token_vectors.append(encoded.cpu())
                

                token_str=tokenizer.decode(token_id)
                token_sample_id = f"{token_str}-{token_nums[token_id]}"#f"{current_sent_id}_{j}"
                all_token_ids.append(token_sample_id)
                
                global_token_count += 1

    # 将所有 token 向量拼接成一个矩阵
    # shape: [num_tokens, hidden_dim]
    #print(type(all_token_vectors))
    all_token_vectors = torch.stack(all_token_vectors)

    # 归一化
    all_token_vectors = F.normalize(all_token_vectors, p=2, dim=1)
    # 余弦相似度矩阵
    cos_sim_matrix = all_token_vectors @ all_token_vectors.T   # shape: [num_tokens, num_tokens]



    # 1 上三角 每个feature的最大cos均值
    n = cos_sim_matrix.shape[0]
    cos_sim_matrix_no_diag = cos_sim_matrix - 2.0 * torch.eye(n, device=cos_sim_matrix.device)
    # 每个 feature 只选择与其最相似的另一个 feature 的 cos 值
    # 在 axis=1 上取最大值
    max_similarities = cos_sim_matrix_no_diag.max(dim=1).values.cpu().numpy()
    mean_cos_sim = max_similarities.mean()
    # 2. 上三角cos均值
    # N = cos_sim_matrix.size(0)
    # mean_off_diag = (cos_sim_matrix.sum() - N) / (N * (N - 1))
    print("余弦相似度矩阵 shape:", cos_sim_matrix.shape)

    print("-"*20)
    if "BatchTopK" in sae_name:
        sae_name4label+="_BatchTopK"
    else:
        sae_name4label+="_TopK"
    print(f"{sae_name4label}_layer_{layer}")
    print("average cos sim")
    print(f"\t{mean_cos_sim}")
    cos_sim_dict[f"{sae_name4label}_layer_{layer}"] = mean_cos_sim.item()

# 存为csv
df = pd.DataFrame(list(cos_sim_dict.items()), columns=["key", "value"])
df.to_csv(f"layer{layer}-token-max_cos_sim_means.csv", index=False)
print(f"have been saved into:\n\tlayer{layer}-token-cos_sim.csv")