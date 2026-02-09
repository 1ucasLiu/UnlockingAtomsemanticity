import torch as t
from nnsight import LanguageModel
import os
import json
import random

from dictionary_learning.training import trainSAE
from dictionary_learning.trainers.standard import StandardTrainer
from dictionary_learning.trainers.top_k import TopKTrainer, AutoEncoderTopK
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE,BatchTopKTrainer
from dictionary_learning.utils import (
    hf_dataset_to_generator,
    local_dataset_to_generator,
    get_nested_folders,
    load_dictionary,
)
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.dictionary import (
    AutoEncoder,
    GatedAutoEncoder,
    AutoEncoderNew,
    JumpReluAutoEncoder,
)
from dictionary_learning.evaluation import evaluate
from datasets import load_from_disk



DEVICE = "cuda:0"
SAVE_DIR = "gemma_2-2b-layer4"
MODEL_NAME =  "google/gemma-2-2b"
#MODEL_PATH = ""
RANDOM_SEED = 42
LAYER = 4
DATASET_NAME = "EleutherAI/fineweb-edu-dedup-10b"
#DATASET_PATH =""





def sae_training():
    """End to end test for training an SAE. Takes ~2 minutes on an RTX 3090.
    This isn't a nice suite of unit tests, but it's better than nothing.
    I have observed that results can slightly vary with library versions. For full determinism,
    use pytorch 2.5.1 and nnsight 0.3.7.
    Unfortunately an RTX 3090 is also required for full determinism. On an H100 the results are off by ~0.3%, meaning this test will
    not be within the EVAL_TOLERANCE."""

    random.seed(RANDOM_SEED)
    t.manual_seed(RANDOM_SEED)

    model = LanguageModel(MODEL_NAME, dispatch=True, device_map=DEVICE)
    #model = LanguageModel(MODEL_PATH, dispatch=True, device_map=DEVICE)
    context_length = 1024
    llm_batch_size = 16 
    sae_batch_size = 8192
    num_contexts_per_sae_batch = sae_batch_size // context_length

    num_inputs_in_buffer = num_contexts_per_sae_batch * 20

    num_tokens = 500_000_000

    # sae training parameters
    k =1024 
    loss_type = "linear_loss" # ['fvu_loss','nce_loss','hsic_loss','linear_loss']
    sae_type = "BatchTopK"
    term_coff = 10
    expansion_factor = 8

    wandb_project_name="test"
    use_wandb=True 

    steps = int(num_tokens / sae_batch_size)  # Total number of batches to train
    save_steps = [12207,24414,36621,48828]
    
    learning_rate = 1e-4

    # topk sae training parameters
    decay_start = None
    auxk_alpha = 1 / 32
    if "gpt2" in MODEL_NAME:
        submodule = model.transformer.h[LAYER] # gpt2
    elif "gemma" in MODEL_NAME:     
        submodule = model.model.layers[LAYER]
    submodule_name = f"resid_post_layer_{LAYER}"
    from datetime import datetime, timezone, timedelta

    tz = timezone(timedelta(hours=8))
    time_str = datetime.now(tz).strftime("%Y%m%d_%H%M")
    short_model_name = MODEL_NAME.split("/")[1]
    if loss_type == "fvu_loss":
        sub_dir = f"{loss_type}-k{k}-{sae_type}-{short_model_name}-{submodule_name}-{num_tokens:0.1e}-{time_str}"
    elif loss_type in ["hsic_loss","linear_loss",'nce_loss']:
        sub_dir = f"{term_coff}{loss_type}-k{k}-{sae_type}-{short_model_name}-{submodule_name}-{num_tokens:0.1e}-{time_str}"
    else:
        sub_dir="test"
    io = "out"
    activation_dim = model.config.hidden_size

    #generator = hf_dataset_to_generator(DATASET_PATH)
    generator = hf_dataset_to_generator(DATASET_NAME)
    
    activation_buffer = ActivationBuffer(
        generator,
        model,
        submodule,
        n_ctxs=num_inputs_in_buffer,
        ctx_len=context_length,
        refresh_batch_size=llm_batch_size,
        out_batch_size=sae_batch_size,
        io=io,
        d_submodule=activation_dim,
        device=DEVICE,
    )
    
    # create the list of configs
    trainer_configs = []
    trainer_configs.extend(
        [
            {
                "trainer": BatchTopKTrainer,
                "dict_class": BatchTopKSAE,
                "lr": learning_rate,
                "activation_dim": activation_dim,
                "dict_size": expansion_factor * activation_dim,
                "k": k,
                "auxk_alpha": auxk_alpha,  # see Appendix A.2
                "warmup_steps": 0,
                "decay_start": decay_start,  # when does the lr decay start
                "steps": steps,  # when when does training end
                "seed": RANDOM_SEED,
                "wandb_name":sub_dir,    
                "device": DEVICE,
                "layer": LAYER,
                "lm_name": MODEL_NAME,
                "submodule_name": submodule_name,
                "loss_type" : loss_type, 
                "term_coff" : term_coff
            },
        ]
    )
    run_cfg_dict = {                
        "num_tokens":num_tokens ,
                "context_length":context_length,
                "llm_batch_size":llm_batch_size,# 64  
                "expansion_factor":expansion_factor,
                }


    print(f"len trainer configs: {len(trainer_configs)}")
    output_dir = f"{SAVE_DIR}/{sub_dir}/{submodule_name}"

    trainSAE(
        data=activation_buffer,
        trainer_configs=trainer_configs,
        use_wandb=use_wandb,
        wandb_project=wandb_project_name,
        steps=steps,
        run_cfg=run_cfg_dict,
        save_steps=save_steps,
        save_dir=output_dir,
        log_steps=10
    )

    folders = get_nested_folders(output_dir)

    #assert len(folders) == 2

    for folder in folders:
        dictionary, config = load_dictionary(folder, DEVICE)

        assert dictionary is not None
        assert config is not None


sae_training()