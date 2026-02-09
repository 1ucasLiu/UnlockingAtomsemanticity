import time 
import torch 
import os

import sae_bench.custom_saes.topk_sae as topk_sae
import sae_bench.custom_saes.relu_sae as relu_sae
import sae_bench.custom_saes.batch_topk_sae as batch_topk_sae
import sae_bench.evals.core.main as core
import sae_bench.evals.absorption.main as absorption
import sae_bench.evals.sparse_probing.main as sparse_probing

import sae_bench.evals.scr_and_tpp.main as scr_and_tpp
import sae_bench.evals.mdl.main as mdl
import sae_bench.evals.ravel.main as ravel

from sae_bench.evals.absorption.eval_config import AbsorptionEvalConfig
from sae_bench.evals.scr_and_tpp.eval_config import ScrAndTppEvalConfig
from sae_bench.evals.sparse_probing.eval_config import SparseProbingEvalConfig
from sae_bench.evals.mdl.eval_config import MDLEvalConfig
from sae_bench.evals.ravel.eval_config import RAVELEvalConfig

from loguru import logger
def evaluate(repo_id,filename,layer,model_name,sae_type,sae_local_dir,eval_folder,eval_types,device,dtype,llm_dtype):
    random_seed = 42
    batch_size_prompts = 16
    n_eval_reconstruction_batches = 20
    n_eval_sparsity_variance_batches = 20
    context_size = 128

    exclude_special_tokens_from_reconstruction = True

    # Relu
    if sae_type =='relu':
        sae = relu_sae.load_dictionary_learning_relu_sae(
            repo_id,
            filename,
            model_name,
            device,  # type: ignore
            dtype,
            layer=layer,
            local_dir=sae_local_dir
        )
    elif sae_type == "topk":
    # TopK
        sae = topk_sae.load_dictionary_learning_topk_sae(
            repo_id,
            filename,
            model_name,
            device,  # type: ignore
            dtype,
            layer=layer,
            local_dir=sae_local_dir
        )
    elif sae_type == "batchTopk":
        # TopK
        sae = batch_topk_sae.load_dictionary_learning_batch_topk_sae(
            repo_id,
            filename,
            model_name,
            device,  # type: ignore
            dtype,
            layer=layer,
            local_dir=sae_local_dir
        )
    filename = filename.replace("ae.pt","")
    selected_saes = [(f"{repo_id}_{filename}", sae)] 


    # core 
    if 'core' in eval_types:
        output_folder=eval_folder+"/"+"core"
        os.makedirs(output_folder, exist_ok=True)
        for sae_name, sae in selected_saes :
            sae.cfg.dtype = "bfloat16"
            
        core.multiple_evals(
            selected_saes=selected_saes ,
            n_eval_reconstruction_batches=n_eval_reconstruction_batches,
            n_eval_sparsity_variance_batches=n_eval_sparsity_variance_batches,
            eval_batch_size_prompts=batch_size_prompts,
            exclude_special_tokens_from_reconstruction=exclude_special_tokens_from_reconstruction,
            compute_featurewise_density_statistics=True,
            compute_featurewise_weight_based_metrics=True,
            dataset="Skylion007/openwebtext",
            context_size=context_size,
            output_folder=output_folder,
            verbose=True,
            dtype=llm_dtype,
        )
        print("..core evaluation finished..")

    # absorption
    if "absorption" in eval_types:
        output_folder = eval_folder+"/"+"absorption"
        os.makedirs(output_folder, exist_ok=True)
        config = AbsorptionEvalConfig(
            random_seed=random_seed,
            model_name=model_name,
        )
        config.llm_batch_size = 16 
        config.llm_dtype = torch.bfloat16
        _ = absorption.run_eval(
            config,
            selected_saes,
            device,
            output_folder,
            force_rerun=True,
        )
        print("..absorption evaluation finished..")

    # # scr_and_tpp
    if "scr" in eval_types:
        output_folder = eval_folder+"/"
        config = ScrAndTppEvalConfig(
            random_seed=random_seed,
            model_name=model_name,
            perform_scr=True,
        )

        config.llm_batch_size = 1 
        config.llm_dtype = torch.bfloat16
        # create output folder
        os.makedirs(output_folder, exist_ok=True)

        # run the evaluation on all selected SAEs
        _ = scr_and_tpp.run_eval(
            config,
            selected_saes,
            device,
            output_folder,
            force_rerun=True,
            clean_up_activations=False,
            save_activations=True,
        )


    # tpp
    if 'tpp' in eval_types:
        print("TPP....")
        output_folder = eval_folder+"/"
        config = ScrAndTppEvalConfig(
            random_seed=random_seed,
            model_name=model_name,
            perform_scr=False,
        )

        config.llm_batch_size = 1 
        config.llm_dtype = torch.bfloat16
        # create output folder
        os.makedirs(output_folder, exist_ok=True)

        _ = scr_and_tpp.run_eval(
            config,
            selected_saes,
            device,
            output_folder,
            force_rerun=True,
            clean_up_activations=False,
            save_activations=True,
        )
        print("..scr_and_tpp evaluation finished..")


    # sparse_probing
    if 'sparse_probing' in eval_types:
        output_folder = eval_folder+"/"+"sparse_probing"
        config = SparseProbingEvalConfig(
            random_seed=random_seed,
            model_name=model_name,
            k_values=[1, 2, 5, 10]
        )

        config.llm_batch_size = 32 
        config.llm_dtype = torch.bfloat16 

        # create output folder
        os.makedirs(output_folder, exist_ok=True)

        # run the evaluation on all selected SAEs
        _ = sparse_probing.run_eval(
            config,
            selected_saes,
            device,
            output_folder,
            force_rerun=True,
            clean_up_activations=False,
            save_activations=True,
        )
        print("..sparse_probing evaluation finished..")



    if "ravel" in eval_types:
        config = RAVELEvalConfig(
            random_seed=random_seed,
            model_name=model_name,
        )

        config.llm_batch_size = 32
        config.llm_dtype = torch.bfloat16
        output_folder=eval_folder+"/"+"ravel"
        # create output folder
        os.makedirs(output_folder, exist_ok=True)

        # run the evaluation on all selected SAEs
        results_dict = ravel.run_eval(
            config,
            selected_saes,
            device,
            output_folder,
            force_rerun=True,
        )
    
    if "unlearning" in eval_types:
        from sae_bench.evals.unlearning.eval_config import UnlearningEvalConfig
        import sae_bench.evals.unlearning.main as unlearning
        config = UnlearningEvalConfig(
            random_seed=random_seed,
            model_name=model_name,
        )

        config.llm_batch_size = 2 #activation_collection.LLM_NAME_TO_BATCH_SIZE[config.model_name]
        config.llm_dtype = torch.bfloat16 #activation_collection.LLM_NAME_TO_DTYPE[config.model_name]


        # create output folder
        output_folder=eval_folder+"/"+"unlearning"
        os.makedirs(output_folder, exist_ok=True)

        # run the evaluation on all selected SAEs
        results_dict = unlearning.run_eval(
            config,
            selected_saes,
            device,
            output_folder,
            force_rerun=True,
            clean_up_artifacts=False,
        )

    if "autointerp" in eval_types:
        from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
        from sae_bench.evals.autointerp.main import run_eval
        from pathlib import Path
        api_key="key"
        base_url = ""
        save_logs_path = Path(__file__).parent / "logs_4.txt"
        save_logs_path.unlink(missing_ok=True)
        # output_path = Path(__file__).parent / "autointerp"
        # output_path.mkdir(exist_ok=True)
        output_folder=eval_folder+"/"+"autointerp"
        cfg = AutoInterpEvalConfig(
        model_name="google/gemma-2-2b", n_latents=100, llm_dtype="bfloat16", llm_batch_size=32)
        save_logs_path = Path(__file__).parent / f"logs_100_random_{repo_id}.txt"
        save_logs_path.unlink(missing_ok=True)
        try:
            results = run_eval(
                cfg,
                selected_saes,
                str(device),
                api_key,
                base_url,
                output_path=output_folder,#str(output_path),
                save_logs_path=str(save_logs_path),
            )  # type: ignore
        except Exception as e:
            print(f"Wrong!/n  {e}")

layer = 8
rootdir = "train_results"

for sae_name in os.listdir(rootdir):
    if os.path.isdir(os.path.join(rootdir, sae_name)):
        print(" * ",sae_name)
        evaluate(repo_id = sae_name,
                filename =  f"resid_post_layer_{layer}/trainer_0/ae.pt",   
                layer = layer,
                model_name = "google/gemma-2-2b",#"openai-community/gpt2"
                sae_local_dir=rootdir,
                sae_type='batchTopk',#  ['topk','batchTopk']
                eval_folder ="eval_results",
                eval_types =['tpp','scr','core',"absorption",'sparse_probing','ravel','unlearning'],
                device = "cuda" if torch.cuda.is_available() else "cpu",
                dtype = torch.float32,#
                llm_dtype = torch.bfloat16)
    
