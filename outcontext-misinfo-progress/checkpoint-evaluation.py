import os
import random
import torch
import numpy as np
import pandas as pd
from inspect_dataloaders import inspect_dataloaders
from models import AITR, MUSE_MLP_CLF
from utils import (eval_dataset, load_ranked,  set_seed, load_ranked_evidence, load_evidence_features, 
                   prepare_dataloader)

def evaluate_checkpoint(checkpoint_path, data_path, evidence_path, eval_path, eval_dataset_name, label_map,
                        encoder='CLIP', encoder_version='ViT-L/14', batch_size=512, 
                        choose_gpu=0, num_workers=1):
    """
    Evaluate a model checkpoint using the original training code structure.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        data_path: Base data path
        evidence_path: Path to evidence data relative to data_path
        verite_path: Path to VERITE data relative to data_path
        encoder: Encoder type ('CLIP')
        encoder_version: Encoder version ('ViT-L/14')
        batch_size: Batch size for evaluation
        choose_gpu: GPU index to use
        num_workers: Number of workers for data loading
    """
    # Set device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device(f"cuda:{choose_gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Process encoder version
    encoder_version_clean = encoder_version.replace("-", "").replace("/", "")
    
    # Determine model type from checkpoint name
    checkpoint_file = os.path.basename(checkpoint_path)
    model_type = 'MUSE_MLP' if 'muse_' in checkpoint_file.lower() and 'aitr' not in checkpoint_file.lower() else 'AITR'
    print(f"Detected model type: {model_type}")
    
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    

    
    print("Loading {} data...".format(eval_dataset_name))
    _test, _image_embeddings, _text_embeddings, X_image_embeddings, X_text_embeddings = load_ranked(
        encoder, encoder_version_clean, eval_path+eval_dataset_name+'/', eval_dataset_name, label_map=label_map
    )



    # Parse parameters from checkpoint name
    params = checkpoint_file.split('_')
    
    # Set default parameters
    emb_dim = 768 if '14' in encoder_version else 512
    tf_h_l = [1, 2, 4, 8]
    tf_dim = 2048  # Default, can be updated based on inspection
    fusion_method = ["concat_1", "add", "sub", "mul"]
    use_evidence = 1
    use_muse = True
    transformer_version = "aitr"
    pooling_method = "attention_pooling"
    sims_to_keep = ['img_txt', 'img_X_img', 'txt_X_img', 'img_X_txt', 'txt_X_txt', 'X_img_X_txt']
    
    # Try to extract parameters from checkpoint name
    if len(params) > 8:
        try:
            # Find use_evidence and use_muse parameters
            for i in range(len(params)):
                if params[i].lower() in ['true', 'false']:
                    use_muse = params[i].lower() == 'true'
                    break
                    
            for i in range(len(params)):
                if params[i].isdigit() and 0 <= int(params[i]) <= 1 and i > 0 and i < len(params) - 1:
                    if params[i+1].lower() in ['true', 'false']:
                        use_evidence = int(params[i])
                        break
                        
            # Try to get transformer version and pooling
            if 'default' in checkpoint_file:
                transformer_version = 'default'
                pooling_method = None
                
            if 'weighted_pooling' in checkpoint_file:
                pooling_method = 'weighted_pooling'
            elif 'max_pooling' in checkpoint_file:
                pooling_method = 'max_pooling'
                
        except Exception as e:
            print(f"Error parsing parameters from filename: {e}")
    
    # Examine state dict to determine parameters
    model_state = checkpoint['model_state_dict']
    
    # Determine transformer dimensions from state dict
    for key in model_state.keys():
        if '.linear2.weight' in key and 'transformer' in key:
            shape = model_state[key].shape
            tf_dim = shape[1]
            print(f"Detected tf_dim = {tf_dim}")
            break
    
    # Determine if model uses MUSE
    if model_type == 'AITR':
        if any('muse_component' in key for key in model_state.keys()):
            use_muse = True
            print("Using MUSE component")
        else:
            use_muse = False
            print("Not using MUSE component")
    
    # Create model based on type
    print(f"Creating {model_type} model...")
    if model_type == 'AITR':
        model = AITR(
            emb_dim=emb_dim,
            fusion_method=fusion_method,
            use_evidence=use_evidence,
            use_muse=use_muse,
            sims_to_keep=sims_to_keep,
            transformer_version=transformer_version,
            tf_h_l=tf_h_l,
            tf_dim=tf_dim,
            pooling_method=pooling_method
        )
    else:
        model = MUSE_MLP_CLF(
            emb_dim=emb_dim,
            sims_to_keep=sims_to_keep
        )
    
    # Load model weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying to load with strict=False...")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Model loaded with some missing keys")
    
    model.to(device)
    model.eval()
    
    # Prepare dataloaders
    print("Preparing dataloaders...")

    

    
    _dataloader = prepare_dataloader(
        input_data=_test,
        visual_features=_image_embeddings,
        textual_features=_text_embeddings,  
        X_visual_features=X_image_embeddings,
        X_textual_features=X_text_embeddings,   
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False,
        use_evidence=use_evidence
       
    )




    print(_test.shape, _image_embeddings.shape, _text_embeddings.shape, X_image_embeddings.shape, X_text_embeddings.shape)
    

    print("\nEvaluating on {}...".format(eval_dataset_name))
    _results = eval_dataset(model, _test, _dataloader, device)

    
    print("\n{} Results:".format(eval_dataset_name))
    print(_results)




    
    return None, _results

def main():
    """Main function to run evaluation on all checkpoints."""
    # Set parameters
    data_path = './data/evidence/'  # Update this path
    evidence_path = 'news_clippings/'

    eval_path = './data/eval/'
    #eval_dataset_name ='DP'
    #label_map={'True': 0, 'Conficted': 1, 'Misleading': 1}
    
    #eval_dataset_name ='FIVEPILS'
    #label_map={True: 0, False: 1, 'out-of-context': 2}

    #eval_dataset_name ='MMFAKEBENCH'
    #label_map={'True': 0, 'Fake': 1, 'out-of-context': 2}

    eval_dataset_name ='VERITE'
    label_map={'true': 0, 'miscaptioned': 1, 'out-of-context': 2}
    
    
    
    
    checkpoints_dir = 'checkpoints'
    
    # Evalutes results both for AITR and MUSE models
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pt')]
    
    all_results = {}
    
    # Evaluate each checkpoint
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_file)
        print(f"\n\n{'='*50}")
        print(f"Evaluating checkpoint: {checkpoint_file}")
        print(f"{'='*50}\n")
        
        #try:
        ds_results = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            data_path=data_path,
            evidence_path=evidence_path,
            eval_path=eval_path, 
            eval_dataset_name = eval_dataset_name,
            label_map = label_map
        )
        
        all_results[checkpoint_file] = {
            'dataset_results': ds_results,
            'dataset': eval_dataset_name
        }
            
       
    
    # Save all results to file
    import json
    with open('evaluation_results_{}.json'.format(eval_dataset_name), 'w') as f:
        json.dump(all_results, f, indent=4, default=str)
    
    print("\nEvaluation complete. Results saved to evaluation_results_{}.json".format(eval_dataset_name))

if __name__ == "__main__":
    main()