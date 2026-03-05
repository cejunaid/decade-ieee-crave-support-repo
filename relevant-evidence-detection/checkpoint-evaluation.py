
# eval_verite_only.py
import os
import numpy as np
import pandas as pd
import torch

from models import RED_DOT
from utils import (
    eval_dataset,
    load_ranked,
    set_seed,
    DatasetIterator_negative_Evidence,
    
)

def infer_tf_layers_from_checkpoint(state_dict: dict) -> int:
    """
    Your run_experiment1 did this for 'transformer.layers.*' keys.
    Falls back to None if not found.
    """
    extracted_layers = 0
    found = False
    for k in state_dict.keys():
        if "transformer.layers" in k:
            # expected like: transformer.layers.0....
            parts = k.split(".")
            try:
                layer_num = int(parts[2])
                extracted_layers = max(extracted_layers, layer_num + 1)
                found = True
            except Exception:
                pass
    return extracted_layers if found else None


def main():
    # -----------------------
    # CONFIG (edit these)
    # -----------------------
    checkpoint_path = "checkpoints/model_news_clippings_balanced_multimodal_0_RED_DOT_1_single_stage_nacc_0_8351_epoch_63.pt"
    
    RED_DOT_version = "single_stage"  # baseline, single_stage, dual_stage, ...
    use_evidence = 1                  # should match training
    use_evidence_neg = 0              # for VERITE generator you used 0 in training
    fuse_evidence = ["concat_1"] if use_evidence else [False]  # match your training choice

    encoder = "CLIP"
    encoder_version = "ViT-L/14"      # ViT-B/32 or ViT-L/14
    
    #data_name = out_csv ="VERITE"
    #verite_path = "/mnt/C4C03417C0341262/decade_paper/data/VERITE/"
    #verite_path = "./data/eval/VERITE/"
    #data_name = out_csv ="NEWSCLIPPINGS"
    #verite_path = "/home/cml-root/Desktop/relevant-evidence-detection/src/data/NEWSCLIPPINGS/"
    #label_map={True: 0, False: 1, 'out-of-context': 2}


    #data_path = './data/evidence/'  # Update this path
    #evidence_path = 'news_clippings/'

    eval_path = './data/eval/'
    
    eval_dataset_name ='DP'
    label_map={'True': 0, 'Conficted': 1, 'Misleading': 1}
    
    #eval_dataset_name ='FIVEPILS'
    #label_map={True: 0, False: 1, 'out-of-context': 2}

    #eval_dataset_name ='MMFAKEBENCH'
    #label_map={'True': 0, 'Fake': 1, 'out-of-context': 2}

    #eval_dataset_name ='VERITE'
    #label_map={'true': 0, 'miscaptioned': 1, 'out-of-context': 2}
    

    choose_gpu = 0
    seed = 0

    # This must match what you trained with (same list order!)
    fusion_method = ["concat_1", "add", "sub", "mul"]

    # These must match training unless you infer layers from checkpoint
    tf_head = 8
    tf_dim = 128

    # -----------------------
    # device / seed
    # -----------------------
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device(f"cuda:{choose_gpu}" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    set_seed(seed)

    # dual_stage uses zero_pad=True in your script
    zero_pad = ("dual_stage" in RED_DOT_version)

    # -----------------------
    # Load VERITE
    # -----------------------
    print("Loading {}...".format(eval_dataset_name))
    _test, _image_embeddings, _text_embeddings, X_image_embeddings, X_text_embeddings = \
            load_ranked(eval_dataset_name, encoder, encoder_version, eval_path+eval_dataset_name+'/', label_map=label_map)

    # Build VERITE generator (same as training)
    _data_generator = DatasetIterator_negative_Evidence(
        _test,
        visual_features=_image_embeddings,
        textual_features=_text_embeddings,
        X_visual_features=X_image_embeddings,
        X_textual_features=X_text_embeddings,
        use_evidence=use_evidence,
        use_evidence_neg=0,          # you used 0 for VERITE
        random_permute=False,
        fuse_evidence=fuse_evidence,
    )

    # -----------------------
    # Determine EMB dim
    # -----------------------
    if encoder_version == "ViT-B/32":
        emb_dim = 512
    elif encoder_version == "ViT-L/14":
        emb_dim = 768
    else:
        raise ValueError(f"Unknown encoder_version: {encoder_version}")

    # -----------------------
    # Load checkpoint
    # -----------------------
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Some people save raw state_dict; others save dict with model_state_dict
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    # Infer tf_layers if possible (like your run_experiment1)
    inferred_layers = infer_tf_layers_from_checkpoint(state_dict)
    if inferred_layers is None:
        # fall back: set this to what you used in training
        tf_layers = 6
        print("Could not infer tf_layers from checkpoint; using fallback:", tf_layers)
    else:
        tf_layers = inferred_layers
        print("Inferred tf_layers from checkpoint:", tf_layers)

    # -----------------------
    # Build model (must match training args)
    # -----------------------
    model = RED_DOT(
        tf_layers=tf_layers,
        tf_head=tf_head,
        tf_dim=tf_dim,
        emb_dim=emb_dim,
        skip_tokens=len(fusion_method) if "concat_1" not in fusion_method else len(fusion_method) + 1,
        use_evidence=use_evidence,
        use_neg_evidence=use_evidence_neg,  # note: VERITE generator used 0 neg
        model_version=RED_DOT_version,
        device=device,
        fuse_evidence=fuse_evidence,
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Loaded model. Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
    model.eval()

    # -----------------------
    # Eval VERITE
    # -----------------------
    print("Running Eval on {}".format(eval_dataset_name))
    with torch.no_grad():
        results = eval_dataset(
            model,
            _data_generator,
            fusion_method,
            use_evidence,
            fuse_evidence,
            device,
            zero_pad=zero_pad
        )
    # Save all results to file
    import json
    with open('evaluation_results_{}.json'.format(eval_dataset_name), 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    print("\nEvaluation complete. Results saved to evaluation_results_{}.json".format(eval_dataset_name))

if __name__ == "__main__":
    main()
