#!/usr/bin/env python3
import os
import torch
import torchvision
import random
import numpy as np
import PIL.Image as PImage
from pathlib import Path

# Disable default parameter init for faster speed
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

from models import VQVAE, build_vae_var
from models.phase_quantization import replace_modules_for_qat, convert_to_inference_mode

def main():
    print("=" * 60)
    print("VAR Image Sampling Demo")
    print("=" * 60)
    
    # ============================================
    # Step 1: Download checkpoints and build models
    # ============================================
    print("\n Building VAR model...")
    
    MODEL_DEPTH = 20  # TODO: =====> please specify MODEL_DEPTH <=====
    assert MODEL_DEPTH in {16, 20, 24, 30}, f"MODEL_DEPTH must be in {{16, 20, 24, 30}}, got {MODEL_DEPTH}"
    
    LOCAL_MODEL_DIR = ""
    vae_ckpt = os.path.join(LOCAL_MODEL_DIR, "vae_ch160v4096z32.pth")
    var_ckpt = os.path.join(LOCAL_MODEL_DIR, f"var_d{MODEL_DEPTH}.pth")
    
    # Build models
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ Using device: {device}")
    
    print("Building VAE and VAR models...")
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    )
    
    # Load checkpoints
    print("Loading VAE checkpoint...")
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    
    checkpoint = torch.load(var_ckpt, map_location='cpu')
    
    # Check if this is a full checkpoint or just model weights
    if 'trainer' in checkpoint:
        model_weights = checkpoint['trainer']['var_wo_ddp']
    else:
        model_weights = checkpoint
        
    ### replace with QAT modules (reorder2)
    replace_modules_for_qat(
        var, 
        method="complex_reorder", 
        skip_head=True, 
        skip_word_embed=True, 
        skip_head_nm=True
        )
    convert_to_inference_mode(var)
    print(f"Inference optimization completed.")

    var.load_state_dict(model_weights, strict=True)

    vae.eval()
    var.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in var.parameters():
        p.requires_grad_(False)
    
    # ============================================
    # Step 2: Sample with classifier-free guidance
    # ============================================
    print("\n Sampling images...")
    
    seed = 0
    num_sampling_steps = 250
    cfg = 4  # classifier-free guidance scale
    
    # ImageNet class labels to sample
    class_labels = (207, 360, 387, 974, 88, 979, 417, 279)
    more_smooth = False  # Set to True for smoother outputs
    
    print(f"CFG scale: {cfg}, Top-k: 900, Top-p: 0.95")
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable faster computation
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')
    
    # Perform sampling
    B = len(class_labels)
    label_B = torch.tensor(class_labels, device=device, dtype=torch.long)
    
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
            recon_B3HW = var.autoregressive_infer_cfg(
                B=B, 
                label_B=label_B, 
                cfg=cfg, 
                top_k=900, 
                top_p=0.95, 
                g_seed=seed, 
                more_smooth=more_smooth
            )
    
    # ============================================
    # Step 3: Save and display results
    # ============================================
    print("\n Processing and saving results...")
    
    # Create grid of images
    chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0)
    chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
    img = PImage.fromarray(chw.astype(np.uint8))
    
    # Save the image
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"var_samples_d{MODEL_DEPTH}_cfg{cfg}.png")
    img.save(output_path)
    print(f"Image saved to: {output_path}")
    
    return img


if __name__ == "__main__":
    img = main()