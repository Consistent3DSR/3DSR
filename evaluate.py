#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
import csv
import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import numpy as np
from torchvision.transforms import Resize
import torchvision
import pyiqa
# from torchmetrics.multimodal import CLIPImageQualityAssessment
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))

def evaluate(img_folder, gt_folder=None):
    metric_musiq = pyiqa.create_metric("musiq").cuda(0)
    metric_niqe = pyiqa.create_metric('niqe', device=device)
    metric_clip = pyiqa.create_metric('clipiqa').to(device=device)
    metric_fid = pyiqa.create_metric('fid').to(device=device)

    img_names_test = sorted([f for f in os.listdir(img_folder) if is_image_file(f)])
    img_names_gt = sorted([f for f in os.listdir(gt_folder) if is_image_file(f)])
    if len(img_names_test) != len(img_names_gt):
        print(f"⚠️ Mismatched file counts! test: {len(img_names_test)} vs gt: {len(img_names_gt)}")

    ssims, psnrs, lpipss = [], [], []
    musiqs, niqes, clips = [], [], []
    image_names = []
    fid_score = metric_fid(args.gt_folder, args.img_folder)
    
    for id in tqdm(range(len(img_names_test)), desc="Evaluating images"):
        if img_names_test[id] not in img_names_gt:
            print(f"⚠️ Skipping {img_names_test[id]}: not found in GT folder")
            continue
                
        gt = tf.to_tensor(Image.open(os.path.join(gt_folder, img_names_gt[id])).convert('RGB')).unsqueeze(0).cuda()
        render = tf.to_tensor(Image.open(os.path.join(img_folder, img_names_test[id])).convert('RGB')).unsqueeze(0).cuda()
        
        ssims.append(ssim(render, gt))
        psnrs.append(psnr(render, gt))

        try:
            lpipss.append(lpips_fn(gt, render).detach())
        except Exception as e:
            print(f"⚠️ LPIPS error on {img_names_test[id]}: {e}")
            lpipss.append(torch.tensor(0.0).cuda())

        musiqs.append(metric_musiq(render).detach())
        niqes.append(metric_niqe(render).float().detach())
        clips.append(metric_clip(render).detach())
        image_names.append(img_names_test[id])
        

    # ✅ Print after loop
    print("\n📊 Final Results:")
    print("  SSIM    :", torch.tensor(ssims).mean().item())
    print("  PSNR    :", torch.tensor(psnrs).mean().item())
    print("  LPIPS   :", torch.tensor(lpipss).mean().item())
    print("  MUSIQ   :", torch.tensor(musiqs).mean().item())
    print("  NIQE    :", torch.tensor(niqes).mean().item())
    print("  CLIPIQA :", torch.tensor(clips).mean().item())
    print(f"FID score: {fid_score}")

    # ✅ Save JSON
    full_dict = {
        "SSIM": torch.tensor(ssims).mean().item(),
        "PSNR": torch.tensor(psnrs).mean().item(),
        "LPIPS": torch.tensor(lpipss).mean().item(),
        "MUSIQ": torch.tensor(musiqs).mean().item(),
        "NIQE": torch.tensor(niqes).mean().item(),
        "CLIPIQA": torch.tensor(clips).mean().item(),
        "FID": fid_score
    }
    per_view_dict = {
        "SSIM": dict(zip(image_names, torch.tensor(ssims).tolist())),
        "PSNR": dict(zip(image_names, torch.tensor(psnrs).tolist())),
        "LPIPS": dict(zip(image_names, torch.tensor(lpipss).tolist())),
        "MUSIQ": dict(zip(image_names, torch.tensor(musiqs).tolist())),
        "NIQE": dict(zip(image_names, torch.tensor(niqes).tolist())),
        "CLIPIQA": dict(zip(image_names, torch.tensor(clips).tolist())),
    }

    output_prefix = os.path.basename(img_folder.rstrip('/'))
    with open(os.path.join(img_folder, f"results_{output_prefix}.json"), 'w') as fp:
        json.dump(full_dict, fp, indent=2)
    with open(os.path.join(img_folder, f"per_view_{output_prefix}.json"), 'w') as fp:
        json.dump(per_view_dict, fp, indent=2)
    
    # Transposed CSV: metrics as columns
    csv_path = os.path.join(img_folder, f"results_transposed.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # First row: header
        writer.writerow(["Model"] + list(full_dict.keys()))
        # Second row: values
        writer.writerow([output_prefix] + [f"{val:.4f}" for val in full_dict.values()])
    
    print(f"✅ Transposed results saved to {csv_path}")
   
if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    # parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    # parser.add_argument('--resolution', '-r', type=int, default=-1)
    # parser.add_argument('--eval_both_dataset', '-e', action="store_true", default=False)
    parser.add_argument('--gt_folder', type=str, default=None)    
    parser.add_argument('--img_folder', type=str, default=None)    
    args = parser.parse_args()
    
    # evaluate(args.model_paths, False, args.gt_folder)
    evaluate(img_folder=args.img_folder, gt_folder=args.gt_folder)