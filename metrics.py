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

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        if fname.endswith(".png"):
            render = Image.open(renders_dir / fname)
            gt = Image.open(gt_dir / fname)
            renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
            gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
            image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths, eval_both_dataset, gt_folder_parent=None):

    if eval_both_dataset:
        eval_dataset = ["train", "test"]
    else:
        eval_dataset = ["test"]

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    metric_musiq = pyiqa.create_metric("musiq").cuda(0)
    metric_niqe = pyiqa.create_metric('niqe', device=device)
    metric_clip = pyiqa.create_metric('clipiqa').to(device)
    metric_fid = pyiqa.create_metric('fid').to(device=device)
    
    # metric_clip = CLIPImageQualityAssessment()
    # lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').cuda(0)

    for scene_dir in model_paths:
        print("scene_dir: ", scene_dir)
        # try:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}        
        # under_scene_dir = os.listdir(scene_dir)
        under_scene_dir = [name for name in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, name))]
        
        iter_dirs = []
        gt_folder = None
        
        for dir in under_scene_dir:
            if "ours" in dir:
                iter_dirs.append(dir)
        
        for dir in iter_dirs:
            if "-1" in dir:
                print(f"Skipping {dir}, it's a video rendering directory")
                continue
            # Check if dir is a directory
            if not os.path.isdir(os.path.join(scene_dir, dir)):
                print(f"Skipping {dir}, not a directory")
                continue
            if "spiral" in dir:
                print(f"Skipping {dir}, it's a video rendering directory")
                continue
            if "interpolation" in dir:
                print(f"Skipping {dir}, it's a video rendering directory")
                continue
            
            method_dir = Path(scene_dir) / dir
            test_dir = Path(method_dir)
            
            for DS_folder in os.listdir(method_dir):
                print("Folder:", DS_folder)
                ds_scale = int(DS_folder.split("_")[-1])
                full_dict[scene_dir][DS_folder] = {}
                try:
                    per_view_dict[scene_dir][DS_folder] = {}    
                except:
                    import pdb; pdb.set_trace()
                
                full_dict_polytopeonly[scene_dir][DS_folder] = {}
                per_view_dict_polytopeonly[scene_dir][DS_folder] = {}

                for dataset_name in eval_dataset:
                    full_dict[scene_dir][DS_folder][dataset_name] = {}
                    per_view_dict[scene_dir][DS_folder][dataset_name] = {}
                    full_dict_polytopeonly[scene_dir][DS_folder][dataset_name] = {}
                    per_view_dict_polytopeonly[scene_dir][DS_folder][dataset_name] = {}
                    DS_dir = test_dir / DS_folder
                    renders_dir = DS_dir / f"{dataset_name}_preds_{ds_scale}"

                    ssims = []
                    psnrs = []
                    lpipss = []
                    musiqs = []
                    niqes = []
                    clips = []
                    image_names = []
                    try:
                        file_names = os.listdir(renders_dir)
                        file_names = sorted(file_names)
                    except:
                        file_names = []
                    
                    # if 'nerf_synthetic' in scene_dir:
                    #     gt_dir = DS_dir / f"gt_{ds_scale}"
                    #     gt_files = sorted(os.listdir(gt_dir))
                    # elif gt_folder_parent is not None:
                    #     llffhold = 8
                        
                    #     if ds_scale == 1:
                    #         gt_folder = os.path.join(gt_folder_parent, f"images")
                    #     else:
                    #         gt_folder = os.path.join(gt_folder_parent, f"images_{ds_scale}")
                                                
                    #     gt_files = sorted(os.listdir(gt_folder))
                    #     all_indices = np.arange(len(gt_files))
                    #     train_indices = all_indices % llffhold != 0
                    #     test_indices = all_indices % llffhold == 0

                    #     if dataset_name == "train":
                    #         gt_indices = all_indices[train_indices]
                    #     else:
                    #         gt_indices = all_indices[test_indices]
                    #     gt_dir = gt_folder
                    #     assert len(file_names) == len(gt_indices)
                    # else:
                    #     gt_dir = DS_dir / f"gt_{dataset_name}_{ds_scale}"
                    gt_dir = Path(str(renders_dir).replace('test_preds', 'gt'))
                    fid_score = metric_fid(gt_dir, renders_dir)
                for idx in tqdm(range(len(file_names)), desc="Metric evaluation progress"):
                    if file_names[idx].endswith(".png"):
                        render = tf.to_tensor(Image.open(renders_dir / file_names[idx]).convert('RGB')).unsqueeze(0)[:, :3, :, :].cuda()
                        if gt_folder is not None:
                            gt_file_name = os.path.join(gt_dir, gt_files[gt_indices[idx]])
                            gt = tf.to_tensor(Image.open(gt_file_name).convert('RGB')).unsqueeze(0)[:, :3, :, :].cuda()
                        else:                            
                            gt = tf.to_tensor(Image.open(gt_dir / file_names[idx]).convert('RGB')).unsqueeze(0)[:, :3, :, :].cuda()
                        ssims.append(ssim(render, gt))
                        psnrs.append(psnr(render, gt))
                        try:
                            lpipss.append(lpips_fn(render, gt).detach())
                        except:                            
                            import pdb; pdb.set_trace()
                        musiqs.append(metric_musiq(render, gt).detach())
                        niqes.append(metric_niqe(render, gt).float().detach())
                        clips.append(metric_clip(render).detach())
                        image_names.append(file_names[idx])
                    torch.cuda.empty_cache()

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("  MUSIQ: {:>12.7f}".format(torch.tensor(musiqs).mean(), ".5"))
                print("  NIQE: {:>12.7f}".format(torch.tensor(niqes).mean(), ".5"))
                print("  CLIPIQA: {:>12.7f}".format(torch.tensor(clips).mean(), ".5"))
                print(f"FID score: {fid_score}")
                print("")
                
                # ✅ Save JSON
                out_dict = {
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "LPIPS": torch.tensor(lpipss).mean().item(),
                    "MUSIQ": torch.tensor(musiqs).mean().item(),
                    "NIQE": torch.tensor(niqes).mean().item(),
                    "CLIPIQA": torch.tensor(clips).mean().item(),
                    "FID": fid_score
                }
                
                out_per_view_dict = {
                    "SSIM": dict(zip(image_names, torch.tensor(ssims).tolist())),
                    "PSNR": dict(zip(image_names, torch.tensor(psnrs).tolist())),
                    "LPIPS": dict(zip(image_names, torch.tensor(lpipss).tolist())),
                    "MUSIQ": dict(zip(image_names, torch.tensor(musiqs).tolist())),
                    "NIQE": dict(zip(image_names, torch.tensor(niqes).tolist())),
                    "CLIPIQA": dict(zip(image_names, torch.tensor(clips).tolist())),
                }

                output_prefix = os.path.basename(str(renders_dir).rstrip('/'))
                with open(os.path.join(renders_dir, f"results_{output_prefix}.json"), 'w') as fp:
                    json.dump(out_dict, fp, indent=2)
                with open(os.path.join(renders_dir, f"per_view_{output_prefix}.json"), 'w') as fp:
                    json.dump(out_per_view_dict, fp, indent=2)
                
                # Transposed CSV: metrics as columns
                csv_path = os.path.join(renders_dir, f"results_transposed.csv")
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # First row: header
                    writer.writerow(["Model"] + list(out_dict.keys()))
                    # Second row: values
                    writer.writerow([output_prefix] + [f"{val:.4f}" for val in out_dict.values()])
                
                print(f"✅ Transposed results saved to {csv_path}")

            #     full_dict[scene_dir][DS_folder][dataset_name].update({"SSIM": torch.tensor(ssims).mean().item(),
            #                                             "PSNR": torch.tensor(psnrs).mean().item(),
            #                                             "LPIPS": torch.tensor(lpipss).mean().item(),
            #                                             "MUSIQ": torch.tensor(musiqs).mean().item(),
            #                                             "NIQE": torch.tensor(niqes).mean().item(),
            #                                             "CLIPIQA": torch.tensor(clips).mean().item()})
            #     per_view_dict[scene_dir][DS_folder][dataset_name].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
            #                                                 "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
            #                                                 "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
            #                                                 "MUSIQ": {name: musiq for musiq, name in zip(torch.tensor(musiqs).tolist(), image_names)},
            #                                                 "NIQE": {name: niqe for niqe, name in zip(torch.tensor(niqes).tolist(), image_names)},
            #                                                 "CLIPIQA": {name: clip for clip, name in zip(torch.tensor(clips).tolist(), image_names)}})
            
            # with open(scene_dir + f"/results_{dir}.json", 'w') as fp:
            #     json.dump(full_dict[scene_dir], fp, indent=True)
            # with open(scene_dir + f"/per_view_{dir}.json", 'w') as fp:
            #     json.dump(per_view_dict[scene_dir], fp, indent=True)
            
            # # Save transposed CSV: metric names as columns, values in one row
            # csv_path = os.path.join(scene_dir, f"results_{dir}_transposed.csv")
            # with open(csv_path, 'w', newline='') as csvfile:
            #     writer = csv.writer(csvfile)
                
            #     # First row: metric names
            #     writer.writerow(["Metric"] + list(full_dict[scene_dir][DS_folder][dataset_name].keys()))
                
            #     # Second row: values
            #     writer.writerow(
            #         [f"{scene_dir}_{DS_folder}_{dataset_name}"] +
            #         [f"{v:.4f}" for v in full_dict[scene_dir][DS_folder][dataset_name].values()]
            #     )

            # print(f"✅ Transposed results saved to {csv_path}")            

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    torch.backends.cudnn.benchmark = True
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--resolution', '-r', type=int, default=-1)
    parser.add_argument('--eval_both_dataset', '-e', action="store_true", default=False)
    parser.add_argument('--gt_folder', '-g', type=str, default=None)    
    args = parser.parse_args()
    
    evaluate(args.model_paths, False, args.gt_folder)
