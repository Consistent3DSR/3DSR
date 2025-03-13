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
    print("")
    metric_musiq = pyiqa.create_metric("musiq").cuda(0)
    metric_niqe = pyiqa.create_metric('niqe', device=device)
    metric_clip = pyiqa.create_metric('clipiqa').to(device)
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
        under_scene_dir = os.listdir(scene_dir)
        iter_dirs = []
        gt_folder = None
        
        for dir in under_scene_dir:
            if "ours" in dir:
                iter_dirs.append(dir)
        
        for dir in iter_dirs:
            method_dir = Path(scene_dir) / dir
            test_dir = Path(method_dir)
            
            for DS_folder in os.listdir(method_dir):
                print("Folder:", DS_folder)
                ds_scale = int(DS_folder.split("_")[-1])
                full_dict[scene_dir][DS_folder] = {}
                per_view_dict[scene_dir][DS_folder] = {}
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
                    
                    if 'nerf_synthetic' in scene_dir:
                        gt_dir = DS_dir / f"gt_{ds_scale}"
                        gt_files = sorted(os.listdir(gt_dir))
                    elif gt_folder_parent is not None:
                        llffhold = 8
                        
                        if ds_scale == 1:
                            gt_folder = os.path.join(gt_folder_parent, f"images")
                        else:
                            gt_folder = os.path.join(gt_folder_parent, f"images_{ds_scale}")
                        
                        gt_files = sorted(os.listdir(gt_folder))
                        all_indices = np.arange(len(gt_files))
                        train_indices = all_indices % llffhold != 0
                        test_indices = all_indices % llffhold == 0

                        if dataset_name == "train":
                            gt_indices = all_indices[train_indices]
                        else:
                            gt_indices = all_indices[test_indices]
                        gt_dir = gt_folder
                        assert len(file_names) == len(gt_indices)
                    else:
                        gt_dir = DS_dir / f"gt_{dataset_name}_{ds_scale}"

                for idx in tqdm(range(len(file_names)), desc="Metric evaluation progress"):
                    if file_names[idx].endswith(".png"):
                        render = tf.to_tensor(Image.open(renders_dir / file_names[idx]).convert('RGB')).unsqueeze(0)[:, :3, :, :].cuda()
                        if gt_folder is not None:
                            gt_file_name = os.path.join(gt_dir, gt_files[gt_indices[idx]])
                            gt = tf.to_tensor(Image.open(gt_file_name).convert('RGB')).unsqueeze(0)[:, :3, :, :].cuda()
                        else:                            
                            gt = tf.to_tensor(Image.open(gt_dir / file_names[idx]).convert('RGB')).unsqueeze(0)[:, :3, :, :].cuda()
                        
                        # try:                            
                        torchvision.utils.save_image(render, "img_output.png")
                        torchvision.utils.save_image(gt, "img_gt.png")
                        # import pdb; pdb.set_trace()
                        # if 'synthetic' in model_paths[0]:
                        #     import pdb; pdb.set_trace()
                        #     image = np.array(Image.open(renders_dir / file_names[idx]).convert('RGBA'))
                        #     gt_img = np.array(Image.open(gt_dir / file_names[idx]).convert('RGBA'))
                        #     alpha = gt_img[:, :, 3]
                        #     image[:, :, 3] = alpha
                        #     image = tf.to_tensor(Image.fromarray(image).convert('RGB')).unsqueeze(0)[:, :3, :, :].cuda()
                        #     image_2 = tf.to_tensor(image).unsqueeze(0)[:, :3, :, :].cuda()
                        #     # gt_img = tf.to_tensor(Image.open(gt_dir / file_names[idx]).convert('RGBA')).cuda()

                        #     alpha = gt_img[-1, :, :]
                        #     mask = alpha > 0
                        #     gt_img = gt_img[:3, :, :]
                        #     mask = mask.unsqueeze(0).expand_as(gt_img)
                        #     gt_info = gt_img[mask]
                        #     renders_info = render[mask.unsqueeze(0)]
                        #     psnr(renders_info, gt_info).mean()
                        ssims.append(ssim(render, gt))
                        psnrs.append(psnr(render, gt))
                        # except:
                        #     gt = Resize(size = render.size()[-2:])(gt)
                        #     torchvision.utils.save_image(render, "img_output.png")
                        #     torchvision.utils.save_image(gt, "img_gt_resized.png")
                        #     ssims.append(ssim(render, gt))
                        #     psnrs.append(psnr(render, gt))
                        #     import pdb; pdb.set_trace()
                        try:
                            # import pdb; pdb.set_trace()
                            lpipss.append(lpips_fn(render, gt).detach())
                            # lpipss.append(lpips(render, gt).detach())
                        except:
                            torch.backends.cudnn.benchmark = True
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
                print("")

                full_dict[scene_dir][DS_folder][dataset_name].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item(),
                                                        "MUSIQ": torch.tensor(musiqs).mean().item(),
                                                        "NIQE": torch.tensor(niqes).mean().item(),
                                                        "CLIPIQA": torch.tensor(clips).mean().item()})
                per_view_dict[scene_dir][DS_folder][dataset_name].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                            "MUSIQ": {name: musiq for musiq, name in zip(torch.tensor(musiqs).tolist(), image_names)},
                                                            "NIQE": {name: niqe for niqe, name in zip(torch.tensor(niqes).tolist(), image_names)},
                                                            "CLIPIQA": {name: clip for clip, name in zip(torch.tensor(clips).tolist(), image_names)}})
            
            with open(scene_dir + f"/results_{dir}.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + f"/per_view_{dir}.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        # except:
        #     print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--resolution', '-r', type=int, default=-1)
    parser.add_argument('--eval_both_dataset', '-e', action="store_true", default=False)
    parser.add_argument('--gt_folder', '-g', type=str, default=None)    
    args = parser.parse_args()
    
    evaluate(args.model_paths, False, args.gt_folder)
