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
import os
os.environ["NUM_ITERS"] = '30000'

import torch
from scene import Scene
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from PIL import Image
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils import camera_utils
from scene.cameras import Camera
import numpy as np
from utils.camera_utils import *
import cv2
import imageio
import json
from glob import glob
import re, copy
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from scene.pose_utils import generate_ellipse_path, generate_spiral_path

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

@torch.no_grad()
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size, scale_factor):    
    render_path = os.path.join(model_path, "ours_{}".format(iteration), f"DS_{scale_factor}", f"{name}_preds_{scale_factor}")
    gts_path = os.path.join(model_path, "ours_{}".format(iteration), f"DS_{scale_factor}", f"gt_{scale_factor}")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if not os.path.exists(os.path.join(render_path, view.image_name + ".png")):
            rendering = render(view, gaussians, pipeline, background, kernel_size=kernel_size)["render"]
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))            

    print("Rendering completed to {}".format(render_path))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        scale_factor = dataset.resolution
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size
        
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, kernel_size, scale_factor=scale_factor)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, kernel_size, scale_factor=scale_factor)

@torch.no_grad()
def render_vis_videos(dataset : ModelParams, iteration, pipeline : PipelineParams):
    scale_factor = dataset.resolution
    render_path = os.path.join(dataset.model_path, f"ours_{iteration}", f"DS_{scale_factor}", f'vis_video')
    makedirs(render_path, exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    kernel_size = dataset.kernel_size
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    views = scene.getTestCameras()
    view = copy.deepcopy(views[0])

    if 'llff' in dataset.source_path:
        render_poses = generate_spiral_path(np.load(dataset.source_path + '/poses_bounds.npy'))
    else:
        render_poses = generate_ellipse_path(views, n_frames=120)
    
    imgs = []
    for idx, pose in enumerate(tqdm(render_poses)):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        rendering = torch.clamp(render(view, gaussians, pipeline, background, kernel_size=kernel_size)["render"], min=0., max=1.)

        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:04d}.png"))
        imgs.append((rendering.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8))
    
    create_video(imgs, 30, os.path.join(dataset.model_path, f"ours_{iteration}", f"DS_{scale_factor}", "vis_video"))

def create_video(imgs, fps, name):
    writer = imageio.get_writer(f'{name}.mp4', fps=fps)
    for img in imgs:
        writer.append_data(img)

    writer.close()
    print(f"====== Saved video to {name}.mp4")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")    
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--vis_video", action="store_true")
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)    
    
    if args.vis_video:
        render_vis_videos(model.extract(args), args.iteration, pipeline.extract(args))    
    else:
        print("Rendering sets ----------")
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)