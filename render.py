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

import torch
from scene import Scene
import os
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

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size, scale_factor):    
    # render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"test_preds_{scale_factor}")
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"gt_{scale_factor}")
    render_path = os.path.join(model_path, "ours_{}".format(iteration), f"DS_{scale_factor}", f"{name}_preds_{scale_factor}")
    gts_path = os.path.join(model_path, "ours_{}".format(iteration), f"DS_{scale_factor}", f"gt_{scale_factor}")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if not os.path.exists(os.path.join(render_path, '{0:05d}'.format(idx) + ".png")):
            rendering = render(view, gaussians, pipeline, background, kernel_size=kernel_size)["render"]
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():        
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        # Super resolving gaussians
        # gaussians.super_resolving_gaussians(2, rendering=True)
        scale_factor = dataset.resolution
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size
        # TODO
        # Extract the camera poses to be the shape=[batch, 3, 4]
        # Apply generate_ellipse_path() in camera_utils to generate camera path

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, kernel_size, scale_factor=scale_factor)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, kernel_size, scale_factor=scale_factor)

def render_video(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        scale_factor = dataset.resolution
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size
        poses = []
        train_cams = scene.getTrainCameras()
        test_cams = scene.getTestCameras()
        all_cams = train_cams + test_cams
        for i in range(len(all_cams)):
            pose = np.zeros((1, 3, 4))
            R = all_cams[i].R
            T = all_cams[i].T
            pose[:,:,:,-1] = R
            pose[:,:,:,-1] = T
            poses.append(pose)
        poses = np.concatenate(poses)

        poses_ellipse = np.load("garden_poses_ellipse.npy")
        render_cams = []
        for id in range(len(poses_ellipse)):
            R = poses_ellipse[id, :, :3]
            T = poses_ellipse[id, :, 3]
            FoVx = all_cams[0].FoVx
            FoVy = all_cams[0].FoVy
            trans = all_cams[0].trans
            data_device = all_cams[0].data_device
            scale = all_cams[0].scale
            image = torch.zeros(all_cams[0].image.shape, device=data_device)            
            cam = Camera(colmap_id=1, R=R, T=T, FoVx=FoVx, FoVy=FoVy, image_name="", uid=id, trans=trans, data_device=data_device, scale=scale, image=image, gt_alpha_mask=None)
            render_cams.append(cam)
            rendering = render(cam, gaussians, pipeline, background, kernel_size=kernel_size)["render"]
            out = Image.fromarray(np.uint8(rendering.cpu().permute(1, 2, 0) * 255))
            out.save(f"video_frame/{id}.png")
        # TODO
        # Extract the camera poses to be the shape=[batch, 3, 4]
        # Apply generate_ellipse_path() in camera_utils to generate camera path
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, kernel_size, scale_factor=scale_factor)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, kernel_size, scale_factor=scale_factor)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--video", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)    
    if args.video:
        render_video(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    else:
        print("Rendering sets ----------")
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)