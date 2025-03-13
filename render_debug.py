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
import re

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size, scale_factor):    
    # render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"test_preds_{scale_factor}")
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"gt_{scale_factor}")
    render_path = os.path.join(model_path, "ours_{}".format(iteration), f"DS_{scale_factor}", f"{name}_preds_{scale_factor}")
    gts_path = os.path.join(model_path, "ours_{}".format(iteration), f"DS_{scale_factor}", f"gt_{scale_factor}")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):            
        # if not os.path.exists(os.path.join(render_path, '{0:05d}'.format(idx) + ".png")):
        rendering = render(view, gaussians, pipeline, background, kernel_size=kernel_size)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        if 'StableSR' in model_path:
            img_name = view.image_name
            scene_name = os.path.basename(model_path)
            if 'synthetic' in model_path:
                gt_folder = os.path.join('/fs/nexus-projects/dyn3Dscene/Codes/datasets/nerf_synthetic/original', scene_name, name)                
            else:
                if scale_factor == 1:
                    img_folder = 'images'
                else:
                    img_folder = f'images_{scale_factor}'
                gt_folder = os.path.join('/fs/nexus-projects/dyn3Dscene/Codes/data/my_new_resize', scene_name, img_folder)
            gt_img = Image.open(os.path.join(gt_folder, img_name+'.png'))
            gt_img.save(os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_interpolate(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, num_frames=15):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)        
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, train_tiny=args.train_tiny)        
        scale_factor = dataset.resolution
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size

        # Output folder
        render_path = os.path.join(dataset.model_path, "ours_interpolation", f"DS_{scale_factor}", f"train_preds_{scale_factor}")
        makedirs(render_path, exist_ok=True)
        
        # Get the two training views
        trainCameras = scene.getTrainCameras().copy()
        cams = []
        for i in range(len(trainCameras)):
            cam = trainCameras[i]
            extrinsic = np.zeros((4,4))
            extrinsic[:3,:3] = cam.R
            extrinsic[:3,-1] = cam.T
            extrinsic[-1,-1] = 1
            cams.append(extrinsic)
        
        # View interpolation
        interpolated_poses = interpolate_camera_poses(cams[0], cams[1], num_frames)        
        
        # import pdb; pdb.set_trace()
        # _, h, w = cam.original_image.shape
        # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # video_writer = cv2.VideoWriter(os.path.join(render_path,'video.mp4'), fourcc, fps=8, frameSize=(w, h))

        video_writer = imageio.get_writer(os.path.join(render_path,'video.mp4'), fps=5, quality=8, codec="libx264")
        # Rendering & saving
        for i in range(num_frames):
            pose = interpolated_poses[i]
            cam_new = Camera(colmap_id=-1, R=pose[:3,:3], T=pose[:3,-1], FoVx=cam.FoVx, FoVy=cam.FoVy, gt_alpha_mask=None, image=cam.original_image, image_name='interpolation', uid=cam.uid, )
            rendering = render(cam_new, gaussians, pipeline, background, kernel_size=kernel_size)["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path,f'interpolation_{i}.png'))
            video_writer.append_data((np.uint8(torch.clamp(rendering.cpu(), min=0, max=1.0).permute(1,2,0)*255)))
        video_writer.close()
    
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, train_tiny=args.train_tiny)
        # Super resolving gaussians
        # gaussians.super_resolving_gaussians(2, rendering=True)
        scale_factor = dataset.resolution
        bg_color = [0, 0, 0]
        # bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
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
        # all_cams =  test_cams + train_cams
        all_cams = test_cams
        
        llffhold = 8
        all_indices = np.arange(len(all_cams))
        # train_indices = all_indices % llffhold != 0
        # test_indices = all_indices % llffhold == 0
        test_indices = [0, 1] 
        w2c_mats = []
        for i in range(len(all_cams)):
            # if i in test_indices:
            pose = np.zeros((1, 3, 4))
            R = all_cams[i].R
            T = all_cams[i].T
            pose[:,:,:-1] = R
            pose[:,:,-1] = T
                # import pdb; pdb.set_trace()
                # # Jamie =============
                # bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
                # w2c = np.concatenate([pose[0], bottom], axis=0)
                # w2c_mats.append(w2c[:, :])
                # # ===================  
                
            # import pdb; pdb.set_trace()
            # pose_ = np.concatenate([pose[0], bottom], axis=0)
            # pose_inv = np.linalg.inv(pose_)
            # pose = np.linalg.inv(pose)
            # pose = pose @ np.diag(np.array([-1, 1, 1, 1]))
            
            # pose[0] = pose_inv[:3]
            poses.append(pose)
        poses = np.concatenate(poses)
        # import pdb; pdb.set_trace()
        n_interp = 30 # Num. frames to interpolate per keyframe.
        render_poses = generate_interpolated_path(poses, n_interp, spline_degree=5, smoothness=.03, rot_weight=.1)
        # # Jamie =============
        # w2c_mats = np.stack(w2c_mats, axis=0)
        # c2w_mats = np.linalg.inv(w2c_mats)
        # poses = c2w_mats[:, :3, :4]
        # poses = poses @ np.diag([1, -1, -1, 1])
        # # ===================  
        
        # Switch from COLMAP (right, down, fwd) to NeRF (right, up, back) frame.
        # poses = poses @ np.diag([1, -1, -1, 1])
        # poses = np.concatenate(poses)
        # poses_new, transform = transform_poses_pca(poses)
        # colmap_to_world_transform = transform
        # poses, transform = recenter_poses(poses)
        
        # # # Jamie =============
        # render_poses = generate_ellipse_path(poses,n_frames=120,z_variation=0,z_phase=0)
        # # import pdb; pdb.set_trace()
        # pose_b = np.zeros((120, 4,4))
        # pose_b[:,:-1,:] = render_poses
        # pose_b[:,-1,-1] = 1
        # pose_b = pose_b @ np.diag([1, -1, -1, 1])
        # w2c_mats_b = np.linalg.inv(pose_b)
        # render_poses = w2c_mats_b[:, :3, :4]
        # # # ===================  
        
        # import pdb; pdb.set_trace()
        # render_poses = generate_spiral_path(poses, bounds, n_frames=config.render_path_frames)
        # np.save("poses_ellipse_recenter.npy", render_poses)
        # poses_ellipse = np.load("garden_poses_ellipse.npy")
        render_cams = []
        
        # for id in enumerate(tqdm(render_poses, desc="Rendering progress")):
        for id in tqdm(range(len(render_poses))):
            # import pdb; pdb.set_trace()
            R = render_poses[id, :, :3]
            T = render_poses[id, :, 3]
            FoVx = all_cams[0].FoVx
            FoVy = all_cams[0].FoVy
            trans = all_cams[0].trans
            data_device = all_cams[0].data_device
            scale = all_cams[0].scale
            image = torch.zeros(all_cams[0].original_image.shape, device=data_device)
            cam = Camera(colmap_id=1, R=R, T=T, FoVx=FoVx, FoVy=FoVy, image_name="", uid=id, trans=trans, data_device=data_device, scale=scale, image=image, gt_alpha_mask=None)
            render_cams.append(cam)
            rendering = render(cam, gaussians, pipeline, background, kernel_size=kernel_size)["render"]
            torchvision.utils.save_image(rendering, os.path.join('frames/interpolated_30/', '{0:05d}'.format(id) + ".png"))
            # out = Image.fromarray(np.uint8(rendering.cpu().permute(1, 2, 0) * 255))
            # out.save(f"video_frame/{id}.png")
            # import pdb; pdb.set_trace()
            
        # TODO
        # Extract the camera poses to be the shape=[batch, 3, 4]
        # Apply generate_ellipse_path() in camera_utils to generate camera path
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, kernel_size, scale_factor=scale_factor)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, kernel_size, scale_factor=scale_factor)

# [Ting]
@torch.no_grad()
def render_scene_smooth_videos(dataset : ModelParams, iteration, pipeline : PipelineParams):
    scale_factor = dataset.resolution
    save_dir = '/fs/nexus-projects/dyn3Dscene/Codes/datasets/mipnerf360_RealBasicVSR'
    scene_name = os.path.basename(dataset.source_path)
    render_path = os.path.join(save_dir, scene_name, f"DS_{scale_factor}", f"ours")
    gts_path = os.path.join(save_dir, scene_name, f"DS_{scale_factor}", f"test_gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree)        
    # dataset.eval = False
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, train_tiny=args.train_tiny)
    print(scene.gaussians._xyz.shape)
    number = scene.gaussians._xyz.shape[0]
    with open(f"{os.path.join(save_dir, scene_name, f'DS_{scale_factor}', 'number.txt')}", "w") as file:
        file.write(str(number))

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    kernel_size = dataset.kernel_size
    train_views = scene.getTrainCameras()
    test_views = scene.getTestCameras()

    views = train_views + test_views

    sorted_views = sorted(views, key=lambda x: int(x.image_name.split('.')[0][4:]))
    test_views_imgname = [x.image_name for x in test_views]

    frame_id = 0    
    tan_fovx = np.tan(sorted_views[0].FoVx / 2.0)
    tan_fovy = np.tan(sorted_views[0].FoVy / 2.0)
    focal_y = sorted_views[0].image_height* 4 / (2.0 * tan_fovy)
    focal_x = sorted_views[0].image_width* 4 / (2.0 * tan_fovx)
    
    res = {
        "camera_model": "OPENCV",
        "fl_x": focal_x,
        "fl_y": focal_y,
        "cx": sorted_views[0].image_width * 2,
        "cy": sorted_views[0].image_height * 2,
        "w": sorted_views[0].image_width * 4,
        "h": sorted_views[0].image_height * 4,
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "k4": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "frames": []
    }
    # res_test = res.copy()

    for idx in tqdm(range(len(sorted_views))):
        extrinsic = np.eye(4)
        extrinsic[:3,:3] = sorted_views[idx].R
        extrinsic[:3,3] = sorted_views[idx].T

        c2w = np.zeros((4, 4))
        c2w[:3, :3] = sorted_views[idx].R.transpose()
        c2w[:3, 3] = sorted_views[idx].T
        c2w[3, 3] = 1.0
        c2w = np.linalg.inv(c2w)
        c2w[:3, 1:3] *= -1

        if sorted_views[idx].image_name in test_views_imgname:
            res['frames'].append({
                "file_path":f'gt/high_res_images/traj_0_{frame_id:04d}.png',
                "transform_matrix": c2w.tolist(),
            })
            # gt = sorted_views[idx].original_image[0:3, :, :]
            gt = Image.open(f"/fs/nexus-projects/dyn3Dscene/Codes/data/my_new_resize/{scene_name}/images_{scale_factor//4}/{sorted_views[idx].image_name}.png")
            gt.save(os.path.join(gts_path, f"traj_0_{frame_id:04d}.png"))
            # torchvision.utils.save_image(gt, os.path.join(gts_path, f"traj_0_{frame_id:04d}.png"))
        else:
            res['frames'].append({
                "file_path":f'images/{frame_id:04d}.png',
                "transform_matrix": c2w.tolist(),
            })

        if idx < len(sorted_views)-1:
            poses = np.zeros((2, 3, 4))
            poses[:,:,:-1] = np.stack([sorted_views[idx].R, sorted_views[idx+1].R])
            poses[:,:,-1] = np.stack([sorted_views[idx].T, sorted_views[idx+1].T])
            render_poses = generate_interpolated_path(poses, n_interp=4, spline_degree=5, smoothness=.03, rot_weight=.1)

            for i in range(len(render_poses)):
                R = render_poses[i, :, :3]
                T = render_poses[i, :, 3]
                FoVx = sorted_views[0].FoVx
                FoVy = sorted_views[0].FoVy
                trans = sorted_views[0].trans
                data_device = sorted_views[0].data_device
                scale = sorted_views[0].scale
                image = torch.zeros(sorted_views[0].original_image.shape, device=data_device)
                cam = Camera(colmap_id=1, R=R, T=T, FoVx=FoVx, FoVy=FoVy, image_name="", uid=idx, trans=trans, data_device=data_device, scale=scale, image=image, gt_alpha_mask=None)
                rendering = render(cam, gaussians, pipeline, background, kernel_size=kernel_size)["render"]
                torchvision.utils.save_image(rendering, os.path.join(render_path, f"{frame_id:04d}.png"))
                frame_id += 1
        else:
            rendering = render(sorted_views[-1], gaussians, pipeline, background, kernel_size=kernel_size)["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, f"{frame_id:04d}.png"))

    transform_path = f"/fs/nexus-projects/dyn3Dscene/Codes/datasets/mipnerf360_RealBasicVSR/{scene_name}/DS_{scale_factor}/transforms.json"
    with open(transform_path, 'w+') as f:
        json.dump(res, f, indent=4)
    # transform_path = f"/fs/nexus-projects/dyn3Dscene/Codes/datasets/mipnerf360_RealBasicVSR/{scene_name}/DS_{scale_factor}/transforms_test.json"
    # with open(transform_path, 'w+') as f:
    #     json.dump(res_test, f, indent=4)

def create_video(imgs, fps, name):
    writer = imageio.get_writer(f'{name}.mp4', fps=fps)
    for img in imgs:
        writer.append_data(img)

    writer.close()

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
    parser.add_argument("--train_tiny", action="store_true")
    parser.add_argument("--interpolate", action="store_true")
    parser.add_argument("--stream", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    if args.stream:
        scale_factor = model.extract(args).resolution
        save_dir = '/fs/nexus-projects/dyn3Dscene/Codes/datasets/mipnerf360_RealBasicVSR'
        scene_name = os.path.basename(model.extract(args).source_path)
        render_path = os.path.join(save_dir, scene_name, f"DS_{scale_factor}", f"ours")

        images_path = glob(f"{render_path}/*.png")
        # pattern = r"(\d+)_(\d+)\.png"

        # def sort_file(filename):
        #     match = re.search(pattern, filename)
        #     if match:
        #         return (int(match.group(1)), int(match.group(2)))
        #     return (0, 0)

        # sorted_filenames = sorted(images_path, key=sort_file)
        sorted_filenames = images_path
        imgs = []
        for img_path in tqdm(sorted_filenames):
            img = np.array(Image.open(img_path), dtype=np.uint8)
            imgs.append(img)
        create_video(imgs, 30, os.path.join(save_dir, scene_name, f"DS_{scale_factor}", "video"))

        quit()

    # Initialize system state (RNG)
    safe_state(args.quiet)    
    if args.video:
        render_scene_smooth_videos(model.extract(args), args.iteration, pipeline.extract(args))
        # render_video(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    elif args.interpolate and args.train_tiny:
        print("Rendering interpolated view ----------")
        render_interpolate(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    else:
        print("Rendering sets ----------")
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)