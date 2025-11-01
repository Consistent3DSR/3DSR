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

import os, glob
import numpy as np
import open3d as o3d
import cv2
import json
import torch
import random
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from torch import autocast
import sys
import copy
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import lpips
import pyiqa
import natsort
# from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
# from scipy.spatial.transform import Rotation as R, Slerp
import torchvision
from scene.cameras import Camera
from PIL import Image
from utils.general_utils import PILtoTorch
try:
    # from torch.utils.tensorboard import SummaryWriter
    from tensorboardX import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
##### Stable SR usage #####
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from utils.stable_sr_utils import instantiate_from_config
from utils.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization
from contextlib import nullcontext
from tqdm import tqdm, trange
from einops import rearrange, repeat
from utils.util_image import ImageSpliterTh
import torch.nn.functional as F
from pathlib import Path
import time

@torch.no_grad()
def create_offset_gt(image, offset):
    height, width = image.shape[1:]
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords).cuda()
    
    id_coords = id_coords.permute(1, 2, 0) + offset
    id_coords[..., 0] /= (width - 1)
    id_coords[..., 1] /= (height - 1)
    id_coords = id_coords * 2 - 1
    
    image = torch.nn.functional.grid_sample(image[None], id_coords[None], align_corners=True, padding_mode="border")[0]
    return image

def prepare_training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args, dataset2=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    
    if args.load_pretrain:
        scene = Scene(dataset, gaussians, load_iteration=30000, shuffle=False)
        scene.model_path = args.output_folder
        dataset_name = os.path.basename(dataset.source_path)
        dataset.model_path = os.path.join(args.output_folder, dataset_name)
        
        tb_writer = prepare_output_and_logger(dataset)
        scene.model_path = dataset.model_path
    else:
        scene = Scene(dataset, gaussians)
    
    if args.load_pretrain:
        gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), dtype=torch.float32, device="cuda")
        gaussians.training_setup(opt)
        print("--- after loading pretrain points:", gaussians._xyz.shape[0])
    else:
        gaussians.training_setup(opt)
        
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)    

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    out_dict = {"scene": scene, "gaussians": gaussians, "tb_writer": tb_writer}
    return out_dict

def training_with_iters(in_dict, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args, dataset2=None, SR_iter=0):
    scene = in_dict['scene']
    gaussians = in_dict['gaussians']
    tb_writer = in_dict['tb_writer']
    
    first_iter = 0    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    testCameras = scene.getTestCameras().copy()
    allCameras = trainCameras + testCameras
    
    # highresolution index
    highresolution_index = []
    for index, camera in enumerate(trainCameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    gaussians.compute_3D_filter(cameras=trainCameras)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Pick a random high resolution camera
        if random.random() < 0.3 and dataset.sample_more_highres:
            viewpoint_cam = trainCameras[highresolution_index[randint(0, len(highresolution_index)-1)]]
            
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        #TODO ignore border pixels
        if dataset.ray_jitter:
            subpixel_offset = torch.rand((int(viewpoint_cam.image_height), int(viewpoint_cam.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
            # subpixel_offset *= 0.0
        else:
            subpixel_offset = None
            
        # Rendering
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size, subpixel_offset=subpixel_offset)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        
        # sample gt_image with subpixel offset
        if dataset.resample_gt_image:
            gt_image = create_offset_gt(gt_image, subpixel_offset)
        
        Ll1 = l1_loss(image, gt_image)
        loss_hr = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = loss_hr
        
        if iteration > opt.iterations - len(trainCameras):
            training_folder = os.path.join(args.output_folder, 'training_views')
            if not os.path.exists(training_folder):
                os.makedirs(training_folder)
            file_name = os.path.join(training_folder, viewpoint_cam.image_name + ".png")
            torchvision.utils.save_image(image, os.path.join(file_name))
        
        if args.fidelity_train_en:
            lr_resolution = dataset.resolution * 4
            gt_path = os.path.join(dataset.source_path, f'images_{lr_resolution}', viewpoint_cam.image_name+'.png')
            image_gt_lr = Image.open(gt_path)
            w_lr, h_lr = image_gt_lr.size
            image_gt_lr = PILtoTorch(image_gt_lr, (w_lr, h_lr)).cuda()
            image_lr = torch.nn.functional.interpolate(image.unsqueeze(0), scale_factor=0.25, mode='bicubic', antialias=True).squeeze(0)
            loss_lr = (1.0 - opt.lambda_dssim) * l1_loss(image_lr, image_gt_lr) + opt.lambda_dssim * (1.0 - ssim(image_lr, image_gt_lr))
            loss += loss_lr * args.wt_lr
                    
        loss.backward()
        iter_end.record()
        
        if iteration == opt.iterations - 1:
            training_folder = os.path.join(args.outdir, 'train_results')
            if not os.path.exists(training_folder):
                os.makedirs(training_folder)
                
            for i in range(len(trainCameras)):                
                cam = trainCameras[i]
                rendering = render(cam, gaussians, pipe, background, kernel_size=dataset.kernel_size, subpixel_offset=subpixel_offset)["render"]
                file_name = os.path.join(training_folder, cam.image_name + f"_step_{3-SR_iter}.png")
                print(file_name)
                torchvision.utils.save_image(rendering, os.path.join(file_name))
            
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, dataset.kernel_size))
            if (iteration in saving_iterations):
                final_iter = (3-SR_iter) * opt.iterations + iteration
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(final_iter)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    gaussians.compute_3D_filter(cameras=trainCameras)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)
        
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    
    out_dict = {"scene": scene, "gaussians": gaussians, "tb_writer": tb_writer, "highresolution_index": highresolution_index}
    
    return out_dict

def load_model_from_config(config, ckpt, verbose=False):
	print(f"Loading model from {ckpt}")
	pl_sd = torch.load(ckpt, map_location="cpu")
	if "global_step" in pl_sd:
		print(f"Global Step: {pl_sd['global_step']}")
	sd = pl_sd["state_dict"]
	model = instantiate_from_config(config.model)
	m, u = model.load_state_dict(sd, strict=False)
	if len(m) > 0 and verbose:
		print("missing keys:")
		print(m)
	if len(u) > 0 and verbose:
		print("unexpected keys:")
		print(u)

	model.cuda()
	model.eval()
	return model

def prepare_model(opt):
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.configs = config
    
    vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
    vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt)
    vq_model = vq_model.to(device)
    vq_model.decoder.fusion_w = opt.dec_w
    
    model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
						  linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
       
    out_dict = {'model': model, 'vq_model': vq_model}
    return out_dict

def space_timesteps(num_timesteps, section_counts):
	"""
	Create a list of timesteps to use from an original diffusion process,
	given the number of timesteps we want to take from equally-sized portions
	of the original process.
	For example, if there's 300 timesteps and the section counts are [10,15,20]
	then the first 100 timesteps are strided to be 10 timesteps, the second 100
	are strided to be 15 timesteps, and the final 100 are strided to be 20.
	If the stride is a string starting with "ddim", then the fixed striding
	from the DDIM paper is used, and only one section is allowed.
	:param num_timesteps: the number of diffusion steps in the original
						  process to divide up.
	:param section_counts: either a list of numbers, or a string containing
						   comma-separated numbers, indicating the step count
						   per section. As a special case, use "ddimN" where N
						   is a number of steps to use the striding from the
						   DDIM paper.
	:return: a set of diffusion steps from the original process to use.
	"""
	if isinstance(section_counts, str):
		if section_counts.startswith("ddim"):
			desired_count = int(section_counts[len("ddim"):])
			for i in range(1, num_timesteps):
				if len(range(0, num_timesteps, i)) == desired_count:
					return set(range(0, num_timesteps, i))
			raise ValueError(
				f"cannot create exactly {num_timesteps} steps with an integer stride"
			)
		section_counts = [int(x) for x in section_counts.split(",")]   #[250,]
	size_per = num_timesteps // len(section_counts)
	extra = num_timesteps % len(section_counts)
	start_idx = 0
	all_steps = []
	for i, section_count in enumerate(section_counts):
		size = size_per + (1 if i < extra else 0)
		if size < section_count:
			raise ValueError(
				f"cannot divide section of {size} steps into {section_count}"
			)
		if section_count <= 1:
			frac_stride = 1
		else:
			frac_stride = (size - 1) / (section_count - 1)
		cur_idx = 0.0
		taken_steps = []
		for _ in range(section_count):
			taken_steps.append(start_idx + round(cur_idx))
			cur_idx += frac_stride
		all_steps += taken_steps
		start_idx += size
	return set(all_steps)

def read_image(im_path):
	im = np.array(Image.open(im_path).convert("RGB"))
	im = im.astype(np.float32)/255.0
	im = im[None].transpose(0,3,1,2)
	im = (torch.from_numpy(im) - 0.5) / 0.5
	return im.cuda()

def visualize_image(latent, rgb_patch, model_dict, out_img_name=None):
    # latent: latent to be decoded
    # rgb_patch: input image rgb patch
    # model_dict: dictionary containing model and vq_model
    # out_img_name: output image name
    
    vq_model = model_dict['vq_model']
    model = model_dict['model']
    _, enc_fea_lq = vq_model.encode(rgb_patch)
    x_samples = vq_model.decode(latent * 1. / model.scale_factor, enc_fea_lq)
    x_samples = wavelet_reconstruction(x_samples, rgb_patch)
    im_sr = torch.clamp((x_samples+1.0)/2.0, min=0.0, max=1.0)
    out = Image.fromarray(np.uint8(im_sr[0, ].permute(1,2,0).cpu().numpy()*255))
    
    if out_img_name is not None:        
        out.save(out_img_name)
    return out
    
def train_proposed(dataset, op, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args, dataset2=None):    
    ####################################
    # Set up for Stable SR
    ####################################
    print('>>>>>>>>>>color correction>>>>>>>>>>>')
    if args.colorfix_type == 'adain':
        print('Use adain color correction')
    elif args.colorfix_type == 'wavelet':
        print('Use wavelet color correction')
    else:
        print('No color correction')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    
    #############################################
    # load StableSR model and scheduler
    #############################################
    # Check input images
    os.makedirs(args.outdir, exist_ok=True)
    outpath = args.outdir
    batch_size = args.n_samples
    images_path_ori = sorted(glob.glob(os.path.join(args.init_img, "*")))
    images_path = np.array(copy.deepcopy(images_path_ori))
    
    # Only taking training views for SR
    llffhold = 8
    all_indices = np.arange(len(images_path))
    train_indices = all_indices % llffhold != 0
    sr_indices = all_indices[train_indices]
    images_path = images_path[sr_indices[:]]
    print(f"Found {len(images_path)} inputs.")
    
    # Prepare model
    out_dict = prepare_model(args)
    model = out_dict['model']
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    sqrt_alphas_cumprod = copy.deepcopy(model.sqrt_alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = copy.deepcopy(model.sqrt_one_minus_alphas_cumprod)
    
    # Modify scheduler for fewer steps
    use_timesteps = set(space_timesteps(1000, [args.ddpm_steps]))
    last_alpha_cumprod = 1.0
    new_betas = []
    timestep_map = []
    for i, alpha_cumprod in enumerate(model.alphas_cumprod):
        if i in use_timesteps:
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
            timestep_map.append(i)
    new_betas = [beta.data.cpu().numpy() for beta in new_betas]
    model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
    model.num_timesteps = 1000
    model.ori_timesteps = list(use_timesteps)
    model.ori_timesteps.sort()
    model = model.to(device)
    
    # Add model and args to out_dict
    out_dict['model'] = model
    out_dict['args'] = args
    precision_scope = autocast if args.precision == "autocast" else nullcontext
    
    #############################################
    # Loading scene and Gaussians
    #############################################    
    op.densify_until_iter = args.densify_end
    input_dict = prepare_training(dataset, op, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args, dataset2)
    scene = input_dict["scene"]
    trainCameras = scene.getTrainCameras()
    
    if 'llff' in dataset.source_path:
        dir_name = dataset.source_path
        lr_resolution = dataset.resolution * 4
        
        orig_folder =  os.path.join(dir_name, 'images')
        orig_files = os.listdir(orig_folder)
        orig_files = natsort.natsorted(orig_files)
        
        cur_files = os.listdir( os.path.join(dir_name, f'images_{lr_resolution}'))
        cur_files = natsort.natsorted(cur_files)        
    #############################################
    # Prepare for SR method
    #############################################
    with model.ema_scope():
        tic = time.time()
        all_samples = list()        
        seed_everything(args.seed)
        
        imgs_per_batch = batch_size
        loop_img_time = len(images_path) // imgs_per_batch
        one_more_time = (len(images_path) % imgs_per_batch) > 0        
        loop_img_time += int(one_more_time)
      
        #############################################
        # Loop by denoising steps
        #############################################
        for iteration in range(args.ddpm_steps-1, -1, -1):
            for loop_id in range(loop_img_time):
                if loop_id == loop_img_time - 1:
                    images_path_small = images_path[loop_id*imgs_per_batch:]
                else:
                    images_path_small = images_path[loop_id*imgs_per_batch : (loop_id+1)*imgs_per_batch]
                
                im_lq_bs = []
                im_path_bs = []
                for img_id in range(len(images_path_small)):
                    cur_image = read_image(images_path_small[img_id])
                    size_min = min(cur_image.size(-1), cur_image.size(-2))
                    upsample_scale = max(args.input_size/size_min,
                                         args.upscale)
                    cur_image = F.interpolate(
                                cur_image,
                                size=(int(cur_image.size(-2)*upsample_scale),
                                        int(cur_image.size(-1)*upsample_scale)),
                                mode='bicubic',
                                )
                    cur_image = cur_image.clamp(-1, 1)                    
                    im_lq_bs.append(cur_image) # 1 x c x h x w, [-1, 1]
                    im_path_bs.append(images_path_small[img_id]) # 1 x c x h x w, [-1, 1]
                im_lq_bs = torch.cat(im_lq_bs, dim=0)                
                ori_h, ori_w = im_lq_bs.shape[2:]
                ref_patch=None
                if not (ori_h % 32 == 0 and ori_w % 32 == 0):
                    flag_pad = True
                    pad_h = ((ori_h // 32) + 1) * 32 - ori_h
                    pad_w = ((ori_w // 32) + 1) * 32 - ori_w
                    im_lq_bs = F.pad(im_lq_bs, pad=(0, pad_w, 0, pad_h), mode='reflect')
                else:
                    flag_pad = False
                    
                if iteration != args.ddpm_steps - 1:
                    #####################################################
                    # Load upsampled image, and encode to latent space
                    #####################################################
                    imgs = []
                    for img_id in range(len(im_path_bs)):
                        img_name = str(Path(im_path_bs[img_id]).name)
                        basename = os.path.splitext(os.path.basename(img_name))[0]
                        training_folder = os.path.join(args.outdir, 'train_results')
                        cur_id = loop_id * imgs_per_batch + img_id
                        imgpath = os.path.join(training_folder, trainCameras[cur_id].image_name + f"_step_{3-int(iteration)-1}.png")                        
                        cur_image = read_image(imgpath)
                        
                        # Add padding to loaded image
                        if not (ori_h % 32 == 0 and ori_w % 32 == 0):
                            pad_h = ((ori_h // 32) + 1) * 32 - ori_h
                            pad_w = ((ori_w // 32) + 1) * 32 - ori_w
                            cur_image = F.pad(cur_image, pad=(0, pad_w, 0, pad_h), mode='reflect')
                        imgs.append(cur_image)
                    imgs = torch.cat(imgs, dim=0)
                    
                print("************** ITERATION", 3-iteration, "**************")
                with torch.no_grad():
                    with precision_scope("cuda"):
                        #############################################
                        # Start of loop for denoised images
                        #############################################
                        for img_id in range(len(im_path_bs)):
                            #############################################
                            # Split image to patches
                            #############################################
                            if im_lq_bs.shape[2] > args.vqgantile_size or im_lq_bs.shape[3] > args.vqgantile_size:
                                im_spliter = ImageSpliterTh(im_lq_bs[img_id].unsqueeze(0), args.vqgantile_size, args.vqgantile_stride, sf=1)
                                if iteration != args.ddpm_steps-1:
                                    im_spliter_x_tilda = ImageSpliterTh(imgs[img_id].unsqueeze(0), args.vqgantile_size, args.vqgantile_stride, sf=1)
                                #############################################
                                # Loop to process each patch in an image   
                                #############################################                         
                                for im_lq_pch, index_infos in im_spliter:
                                    if iteration == args.ddpm_steps-1:
                                        init_latent = model.get_first_stage_encoding(model.encode_first_stage(im_lq_pch))  # move to latent space
                                        text_init = ['']*args.n_samples
                                        semantic_c = model.cond_stage_model(text_init)
                                        noise = torch.randn_like(init_latent)
                                        # If you would like to start from the intermediate steps, you can add noise to LR to the specific steps.
                                        t = repeat(torch.tensor([999]), '1 -> b', b=im_lq_pch.size(0))
                                        t = t.to(device).long()
                                        # Apply the noise to the latent space (sqrt(alpha) * z + sqrt(1-alpha) * x) to create x_T
                                        x_T = model.q_sample_respace(x_start=init_latent, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, 
                                                sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)
                                        _, x0_head = model.sample_canvas_one_iter(iteration=iteration, cond=semantic_c, struct_cond=init_latent, 
                                                                                    batch_size=im_lq_pch.size(0), timesteps=args.ddpm_steps, time_replace=args.ddpm_steps, 
                                                                                    x_T=x_T, tile_size=int(args.input_size/8), tile_overlap=args.tile_overlap, 
                                                                                    batch_size_sample=args.n_samples, return_x0=True)
                                    else:
                                        #############################################
                                        # Encode image to latent space
                                        #############################################
                                        im_lq_pch_tilda, index_infos_tilda = next(im_spliter_x_tilda)
                                        x0_tilda_latent = model.get_first_stage_encoding(model.encode_first_stage(im_lq_pch_tilda))  # move to latent space
                                        text_init = ['']*args.n_samples
                                        semantic_c = model.cond_stage_model(text_init)
                                        init_latent = model.get_first_stage_encoding(model.encode_first_stage(im_lq_pch))  # move to latent space
                                        x_T_1 = model.sample_canvas_one_iter(iteration=iteration+1, cond=semantic_c, struct_cond=init_latent, 
                                                        batch_size=im_lq_pch.size(0), timesteps=args.ddpm_steps, time_replace=args.ddpm_steps, 
                                                        x_T=x_T, tile_size=int(args.input_size/8), tile_overlap=args.tile_overlap, 
                                                        batch_size_sample=args.n_samples, return_x0=False, x0_input=x0_tilda_latent)
                                        _, x0_head = model.sample_canvas_one_iter(iteration=iteration, cond=semantic_c, struct_cond=init_latent, 
                                                                                    batch_size=im_lq_pch.size(0), timesteps=args.ddpm_steps, time_replace=args.ddpm_steps, 
                                                                                    x_T=x_T_1, tile_size=int(args.input_size/8), tile_overlap=args.tile_overlap, 
                                                                                    batch_size_sample=args.n_samples, return_x0=True)
                                    # Decode the latent space to image space
                                    vq_model = out_dict['vq_model']
                                    _, enc_fea_lq = vq_model.encode(im_lq_pch)
                                    x_samples = vq_model.decode(x0_head * 1. / model.scale_factor, enc_fea_lq)
                                    
                                    if args.colorfix_type == 'adain':
                                        x_samples = adaptive_instance_normalization(x_samples, im_lq_pch)
                                    elif args.colorfix_type == 'wavelet':
                                        x_samples = wavelet_reconstruction(x_samples, im_lq_pch)
                                    im_spliter.update_gaussian(x_samples, index_infos)

                                im_sr = im_spliter.gather()
                                im_sr = torch.clamp((im_sr+1.0)/2.0, min=0.0, max=1.0)
                                
                                if upsample_scale > args.upscale:
                                    im_sr = F.interpolate(
                                                im_sr,
                                                size=(int(im_lq_bs.size(-2)*args.upscale/upsample_scale),
                                                    int(im_lq_bs.size(-1)*args.upscale/upsample_scale)),
                                                mode='bicubic',)
                                    im_sr = torch.clamp(im_sr, min=0.0, max=1.0)
                                
                                if flag_pad:
                                    im_sr = im_sr[:, :, :ori_h, :ori_w, ]

                                im_sr = im_sr.cpu().numpy().transpose(0,2,3,1)*255   # b x h x w x c                                
                                img_name = str(Path(im_path_bs[img_id]).name)
                                basename = os.path.splitext(os.path.basename(img_name))[0]
                                outpath = str(Path(args.outdir)) + '/' + basename + f'_step_{3-int(iteration)}.png'
                                print('Finished:', outpath)
                                Image.fromarray(im_sr[0, ].astype(np.uint8)).save(outpath)
                            
                            #######################################################################
                            # Take the entire image as SR input (when input image is small enough)
                            #######################################################################
                            else:
                                if iteration == args.ddpm_steps-1:
                                    init_latent = model.get_first_stage_encoding(model.encode_first_stage(im_lq_bs[img_id].unsqueeze(0)))  # move to latent space
                                    text_init = ['']*args.n_samples
                                    semantic_c = model.cond_stage_model(text_init)
                                    noise = torch.randn_like(init_latent)
                                    # If you would like to start from the intermediate steps, you can add noise to LR to the specific steps.
                                    t = repeat(torch.tensor([999]), '1 -> b', b=1)
                                    t = t.to(device).long()
                                    x_T = model.q_sample_respace(x_start=init_latent, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)
                                    _, x0_head = model.sample_canvas_one_iter(iteration=iteration, cond=semantic_c, struct_cond=init_latent, 
                                                                            batch_size=1, timesteps=args.ddpm_steps, time_replace=args.ddpm_steps, 
                                                                            x_T=x_T, tile_size=int(args.input_size/8), tile_overlap=args.tile_overlap, 
                                                                            batch_size_sample=args.n_samples, return_x0=True)
                                else:
                                    #############################################
                                    # Encode image to latent space
                                    #############################################
                                    x0_tilda_latent = model.get_first_stage_encoding(model.encode_first_stage(imgs[img_id].unsqueeze(0)))  # move to latent space
                                    text_init = ['']*args.n_samples
                                    semantic_c = model.cond_stage_model(text_init)
                                    init_latent = model.get_first_stage_encoding(model.encode_first_stage(im_lq_bs[img_id].unsqueeze(0)))  # move to latent space
                                    # Get x_{t-1}
                                    x_T_1 = model.sample_canvas_one_iter(iteration=iteration+1, cond=semantic_c, struct_cond=init_latent, 
                                                    batch_size=1, timesteps=args.ddpm_steps, time_replace=args.ddpm_steps, 
                                                    x_T=x_T, tile_size=int(args.input_size/8), tile_overlap=args.tile_overlap, 
                                                    batch_size_sample=args.n_samples, return_x0=False, x0_input=x0_tilda_latent)
                                    # Predict x0_head
                                    _, x0_head = model.sample_canvas_one_iter(iteration=iteration, cond=semantic_c, struct_cond=init_latent, 
                                                                                batch_size=1, timesteps=args.ddpm_steps, time_replace=args.ddpm_steps, 
                                                                                x_T=x_T_1, tile_size=int(args.input_size/8), tile_overlap=args.tile_overlap, 
                                                                                batch_size_sample=args.n_samples, return_x0=True)
                                    
                                vq_model = out_dict['vq_model']
                                _, enc_fea_lq = vq_model.encode(im_lq_bs[img_id].unsqueeze(0))
                                x_samples = vq_model.decode(x0_head * 1. / model.scale_factor, enc_fea_lq)
                                if args.colorfix_type == 'adain':
                                    x_samples = adaptive_instance_normalization(x_samples, im_lq_bs[img_id].unsqueeze(0))
                                elif args.colorfix_type == 'wavelet':
                                    x_samples = wavelet_reconstruction(x_samples, im_lq_bs[img_id].unsqueeze(0))
                                im_sr = torch.clamp((x_samples+1.0)/2.0, min=0.0, max=1.0)
                                if flag_pad:
                                    im_sr = im_sr[:, :, :ori_h, :ori_w, ]

                                im_sr = im_sr.cpu().numpy().transpose(0,2,3,1)*255   # b x h x w x c                                
                                img_name = str(Path(im_path_bs[img_id]).name)
                                basename = os.path.splitext(os.path.basename(img_name))[0]
                                outpath = str(Path(args.outdir)) + '/' + basename + f'_step_{3-int(iteration)}.png'
                                Image.fromarray(im_sr[0, ].astype(np.uint8)).save(outpath)
                                print('Finished:', outpath)
                                
                                if iteration == 0:
                                    final_sr_path = os.path.join(args.outdir, 'final_sr_results')
                                    os.makedirs(final_sr_path, exist_ok=True)
                                    outpath = final_sr_path + '/' + basename + f'.png'
                                    Image.fromarray(im_sr[0, ].astype(np.uint8)).save(outpath)                        
                    #############################################
                    # End of loop for denoised images
                    #############################################                
            
            #############################################
            # Update ground truth image in trainCameras  
            #############################################
            for img_id in range(len(trainCameras)):
                # If you read from the saved image, you can use the following code
                # cam_id =  loop_id * imgs_per_batch + img_id
                
                # if 'llff' in dataset.source_path:
                #     matching_index = next((i for i, name in enumerate(orig_files) if trainCameras[img_id].image_name in name), None)
                #     img_name = cur_files[matching_index].split('.')[0]
                img_name = trainCameras[img_id].image_name
                img_path = str(Path(args.outdir)) + '/' + img_name + f'_step_{3-int(iteration)}.png'
                img_transfer = Image.open(img_path).convert("RGB")
                width, height = img_transfer.size
                loaded_image = PILtoTorch(img_transfer, (width, height)).cuda()
                # print(img_path)
                # torchvision.utils.save_image(loaded_image, 'vis.png')
                # torchvision.utils.save_image(trainCameras[img_id].original_image, 'vis_2.png')
                trainCameras[img_id].original_image = loaded_image.clone()
                
            # #############################################
            # # Train GS
            # #############################################
            input_dict = training_with_iters(input_dict, dataset, op, pipe, testing_iterations, saving_iterations,
                                            checkpoint_iterations, checkpoint, debug_from, args, dataset2, SR_iter=iteration,) 

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args, dataset2=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(" ----- checkpoint loaded from", checkpoint)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    testCameras = scene.getTestCameras().copy()
    allCameras = trainCameras + testCameras
    
    # highresolution index
    highresolution_index = []
    for index, camera in enumerate(trainCameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    gaussians.compute_3D_filter(cameras=trainCameras)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    
    first_iter += 1

    num_points = {}
    
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        pop_id = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack.pop(pop_id)
        
        if random.random() < 0.3 and dataset.sample_more_highres:
            viewpoint_cam = trainCameras[highresolution_index[randint(0, len(highresolution_index)-1)]]
            
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        #TODO ignore border pixels
        if dataset.ray_jitter:
            subpixel_offset = torch.rand((int(viewpoint_cam.image_height), int(viewpoint_cam.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
            # subpixel_offset *= 0.0
        else:
            subpixel_offset = None
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size, subpixel_offset=subpixel_offset)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # sample gt_image with subpixel offset
        if dataset.resample_gt_image:
            gt_image = create_offset_gt(gt_image, subpixel_offset)
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))        
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, dataset.kernel_size))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            if (iteration == opt.iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            if iteration % 1000 == 0:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, output_folder="iteration_29000")

            if not args.freeze_point:
                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                        gaussians.compute_3D_filter(cameras=trainCameras)

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    gaussians.compute_3D_filter(cameras=trainCameras)
            
            if iteration % 500 == 0:
                num_points[iteration] = gaussians.get_xyz.shape[0]
                print("number of points:", gaussians._xyz.shape[0])
            
            if iteration == opt.iterations:
                with open(os.path.join(args.output_folder, "num_points.json"), "w") as f:
                    json.dump(num_points, f)
        
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            try:
                tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            except:
                pass
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def parse_args():
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--load_pretrain", action="store_true")
    parser.add_argument("--freeze_point", action="store_true")
    parser.add_argument("--SR_GS", action="store_true")
    parser.add_argument("--fidelity_train_en", action="store_true")
    parser.add_argument("--prune_init_en", action="store_true")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--edge_aware_loss_en", action="store_true")
    parser.add_argument("--lpips_wt", type=float, default=0.2)
    parser.add_argument("--wt_lr", type=float, default=0.4)
    parser.add_argument("--densify_end", type=int, default=15000)
    parser.add_argument("--original", action="store_true")
    #############################################
    #### From Stable SR code ####
    #############################################
    parser.add_argument(
		"--init-img",
		type=str,
		nargs="?",
		help="path to the input image",
		default="inputs/user_upload"
	)
    parser.add_argument(
		"--outdir",
		type=str,
		nargs="?",
		help="dir to write results to",
		default="outputs/user_upload"
	)
    parser.add_argument(
		"--ddpm_steps",
		type=int,
		default=1000,
		help="number of ddpm sampling steps",
	)
    parser.add_argument(
		"--n_iter",
		type=int,
		default=1,
		help="sample this often",
	)
    parser.add_argument(
		"--C",
		type=int,
		default=4,
		help="latent channels",
	)
    parser.add_argument(
		"--f",
		type=int,
		default=8,
		help="downsampling factor, most often 8 or 16",
	)
    parser.add_argument(
		"--n_samples",
		type=int,
		default=1,
		help="how many samples to produce for each given prompt. A.k.a batch size",
	)
    parser.add_argument(
		"--config",
		type=str,
		default="configs/stable-diffusion/v1-inference.yaml",
		help="path to config which constructs model",
	)
    parser.add_argument(
		"--ckpt",
		type=str,
		default="./stablesr_000117.ckpt",
		help="path to checkpoint of model",
	)
    parser.add_argument(
		"--vqgan_ckpt",
		type=str,
		default="./vqgan_cfw_00011.ckpt",
		help="path to checkpoint of VQGAN model",
	)  
    parser.add_argument(
		"--precision",
		type=str,
		help="evaluate at this precision",
		choices=["full", "autocast"],
		default="autocast"
	)
    parser.add_argument(
		"--dec_w",
		type=float,
		default=0.5,
		help="weight for combining VQGAN and Diffusion",
	)
    parser.add_argument(
		"--tile_overlap",
		type=int,
		default=32,
		help="tile overlap size (in latent)",
	)
    parser.add_argument(
		"--upscale",
		type=float,
		default=4.0,
		help="upsample scale",
	)
    parser.add_argument(
		"--colorfix_type",
		type=str,
		default="nofix",
		help="Color fix type to adjust the color of HR result according to LR input: adain (used in paper); wavelet; nofix",
	)
    parser.add_argument(
		"--vqgantile_stride",
		type=int,
		default=1000,
		help="the stride for tile operation before VQGAN decoder (in pixel)",
	)
    parser.add_argument(
		"--vqgantile_size",
		type=int,
		default=1280,
		help="the size for tile operation before VQGAN decoder (in pixel)",
	)
    parser.add_argument(
        "--input_size",
        type=int,
        default=512,
        help="input size",
    )
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    return lp, op, pp, args

if __name__ == "__main__":
    lp, op, pp, args = parse_args()
    print("Optimizing " + args.model_path)    
    # Set up random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    seed_everything(args.seed)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)    
        
    if args.original:
        training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)
    else:
        train_proposed(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)    
    # All done
    print("\nTraining complete.")
