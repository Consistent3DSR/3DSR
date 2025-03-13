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
import numpy as np
import open3d as o3d
import cv2
import json
import torch
import random
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import lpips
import pyiqa
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torchvision
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image
from utils.general_utils import PILtoTorch
try:
    # from torch.utils.tensorboard import SummaryWriter
    from tensorboardX import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

prune_ratio = float(os.environ["PRUNE_RATIO"]) if "PRUNE_RATIO" in os.environ else 1.0
min_opacity = float(os.environ["MIN_OPACITY"]) if "MIN_OPACITY" in os.environ else 0.005
consecutive_timesteps = int(os.environ["CONSEC_TIMESTEPS"])
num_inference_steps = int(os.environ["NUM_INFERENCE_STEPS"])

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
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)    
     
    if args.load_pretrain:
        scene = Scene(dataset, gaussians, load_iteration=30000, shuffle=False, train_tiny=args.train_tiny)
        scene.model_path = args.output_folder
        dataset_name = os.path.basename(dataset.source_path)
        dataset.model_path = os.path.join(args.output_folder, dataset_name)
        tb_writer = prepare_output_and_logger(dataset)
        scene.model_path = dataset.model_path
        
        if args.prune_init_en:
            num_points = scene.gaussians._xyz.shape[0]
            valid_ids = torch.randperm(num_points)[:int(num_points * prune_ratio+0.5)]
            # Prune points
            gaussians._xyz = gaussians._xyz[valid_ids].clone().detach()
            gaussians._features_dc = gaussians._features_dc[valid_ids].clone().detach()
            gaussians._features_rest = gaussians._features_rest[valid_ids].clone().detach()
            gaussians._scaling = gaussians._scaling[valid_ids].clone().detach()
            gaussians._rotation = gaussians._rotation[valid_ids].clone().detach()
            gaussians._opacity = gaussians._opacity[valid_ids].clone().detach()
    else:
        scene = Scene(dataset, gaussians)    
        
    gaussians.training_setup(opt)
    if args.load_pretrain:
        gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), dtype=torch.float32, device="cuda")
        gaussians.training_setup(opt)
        print("--- after loading pretrain points:", gaussians._xyz.shape[0])

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(" ----- checkpoint loaded from", checkpoint)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
   
    trainCameras = scene.getTrainCameras().copy()
    testCameras = scene.getTestCameras().copy()
    # allCameras = trainCameras + testCameras
    
    # highresolution index
    highresolution_index = []
    for index, camera in enumerate(trainCameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    gaussians.compute_3D_filter(cameras=trainCameras)

    out_dict = {"scene": scene, "gaussians": gaussians, "tb_writer": tb_writer, "highresolution_index": highresolution_index}
    
    return out_dict

def training_with_iters(in_dict, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args, dataset2=None, SR_iter=0, SR_step=1):
    scene = in_dict["scene"]
    gaussians = in_dict["gaussians"]
    tb_writer = in_dict["tb_writer"]
    highresolution_index = in_dict["highresolution_index"]
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    trainCameras = scene.getTrainCameras().copy()

    ema_loss_for_log = 0.0
    first_iter = 0
    viewpoint_stack = None
    try:
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    except:
        import pdb; pdb.set_trace()
    
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
        pop_id = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack.pop(pop_id)
        
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
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size, subpixel_offset=subpixel_offset)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # sample gt_image with subpixel offset
        if dataset.resample_gt_image:
            gt_image = create_offset_gt(gt_image, subpixel_offset)

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        #############################################
        # Save rendered images as blending results
        #############################################
        if iteration > opt.iterations - len(trainCameras):
            training_folder = os.path.join(args.output_folder, 'training_with_step',f"timesteps_{SR_iter * SR_step}_{(SR_iter + 1) * SR_step}", "GS_training_results")
            if not os.path.exists(training_folder):
                os.makedirs(training_folder)
            file_name = os.path.join(training_folder, viewpoint_cam.image_name + ".png")
            torchvision.utils.save_image(image, os.path.join(file_name))
            
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
            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)
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
                        # gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                        gaussians.densify_and_prune(opt.densify_grad_threshold, min_opacity, scene.cameras_extent, size_threshold)
                        gaussians.compute_3D_filter(cameras=trainCameras)

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)
            
            if iteration % 500 == 0:
                num_points[iteration] = gaussians.get_xyz.shape[0]
                print("number of points:", gaussians._xyz.shape[0])
            
            # if iteration == opt.iterations:
            #     with open(os.path.join(args.output_folder, "num_points.json"), "w") as f:
            #         json.dump(num_points, f)
        
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    out_dict = {"scene": scene, "gaussians": gaussians, "tb_writer": tb_writer, "highresolution_index": highresolution_index}
    return out_dict

def train_proposed(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args, dataset2=None):
    # load model and scheduler
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        model_id
    )
    # pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    #     model_id, variant="fp16", torch_dtype=torch.float16
    # )
    pipeline = pipeline.to("cuda")
    # import pdb; pdb.set_trace()
    
    #############################################
    # Loading scene and Gaussians
    #############################################
    input_dict = prepare_training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args, dataset2)
    scene = input_dict["scene"]
    trainCameras = scene.getTrainCameras().copy()
    import pdb; pdb.set_trace()
    #############################################
    # Prepare for SR method
    #############################################
    prompt = ["",""]
    # SR_infos = {}
    img_input = []
    for i in range(len(trainCameras)):
        low_res_img = Image.open(os.path.join('/fs/nexus-projects/dyn3Dscene/Codes/data/orig/bicycle_orig', f"images_{int(args.resolution)*4}", trainCameras[i].image_name+".JPG")).convert("RGB")
        img_input.append(low_res_img)
    prepared_info = pipeline.prepare_proposed(prompt=prompt, image=img_input, num_inference_steps=num_inference_steps)
    # SR_infos[trainCameras[i].image_name] = prepared_info
    
    timesteps = pipeline.scheduler.timesteps
    iters = len(timesteps) // consecutive_timesteps
    
    for iteration in range(iters):
        print("*** Iteration:", iteration, '***')
        #############################################
        # Run the SR pipeline with loop (predicting x0_head)
        #############################################   
        input_timesteps = timesteps[iteration * consecutive_timesteps: (iteration + 1) * consecutive_timesteps]                
        prepared_info['timesteps'] = input_timesteps
        prepared_info['start_iter'] = iteration * consecutive_timesteps                
        # with torch.no_grad():
        prepared_info = pipeline.proposed(in_dict=prepared_info)
        
       
        #############################################
        # Create folder to save x0_head as GS input training images
        #############################################
        training_folder = os.path.join(args.output_folder, 'training_with_step',f"timesteps_{iteration * consecutive_timesteps}_{(iteration + 1) * consecutive_timesteps}", "input_training")
        if os.path.exists(training_folder) == False:
            os.makedirs(training_folder)
            
        #############################################
        # Decode and save the x0_head as GS input training images
        #############################################
        # decode_info = prepared_info.copy()
        # decode_info['latents'] = prepared_info['x0_head'].clone()
        out_img = pipeline.postprocess_proposed(latents=prepared_info['x0_head'], prompt_embeds=prepared_info['prompt_embeds'])
        for i in range(len(trainCameras)):
            out_img[0][i].save(os.path.join(training_folder, trainCameras[i].image_name + ".png"))
            # import pdb; pdb.set_trace()

                
        x0_tilde = []
        for i in range(len(trainCameras)):            
            # Read GS rendering as blended x0_tilde
            file_name = os.path.join(training_folder, trainCameras[i].image_name + ".png")
            loaded_image = Image.open(file_name).convert("RGB")
            width, height = loaded_image.size
            im = PILtoTorch(loaded_image, (width, height)).cuda()
            # import pdb; pdb.set_trace()
            im = pipeline.image_processor.normalize(im)
            # im = (im - 0.5) * 2
            
            im_latent = None
            with torch.no_grad():
                # import pdb; pdb.set_trace() 
                im_latent = pipeline.vae.encode(im.unsqueeze(0), return_dict=False)[0].sample()                
                im_latent = pipeline.vae.config.scaling_factor * im_latent
            x0_tilde.append(im_latent)
        
        x0_tilde = torch.concat(x0_tilde, dim=0)   
        latents_t_1 = pipeline.calc_x_t_1(in_dict=prepared_info, x0_head_new=x0_tilde)     
        prepared_info['latents'] = latents_t_1
        prepared_info['x0_head'] = x0_tilde
        """
        #############################################
        # Update ground truth image in trainCameras  
        #############################################      
        for i in range(len(trainCameras)):
            # If you read from the saved image, you can use the following code
            # file_name = os.path.join(training_folder, trainCameras[i].image_name + ".png")
            # img_transfer = Image.open(file_name).convert("RGB")
            
            # If you use the SD SR output x0_head image
            img_transfer = out_img[0][i]
            width, height = img_transfer.size
            loaded_image = PILtoTorch(img_transfer, (width, height)).cuda()
            # import pdb; pdb.set_trace()
            trainCameras[i].original_image = loaded_image.clone()
        
        #############################################
        # Train GS
        #############################################
        input_dict = training_with_iters(input_dict, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args, dataset2, SR_iter=iteration, SR_step=consecutive_timesteps)
        # torch.cuda.empty_cache()
        
        
        #############################################
        # Update blended x0_tilde to pipeline
        #############################################
        training_folder = os.path.join(args.output_folder, 'training_with_step',f"timesteps_{iteration * consecutive_timesteps}_{(iteration + 1) * consecutive_timesteps}", "GS_training_results")
        x0_tilde = []
        for i in range(len(trainCameras)):
            # Read GS rendering as blended x0_tilde
            file_name = os.path.join(training_folder, trainCameras[i].image_name + ".png")
            loaded_image = Image.open(file_name).convert("RGB")
            width, height = loaded_image.size
            im = PILtoTorch(loaded_image, (width, height)).cuda()
            im = pipeline.image_processor.normalize(im)
            # import pdb; pdb.set_trace()
            # im = (im - 0.5) * 2
            
            im_latent = None
            with torch.no_grad():
                im_latent = pipeline.vae.encode(im.unsqueeze(0), return_dict=False)[0].sample()                
                im_latent = pipeline.vae.config.scaling_factor * im_latent
            x0_tilde.append(im_latent)
        
        x0_tilde = torch.concat(x0_tilde, dim=0)        
        
        # import pdb; pdb.set_trace()
        x0_tilde = prepared_info['x0_head']
        latents_t_1 = pipeline.calc_x_t_1(in_dict=prepared_info, x0_head_new=x0_tilde)
        # x_t_1_head = pipeline.scheduler.step( prepared_info['noise_pred'], input_timesteps, prepared_info['latents'], **prepared_info['extra_step_kwargs'], return_dict=True, x0_head=im_latent)['prev_sample']        
        prepared_info['latents'] = latents_t_1
        prepared_info['x0_head'] = x0_tilde
        """
        # width, height = loaded_image.size
        # loaded_image = PILtoTorch(loaded_image, (width, height)).cuda()
        # trainCameras[i].original_image = loaded_image
    # import pdb; pdb.set_trace()

            
        
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args, dataset2=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)   
    
    if args.fidelity_train_en:
        dataset2.source_path = dataset.source_path.split("_SR")[0]
        dataset2.resolution = dataset.resolution * 4
        gaussians2 = GaussianModel(dataset2.sh_degree)
        scene2 = Scene(dataset2, gaussians2, shuffle=False)

        trainCameras2 = scene2.getTrainCameras().copy()
        testCameras2 = scene2.getTestCameras().copy()
        allCameras2 = trainCameras2 + testCameras2
        viewpoint_stack2 = None
    
    if args.load_pretrain:
        scene = Scene(dataset, gaussians, load_iteration=30000, shuffle=False, train_tiny=args.train_tiny)
        scene.model_path = args.output_folder
        dataset_name = os.path.basename(dataset.source_path)
        dataset.model_path = os.path.join(args.output_folder, dataset_name)
        tb_writer = prepare_output_and_logger(dataset)
        scene.model_path = dataset.model_path
        
        if args.prune_init_en:
            num_points = scene.gaussians._xyz.shape[0]
            valid_ids = torch.randperm(num_points)[:int(num_points * prune_ratio+0.5)]
            # Prune points
            gaussians._xyz = gaussians._xyz[valid_ids].clone().detach()
            gaussians._features_dc = gaussians._features_dc[valid_ids].clone().detach()
            gaussians._features_rest = gaussians._features_rest[valid_ids].clone().detach()
            gaussians._scaling = gaussians._scaling[valid_ids].clone().detach()
            gaussians._rotation = gaussians._rotation[valid_ids].clone().detach()
            gaussians._opacity = gaussians._opacity[valid_ids].clone().detach()
    else:
        scene = Scene(dataset, gaussians)

    print("--- before super resolving points:", gaussians._xyz.shape[0])
    gaussians.training_setup(opt)
    if args.SR_GS:
        gaussians.super_resolving_gaussians(2)
        gaussians.training_setup(opt)
        print("--- after super resolving points:", gaussians._xyz.shape[0])
    elif args.load_pretrain:
        gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), dtype=torch.float32, device="cuda")
        gaussians.training_setup(opt)
        print("--- after loading pretrain points:", gaussians._xyz.shape[0])

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
    # allCameras = trainCameras + testCameras
    
    # highresolution index
    highresolution_index = []
    for index, camera in enumerate(trainCameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    gaussians.compute_3D_filter(cameras=trainCameras)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    try:
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    except:
        import pdb; pdb.set_trace()
    first_iter += 1

    if args.lpips_train_en:
        lpips_fn = lpips.LPIPS(net='vgg').cuda(0)
    if args.musiq_train_en:
        metric_musiq = pyiqa.create_metric("musiq").cuda(0)
    
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
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size, subpixel_offset=subpixel_offset)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # sample gt_image with subpixel offset
        if dataset.resample_gt_image:
            gt_image = create_offset_gt(gt_image, subpixel_offset)

        # edge_aware_loss_en = False
        if args.edge_aware_loss_en:
            dx = torch.abs(image[:, :,1:] - image[:,:,:-1])
            dy = torch.abs(image[:,1:,:] - image[:,:-1,:])
            dx_norm = dx / dx.max()
            dy_norm = dy / dy.max()
                        
            dx2 = torch.zeros_like(image)
            dy2 = torch.zeros_like(image)
            dx2[:,:,1:] = dx_norm
            dy2[:,1:,:] = dy_norm
            mix = dx2 + dy2
            wt = torch.exp(mix)
            L1_loss = torch.abs((image - gt_image))
            
            Ll1 = (L1_loss * wt).mean()
        else:
            Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # Save GT image
        # torchvision.utils.save_image(gt_image, os.path.join('gt.png'))
        
        if args.musiq_train_en:
            musiq_scroe = metric_musiq(image)
            loss += 1 / musiq_scroe[0][0] * 5
        
        # Add LPIPS loss
        if args.lpips_train_en:
            lpips_loss = lpips_fn(image, gt_image)[0][0][0][0]
            lpips_weight = 0.2
            loss += lpips_loss * lpips_weight
        
        # Fidelity training
        if args.fidelity_train_en:
            if not viewpoint_stack2:
                viewpoint_stack2 = scene2.getTrainCameras().copy()
            viewpoint_cam2 = viewpoint_stack2.pop(pop_id)
            avg_pool = torch.nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
            img_small = avg_pool(image)
            gt_img_small = view_point_cam2.original_image.cuda()
            # torchvision.utils.save_image(img_small, "img_small.png")
            # torchvision.utils.save_image(gt_img_small, "gt_img_small.png")
            # torchvision.utils.save_image(image, "img.png")
            # torchvision.utils.save_image(gt_image, "gt_img.png")
            L1_2 = l1_loss(img_small, gt_img_small)
            loss += (1.0 - opt.lambda_dssim) * L1_2 + opt.lambda_dssim * (1.0 - ssim(img_small, gt_img_small))
        
        if iteration > opt.iterations - len(trainCameras):
            training_folder = os.path.join(args.output_folder, 'training_with_step')
            if not os.path.exists(training_folder):
                os.makedirs(training_folder)
            file_name = os.path.join(training_folder, viewpoint_cam.image_name + ".png")
            torchvision.utils.save_image(image, os.path.join(file_name))
            
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
            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)
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
                        # gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                        gaussians.densify_and_prune(opt.densify_grad_threshold, min_opacity, scene.cameras_extent, size_threshold)
                        gaussians.compute_3D_filter(cameras=trainCameras)

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
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

    import pdb; pdb.set_trace()
        
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

if __name__ == "__main__":
    # Set up command line argument parser
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
    parser.add_argument("--musiq_train_en", action="store_true")
    parser.add_argument("--lpips_train_en", action="store_true")
    parser.add_argument("--prune_init_en", action="store_true")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--train_tiny", action="store_true")
    parser.add_argument("--edge_aware_loss_en", action="store_true")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Set up random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # if args.fidelity_train_en:
    #     training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args, dataset2=lp.extract(args))
    # else:
    #     training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)
    train_proposed(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)
    # All done
    print("\nTraining complete.")
