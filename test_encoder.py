import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch
import torch.nn.functional as F
from utils.general_utils import PILtoTorch
import numpy as np
import os

if __name__ == "__main__":
    factors = [8]
    for res_id in range(len(factors)):
        input_folder = f"/fs/nexus-projects/dyn3Dscene/Codes/data/orig/bicycle_orig/images_{factors[res_id]}"
        output_folder = f"/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting/test_encoder_decoder_new"
        if os.path.exists(output_folder) == False:
            os.makedirs(output_folder)
        img_files = os.listdir(input_folder)
        img_files.sort()
        
        # load model and scheduler
        model_id = "stabilityai/stable-diffusion-x4-upscaler"
        pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            model_id
        )
        
        pipeline = pipeline.to("cuda")
        
        input_path = os.path.join(input_folder, img_files[0])
        output_path = os.path.join(output_folder, img_files[0])
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        low_res_img = Image.open(input_path).convert("RGB")
        
        for i in range(100):
            print("Processing image ", i)
            out_file_name = os.path.join(output_folder, img_files[0].split(".")[0] + f"_t_{i}.png")
            loaded_image = Image.open(input_path).convert("RGB")
            width, height = loaded_image.size
            im = PILtoTorch(loaded_image, (width, height)).cuda()            
            im = pipeline.image_processor.normalize(im)
            im_latent = None
            with torch.no_grad():                
                im_latent = pipeline.vae.encode(im.unsqueeze(0), return_dict=False)[0].sample()
                # im_latent = pipeline.vae.config.scaling_factor * im_latent
                image_out = pipeline.vae.decode(im_latent, return_dict=False)[0]
                output = pipeline.image_processor.denormalize(image_out)
                # import pdb; pdb.set_trace()
                output = pipeline.image_processor.pt_to_numpy(output)
                out_img = pipeline.image_processor.numpy_to_pil(output)
                
            # out_img = pipeline.postprocess_proposed(latents=im_latent, prompt_embeds="")
            
            out_img[0].save(out_file_name)
            input_path = out_file_name