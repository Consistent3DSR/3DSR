#!/bin/bash

gpu=0
factors=4
scenes=("bicycle")  # List of scenes
export NUM_ITERS=5000
lambda_hrs=(0.6 1)

dataset="/fs/nexus-projects/dyn3Dscene/Codes/data/my_new_resize"
output_base="/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/my_data/new_resize/original_setting"

# Loop through each lambda_hr
for lambda_hr in "${lambda_hrs[@]}"; do
    # Ensure proper floating point representation
    lambda_hr_formatted=$(printf "%g" "$lambda_hr")
    
    # Define variation name dynamically
    variation="fidelity_train_${lambda_hr_formatted}_lambda_hr"
    
    exp_base="/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/mip-splatting-multiresolution"
    exp_dir="${exp_base}/load_DS_$((factors * 4))/train_proposed_DS_$((factors))_full_dataset_${variation}"

    # Loop through each scene
    for scene in "${scenes[@]}"; do
        echo "Processing scene: $scene with $NUM_ITERS iterations"
        echo "Experiment directory: ${exp_dir}/${scene}"

        # Training
        OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$gpu python train_proposed_2025.py \
            -s ${dataset}/${scene} \
            -m ${output_base}/input_DS_$((factors * 4))/${scene} \
            -r $factors \
            --port $((6009 + gpu)) \
            --eval \
            --output_folder ${exp_dir} \
            --load_pretrain \
            --kernel_size 0.1 \
            --config /fs/nexus-projects/dyn3Dscene/Codes/StableSR/configs/stableSRNew/v2-finetune_text_T_512.yaml \
            --ckpt /fs/nexus-projects/dyn3Dscene/Codes/StableSR/weights/stablesr_turbo.ckpt \
            --init-img ${dataset}/${scene}/images_$((factors * 4)) \
            --outdir ${exp_dir}/${scene}/outputs_proposed_load_DS_$((factors * 4))/ \
            --ddpm_steps 4 \
            --dec_w 0.5 \
            --seed 42 \
            --n_samples 1 \
            --vqgan_ckpt /fs/nexus-projects/dyn3Dscene/Codes/StableSR/weights/vqgan_cfw_00011.ckpt \
            --colorfix_type wavelet \
            --upscale 4 \
            --fidelity_train
            # --lambda_hr "$lambda_hr_formatted" \
            

        # Rendering
        OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${gpu} python render.py -m ${exp_dir}/${scene} -r ${factors} --data_device cpu --skip_train

        # Evaluation
        OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${gpu} python metrics.py -m ${exp_dir}/${scene} -g ${dataset}/${scene}

        echo "Completed processing for scene: $scene with $NUM_ITERS iterations and lambda_hr=${lambda_hr_formatted}"

    done
done