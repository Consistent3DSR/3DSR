# dataset_name="deep_blending"
# scenes=(
#     # Aqua
#     # Bedroom
#     # Boats
#     # Bridge
#     # CreepyAttic
#     # DrJohnson
#     # Hugo-1
#     # Museum-1
#     # Museum-2
#     # NightSnow
#     # Playroom
#     # Ponche
#     # SaintAnne
#     # Street-10
#     # Tree-18
#     # Yellowhouse-12
#     # flower
#     )
# dataset_name="llff"
# scenes=(
#     # fern
#     flower
#     # fortress
#     horns
#     # leaves
#     orchids
#     # room
#     trex
# )
factor=8
dataset_name="mipnerf360"
scenes=(
    # bicycle
    # bonsai
    # counter     
    flowers    
    # garden   
    # kitchen  
    # room
    # stump
    # treehill
)
# methods=(
#     # "SinSR"
#     # "SUPIR"
#     # "StableSR"
#     # "resshift"
#     # "OSEDiff"
# )
# dataset_name="llff"
for scene in "${scenes[@]}"; do    
    # Ours
    # LLFF
    # python evaluate.py \
    #     --gt_folder "/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/${dataset_name}/proposed/load_DS_$((factor * 4))/train_proposed_DS_2_fidelity_wt_1_iter_5000_stop_densify_2500_0305/${scene}/ours_15000/DS_${factor}/gt_${factor}" \
    #     --img_folder "/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/${dataset_name}/proposed/load_DS_$((factor * 4))/train_proposed_DS_2_fidelity_wt_1_iter_5000_stop_densify_2500_0305/${scene}/ours_15000/DS_${factor}/test_preds_${factor}"

    # MipNeRF360        
    # python evaluate.py \
    #     --gt_folder "/fs/nexus-projects/SR3D/Results/outputs/mip-splatting-multiresolution/load_DS_8/train_proposed_DS_2_fidelity_wt_1_iter_5000_stop_densify_2500_0304/${scene}/ours_10000/DS_2/gt_2" \
        # --img_folder "/fs/nexus-projects/SR3D/Results/outputs/mip-splatting-multiresolution/load_DS_8/train_proposed_DS_2_fidelity_wt_1_iter_5000_stop_densify_2500_0304/${scene}/ours_10000/DS_2/test_preds_2"

        # --gt_folder "/fs/nexus-projects/SR3D/Results/outputs/mip-splatting-multiresolution/load_DS_8/current_best/fidelity_w_1_iter_5000_stop_densify_2500/${scene}/ours_20000/DS_2/gt_2" \
        # --img_folder "/fs/nexus-projects/SR3D/Results/outputs/mip-splatting-multiresolution/load_DS_8/current_best/fidelity_w_1_iter_5000_stop_densify_2500/${scene}/ours_20000/DS_2/test_preds_2"

    # python evaluate.py \
    #     --gt_folder "/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/${dataset_name}/proposed/load_DS_$((factor * 4))/train_proposed_DS_${factor}_fidelity_wt_1_lr_wt_1_iter_5000_0306_0427/${scene}/ours_35000/DS_${factor}/gt_${factor}" \
    #     --img_folder "/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/${dataset_name}/proposed/load_DS_$((factor * 4))/train_proposed_DS_${factor}_fidelity_wt_1_lr_wt_1_iter_5000_0306_0427/${scene}/ours_35000/DS_${factor}/test_preds_${factor}"
        
    # python evaluate.py \
    #     --gt_folder "/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/${dataset_name}/proposed/load_DS_$((factor * 4))/train_proposed_DS_${factor}_fidelity_wt_1_lr_wt_1_iter_5000_0306_0427/${scene}/ours_37000/DS_${factor}/gt_${factor}" \
    #     --img_folder "/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/${dataset_name}/proposed/load_DS_$((factor * 4))/train_proposed_DS_${factor}_fidelity_wt_1_lr_wt_1_iter_5000_0306_0427/${scene}/ours_37000/DS_${factor}/test_preds_${factor}"
            
    # StableSR
    # python evaluate.py \
    #     --gt_folder "/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/independent_SR/StableSR/original_setting/input_DS_2/${scene}/ours_30000/DS_2/gt_2" \
    #     --img_folder "/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/independent_SR/StableSR/original_setting/input_DS_2/${scene}/ours_30000/DS_2/test_preds_2"
        # --gt_folder "/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/deep_blending/proposed/load_DS_4/train_proposed_DS_1_fidelity_wt_1_lr_wt_1_iter_5000_0306_0427/${scene}/ours_15000/DS_1/gt_1" \
        # --img_folder "/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/independent_SR/deep_blending/StableSR/input_DS_1/${scene}/ours_30000/DS_1/test_preds_1"

    # SRGS
    python evaluate.py \
        --gt_folder "/fs/nexus-projects/dyn3Dscene/Codes/SRGS/outputs/mipnerf360/input_DS_${factor}/${scene}/test/ours_30000/gt" \
        --img_folder "/fs/nexus-projects/dyn3Dscene/Codes/SRGS/outputs/mipnerf360/input_DS_${factor}/${scene}/test/ours_30000/renders"

    # Render SuperGaussian
    # cd /fs/nexus-projects/dyn3Dscene/Codes/SuperGaussian/third_parties/gaussian-splatting
    #     CUDA_VISIBLE_DEVICES=0 /fs/nexus-scratch/ytchen/miniconda3/envs/superGS/bin/python render.py \
    #         --model_path /fs/nexus-projects/dyn3Dscene/Codes/SuperGaussian/outputs/DS_$factor/$scene --evaluate
    
    # cd /fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine
    # # SuperGaussian
    # python evaluate.py \
    #     --gt_folder "/fs/nexus-projects/dyn3Dscene/Codes/SuperGaussian/outputs/DS_8/${scene}/ours_30000/gt" \
    #     --img_folder "/fs/nexus-projects/dyn3Dscene/Codes/SuperGaussian/outputs/DS_8/${scene}/ours_30000/renders"
    
    # python evaluate.py \
    #     --gt_folder "/fs/nexus-projects/dyn3Dscene/Codes/SuperGaussian/outputs/DS_8/room/point_cloud/iteration_30000/renders/gt" \
    #     --img_folder "/fs/nexus-projects/dyn3Dscene/Codes/SuperGaussian/outputs/DS_8/room/point_cloud/iteration_30000/renders/test"
    
    
    #SRGS                      
    # python evaluate.py \
    #     --img_folder "/fs/nexus-projects/dyn3Dscene/Codes/SRGS/outputs/llff/${scene}/test/ours_30000/renders" \
    #     --gt_folder "/fs/nexus-projects/dyn3Dscene/Codes/SRGS/outputs/llff/${scene}/test/ours_30000/gt"
                     
    #SRGS-SwinIR
    # python evaluate.py \
    #     --img_folder "/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/${dataset_name}/SRGS_SwinIR/input_DS_${factor}/no_pretrain/${scene}/ours_30000/DS_${factor}/test_preds_${factor}" \
    #     --gt_folder "/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/${dataset_name}/SRGS_SwinIR/input_DS_${factor}/no_pretrain/${scene}/ours_30000/DS_${factor}/gt_${factor}"
    
    # DiSR-NeRF
    # python evaluate.py \
    #     --gt_folder "/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/${dataset_name}/proposed/load_DS_$((factor * 4))/train_proposed_DS_${factor}_fidelity_wt_1_iter_5000_stop_densify_2500_0305/${scene}/ours_35000/DS_${factor}/gt_${factor}" \
    #     --img_folder "/fs/nexus-projects/dyn3Dscene/Codes/DiSR-NeRF/outputs/LLFF_rm_val_pose_0515_${scene}/save/sr_val/results"
        
    
    
done
# for method in "${methods[@]}"; do
#     for scene in "${scenes[@]}"; do
#         # CUDA_VISIBLE_DEVICES=0,1 python test.py \
#         #     --img_dir "/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/my_data/new_resize/original_setting/input_DS_${factor}/${s}/ours_-1/DS_${factor}/vis_video" \
#         #     --save_dir "/fs/nexus-projects/dyn3Dscene/Codes/data/SR/${method}/video/upsample_${factor}/${s}" \
#         #     --SUPIR_sign Q \
#         #     --upscale ${factor} \
#         #     --no_llava
#         # echo " Done ${s} upsampling ---------"
#         python evaluate.py \
#             --gt_folder "/fs/nexus-projects/dyn3Dscene/Codes/datasets/${dataset_name}/${scene}/images" \
#             --img_folder "/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/${dataset_name}/load_DS_4/train_proposed_SR_0423/${scene}/ours_30000/DS_1/test_preds_1"
        
#         # python evaluate.py \
#         #     --gt_folder "/fs/nexus-projects/dyn3Dscene/Codes/data/my_new_resize/${scene}/images_${factor}" \
#         #     --img_folder "/fs/nexus-projects/dyn3Dscene/Codes/data/${method}/${scene}/images_${factor}"
#         #     # --img_folder "/fs/nexus-projects/dyn3Dscene/Codes/data/SR/${method}/${scene}"

#     done
# done
