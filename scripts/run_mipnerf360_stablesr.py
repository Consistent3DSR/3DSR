# single-scale trainig and single-scale testing as in the original mipnerf-360

import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import queue
import time

# scenes = ["bicycle", "bonsai", "counter", "flowers", "garden", "stump", "treehill", "kitchen", "room"]
# factors = [4, 2, 2, 4, 4, 4, 4, 2, 2]

# scenes = ["bonsai", "counter", "flowers", "garden", "stump", "treehill", "kitchen"]
# factors = [16, 16, 16, 16, 16, 16, 16]
# factors = [4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2]#[8, 16, 8, 16, 8, 16,  8, 16, 8, 16, 8, 16, 8, 16,]
# scenes = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"] #["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex" ] #"fern", 
# scenes = ["Aqua", "Bedroom", "Boats", "Bridge", "CreepyAttic", "Hugo-1", "Library", "Museum-1", 
#           "Museum-2", "NightSnow", "Playroom", "Ponche", "SaintAnne", "Shed", "Street-10", 
#           "Tree-18", "Yellowhouse-12"]
# scenes = ["Family", "Horse", "M60", "Playground", "Train", "Truck", "Lighthouse"]
scenes=["bicycle"]
# scenes = [  
#             # "Library",
#             # "Shed",
#             "Street-10",
#             # "Yellowhouse-12"
#          ]
# factors = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
factors = [2] #[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4] #[2,2,2,2,2,2,2,2] #[8, 8,8,8,8,8,8,8]

####################################
# Status Check *******************
# Run original data 
# Run Stable SR data (v)
# Run proposed SR data
# Run bicubic data 
####################################

excluded_gpus = set([])
os.environ["NUM_ITERS"] = '30000'

dry_run = False

jobs = list(zip(scenes, factors))

def train_scene(gpu, scene, factor):
    ####################################
    # Folder information
    ####################################
    # ------------ Dataset folder
    # dataset = "/fs/nexus-projects/dyn3Dscene/Codes/data/my_new_resize"
    # dataset = "/fs/nexus-projects/dyn3Dscene/Codes/data/bicubic"
    # dataset = "/fs/nexus-projects/dyn3Dscene/Codes/data/proposed"
    dataset_name = "mipnerf360"
    dataset_gt = f"/fs/nexus-projects/dyn3Dscene/Codes/datasets/{dataset_name}"
    dataset = f"/fs/nexus-projects/dyn3Dscene/Codes/datasets/SR_results/StableSR/{dataset_name}"
    # dataset_gt = "/fs/nexus-projects/dyn3Dscene/Codes/datasets/mipnerf360"
    # ------------ Output folder
    # Original data
    # output_dir = f"/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/{dataset_name}/orig_new/input_DS_{int(factor)}"
    # Stable SR data
    # output_dir = f"/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/independent_SR/StableSR/original_setting/input_DS_{int(factor)}"
    output_dir = f"/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/independent_SR/mipnerf360/StableSR/original_setting/input_DS_{int(factor)}"
    # output_dir = f"/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/llff/StableSR/input_DS_{int(factor)}"
    # output_dir = f"/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/independent_SR/llff/StableSR/input_DS_{int(factor)}"
    # output_dir = f"/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/independent_SR/{dataset_name}/StableSR/consistent_noise/input_DS_{int(factor)}"
    # Proposed SR data
    # output_dir = f"/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/proposed_SR/x4_upsample_to_DS_{int(factor)}_3DGS_iter_2000_fidelity_ratio_0.5"
    # Bicubic data
    # output_dir = f"/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/bicubic/x4_upsample_to_DS_{int(factor)}"
    
    

    ####################################
    # Training command
    ###################################
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s {dataset}/{scene} -m {output_dir}/{scene} --eval -r {factor} --port {6002+int(gpu)} --kernel_size 0.1 --output_folder {output_dir}/{scene}"
    print(cmd)
    if not dry_run:
        os.system(cmd)

    # ####################################
    # # Rendering command
    # ####################################
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {output_dir}/{scene} -r {int(factor)} --data_device cpu --skip_train"
    print(cmd)
    if not dry_run:
        os.system(cmd)
    
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {output_dir}/{scene} -r {int(factor)} --data_device cpu --vis_video"
    print(cmd)
    if not dry_run:
        os.system(cmd)
    
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {output_dir}/{scene} -r {int(factor*4)} --data_device cpu --skip_train"
    # print(cmd)
    # if not dry_run:
    #     os.system(cmd)
    
    ####################################
    # Evaluation command
    ####################################
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {output_dir}/{scene} -g {dataset_gt}/{scene}"    
    print(cmd)
    if not dry_run:
        os.system(cmd)
    
    cmd = f"python evaluate.py --gt_folder /fs/nexus-projects/SR3D/Results/outputs/mip-splatting-multiresolution/load_DS_8/train_proposed_DS_2_fidelity_wt_1_iter_5000_stop_densify_2500_0304/${scene}/ours_10000/DS_2/gt_2 --img_folder /fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/independent_SR/StableSR/original_setting/input_DS_2/${scene}/ours_30000/DS_2/test_preds_2"
    print(cmd)
    if not dry_run:
        os.system(cmd)
    
    return True


def worker(gpu, scene, factor):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, factor)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.
    
def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1))
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)
        
        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, *job)  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., rellasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(5)
        
    print("All jobs have been processed.")


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)
