# single-scale trainig and single-scale testing as in the original mipnerf-360

import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import queue
import time
from pathlib import Path

# scenes = ["bicycle", "bonsai", "counter", "flowers", "garden", "stump", "treehill", "kitchen", "room"]
# factors = [4, 2, 2, 4, 4, 4, 4, 2, 2]

scenes = ["bicycle"]
factors = [4] #[2, 4, 8]

# num_iters = int(os.environ["NUM_ITERS"])
# consecutive_timesteps = int(os.environ["CONSEC_TIMESTEPS"])
# num_inference_steps = int(os.environ["NUM_INFERENCE_STEPS"])
# prune_ratio = os.environ["PRUNE_RATIO"] if "PRUNE_RATIO" in os.environ else 1.0
# min_opacity = os.environ["MIN_OPACITY"] if "MIN_OPACITY" in os.environ else 0.005

excluded_gpus = set([])


# output_dir = "mip-splatting-ouptut"

dry_run = False

jobs = list(zip(scenes, factors))

def train_scene(gpu, scene, factor):
    ####################################
    # Folder information
    ####################################
    # ------------ Dataset folder
    dataset = "/fs/nexus-projects/dyn3Dscene/Codes/data/proposed"
    dataset_gt = "/fs/nexus-projects/dyn3Dscene/Codes/data/my_new_resize"
    
    # ------------ Loaded pretrained model folder
    # output_dir = f"/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/my_data/new_resize/original_setting/input_DS_{int(factor)*4}"
    output_dir = f"/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/my_data/new_resize/original_setting/input_DS_{int(factor)}"

    # ------------ Current experiment folder
    # Original data
    # exp_dir = f"/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/my_data/new_resize/original_setting/input_DS_{factor}"
    # Stable SR
    # exp_dir = f"/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/independent_SR/StableSR/original_setting/input_DS_{factor}"
    # Proposed method
    exp_dir = f"/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/mip-splatting-multiresolution/load_DS_{int(factor)*4}/train_proposed_SR_DS_{int(factor)}_fidelity_ratio_1"
    
    
    ####################################
    # Training command
    ####################################
    # Load preatrained model
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s {dataset}/{scene} -m {output_dir}/{scene} --eval -r {int(factor)} --port {6009+int(gpu)} --kernel_size 0.1 --output_folder {exp_dir}  --fidelity_train_en --load_pretrain"
    # Train original
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s {dataset}/{scene} -m {output_dir}/{scene} --eval -r {int(factor)} --port {6009+int(gpu)} --kernel_size 0.1 --output_folder {exp_dir}"
    print(cmd)
    if not dry_run:
        os.system(cmd)

    ####################################
    # Rendering command
    ####################################
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {output_dir}/{exp_dir}/{scene} -r 1 --data_device cpu --skip_train"
    
    # ---------- Render test data
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {exp_dir}/{scene} -r {factor} --data_device cpu --skip_train"    
    # # Render training data
    # # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {exp_dir}/{scene} -r 2 --data_device cpu --skip_test --train_tiny"
    # print(cmd)
    # if not dry_run:
    #     os.system(cmd)
    
    # Render training data
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {exp_dir}/{scene} -r 2 --data_device cpu --skip_test --train_tiny"
    # print(cmd)
    # if not dry_run:
    #     os.system(cmd)
    
    # ---------- Render training data
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {exp_dir}/{scene} -r {factor} --data_device cpu --skip_test"
    # print(cmd)
    # if not dry_run:
    #     os.system(cmd)
        
    # ---------- Render interpolated views
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {exp_dir}/{scene} -r 2 --data_device cpu --skip_train --train_tiny --interpolate"        
    # print(cmd)    
    # if not dry_run:
    #     os.system(cmd)
    
    ####################################
    # Evaluation command
    ####################################
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {exp_dir}/{scene} -g {dataset_gt}/{scene}"
    # print(cmd)
    # if not dry_run:
    #     os.system(cmd)
    
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

