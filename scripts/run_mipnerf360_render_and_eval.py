# single-scale trainig and single-scale testing as in the original mipnerf-360

import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import queue
import time

# scenes = [ "counter", "flowers", "garden", "stump", "treehill", "kitchen", "room"]#["bicycle", "bonsai", "counter", "flowers", "garden", "stump", "treehill", "kitchen", "room"]
# factors = [4, 4, 4, 4, 4, 4, 4]

scenes = ['fern']#["drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"] #["bicycle", "bonsai", "counter", "flowers", "garden", "stump", "treehill", "kitchen", "room"]
factors = [2] # [4, 4, 4, 4, 4, 4, 4]

excluded_gpus = set([])

dry_run = False

jobs = list(zip(scenes, factors))

def train_scene(gpu, scene, factor):
    ####################################
    # Folder information
    ####################################
    # ------------ Dataset folder
    dataset = "/fs/nexus-projects/dyn3Dscene/Codes/datasets/nerf_llff_data"
    # dataset = "/fs/nexus-projects/dyn3Dscene/Codes/datasets/mipnerf360"
    
    # ------------ Loaded pretrained model folder
    # output_dir = f"/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/my_data/new_resize/original_setting/input_DS_{int(factor)*4}"
    # output_dir = f"/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/my_data/new_resize/original_setting/input_DS_{int(factor)}"
    output_dir = f"/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/my_data/new_resize/original_setting/input_DS_{int(factor)}"

    # ------------ Current experiment folder
    # Original data
    # exp_dir = f"/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/my_data/new_resize/original_setting/input_DS_{int(factor)*4}"
    exp_dir = f"/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/llff/proposed/load_DS_8/train_proposed_DS_2_fidelity_wt_1_iter_5000_stop_densify_2500_0305"
    # Stable SR
    # exp_dir = f"/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/independent_SR/StableSR/original_setting/input_DS_{factor}"
    # Proposed method
    # exp_dir = f"/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/mip-splatting-multiresolution/load_DS_{int(factor)*4}/train_proposed_DS_{int(factor)}_full_dataset_0226"
    
    ####################################
    # Training command
    ####################################
    # Load preatrained model
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s {dataset}/{scene} -m {output_dir}/{scene} --eval -r {int(factor)} --port {6009+int(gpu)} --kernel_size 0.1 --output_folder {exp_dir} --load_pretrain"
    # Train original
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s {dataset}/{scene} -m {output_dir}/{scene} --eval -r {int(factor)} --port {6009+int(gpu)} --kernel_size 0.1 --output_folder {exp_dir}"
    # print(cmd)
    # if not dry_run:
    #     os.system(cmd)

    ####################################
    # Rendering command
    ####################################
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render_debug.py -m {exp_dir}/{scene} -r 1 --data_device cpu --skip_train"
    
    # ---------- Render test data
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {exp_dir}/{scene} -r {factor} --data_device cpu --skip_train --iteration 5000"
    # Render training data
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {exp_dir}/{scene} -r 2 --data_device cpu --skip_test --train_tiny"
    print(cmd)
    if not dry_run:
        os.system(cmd)
        
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {exp_dir}/{scene} -r {factor} --data_device cpu --skip_train --iteration 10000"
    # Render training data
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {exp_dir}/{scene} -r 2 --data_device cpu --skip_test --train_tiny"
    print(cmd)
    if not dry_run:
        os.system(cmd)
    
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {exp_dir}/{scene} -r {factor} --data_device cpu --skip_train --iteration 15000"
    # Render training data
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {exp_dir}/{scene} -r 2 --data_device cpu --skip_test --train_tiny"
    print(cmd)
    if not dry_run:
        os.system(cmd)
    
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {exp_dir}/{scene} -r {factor} --data_device cpu --skip_train --iteration 20000"
    # Render training data
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {exp_dir}/{scene} -r 2 --data_device cpu --skip_test --train_tiny"
    print(cmd)
    if not dry_run:
        os.system(cmd)
    
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {exp_dir}/{scene} -r {factor} --data_device cpu --skip_train --iteration 35000"
    # Render training data
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {exp_dir}/{scene} -r 2 --data_device cpu --skip_test --train_tiny"
    print(cmd)
    if not dry_run:
        os.system(cmd)

    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {exp_dir}/{scene} -r {factor} --data_device cpu --skip_train --iteration 37000"
    # Render training data
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {exp_dir}/{scene} -r 2 --data_device cpu --skip_test --train_tiny"
    print(cmd)
    if not dry_run:
        os.system(cmd)
    
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {exp_dir}/{scene} -r {factor} --data_device cpu --skip_train"
    # # Render training data
    # # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {exp_dir}/{scene} -r 2 --data_device cpu --skip_test --train_tiny"
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
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics_synthetic.py -m {exp_dir}/{scene} -g {dataset}/{scene}"
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
# with ThreadPoolExecutor(max_workers=8) as executor:
#     dispatch_jobs(jobs, executor)

reserved_gpus = set()
import torch
all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1))
available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)

gpu = available_gpus.pop(0)
for i in range(len(scenes)):
    scene = scenes[i]
    factor = factors[i]
    train_scene(gpu, scene, factor)