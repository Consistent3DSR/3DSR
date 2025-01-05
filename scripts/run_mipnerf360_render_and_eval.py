# single-scale trainig and single-scale testing as in the original mipnerf-360

import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import queue
import time

# scenes = ["bicycle", "bonsai", "counter", "flowers", "garden", "stump", "treehill", "kitchen", "room"]
# factors = [4, 2, 2, 4, 4, 4, 4, 2, 2]

scenes = ["bicycle"]
factors = [8] #[2, 4, 8, 16, 32] #[1, 2, 4, 8] #[1, 2, 4, 8, 16, 32]

excluded_gpus = set([])

dry_run = False

jobs = list(zip(scenes, factors))

def train_scene(gpu, scene, factor):
    # output_dir = f"outputs/original_data/load_from_DS_1_then_resize/input_DS_{int(factor)}"
    # output_dir = f"outputs/original_data/original_setting/input_DS_{int(factor)}"
    output_dir = f"/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting/outputs/my_data/original_setting/input_DS_{int(factor)*4}"
    exp_dir = f"/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting/outputs/mip-splatting-multiresolution/load_DS_{int(factor)*4}/apply_edge_aware_loss"
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python visualize.py -m {output_dir}/{scene} -r {int(factors[0])} --data_device cpu --skip_train"
    # print(cmd)    
    # if not dry_run:
    #     os.system(cmd)
    
    # # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s /fs/nexus-projects/dyn3Dscene/Codes/data/{scene}_orig -m {output_dir}/{scene} --eval -r {factor} --port {6009+int(gpu)} --kernel_size 0.1 --output_folder {output_dir}/{scene}"
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s /fs/nexus-projects/dyn3Dscene/Codes/data/my_new_resize/{scene} -m {output_dir}/{scene} --eval -r {factor} --port {6009+int(gpu)} --kernel_size 0.1 --output_folder {output_dir}/{scene}"
    # print(cmd)
    # if not dry_run:
    #     os.system(cmd)

    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {output_dir}/{scene} -r 4 --data_device cpu --skip_train"
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {exp_dir}/{scene} -r 8 --data_device cpu --skip_train --video"
    print(cmd)    
    if not dry_run:
        os.system(cmd)
    
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {exp_dir}/{scene} -g /fs/nexus-projects/dyn3Dscene/Codes/data/my_resize/{scene}"
    # # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {output_dir}/{scene} -g /fs/nexus-projects/dyn3Dscene/Codes/data/{scene}"
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