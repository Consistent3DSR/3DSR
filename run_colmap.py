import pycolmap
import natsort
# # Run COLMAP's sparse reconstruction on a dataset
# pycolmap.process_structure_from_motion(
#     path="/fs/nexus-projects/dyn3Dscene/Codes/datasets/nerf_llff_data/fern",
#     camera_model="OPENCV",  # Specify camera model
#     single_camera=True,  # Assume all images use the same intrinsics
# )

# sfm = pycolmap.SfmPipeline(
#     dataset_path="/fs/nexus-projects/dyn3Dscene/Codes/datasets/nerf_llff_data/fern",  # Folder containing images
#     output_path="/fs/nexus-projects/dyn3Dscene/Codes/datasets/nerf_llff_data/fern/processed",    # Where COLMAP outputs the results
#     camera_model="OPENCV",           # Camera model (PINHOLE, SIMPLE_PINHOLE, etc.)
#     single_camera=True                # Assume all images use the same intrinsics
# )
from os import makedirs
# scenes = ["flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
scenes = ["Family", "Horse", "M60", "Panther", "Playground", "Train", "Truck"]

for scene in scenes:
    output_path = f'/fs/nexus-projects/dyn3Dscene/Codes/datasets/TanksAndTempleBG/{scene}/colmap'
    image_dir = f'/fs/nexus-projects/dyn3Dscene/Codes/datasets/TanksAndTempleBG/{scene}/rgb'

    makedirs(output_path, exist_ok=True)
    mvs_path = f"{output_path}/mvs"
    database_path = f"{output_path}/database.db"

    pycolmap.extract_features(database_path, image_dir)
    pycolmap.match_exhaustive(database_path)
    maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
    maps[0].write(output_path)
    # dense reconstruction
    pycolmap.undistort_images(mvs_path, output_path, image_dir)
    pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
    pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)