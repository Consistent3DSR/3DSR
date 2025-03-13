import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

def interpolate_camera_poses(cam1, cam2, num_frames):
    """
    Interpolates num_frames intermediate camera extrinsics between cam1 and cam2.

    Args:
        cam1 (numpy.ndarray): (4,4) Camera extrinsic matrix (R|t)
        cam2 (numpy.ndarray): (4,4) Camera extrinsic matrix (R|t)
        num_frames (int): Number of interpolated frames to generate.

    Returns:
        list of numpy.ndarray: A list of (4,4) interpolated camera extrinsic matrices.
    """
    # Extract rotation matrices (3x3) and translation vectors (3x1)
    R1, t1 = cam1[:3, :3], cam1[:3, 3]
    R2, t2 = cam2[:3, :3], cam2[:3, 3]

    # Convert rotation matrices to quaternions
    rot1 = R.from_matrix(R1)
    rot2 = R.from_matrix(R2)
    
    # Define keyframes for SLERP (rotation interpolation)
    key_times = [0, 1]  # Start and end
    key_rots = R.from_quat([rot1.as_quat(), rot2.as_quat()])  # Convert to quaternions
    slerp = Slerp(key_times, key_rots)  # Create SLERP object

    interpolated_poses = []

    for i in range(1, num_frames + 1):
        alpha = i / (num_frames + 1)  # Normalized interpolation factor

        # Interpolate rotation using SLERP
        interp_R = slerp(alpha).as_matrix()

        # Interpolate translation using linear interpolation (LERP)
        interp_t = (1 - alpha) * t1 + alpha * t2

        # Construct interpolated extrinsic matrix
        interp_extrinsic = np.eye(4)
        interp_extrinsic[:3, :3] = interp_R
        interp_extrinsic[:3, 3] = interp_t

        interpolated_poses.append(interp_extrinsic)

    return interpolated_poses

# Example usage:
cam1 = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

cam2 = np.array([
    [0.866, -0.5, 0, 1],
    [0.5, 0.866, 0, 2],
    [0, 0, 1, 3],
    [0, 0, 0, 1]
])

num_frames = 5
interpolated_cameras = interpolate_camera_poses(cam1, cam2, num_frames)

for i, cam in enumerate(interpolated_cameras):
    print(f"Frame {i+1}:\n{cam}\n")