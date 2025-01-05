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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import scipy

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size
    if args.resolution in [1, 2, 4, 8, 16, 32, 64]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        # Jamie
        if not (args.resolution > 1 and f"images_{args.resolution}" in cam_info.image_path):
            resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        else:
            resolution = (orig_w, orig_h)
        
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    
    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    cam = Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                image=gt_image, gt_alpha_mask=loaded_mask,
                image_name=cam_info.image_name, uid=id, data_device=args.data_device)
    
    return cam

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        try:
            camera_list.append(loadCam(args, id, c, resolution_scale))
        except:
            print(f"Error loading camera {id} with image {c.image_name}, resolution {resolution_scale}")
            import pdb; pdb.set_trace()

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


# Constants for generate_spiral_path():
NEAR_STRETCH = .9  # Push forward near bound for forward facing render path.
FAR_STRETCH = 5.  # Push back far bound for forward facing render path.
FOCUS_DISTANCE = .75  # Relative weighting of near, far bounds for render path.

def focus_point_fn(poses):
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt

def average_pose(poses):
    """New pose using average position, z-axis, and up vector of input poses."""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world

def integrate_weights_np(w):
    """Compute the cumulative sum of w, assuming all weight vectors sum to 1.

  The output's size on the last dimension is one greater than that of the input,
  because we're computing the integral corresponding to the endpoints of a step
  function, not the integral of the interior/bin values.

  Args:
    w: Tensor, which will be integrated along the last axis. This is assumed to
      sum to 1 along the last axis, and this function will (silently) break if
      that is not the case.

  Returns:
    cw0: Tensor, the integral of w, where cw0[..., 0] = 0 and cw0[..., -1] = 1
  """
    cw = np.minimum(1, np.cumsum(w[..., :-1], axis=-1))
    shape = cw.shape[:-1] + (1,)
    # Ensure that the CDF starts with exactly 0 and ends with exactly 1.
    cw0 = np.concatenate([np.zeros(shape), cw,
                          np.ones(shape)], axis=-1)
    return cw0

def invert_cdf_np(u, t, w_logits):
    """Invert the CDF defined by (t, w) at the points specified by u in [0, 1)."""
    # Compute the PDF and CDF for each weight vector.
    w = np.exp(w_logits) / np.exp(w_logits).sum(axis=-1, keepdims=True)
    cw = integrate_weights_np(w)
    # Interpolate into the inverse CDF.
    interp_fn = np.interp
    t_new = interp_fn(u, cw, t)
    return t_new

def recenter_poses(poses):
    """Recenter poses around the origin."""
    cam2world = average_pose(poses)
    transform = np.linalg.inv(pad_poses(cam2world))
    poses = transform @ pad_poses(poses)
    return unpad_poses(poses), transform

def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)

def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)

def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]

def viewmatrix(lookdir, up, position):
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m

def sample_np(rand,
              t,
              w_logits,
              num_samples,
              single_jitter=False,
              deterministic_center=False):
    """
    numpy version of sample()
  """
    eps = np.finfo(np.float32).eps

    # Draw uniform samples.
    if not rand:
        if deterministic_center:
            pad = 1 / (2 * num_samples)
            u = np.linspace(pad, 1. - pad - eps, num_samples)
        else:
            u = np.linspace(0, 1. - eps, num_samples)
        u = np.broadcast_to(u, t.shape[:-1] + (num_samples,))
    else:
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u_max = eps + (1 - eps) / num_samples
        max_jitter = (1 - u_max) / (num_samples - 1) - eps
        d = 1 if single_jitter else num_samples
        u = np.linspace(0, 1 - u_max, num_samples) + \
            np.random.rand(*t.shape[:-1], d) * max_jitter

    return invert_cdf_np(u, t, w_logits)

def generate_ellipse_path(poses, n_frames=120, const_speed=True, z_variation=0., z_phase=0.):
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0], center[1], 0])

    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack([
            low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
            low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
            z_variation * (z_low[2] + (z_high - z_low)[2] *
                           (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
        ], -1)

    theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    if const_speed:
        # Resample theta angles so that the velocity is closer to constant.
        lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
        theta = sample_np(None, theta, np.log(lengths), n_frames + 1)
        positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])    
    return np.stack([viewmatrix(p - center, up, p) for p in positions])

def generate_interpolated_path(poses, n_interp, spline_degree=5,
                               smoothness=.03, rot_weight=.1):
    """Creates a smooth spline path between input keyframe camera poses.

  Spline is calculated with poses in format (position, lookat-point, up-point).

  Args:
    poses: (n, 3, 4) array of input pose keyframes.
    n_interp: returned path will have n_interp * (n - 1) total poses.
    spline_degree: polynomial degree of B-spline.
    smoothness: parameter for spline smoothing, 0 forces exact interpolation.
    rot_weight: relative weighting of rotation/translation in spline solve.

  Returns:
    Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
  """

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(points,
                        n_interp * (points.shape[0] - 1),
                        k=spline_degree,
                        s=smoothness)
    return points_to_poses(new_points)


def generate_spiral_path(poses, bounds, n_frames=120, n_rots=2, zrate=.5):
    """Calculates a forward facing spiral path for rendering."""
    # Find a reasonable 'focus depth' for this dataset as a weighted average
    # of conservative near and far bounds in disparity space.
    near_bound = bounds.min() * NEAR_STRETCH
    far_bound = bounds.max() * FAR_STRETCH
    # All cameras will point towards the world space point (0, 0, -focal).
    focal = 1 / (((1 - FOCUS_DISTANCE) / near_bound + FOCUS_DISTANCE / far_bound))

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), 90, 0)
    radii = np.concatenate([radii, [1.]])

    # Generate poses for spiral path.
    render_poses = []
    cam2world = average_pose(poses)
    up = poses[:, :3, 1].mean(0)
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
        t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
        position = cam2world @ t
        lookat = cam2world @ [0, 0, -focal, 1.]
        z_axis = position - lookat
        render_poses.append(viewmatrix(z_axis, up, position))
    render_poses = np.stack(render_poses, axis=0)
    import pdb; pdb.set_trace()
    return render_poses


def transform_poses_pca(poses):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)
    import pdb; pdb.set_trace()
    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform