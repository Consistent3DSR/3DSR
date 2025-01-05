import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from utils.sh_utils import eval_sh
import torch

def create_point_cloud(points, rgbs=None):
    '''
    Input:
        points: np.array [N, 3]
        rgbs:   np.array [N, 3] (optional)
    Return:
        Open3D PointCloud object
    '''
    if rgbs is None:
        rgbs = np.zeros_like(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgbs)
    return pcd

def draw_origin():
    '''
    Draw xyz axes at origin
    Return:
        List of [Open3D LineSet objects]
    '''
    # coord axes
    coord_x = o3d.geometry.LineSet()
    coord_x.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0]])
    coord_x.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
    coord_x.lines = o3d.utility.Vector2iVector([[0, 1]])
    
    coord_y = o3d.geometry.LineSet()
    coord_y.points = o3d.utility.Vector3dVector([[0, 0, 0], [0, 1, 0]])
    coord_y.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
    coord_y.lines = o3d.utility.Vector2iVector([[0, 1]])
    
    coord_z = o3d.geometry.LineSet()
    coord_z.points = o3d.utility.Vector3dVector([[0, 0, 0], [0, 0, 1]])
    coord_z.colors = o3d.utility.Vector3dVector([[0, 0, 1]])
    coord_z.lines = o3d.utility.Vector2iVector([[0, 1]])
    return [coord_x, coord_y, coord_z]

def draw_camera_frustum(K, T, height, width, color=[1, 0, 0], scale=1.0):
    '''
    K: camera intrinsic, np.array [3, 3]
    T: camera pose, np.array [3, 4] or [4, 4]
    height: image height
    width: image width
    color: rgb color of the visualizing lines
    scale: frustum scale
    '''
    camera = o3d.geometry.LineSet.create_camera_visualization(
            width, height, K, np.eye(4), scale=scale,
    )
    T_ = np.eye(4)
    T_[:3, :4] = T[:3, :4]
    camera = camera.transform(T_)
    camera.paint_uniform_color(color)
    return camera

def convert_o3d_to_go(o3d_objs, show_origin=True):
    '''
    Input:
        o3d_objs: List of Open3D Geometry objects
        show_origin: draw xyz axes at origin or not
    Return:
        go_objs: List of Plotly graph objects
    '''
    go_objs = []

    if show_origin: o3d_objs += draw_origin()

    for gm in o3d_objs:
        geometry_type = gm.get_geometry_type()
        
        if geometry_type == o3d.geometry.Geometry.Type.PointCloud:
            pts = np.asarray(gm.points)
            clr = None  #for colors
            if gm.has_colors():
                clr = np.asarray(gm.colors)
            elif gm.has_normals():
                clr = (0.5, 0.5, 0.5) + np.asarray(gm.normals) * 0.5
            else:
                gm.paint_uniform_color((1.0, 0.0, 0.0))
                clr = np.asarray(gm.colors)

            sc = go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='markers', marker=dict(size=1, color=clr))
            go_objs.append(sc)

        elif geometry_type == o3d.geometry.Geometry.Type.TriangleMesh:
            tri = np.asarray(gm.triangles)
            vert = np.asarray(gm.vertices)
            clr = None
            if gm.has_triangle_normals():
                clr = (0.5, 0.5, 0.5) + np.asarray(gm.triangle_normals) * 0.5
                clr = tuple(map(tuple, clr))
            else:
                clr = (1.0, 0.0, 0.0)
            
            mesh = go.Mesh3d(x=vert[:,0], y=vert[:,1], z=vert[:,2], i=tri[:,0], j=tri[:,1], k=tri[:,2], facecolor=clr, opacity=0.50)
            go_objs.append(mesh)
        
        elif geometry_type == o3d.geometry.Geometry.Type.LineSet:
            pts = np.asarray(gm.points)
            lines = np.asarray(gm.lines).astype(int)
            colors = np.asarray(gm.colors).astype(float)
            for L in lines:
                l = pts[L]
                lineset = go.Scatter3d(
                    x=l[:, 0], y=l[:, 1], z=l[:, 2],
                    line=dict(color=colors, width=2),
                    mode='lines',
                )
                go_objs.append(lineset)
                
        else:
            print(geometry_type)
            raise NotImplementedError
        
    return go_objs

def visualize(gs, viewpoint_camera, visibility_mask, save_path='3d_visualization.html'):
    
    o3d_objs = []    

    # TODO get point cloud by your pose and depth
    # points: np.array [N, 3]
    # rgbs:   np.array [N, 3]
    mask = visibility_mask.cpu().numpy()
    # num_pts = mask.shape[0]
    # keep_ratio = 0.05
    # valid_ids = torch.randperm(num_pts)[:int(num_pts*keep_ratio)].numpy()
    # mask2 = np.zeros_like(mask)
    # mask2[valid_ids] = True
    # mask *= mask2
    # x_min = gs._xyz[:,0] > 0
    # x_max = gs._xyz[:,0] < 0.75
    x_min = gs._xyz[:,0] > 0.11
    x_max = gs._xyz[:,0] < 0.22
    x_valid = x_max * x_min
    # y_min = gs._xyz[:,1] > 0
    # y_max = gs._xyz[:,1] < 2.0
    y_min = gs._xyz[:,1] > 0.41
    y_max = gs._xyz[:,1] < 0.986
    y_valid = y_max * y_min
    # z_min = gs._xyz[:,2] > -1
    # z_max = gs._xyz[:,2] < 1.2
    z_min = gs._xyz[:,2] > 0.12
    z_max = gs._xyz[:,2] < 0.450
    z_valid = z_max * z_min
    xyz_valid = x_valid * y_valid * z_valid
    mask = xyz_valid.cpu().numpy()
    # import pdb; pdb.set_trace()
    # mask = mask * xyz_valid.cpu().numpy()
    
    points = gs._xyz[mask].detach().cpu().numpy()    
    shs_view = gs.get_features[mask].transpose(1, 2).view(-1, 3, (gs.max_sh_degree+1)**2)    
    dir_pp = (gs.get_xyz[mask] - viewpoint_camera.camera_center.repeat(gs.get_features[mask].shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(gs.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    pcd = create_point_cloud(points, colors_precomp.detach().cpu().numpy())
    o3d_objs += [pcd]
    print(" ------ point cloud done --------")
    # TODO draw camera by your pose and intrinsic
    # T: camera pose, np.array [3, 4] or [4, 4] are fine
    # K: intrinsic, np.array [3, 3]
    # h: image height
    # w: image width
    T = viewpoint_camera.T
    R = viewpoint_camera.R
    extrinsic = np.eye(4)
    extrinsic[:3,:3] = R
    extrinsic[:3,3] = T
    width = viewpoint_camera.image_width
    height = viewpoint_camera.image_height
    K = np.identity(3)
    K[0, 0] = viewpoint_camera.focal_x
    K[1, 1] = viewpoint_camera.focal_y
    K[0, 2] = width / 2
    K[1, 2] = height / 2
    camera = draw_camera_frustum(K, extrinsic, height, width)
    o3d_objs += [camera]
    print(" ------ camera done --------")
    go_objs = convert_o3d_to_go(o3d_objs)

    # fig = go.Figure(data=go_objs,layout=dict(
    #     scene=dict(aspectmode='data',aspectratio=dict(x=1, y=1, z=1),),
    #     showlegend=False,width=width,height=height,))
    fig = go.Figure(data=go_objs,layout=dict(
        scene=dict(aspectmode='data',aspectratio=dict(x=1, y=1, z=1),),
        showlegend=False,width=2400,height=1600,))
    # if width < 1000:
    #     fig = go.Figure(data=go_objs,layout=dict(
    #     scene=dict(aspectmode='data',aspectratio=dict(x=1, y=1, z=1),),
    #     showlegend=False,width=2400,height=1600,))
    # else:
    #     fig = go.Figure(data=go_objs,layout=dict(
    #         scene=dict(aspectmode='data',aspectratio=dict(x=1, y=1, z=1),),
    #         showlegend=False,width=width,height=height,))
    fig.write_html(save_path)
    # import pdb; pdb.set_trace()
    # fig_json = fig.to_json()
    # with open("vis_gaussians.json", 'w') as f:
    #     f.write(fig_json)
    # fig.write_image(save_path.replace('.html', '.png'))
    # fig.write_image("out.png", engine="orca")
    # import pdb; pdb.set_trace()