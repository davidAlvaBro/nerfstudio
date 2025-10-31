import open3d as o3d
from pathlib import Path
import numpy as np 
import json 

with open("/home/dbl@grazper.net/david-thesis/data/test/mvgen/transforms.json", "r") as f: # Notice that the mvgen intrinsics are rescaled
    data = json.load(f)
extrinsics = []
intrinsics = []
for cam in data["frames"]: 
    intrinsic = np.array([[cam["fl_x"], 0, cam["cx"]],
                        [0, cam["fl_y"], cam["cy"]],
                        [0, 0, 1]])
    F = np.array(cam["transform_matrix"])
    extrinsics.append(F)
    intrinsics.append(intrinsic) 
# reference cam 
ref_extrinsics = extrinsics[data["trajectory_ref"]]
ref_intrinsics = intrinsics[data["trajectory_ref"]]

cameras_points = []
ref_extrinsics_inv = np.linalg.inv(ref_extrinsics)
for c2w in extrinsics: 
    # World coords (center) to ref_cam coords 
    cameras_points.append(c2w[:3, 3] / c2w[3, 3])
cameras_points = np.stack(cameras_points)

# ply = "/home/dbl@grazper.net/david-thesis/data/test/mvgen/pcd.ply"
ply = "/home/dbl@grazper.net/david-thesis/data/test/mvgen/depth_point_cloud.ply"
pcd = o3d.io.read_point_cloud(ply)
if not pcd.has_normals():
    pcd.estimate_normals(fast_normal_computation=True)
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

# pick a radius relative to scene size
scene_diag = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())
cam_sphere_radius = max(1e-6, 0.01 * scene_diag)   # adjust 0.01 as you like (bigger → more visible)

cam_spheres = []
for p in np.asarray(cameras_points):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=cam_sphere_radius)
    s.compute_vertex_normals()
    s.paint_uniform_color([1.0, 0.0, 0.0])  # red
    s.translate(p)                          # move sphere to camera center
    cam_spheres.append(s)

# --- viewer ---
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="PLY preview", width=1280, height=800)
vis.add_geometry(pcd); vis.add_geometry(axes)
for s in cam_spheres:
    vis.add_geometry(s)

opt = vis.get_render_option()
opt.point_size = 2.0  # only affects the .ply points, not the spheres
vis.run()
vis.destroy_window()