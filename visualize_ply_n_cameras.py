import open3d as o3d
from pathlib import Path
import numpy as np 
import json 
# USUAL 
transforms_path = "/home/dbl@grazper.net/david-thesis/data/test/transforms.json"
ply = "/home/dbl@grazper.net/david-thesis/data/test/mvgen/depth_point_cloud.ply"
ply = "/home/dbl@grazper.net/david-thesis/data/test/people_only.ply"
ply = "/home/dbl@grazper.net/david-thesis/data/test/joint_point_cloud.ply"
transforms_path = "/home/dbl@grazper.net/david-thesis/data/test/mvgen/transforms.json"
ply = "/home/dbl@grazper.net/david-thesis/data/test/mvgen/joint_point_cloud.ply"
ply = "/home/dbl@grazper.net/david-thesis/data/test/mvgen/mixed2.ply"
# ply = "/home/dbl@grazper.net/david-thesis/data/test/people_only.ply"
# ply = "/home/dbl@grazper.net/david-thesis/data/test/fused.ply"
# transforms_path = "/home/dbl@grazper.net/david-thesis/data/pointcloud/transforms.json"
# ply = "/home/dbl@grazper.net/david-thesis/data/pointcloud/depth_point_cloud.ply"
# ply = "/home/dbl@grazper.net/david-thesis/data/pointcloud/joint_point_cloud.ply"

# Presentation success1
# MVGen point cloud 
# transforms_path = "/home/dbl@grazper.net/david-thesis/data/success1/mvgen/transforms.json"
# ply = "/home/dbl@grazper.net/david-thesis/data/success1/mvgen/depth_point_cloud.ply"

# # Inferred Grazper point cloud 
# transforms_path = "/home/dbl@grazper.net/david-thesis/data/success1/mvgen/transforms.json"
# ply = "/home/dbl@grazper.net/david-thesis/data/success1/depth_point_cloud.ply"

# # Dense pointcloud from COLMAP 
# transforms_path = "/home/dbl@grazper.net/david-thesis/data/success1/mvgen/transforms.json"
# ply = "/home/dbl@grazper.net/david-thesis/data/success1/fused.ply"

with open(transforms_path, "r") as f: # Notice that the mvgen intrinsics are rescaled
    data = json.load(f)
extrinsics = []
# intrinsics = []
for cam in data["frames"]: 
    # intrinsic = np.array([[cam["fl_x"], 0, cam["cx"]],
    #                     [0, cam["fl_y"], cam["cy"]],
    #                     [0, 0, 1]])
    F = np.array(cam["transform_matrix"])
    extrinsics.append(F)
    # intrinsics.append(intrinsic) 
# reference cam 
# ref_extrinsics = extrinsics[data["trajectory_ref"]]
ref_extrinsics = extrinsics[data["ref"]]
# ref_intrinsics = intrinsics[data["trajectory_ref"]]

cameras_points = []
ref_extrinsics_inv = np.linalg.inv(ref_extrinsics)
for c2w in extrinsics: 
    # World coords (center) to ref_cam coords 
    cameras_points.append(c2w[:3, 3] / c2w[3, 3])
cameras_points = np.stack(cameras_points)

# ply = "/home/dbl@grazper.net/david-thesis/data/test/mvgen/pcd.ply"
# ply = "/home/dbl@grazper.net/david-thesis/data/test/mvgen/depth_point_cloud.ply"

pcd = o3d.io.read_point_cloud(ply)
if not pcd.has_normals():
    pcd.estimate_normals(fast_normal_computation=True)
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

cam_sphere_radius = 0.1

cam_spheres = []
for i, p in enumerate(np.asarray(cameras_points)):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=cam_sphere_radius)
    s.compute_vertex_normals()
    s.paint_uniform_color([1.0, 0.0, 0.0])  # red
    if i == data["ref"]: s.paint_uniform_color([0.0,1.0,0.0])
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
# opt.point_show_normal = True
vis.run()
vis.destroy_window()