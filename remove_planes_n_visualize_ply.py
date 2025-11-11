import open3d as o3d
from pathlib import Path
import numpy as np 
import json 

# Dense pointcloud from COLMAP 
transforms_path = "/home/dbl@grazper.net/david-thesis/data/success1/mvgen/transforms.json"
ply = "/home/dbl@grazper.net/david-thesis/data/success1/fused.ply"

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
ref_extrinsics = extrinsics[data["trajectory_ref"]]
# ref_intrinsics = intrinsics[data["trajectory_ref"]]

cameras_points = []
ref_extrinsics_inv = np.linalg.inv(ref_extrinsics)
for c2w in extrinsics: 
    # World coords (center) to ref_cam coords 
    cameras_points.append(c2w[:3, 3] / c2w[3, 3])
cameras_points = np.stack(cameras_points) 


# Load Point cloud
pcd = o3d.io.read_point_cloud(ply)
if not pcd.has_normals():
    pcd.estimate_normals(fast_normal_computation=True)
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

# Remove everything outside planes - In grazper scenes there are 5 (walls + floor)
n_walls = 5
normals = np.zeros((n_walls, 3))
d = np.zeros((n_walls))
# temporary_walls_removed = pcd.clone()
temporary_walls_removed = o3d.geometry.PointCloud(pcd) 
for i in range(n_walls): 
    plane_param, inliers = temporary_walls_removed.segment_plane(
        distance_threshold=0.08, # 10 cm - more and the people count as a plane... 
        ransac_n=3, # Floor and walls are very precise so no more than 3 points are needed
        num_iterations=20000
    )
    # Record normal to later use for "inside scene inliers"
    normals[i] = plane_param[:3]
    d[i] = plane_param[3]

    temporary_walls_removed = temporary_walls_removed.select_by_index(inliers, invert=True)
    # if i > 3: 
    #     vis = o3d.visualization.Visualizer()
    #     vis.create_window(window_name="PLY preview", width=1280, height=800)
    #     # vis.add_geometry(pcd); vis.add_geometry(axes)
    #     vis.add_geometry(temporary_walls_removed); vis.add_geometry(axes)

    #     opt = vis.get_render_option()
    #     opt.point_size = 2.0  # only affects the .ply points, not the spheres
    #     vis.run()
    #     vis.destroy_window()

# Figure out if scene is on positive or negative side of plane 
camera_points_inner_prod = cameras_points @ normals.T + d[None, :]
point_normals_inner_prod = np.array(pcd.points) @ normals.T + d[None, :]
signs = np.sign(np.mean(point_normals_inner_prod, axis=0))

oriented_inner_prods = point_normals_inner_prod * signs
tol_floor = 0.1 
tol_wall = 0.5
tol = np.ones((5))*tol_wall 
tol[0] = tol_floor 
inside_mask = np.all(oriented_inner_prods >= tol, axis=1) 
inside_points = pcd.select_by_index(np.where(inside_mask)[0])


cam_sphere_radius = 0.1

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
# vis.add_geometry(pcd); vis.add_geometry(axes)
vis.add_geometry(inside_points); vis.add_geometry(axes)
for s in cam_spheres:
    vis.add_geometry(s)

opt = vis.get_render_option()
opt.point_size = 2.0  # only affects the .ply points, not the spheres
vis.run()
vis.destroy_window()