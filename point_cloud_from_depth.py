from pathlib import Path
from typing import Tuple, List

import numpy as np
import struct
import open3d as o3d 

def generate_point_cloud(img: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, c2w: np.ndarray, l2_depth: bool = True) -> np.ndarray:
    """
    This function projects image pixels to 3d space and stores the coordinates and colors of each pixel in a numpy array 
    """
    # Nerfstudio is using  convention y+up and z-forward that means that my 
    # image should go from high to low in the y direction and be negative in the z direction
    flip_ynz = np.diag([1, -1, -1])
    K = intrinsics @ flip_ynz # A bit weird to have a negative focal length right?
    K_inv = np.linalg.inv(K)

    # Convert image coordinates to camera 3d coordinates without scale
    H, W, _ = img.shape 
    u = np.arange(W) + 0.5
    v = np.arange(H) + 0.5
    uu, vv = np.meshgrid(u, v)
    scale_direction = np.ones_like(uu)
    img_coords = np.stack([uu, vv, scale_direction], axis=-1)

    camera_coords = img_coords @ K_inv.T

    # Current depth estimator is L2, but another could be z-depth
    if l2_depth: # We need the z-depth 
        rescale_depth = np.abs(camera_coords[:,:,2]) / np.linalg.norm(camera_coords, axis=-1) 
        depth = depth * rescale_depth

    # Now we can get scale from the z-coordinates 
    camera_coords_scaled = camera_coords*(depth / np.abs(camera_coords[:,:,2]))[:,:,None]

    # Project to world coordinates
    R = c2w[:3,:3]
    t = c2w[:3, 3]
    world_coords = camera_coords_scaled @ R.T + t
    
    # At last we need to associate these 3d point with pixel RGB values 
    xyzrgb = np.zeros((H, W, 6))
    xyzrgb[:,:,:3] = world_coords
    xyzrgb[:,:,3:] = img 
    xyzrgb = xyzrgb.reshape(-1, 6)

    return xyzrgb


def multiple_point_clouds(imgs: List[np.ndarray], depths: List[np.ndarray], intrinsics: List[np.ndarray], c2ws: List[np.ndarray]) -> np.ndarray: 
    """
    Wrapper to 'generate_point_cloud' to apply it on len(imgs) frames. 
    """
    point_clouds = []
    for (img, depth, K, c2w) in zip(imgs, depths, intrinsics, c2ws): 
        xyzrgb = generate_point_cloud(img=img, depth=depth, intrinsics=K, c2w=c2w)
        point_clouds.append(xyzrgb)
    
    return np.vstack(point_clouds)



def store_point_cloud_as_ply(point_cloud: np.ndarray, path: Path) -> None: 
    with open(path, "wb") as f:
        header = f"""ply
format binary_little_endian 1.0
element vertex {point_cloud.shape[0]}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        f.write(header.encode("ascii"))
        for x, y, z, r, g, b in point_cloud: 
            f.write(struct.pack("<fffBBB", float(x), float(y), float(z), int(r), int(g), int(b)))    

def depth_map_from_point_cloud(K: np.ndarray, img_shape: Tuple[int, int], point_cloud: np.ndarray): 
    """
    Given the coordinates of a point cloud compute a depth map of shape 'img_shape'
    """
    H, W = img_shape 

    weights = np.zeros((H,W))
    depths = np.zeros((H,W))

    # Project points to image plane - convention is z- forward, but we want positive depths
    img_points = point_cloud @ (np.diag([1,-1,-1]) @ K.T)  
    
    for (uz,vz,z) in img_points: 
        # TODO could substract 0.5 simulate centers of pixels? 
        u = uz/z 
        v = vz/z 

        # TODO could be less strict, but the people should be in the middle
        # Boundery removal so all points are fully within the image 
        if (0 >= u or u > W -1 or 0 >= v or v > H-1 or z < 0): 
            continue 
        
        ulu, ulv = int(np.floor(u)), int(np.floor(v))
        delta_u, delta_v = u - ulu, v - ulv 

        # u:upper, l:lower - l:left, r:right - [ul, ur], [ll, lr]
        weight = np.array([[(1-delta_u)*(1-delta_v), delta_u*(1-delta_v)], 
                           [(1-delta_u)*delta_v, delta_u*delta_v]])

        depths[ulv:ulv + 2, ulu:ulu + 2] += abs(z)*weight
        weights[ulv:ulv + 2, ulu:ulu + 2] += weight
    
    # Instead of adding a small number in the denominator 
    mask = weights != 0
    depth_map = np.divide(depths, weights,
                  out=np.zeros_like(depths),
                  where=mask)

    return depth_map, mask 


def depth_from_pc_and_camera(c2w: np.ndarray, K: np.ndarray, img_shape: Tuple[int, int], point_cloud): 
    """
    Aligns pointcloud to camera coordinates before running 'depth_map_from_point_cloud' and extracts only the points
    """
    # point cloud coordinates 
    pcc = np.asarray(point_cloud.points)
    w2c = np.linalg.inv(c2w)
    R, t = w2c[:3, :3], w2c[:3, 3]
    pcd_aligned = pcc @ R.T + t 

    return depth_map_from_point_cloud(K=K, img_shape=img_shape, point_cloud=pcd_aligned) 

def align_median_depth(target_depth: np.ndarray, source_depth: np.ndarray, mask: np.ndarray = None): 
    """
    Given a target and source depth find the medians and compute m_t/m_s and multiply the source_depth with this.
    """
    if mask is None: 
        mask = np.ones_like((target_depth), dtype=bool)
    # median_t = np.median(target_depth[mask])
    # median_s = np.median(source_depth[mask])
    median_t = np.median(target_depth[mask])
    median_s = np.median(source_depth[mask])

    scale = median_t/median_s 

    return source_depth * scale 

def adjust_depth_estimates_with_gt_point_cloud(metadata: dict, working_dir: Path, target_pc): 
    """
    For each frame in metadata["frames"] with a depth estimate, adjust it through the median with a target pc. 
    """

    for frame in metadata["frames"]: 
        # TODO Check that it has a depth estimate, otherwise pass? 
        depth_estimate = np.load(working_dir / frame["depth_path"])

        # Get reference depth from point cloud 
        K = np.array([[frame["fl_x"], 0, frame["cx"]],
                    [0, frame["fl_y"], frame["cy"]],
                    [0,0,1]])
        c2w = np.array(frame["transform_matrix"])
        (H,W) = frame["h"], frame["w"]
        depth_target, mask = depth_from_pc_and_camera(c2w=c2w, K=K, img_shape=(H,W), point_cloud=target_pc)

        aligned_depth = align_median_depth(target_depth=depth_target, source_depth=depth_estimate, mask=mask)

        np.save(working_dir / frame["depth_path"], aligned_depth)
    


# if __name__ == "__main__":
#     import cv2
#     import json 

#     # metadata_path = Path("../data/success1/mvgen/transforms.json")
#     # data_folder = Path("../data/success1/mvgen")
#     # output_dir = Path("../data/success1")
#     metadata_path = Path("data/pointcloud/transforms.json")
#     data_folder = Path("data/pointcloud")
#     output_dir = Path("data/pointcloud")

#     with open(metadata_path, "r") as f:
#         metadata = json.load(f)

#     # depth_map = np.load(data_folder / metadata["depth_map"]) 
#     depth_map = np.load(data_folder / "ref_depth_map.npy") 
#     ref_frame = metadata["frames"][metadata["ref"]]
#     ref_img = cv2.cvtColor(cv2.imread(data_folder / ref_frame["file_path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
#     ref_intrinsics = np.array([[ref_frame["fl_x"], 0, ref_frame["cx"]], 
#                                 [0, ref_frame["fl_y"], ref_frame["cy"]], 
#                                 [0, 0, 1]])
#     ref_c2w = np.array(ref_frame["transform_matrix"])
    
#     point_cloud = generate_point_cloud(img=ref_img, depth=depth_map, intrinsics=ref_intrinsics, c2w=ref_c2w)
#     ply_file_path = Path(output_dir / "depth_point_cloud.ply")
#     store_point_cloud_as_ply(point_cloud=point_cloud, path=ply_file_path)

if __name__ == "__main__": 
    import json 
    import matplotlib.pyplot as plt 
    from matplotlib.colors import TwoSlopeNorm

    point_cloud_path = Path("/home/dbl@grazper.net/david-thesis/data/pointcloud/fused.ply")
    point_cloud_path = Path("/home/dbl@grazper.net/david-thesis/data/pointcloud/only_people.ply")
    # point_cloud_path = Path("data/pointcloud/depth_point_cloud.ply")
    pcd = o3d.io.read_point_cloud(str(point_cloud_path))

    working_dir = Path("/home/dbl@grazper.net/david-thesis/data/pointcloud")
    transforms_path = working_dir / "transforms.json"
    # transforms_path = Path("data/pointcloud/transforms.json")
    with open(transforms_path, "r") as f:
        metadata = json.load(f)

    adjust_depth_estimates_with_gt_point_cloud(metadata=metadata, working_dir=working_dir, target_pc=pcd)

    for frame in metadata["frames"]: 
        K = np.array([[frame["fl_x"], 0, frame["cx"]],
                    [0, frame["fl_y"], frame["cy"]],
                    [0,0,1]])
        
        c2w = np.array(frame["transform_matrix"])
        (H,W) = frame["h"], frame["w"]

        depth_from_pc, mask = depth_from_pc_and_camera(c2w=c2w, K=K, img_shape=(H,W), point_cloud=pcd)

        # Compare to estimated depths
        depth_estimate = np.load(working_dir / frame["depth_path"])

        dif = np.zeros_like(depth_from_pc)
        dif[mask] = depth_from_pc[mask] - depth_estimate[mask]

        alpha = mask.astype(float)

        amp = np.nanpercentile(np.abs(dif[mask]), 99) if mask.any() else 1.0
        norm = TwoSlopeNorm(0.0, vmin=-amp, vmax=amp)

        plt.imshow(np.zeros_like(depth_from_pc), cmap=plt.cm.gray, vmin=0, vmax=1)
        plt.imshow(dif, cmap='coolwarm', norm=norm, alpha=alpha)
        plt.colorbar(label='Depth difference (m)')
        plt.title(frame["file_path"])
        plt.axis('off')
        plt.show()

    



# if __name__ == "__main__": 
#     import json 
#     import cv2 

#     transforms_path = Path("/home/dbl@grazper.net/david-thesis/data/pointcloud/transforms.json")
#     base_dir = Path("/home/dbl@grazper.net/david-thesis/data/pointcloud")
    
#     with open(transforms_path, "r") as f:
#         metadata = json.load(f)

#     # Adjust depth maps 
#     # point_cloud_path = Path("/home/dbl@grazper.net/david-thesis/data/pointcloud/only_people.ply")
#     # pcd = o3d.io.read_point_cloud(str(point_cloud_path))
#     # adjust_depth_estimates_with_gt_point_cloud(metadata=metadata, working_dir=base_dir, target_pc=pcd)
    
#     imgs = [] 
#     Ks = [] 
#     c2ws = [] 
#     depths = []
#     for frame in metadata["frames"]: 
#         depth = np.load(base_dir / frame["depth_path"])
#         img = cv2.cvtColor(cv2.imread(base_dir / frame["file_path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
#         K = np.array([[frame["fl_x"], 0, frame["cx"]],
#                     [0, frame["fl_y"], frame["cy"]],
#                     [0,0,1]])
#         c2w = np.array(frame["transform_matrix"])

#         imgs.append(img)
#         depths.append(depth)
#         Ks.append(K)
#         c2ws.append(c2w)
    
#     point_cloud_np = multiple_point_clouds(imgs=imgs, depths=depths, intrinsics=Ks, c2ws=c2ws)
#     store_point_cloud_as_ply(point_cloud_np, path=base_dir / "joint_point_cloud.ply")