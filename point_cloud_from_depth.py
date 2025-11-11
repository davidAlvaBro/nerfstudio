from pathlib import Path

import numpy as np
import struct

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


if __name__ == "__main__":
    import cv2
    import json 

    metadata_path = Path("../data/success1/mvgen/transforms.json")
    data_folder = Path("../data/success1/mvgen")
    output_dir = Path("../data/success1")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    depth_map = np.load(data_folder / metadata["depth_map"]) 
    ref_frame = metadata["frames"][metadata["trajectory_ref"]]
    ref_img = cv2.cvtColor(cv2.imread(data_folder / ref_frame["file_path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    ref_intrinsics = np.array([[ref_frame["fl_x"], 0, ref_frame["cx"]], 
                                [0, ref_frame["fl_y"], ref_frame["cy"]], 
                                [0, 0, 1]])
    ref_c2w = np.array(ref_frame["transform_matrix"])
    
    point_cloud = generate_point_cloud(img=ref_img, depth=depth_map, intrinsics=ref_intrinsics, c2w=ref_c2w)
    ply_file_path = Path(output_dir / "depth_point_cloud.ply")
    store_point_cloud_as_ply(point_cloud=point_cloud, path=ply_file_path)