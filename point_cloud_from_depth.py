from pathlib import Path

import numpy as np
import struct

def generate_point_cloud(img: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, c2w: np.ndarray, l2_depth: bool = True) -> np.ndarray:
    """
    This function projects image pixels to 3d space and stores the coordinates and colors of each pixel in a numpy array 
    """
    # Convert image coordinates to camera 3d coordinates without scale
    H, W, _ = img.shape 
    K_inv = np.linalg.inv(intrinsics)
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
    camera_coords_scaled = camera_coords*(depth / camera_coords[:,:,2])[:,:,None]

    # Nerfstudio is using  convention y+up and z-forward that means that my 
    # image should go from high to low in the y direction and be negative in the z direction
    flip_ynz = np.diag([1, -1, -1])
    camera_coords_nerf_conv = camera_coords_scaled @ flip_ynz.T 

    # Project to world coordinates
    R = c2w[:3,:3]
    t = c2w[:3, 3]
    world_coords = camera_coords_nerf_conv @ R.T + t
    
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
    H, W = 100, 200
    img = np.ones((H,W, 3))
    img[:,:1] = 2
    img[:,:2] = 3
    depth = np.ones((H,W))*0.5
    K = np.array([[1175.7232666015625,0,738.018310546875],
                  [0,1176.2763671875, 554.6907348632812],
                  [0,0,1]]) 
    c2w = np.array([[0.9455424547195435, -0.06263718008995056, 0.3194151818752289, 2.9413599967956543],
                    [0.03294133022427559, 0.99468594789505, 0.09754357486963272, 1.9202120304107666],
                    [-0.3238276243209839, -0.08170963078737259, 0.9425811767578125, 6.2920660972595215],
                    [0.0, 0.0, 0.0, 1.0]])
    
    generate_point_cloud(img=img, depth=depth, intrinsics=K, c2w=c2w)