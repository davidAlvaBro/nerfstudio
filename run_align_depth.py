from pathlib import Path
import argparse

import numpy as np
import open3d as o3d 

from utils import load_transforms_json
from point_cloud_from_depth import depth_from_pc_and_camera, align_median_depth



def run_pipelines(): 
    parser = argparse.ArgumentParser(description="Aligns ref only or all depths in working_dir/transforms.json to gt_pointcloud.")
    parser.add_argument("--gt_pointcloud", required=True, type=str, help="Path to the point cloud to compare the depths with.")
    parser.add_argument("--working_dir", required=True, type=str, help="Path parent folder of transforms.json file and ./depths.")
    parser.add_argument("--only_ref", action="store_true", help="Update only the reference image.")
    args = parser.parse_args()
    
    data_folder = Path(args.working_dir)
    metadata_path = data_folder / "transforms.json"
    metadata = load_transforms_json(metadata_path)
    
    pc_colmap = o3d.io.read_point_cloud(args.gt_pointcloud)

    for frame in metadata["frames"]: 
        if args.only_ref and not frame == metadata["frames"][metadata["ref"]]: 
            continue

        # update the depth 
        depth = np.load(data_folder / frame["depth_path"])
        K = np.array([[frame["fl_x"], 0, frame["cx"]],
                    [0, frame["fl_y"], frame["cy"]],
                    [0,0,1]])
        c2w = np.array(frame["transform_matrix"])
        img_shape = (frame["h"],frame["w"])
        
        target_depth, mask = depth_from_pc_and_camera(c2w=c2w, K=K, img_shape=img_shape, point_cloud=pc_colmap)

        depth_aligned = align_median_depth(target_depth=target_depth, source_depth=depth, mask=mask)

        np.save(data_folder / frame["depth_path"], depth_aligned)



if __name__ == "__main__": 
    run_pipelines()