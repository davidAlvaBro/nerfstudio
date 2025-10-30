import json
from pathlib import Path
import argparse

import numpy as np
import torch as th 
import cv2 

from nerfstudio.cameras.cameras import Cameras, CameraType

from colmap_pipeline import run_colmap_frozen_poses
from gsplat_pipeline import train_gsplat, render_gsplat
from utils import remove, load_transforms_json
# from flythrough_cameras import intermediate_poses_between_training_views
from yolo_pipeline import apply_yolo
from point_cloud_from_depth import generate_point_cloud, store_point_cloud_as_ply



def run_pipelines(): 
    parser = argparse.ArgumentParser(description="Run the COLMAP-pipeline, the Gaussian Splatting pipeline, Construct new views and find poses with the YOLO-pipeline.")
    parser.add_argument("--data_folder", required=True, type=str, help="Path parent folder of transforms.json file and ./images.")
    parser.add_argument("--output_dir", required=True, type=str, help="Path to a directory where the point clouds, 'renders', 'views.json', 'annotations.npz' go.")
    parser.add_argument("--clean_working_dir", default=True, type=bool, help="Whether or not the data in the working dir is deleted after use") # TODO actually do this :0
    parser.add_argument("--gaussian_splat_steps", default=4000, type=int, help="Max steps for training the Gaussian Splatting")
    parser.add_argument("--run_colmap", action="store_true", help="If not used either a depth map is provided in transform.json or a .ply file already exists and is referenced in transform.json")
    parser.add_argument("--name", default="test", type=str, help="Used to identify Gaussian Splat. Not relevant if 'clean_working_dir'.")
    args = parser.parse_args()
    data_folder = Path(args.data_folder)
    metadata_path = data_folder / "transforms.json"
    working_dir = Path("temp")
    output_dir = Path(args.output_dir)
    renders = output_dir / "renders"

    
    # 1. Read the transforms.json and check that every entry is valid 
    run_args = load_transforms_json(metadata_path)
    # TODO make the check
    
    # TODO make a flag so step 2 can be skipped if the views.json already exists and load them instead - How do I make the names match? 
    # TODO do something here 
    # This is temporary, but I should just get the similar stuff while testing ^
    # first_frame = run_args["frames"][0]
    # H, W = first_frame["h"], first_frame["w"]
    # fx, fy = first_frame["fl_x"], first_frame["fl_y"]
    # cx, cy = first_frame["cx"], first_frame["cy"]

    # 2. Compute novel extrinsics to be used later - and a corresponding NerfStudio object  
    
  
    
    # 3. Run the colmap pipeline 
    # This puts a fused.ply and a ply.ply in the dataset_path for the g-splat
    if args.run_colmap:
        _, _ = run_colmap_frozen_poses(metadata_path=metadata_path, data_folder=data_folder, workdir=working_dir / "colmap", out_path=output_dir, cleanup=args.clean_working_dir)
        # Add the fused.ply to the metadata.json to use it in the GS 
        run_args["ply_file_path"] = str(output_dir / "fused.ply")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(run_args, f, ensure_ascii=False, indent=2)
    elif "depth_map" in run_args:
        # Fetch image, depth, intrinsics and extrinsics 
        depth_map = np.load(data_folder / run_args["depth_map"]) 
        ref_frame = run_args["frames"][run_args["ref"]]
        ref_img = cv2.cvtColor(cv2.imread(data_folder / ref_frame["file_path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        ref_intrinsics = np.array([[ref_frame["fl_x"], 0, ref_frame["cx"]], 
                                   [0, ref_frame["fl_y"], ref_frame["cy"]], 
                                   [0, 0, 1]])
        ref_c2w = np.array(ref_frame["transform_matrix"])
        
        point_cloud = generate_point_cloud(img=ref_img, depth=depth_map, intrinsics=ref_intrinsics, c2w=ref_c2w)
        ply_file_path = Path(output_dir / "depth_point_cloud.ply")
        store_point_cloud_as_ply(point_cloud=point_cloud, path=ply_file_path)

        # Store the path to the point cloud for gaussian splatting pipeline 
        run_args["ply_file_path"] = str(ply_file_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(run_args, f, ensure_ascii=False, indent=2)
    # If none of the two options there should already exists a 'run_args["ply_file_path"]' along with the file

    # 4. Train a Gaussian splatting 
    ckpt_path = train_gsplat(data_folder=data_folder, working_dir=working_dir, max_steps = args.gaussian_splat_steps, experiment_name = "gaussian", project_name = args.name)
    # TODO Should I move this outside the normal file structure so the normal structure can be purged while keeping the gsplat? 
    
    # 5. Render the new views with the Gaussian splatting. 
    _, rendered_image_names = render_gsplat(ckpt_path=ckpt_path, 
                                                          working_dir=working_dir, 
                                                          data_folder=data_folder, 
                                                          out_dir=renders,  
                                                          experiment_name="gaussian", 
                                                          project_name=args.name)
    
    # 6. Run YOLO pipeline 
    yolo_annotations = apply_yolo(data_folder=data_folder, out_path=output_dir)
    
    # # 7. Save evaluation camera views so that they can be compared 
    # my_dict = {"frame_idx": run_args["frame_idx"],
    #            "recording_key": run_args["recording_key"],
    #            "views": {}} 
    # for (view, name) in zip(frames_to_validate_on, rendered_image_names): 
    #     my_dict["views"][str(name)] = list(view.flatten())
    # with open(output_dir / "views.json", "w", encoding="utf-8") as f:
    #     json.dump(my_dict, f, ensure_ascii=False, indent=2)

    # TODO add cleanup if I want it 



if __name__ == "__main__": 
    run_pipelines()