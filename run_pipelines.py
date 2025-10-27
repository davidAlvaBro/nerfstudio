import json
from pathlib import Path
import argparse

import numpy as np
import torch as th 

from nerfstudio.cameras.cameras import Cameras, CameraType

from colmap_pipeline import run_colmap_frozen_poses
from gsplat_pipeline import train_gsplat, render_gsplat
from utils import remove, load_transforms_json
# from flythrough_cameras import intermediate_poses_between_training_views
from yolo_pipeline import apply_yolo



def run_pipelines(): 
    parser = argparse.ArgumentParser(description="Run the COLMAP-pipeline, the Gaussian Splatting pipeline, Construct new views and find poses with the YOLO-pipeline.")
    parser.add_argument("--transforms", required=True, type=str, help="Path to a transforms.json file.")
    parser.add_argument("--output_dir", required=True, type=str, help="Path to a directory where the point clouds, 'renders', 'views.json', 'annotations.npz' go.")
    parser.add_argument("--clean_working_dir", default=True, type=bool, help="Whether or not the data in the working dir is deleted after use") # TODO actually do this :0
    parser.add_argument("--gaussian_splat_steps", default=4000, type=int, help="Max steps for training the Gaussian Splatting")
    parser.add_argument("--name", default="test", type=str, help="Used to identify Gaussian Splat. Not relevant if 'clean_working_dir'.")
    args = parser.parse_args()
    metadata_path = Path(args.transforms)
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
    _, _ = run_colmap_frozen_poses(metadata_path=metadata_path, workdir=working_dir / "colmap", out_path=output_dir, cleanup=args.clean_working_dir)
    # Add the fused.ply to the metadata.json to use it in the GS 
    run_args["ply_file_path"] = str(output_dir / "fused.ply")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(run_args, f, ensure_ascii=False, indent=2)

    # 4. Train a Gaussian splatting 
    ckpt_path = train_gsplat(metadata_path=metadata_path, working_dir=working_dir, max_steps = args.gaussian_splat_steps, experiment_name = "gaussian", project_name = args.name)
    # TODO Should I move this outside the normal file structure so the normal structure can be purged while keeping the gsplat? 

    # 5. Render the new views with the Gaussian splatting. 
    _, rendered_image_names = render_gsplat(ckpt_path=ckpt_path, 
                                                          working_dir=working_dir, 
                                                          metadata_path=metadata_path, 
                                                          out_dir=renders, 
                                                          cameras=eval_nerfstudio_cameras, 
                                                          experiment_name="gaussian", 
                                                          project_name=args.name)
    
    # 6. Run YOLO pipeline 
    yolo_annotations = apply_yolo(metadata_path=metadata_path, out_path=output_dir)

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