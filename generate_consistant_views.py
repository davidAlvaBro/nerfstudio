from typing import List, Sequence, Tuple 
import json
from pathlib import Path

import numpy as np
import torch as th 

from nerfstudio.cameras.cameras import Cameras, CameraType

from colmap_pipeline import run_colmap_frozen_poses
from gsplat_pipeline import train_gsplat, render_gsplat
from utils import remove
from flythrough_cameras import intermediate_poses_between_training_views


# TODO have not looked at this function 
def _make_cameras_from_c2w(
    c2w_list: Sequence[np.ndarray],
    intrinsics: Tuple[float, float, float, float],  # fx, fy, cx, cy
    hw: Tuple[int, int],  # (H, W)
) -> Cameras:
    """Create a Nerfstudio Cameras object from NeRF/OpenGL c2w matrices and pinhole intrinsics."""
    H, W = hw
    fx, fy, cx, cy = intrinsics
    c2w = th.from_numpy(np.stack(c2w_list, axis=0)).float()  # (N,4,4)
    n = c2w.shape[0]
    return Cameras(
        camera_to_worlds=c2w[:, :3, :4],
        fx=th.full((n, 1), float(fx)),
        fy=th.full((n, 1), float(fy)),
        cx=th.full((n, 1), float(cx)),
        cy=th.full((n, 1), float(cy)),
        height=H,
        width=W,
        camera_type=CameraType.PERSPECTIVE,
    )


def generate_consistant_views(experiment_path: Path, 
                              dataset_path: Path, 
                              experiment_name = "gsplat_exp", 
                              project_name = "nerfstudio-project",
                              run_colmap: bool = True, 
                              cleanup: bool = True, 
                              save_renders: bool = True, 
                              views: List[np.ndarray] | None = None, 
                              save_dir: Path = Path("rendered")) -> List[th.Tensor] : 
    """
    This function generates consistant views of a scene. 
    It runs a SfM (COLMAP) pipeline (with fixed camera extrinsics).
    Then uses the sparse point cloud and the images to create a GaussianSplat. 
    From this GaussianSplat the n views in "views" is rendered. 

    Args : 
        - Experiment_path : the path where colmap dumps all the files needed during the SfM 
        - dataset : the path of the images and initial poses. This directory must have a "images" folder and a "transformations.json". 
        - views : the disired rendered views. 
        - cleanup : if the helpers should be removed after use (colmap). 
        - save_renders : if the new rendered images should be saved on disc. 
        - save_dir : the subdirectory of "dataset_path" to save to. Only used if "save_renders" are true.
    
    Returns : 
        - List of rendered images as a torch tensor
    """
    images_dir = dataset_path / Path("images")
    cameras_path = dataset_path / Path("transforms.json")
    save_dir = dataset_path / save_dir

    assert images_dir.exists() 
    assert cameras_path.exists() 
    
    # If no views are given evaluate on the training perspectives 
    if views is None: 
        with open(cameras_path, 'r') as file:
            all_cameras = json.load(file)
            views = [np.array(c["transform_matrix"]) for c in all_cameras['frames']]

    # check that each view is a (4x4) matrix and convert them to "NerfStudio" cameras  
    for view in views: 
        assert type(view) is np.ndarray and view.shape == (4, 4)
    cameras = _make_cameras_from_c2w(c2w_list=views, intrinsics=[1175.72322010302, 1176.27640974755, 738.018326950399, 554.69070948643], hw=(1080, 1440)) # Hmm just hardcoded these but that is fine right?

    # Run the colmap pipeline. The resulting ply point cloud is put in the dataset_path. 
    if run_colmap: 
        (t_sparse, t_dense) = run_colmap_frozen_poses(data=dataset_path, workdir=experiment_path, cleanup=False)
        print(f"Sparse time : {t_sparse:0.4f}s, \nDense time : {t_dense:.4f}s")

    # Train a gsplat to the current scene 
    ckpt_path = train_gsplat(data_dir=dataset_path, working_dir=experiment_path, max_steps = 4000, experiment_name = experiment_name, project_name = project_name, disable_viewer = True, downscale_factor = 1)

    # Render new images for from this gsplat 
    imgs = render_gsplat(ckpt_path=ckpt_path, data_dir=dataset_path, working_dir=experiment_path, experiment_name = experiment_name, project_name = project_name, out_dir=save_dir, cameras=cameras, save_images=save_renders, downscale_factor=1)

    if cleanup : 
        remove(experiment_path)

    return imgs


if __name__ == "__main__": 
    dataset_path = Path("dataset")
    experiment_path = Path("temp") 
    experiment_name = "test" 
    project_name = "test"
    interpolations = intermediate_poses_between_training_views(scene_path=dataset_path, n_between=8)
    views = [c[3] for c in interpolations]
    generate_consistant_views(experiment_path=experiment_path, 
                              dataset_path=Path("dataset"), 
                              views=views, 
                              experiment_name=experiment_name,
                              project_name=project_name,
                              run_colmap=False, 
                              cleanup=False, 
                              save_renders=True, 
                              save_dir=Path("rendered"))
    
    from make_gif import make_gif
    setting = "no"
    make_gif(path=dataset_path / "rendered", out_path=Path(f"gifs/gsplat_{setting}_fly_around.gif"))
    make_gif(path=Path("temp/images/train"), out_path=Path(f"gifs/gsplat_{setting}_train.gif"))
    make_gif(path=Path("temp/images/val"), out_path=Path(f"gifs/gsplat_{setting}_val.gif"))