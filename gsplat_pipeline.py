from __future__ import annotations
from pathlib import Path
from typing import List
import copy
import re 

import numpy as np
import torch
from PIL import Image

from nerfstudio.configs.method_configs import method_configs 
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackLocation
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig 

from utils import ensure_dir


# TODO rewrite this function. 
def _latest_ckpt_path(run_dir: Path, relative_model_dir: str = "nerfstudio_models") -> Path | None:
    """Return the newest 'step-XXXXX.ckpt' in the run, or None if none exist."""
    ckpt_dir = run_dir / relative_model_dir
    ckpts = sorted(ckpt_dir.glob("step-*.ckpt"))
    if not ckpts:
        return None
    # Prefer highest step number (robust if mtimes are weird)
    def step_num(p: Path) -> int:
        m = re.search(r"step-(\d+)\.ckpt$", p.name)
        return int(m.group(1)) if m else -1
    return max(ckpts, key=step_num)

######
# This is a callback for rendering images during training. 
def add_fixed_train_render_callback(
    trainer,
    camera_index: int = 0,
    every_n_steps: int = 100,
    out_dir: Path | None = None,
):
    """
    Registers a TrainingCallback that renders a fixed *training* image every `every_n_steps`.
    - camera_index: which training view to render (stable across calls)
    - out_dir: where to save PNGs (defaults to <run>/renders_train_callback)
    """
    pipeline = trainer.pipeline
    model = pipeline.model
    train_cameras = pipeline.datamanager.train_dataset.cameras
    val_cameras = pipeline.datamanager.eval_dataset.cameras

    # make sure index is in range
    if len(train_cameras) == 0:
        raise RuntimeError("No training cameras found.")
    cam_idx = int(camera_index) % len(train_cameras)

    # where to save images
    save_dir = Path(out_dir) if out_dir is not None else (Path(trainer.base_dir) / "renders_train_callback")
    save_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def _render_and_log(step):
        """This function is called by the trainer at the configured cadence."""
        # step = getattr(callback_attrs, "step", None)
        # Pull writer from callback attrs if available (preferred), otherwise fall back to trainer.writer
        # writer = getattr(callback_attrs, "writer", None) or getattr(trainer, "writer", None)

        # grab the one fixed camera and send to model's device if supported
        cams = train_cameras[cam_idx : cam_idx + 1]
        val_cams = val_cameras[0 : 1] # doesn't do anything
        if hasattr(cams, "to"):
            cams = cams.to(model.device)

        model.eval()
        out_train = model.get_outputs_for_camera(cams)
        out_val = model.get_outputs_for_camera(val_cams)
        rgb_train = out_train["rgb"]  # H x W x 3, float in [0,1]
        rgb_val = out_val["rgb"]  # H x W x 3, float in [0,1]

        # Choose one 
        if isinstance(rgb_train, torch.Tensor):
            rgb_np_train = rgb_train.clamp(0, 1).detach().cpu().numpy()
            rgb_np_val = rgb_val.clamp(0, 1).detach().cpu().numpy()
        else:
            rgb_np_train = np.clip(rgb_train, 0.0, 1.0)
            rgb_np_val = np.clip(rgb_val, 0.0, 1.0)

        img_train = (rgb_np_train * 255.0 + 0.5).astype(np.uint8)  # H x W x 3, uint8
        img_val = (rgb_np_val * 255.0 + 0.5).astype(np.uint8)  # H x W x 3, uint8

        # Save PNG
        fname = save_dir / f"train/cam_step{(step if step is not None else 0):06d}.png"
        fname.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img_train).save(fname)
        fname = save_dir / f"val/cam_step{(step if step is not None else 0):06d}.png"
        fname.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img_val).save(fname)

        # Log to TensorBoard (optional)
        # if log_to_tensorboard and writer is not None:
        #     # CHW for TB
        #     chw = torch.from_numpy(img).permute(2, 0, 1)
        #     # Most EventWriter implementations accept uint8 or float [0,1]
        #     writer.put_image(f"train_render/cam_{cam_idx:04d}", chw, step=step if step is not None else 0)

    # Register the callback with the trainer loop
    trainer.callbacks.append(
        TrainingCallback(
            where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
            update_every_num_iters=int(every_n_steps),
            func=_render_and_log,
        )
    )
############### TO here 


def initialize_gsplat_config(data_dir: Path, working_dir: Path, experiment_name: str, project_name: str, downscale_factor: int = 1):
    """
    Initializes the part necessesary for both training and rendering.   
    """
    # Initialize the configurations for the trainer object (Nerfactos Gaussian splatter)
    base_cfg: TrainerConfig = copy.deepcopy(method_configs["splatfacto"])  

    # Customized settings  
    base_cfg.data = data_dir
    # TODO figure out which file system layout I actually want for my experiments.
    # (Outer most could be pipeline, then )
    # Setting the name of the working dir 
    base_cfg.output_dir = working_dir
    base_cfg.project_name = project_name
    base_cfg.experiment_name = experiment_name
    base_cfg.timestamp = "nah"
    dp = NerfstudioDataParserConfig(
        data=data_dir,
        downscale_factor=downscale_factor,
        orientation_method="none",
        center_method="none",
        auto_scale_poses=False,
        load_3D_points=True, # IMPORTANT : without this it does not initialize to the ".ply" cloud
    )
    base_cfg.pipeline.datamanager.dataparser = dp  
    return base_cfg 


def train_gsplat(data_dir: Path, 
                     working_dir: Path,
                     max_steps: int = 10_000, 
                     experiment_name: str = "gsplat_exp",
                     project_name: str = "nerfstudio-project",
                     downscale_factor: int = 1,
                     disable_viewer: bool = True,
                     track_training: bool = False, 
                     viewer_port: int = 7007) -> Path:
    """
    This function trains a Gsplat on the data given in "data_dir". 
    This directory has to have an "image" folder with images, a ".ply" point cloud file and a "transforms.json",
    that references both and has the camera extrinsics for each view.
    """
    # # Initialize the configurations for the trainer object (Nerfactos Gaussian splatter)
    # base_cfg: TrainerConfig = copy.deepcopy(method_configs["splatfacto"])  

    # # Edit settings which have been passed in 
    # base_cfg.data = data_dir
    base_cfg = initialize_gsplat_config(data_dir=data_dir, working_dir=working_dir, experiment_name=experiment_name, project_name=project_name, downscale_factor=downscale_factor)
    base_cfg.max_num_iterations = int(max_steps)

    # TODO remove the below if it does nothing, otherwise purge the if statement and move it up under dp 
    # Optional: ensure pose optimizer is OFF for fixed-poses workflow
    if hasattr(base_cfg.pipeline, "model") and hasattr(base_cfg.pipeline.model, "camera_optimizer"):
        base_cfg.pipeline.model.camera_optimizer = CameraOptimizerConfig(mode="off")  # type: ignore[attr-defined]

# ##########################
#     # Experimental settings to get better Gsplats : 
#     # Remove more  # Seems to remove some jitter 
#     base_cfg.pipeline.model.cull_screen_size = 0.30 # 0.2 # og 0.15
#     base_cfg.pipeline.model.cull_alpha_thresh = 0.25 # 0.15 # og 0.10
#     base_cfg.pipeline.model.cull_scale_thresh = 0.35
#     base_cfg.pipeline.model.reset_alpha_every = 150

#     # More Gaussians in holes? # cant really see the difference 
#     base_cfg.pipeline.model.densify_grad_thresh = 0.0005
#     base_cfg.pipeline.model.densify_size_thresh = 0.02
#     base_cfg.pipeline.model.n_split_samples = 2
#     base_cfg.pipeline.model.max_gauss_ratio = 12.0

#     # Reduce opacity learning rate, and increase positional # can't really see the difference
#     base_cfg.optimizers["opacities"]["optimizer"].lr = 0.01
#     base_cfg.optimizers["opacities"]["scheduler"] = ExponentialDecaySchedulerConfig(lr_pre_warmup=0.01, lr_final=0.002, warmup_steps=0, max_steps=int(max_steps), ramp="cosine")
#     base_cfg.optimizers["means"]["optimizer"].lr = 0.00025
#     base_cfg.optimizers["scales"]["optimizer"].lr = 0.003

#     # # Spherical harmonics - reduce early??? # No difference I could spot 
#     # base_cfg.pipeline.model.sh_degree = 0
#     # base_cfg.pipeline.model.sh_degree_interval = 1000
#     # base_cfg.pipeline.model.ssim_lambda = 0.1

#     # # Initialize Gaussians uniformly sometimes 
#     # base_cfg.pipeline.model.random_init = True 
#     # base_cfg.pipeline.model.num_random = 100
#     # base_cfg.pipeline.model.random_scale = 5.0
    
#     # Stop splitting later in run - eh feels like a bad idea
#     base_cfg.pipeline.model.stop_split_at = int(0.6 * max_steps)
#     base_cfg.pipeline.model.stop_screen_size_at = int(max_steps) 
    
#     # Collider - not only for training ? 
#     base_cfg.pipeline.model.enable_collider = True
#     base_cfg.pipeline.model.collider_params = {"near_plane": 2.0, "far_plane": 6.0}

# #######################

    base_cfg.viewer = ViewerConfig()
    base_cfg.viewer.quit_on_train_completion = True # NOTE if debuggin might want turn to False
    if disable_viewer: 
    # No viewer 
        base_cfg.vis = "tensorboard"
    else :
        # NOTE : for debugging only :)
        base_cfg.vis = "viewer" # use the web viewer
        base_cfg.viewer.websocket_port = viewer_port 

    # Train
    trainer = base_cfg.setup()
    ensure_dir(trainer.base_dir)
    trainer.setup(test_mode="val")

    if track_training: 
        add_fixed_train_render_callback(
            trainer,
            camera_index=0,
            every_n_steps=100,
            out_dir="temp/images"
        )

    trainer.train()

    # Return the location of the trained gaussian splt 
    return _latest_ckpt_path(run_dir=trainer.base_dir)


@torch.no_grad()
def render_gsplat(ckpt_path: Path, 
                  working_dir: Path,
                  data_dir: Path, 
                  out_dir: Path, 
                  cameras: Cameras,
                  experiment_name: str = "gsplat_exp",
                  project_name: str = "nerfstudio-project", 
                  save_images: bool = True, 
                  downscale_factor: int = 1) -> List[Path]: 
    """
    This function takes in parameters to an already trained Gsplat and renders the given views. 
    """
    ensure_dir(out_dir)
    # Standard gsplat settings 
    base_cfg = initialize_gsplat_config(data_dir=data_dir, working_dir=working_dir, experiment_name=experiment_name, project_name=project_name, downscale_factor=downscale_factor)

    base_cfg.load_checkpoint = ckpt_path 

    # Get trainer (which wraps the gsplat model)
    trainer = base_cfg.setup()
    trainer.setup(test_mode="inference")
    model = trainer.pipeline.model.eval()

    # Render views 
    images: List[np.ndarray] = []
    for i in range(len(cameras)):
        out = model.get_outputs_for_camera(cameras[i:i+1])
        rgb = out['rgb'].clamp(0,1).cpu().numpy()
        img = (rgb*255.0 + 0.5).astype(np.uint8)
        if save_images: 
            p = out_dir / f"render_{i:04d}.png"
            Image.fromarray(img).save(p)
        images.append(np.transpose(img, (2, 0, 1)))
    return torch.tensor(np.stack(images))



if __name__ == "__main__":
    data = Path("/workspace/dataset") 
    working_dir = Path("temp")
    # from time import perf_counter
    # t0 = perf_counter()
    cfg_path = train_gsplat(data, working_dir=working_dir, max_steps=6000, disable_viewer=True, track_training=True, experiment_name="demo_gsplat")
    # print(f"Time : {perf_counter() - t0:0.4f}s")

    from make_gif import make_gif
    make_gif(path=Path("temp/images/train"), out_path=Path("gifs/gsplat_train_no.gif"))
    make_gif(path=Path("temp/images/val"), out_path=Path("gifs/gsplat_val_no.gif"))