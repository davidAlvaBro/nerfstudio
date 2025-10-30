from pathlib import Path 

import numpy as np 
import torch as th 
from ultralytics import YOLO 

from utils import load_transforms_json

# 
YOLO_TO_CONTROLNET = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3] 
# YOLO_TO_CONTROLNET = [0, 17, 5, 7, 9, 6, 8, 10, 11, 13, 15, 12, 14, 16, 1, 2, 3, 4] 
N_DATAPOINTS_CONTROLNET = 18

def apply_yolo(data_folder: Path, out_path: Path) -> None: 
    """
    Given a directory runs YOLOv11 on each image, 
    and stores the ControlNet-version keypoint in a metadata file in this folder. 
    """
    model = YOLO("yolo11n-pose.pt") # Just standard pose yolo model 
    run_args = load_transforms_json(data_folder / "transforms.json")
    pngs = [data_folder / frame["file_path"] for frame in run_args["frames"]] + [data_folder / frame["file_path"] for frame in run_args["eval"]]

    # pngs = [p for p in images_path.glob("*.[pP][nN][gG]") if p.is_file()]
    
    # Book keeping 
    index = 0 
    person = 0
    image_to_people = {} # Pointers to subset 
    image_to_imsize = {} # Pointers to shape of each image 
    subset_full = np.zeros((0, N_DATAPOINTS_CONTROLNET)) # points to each point 
    candidate_full = np.zeros((0, 4)) # Each point 
    
    # Here ultralytics handle the batching of images 
    for r in model.predict(
            source=pngs,
            task="pose",       # optional; inferred from model, but explicit is fine
            batch=16,          # try 32/64 depending on VRAM
            imgsz=640,         # adjust as needed
            device=0,          # GPU 0 (use "mps" on Apple, or "cpu" if no GPU)
            half=True,         # FP16 on CUDA for speed/VRAM savings
            workers=4,         # dataloader worker threads
            stream=True,       # generator-style iteration
            verbose=False
        ):
        # r is the results object which has the keypoints for each instance 
        kpts_xy  = getattr(r.keypoints, "xy", None).detach().cpu()
        img_path = r.path
        h, w = r.orig_img.shape[:2]
        n_characters = kpts_xy.shape[0]
        
        # Construct "extra" point (between shoulders)
        new_points = ((kpts_xy[:, 5, :] + kpts_xy[:, 6, :])/2).unsqueeze(1)
        kpts_xy = th.hstack((kpts_xy, new_points))

        # Put in the "correct/controlnet" ordering 
        kpts_xy = kpts_xy[:, YOLO_TO_CONTROLNET].numpy().astype(int)

        # Store as candidates and subsets 
        candidate = np.ones((n_characters*N_DATAPOINTS_CONTROLNET, 4)) 
        subset = np.zeros((n_characters, N_DATAPOINTS_CONTROLNET))

        for i, person_kpts in enumerate(kpts_xy): 
            for j, joint in enumerate(person_kpts): 
                # Each joint is stored along with the pointer in subset  
                candidate[i*N_DATAPOINTS_CONTROLNET + j, :2] = joint 
                subset[i, j] = i*N_DATAPOINTS_CONTROLNET + j + index 
            
        # Update full accounts (image_to_people, subset, candidate) 
        image_to_people[str(img_path)] = list(range(person, person + n_characters))
        image_to_imsize[str(img_path)] = (h, w)
        person += n_characters 
        index += n_characters*N_DATAPOINTS_CONTROLNET 
        subset_full = np.concatenate((subset_full, subset)) 
        candidate_full = np.concatenate((candidate_full, candidate)) 
        
    # Afterwards this is stored in the data folder 
    annotated_data = {"candidate" : candidate_full, 
                      "subset": subset_full,
                      "image_to_people": image_to_people, 
                      "image_to_size": image_to_imsize}
    np.savez(out_path / "pred_annotation.npz", **annotated_data)

    return out_path / "pred_annotation.npz"
    
if __name__ == "__main__": 
    apply_yolo(Path("dataset/rendered"), Path("dataset/rendered"))