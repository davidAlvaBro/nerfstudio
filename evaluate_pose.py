from typing import Tuple
import math

from pathlib import Path 
import numpy as np 
import cv2 

from utils import ensure_dir

# draw the body keypoint and lims
def draw_bodypose(canvas, candidate, subset):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    # plt.imsave("preview.jpg", canvas[:, :, [2, 1, 0]])
    # plt.imshow(canvas[:, :, [2, 1, 0]])
    return canvas

def compare_poses(gt_pose_path: Path, pred_pose_path: Path, threshold: float = 20.0, verbose: bool = False) -> Tuple[float, float]: 
    """
    This function loads the annotations for a ground truth and the predicted, 
    then compares the keypoints of each image to print the Percent Correct Keypoints (PCK), 
    and the Mean Per Joint Position Error (MPJPE).

    The threshold is used for PCK. 
    """
    # Load pose data (candidates/keypoint_xy, subset/pointer, image_to_people, image_to_size)
    gt_loaded = np.load(gt_pose_path, allow_pickle=True)
    gt_candidate_full = gt_loaded["candidate"]
    gt_subset_full = gt_loaded["subset"].astype(int)
    gt_image_to_people = gt_loaded["image_to_people"].item() 
    pred_loaded = np.load(pred_pose_path, allow_pickle=True)
    pred_candidate_full = pred_loaded["candidate"]
    pred_subset_full = pred_loaded["subset"].astype(int)
    pred_image_to_people = pred_loaded["image_to_people"].item() 

    # Print initial stats of images that cannot be compared (missing data)
    gt_image_to_people = {Path(k).name: gt_image_to_people[k] for k in gt_image_to_people}
    pred_image_to_people = {Path(k).name: pred_image_to_people[k] for k in pred_image_to_people}
    gt_keys = set(map(str, gt_image_to_people.keys()))
    pred_keys = set(map(str, pred_image_to_people.keys()))
    only_in_gt = gt_keys - pred_keys
    only_in_pred = pred_keys - gt_keys
    print(f"Found missmatch between images to evaluate.\nGT images not in pred: {only_in_gt},\nPred not in GT: {only_in_pred}")
    images_to_check = gt_keys & pred_keys 
    
    # Go through each image bot have and compare the keypoints 
    key_point_distances = [] # all keypoints 
    unmatched_counter = 0

    for img_path in images_to_check: 
        gt_people = gt_image_to_people[img_path]
        pred_people = pred_image_to_people[img_path]
        # TODO if this is slow (which it is not) do it in different way 
        gt_distances = {}
        for gt_person in gt_people: 
            gt_distances[gt_person] = {}
            for pred_person in pred_people: 
                gt_keypoints = gt_candidate_full[gt_subset_full[gt_person]][:,:2]
                pred_keypoints = pred_candidate_full[pred_subset_full[pred_person]][:,:2]
                gt_distances[gt_person][pred_person] = np.linalg.norm(gt_keypoints - pred_keypoints) # TODO check this norm 
        
        # Match them with the smallest distance first 
        while True: 
            # Stop when there are no more matches 
            if len(gt_people) == 0 or len(pred_people) == 0: 
                unmatched_counter += len(gt_people) + len(pred_people)
                break  

            minimum_dist = np.inf 
            min_idx = (-1,-1)
            for gt_person in gt_people: 
                for pred_person in pred_people: 
                    if gt_distances[gt_person][pred_person] < minimum_dist: 
                        min_idx = (gt_person, pred_person)
                        minimum_dist = gt_distances[gt_person][pred_person]
            # Put this down as a match and remove them from data structure 
            gt_keypoints = gt_candidate_full[gt_subset_full[min_idx[0]]][:,:2]
            pred_keypoints = pred_candidate_full[pred_subset_full[min_idx[1]]][:,:2]
            key_point_distances.extend(np.linalg.norm(gt_keypoints - pred_keypoints, axis=1).tolist()) # TODO check this norm 

            
            gt_people.remove(min_idx[0])
            pred_people.remove(min_idx[1])
            
    key_point_distances = np.array(key_point_distances)
    PCK = np.mean(key_point_distances < threshold)
    MPJPE = np.mean(key_point_distances)
    return PCK, MPJPE, unmatched_counter


# TODO look at the colors, they seem off? 
def project_poses(poses_path: Path, output_path: Path | None = None) -> None : 
    """
    This function draws the poses of each image in poses_path.images_to_people.keys()
    and stores them in "output_path" or "renders" in the parent directory of the input paths. 
    """
    # Load pose data (candidates/keypoint_xy, subset/pointer, image_to_people, image_to_size)
    loaded = np.load(poses_path, allow_pickle=True)
    candidate_full = loaded["candidate"]
    subset_full = loaded["subset"].astype(int)
    image_to_people = loaded["image_to_people"].item() 
    
    # Go through each image 
    images = list(image_to_people.keys())

    for img_path in images: 
        img = cv2.imread(img_path)
        people = image_to_people[img_path]
        subset = subset_full[people]
        # candidate = candidate_full[subset.reshape(-1)]

        # TODO : do I want any resizing? 
        img = draw_bodypose(img, candidate_full, subset)
        
        img_path = Path(img_path)
        if output_path is None: 
            output_path = img_path.parent / "renders"
        ensure_dir(output_path)
        cv2.imwrite(output_path / img_path.name, img)

if __name__ == "__main__": 
    project_poses(Path("dataset/rendered/pred_annotation.npz"))

# if __name__ == "__main__":
#     gt_pose_path = Path("dataset2/rendered/gt_annotation.npz")
#     pred_pose_path = Path("dataset2/rendered/pred_annotation.npz") 
#     PCK, MPJPE, unmatched_counter = compare_poses(gt_pose_path=gt_pose_path, pred_pose_path=pred_pose_path, verbose=True)
#     print(f"Percent Consistant Keypoints {PCK}, Mean Error {MPJPE}, unmatched people {unmatched_counter}")