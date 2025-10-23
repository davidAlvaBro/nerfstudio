import sys
import math 
import os 
from time import perf_counter
from typing import Tuple

from pathlib import Path
from collections import OrderedDict
import numpy as np

from utils import run, ensure_dir, remove, load_transforms_json

# ------------ Helpers --------------
def ensure_4x4(m):
    M = np.asarray(m, dtype=np.float64)
    if M.shape == (4, 4):
        return M
    if M.shape == (3, 4):
        bot = np.array([[0, 0, 0, 1.0]], dtype=np.float64)
        return np.vstack([M, bot])
    raise ValueError(f"Pose must be 3x4 or 4x4, got {M.shape}")

# Helpers for computervision conversion 
# TODO look at me 
def mat3_to_hamilton_quat(R):
    """Return quaternion in COLMAP's (qw, qx, qy, qz) Hamilton convention."""
    # Robust conversion
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    # Normalize and make qw >= 0 (COLMAP convention prefers this)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q *= -1.0
    return q

# TODO look at me 
def to_colmap_from_nerf_c2w(c2w_nerf):
    """
    Convert NeRF(OpenGL) c2w -> COLMAP/OpenCV c2w.
    NeRF camera: +X right, +Y up, +Z back (look along -Z).
    COLMAP camera: +X right, +Y down, +Z forward.
    Mapping (camera frame): v_colmap = S * v_nerf, with S = diag(1, -1, -1).
    For c2w matrices (camera->world): R_cv = R_gl * S, t unchanged.
    """
    S = np.diag([1.0, -1.0, -1.0, 1.0])
    # Apply on camera basis (right/up/forward columns)
    R_gl = c2w_nerf[:3, :3]
    t = c2w_nerf[:3, 3]
    R_cv = R_gl @ np.diag([1.0, -1.0, -1.0])  # strip to 3x3 version of S
    return R_cv, t
# TODO look at me 
def c2w_to_w2c(R_c2w, t_c2w):
    """Standard inversion: w2c = [R^T | -R^T t]."""
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w
    return R_w2c, t_w2c
# TODO look at me 
def fmt(*vals):
    return " ".join(f"{v:.12g}" for v in vals)

# TODO look at me 
def build_and_write_colmap_text(transforms_path, out_dir, model="PINHOLE"):
    os.makedirs(out_dir, exist_ok=True)
    data = load_transforms_json(transforms_path)
    frames = data.get("frames", [])
    parent_folder = None

    # Global intrinsics (if present)
    global_w = data.get("w", None)
    global_h = data.get("h", None)
    flx = data.get("fl_x", None)
    fly = data.get("fl_y", None)
    cxg = data.get("cx", None)
    cyg = data.get("cy", None)

    # Collect per-image records
    records = []
    for f in frames:
        # Resolve image file (transforms usually store relative paths without extension sometimes)
        file_path = Path(f.get("file_path"))

        # normalize to basename with extension found on disk
        img_name = file_path.name
        if parent_folder is None: 
            parent_folder = file_path.parent

        # Intrinsics (per-frame overrides if present)
        w = f.get("w", global_w)
        h = f.get("h", global_h)
        fx = f.get("fl_x", flx)
        fy = f.get("fl_y", fly)
        cx = f.get("cx", cxg)
        cy = f.get("cy", cyg)
        if None in (w, h, fx, fy, cx, cy):
            raise RuntimeError(f"Missing intrinsics for frame {img_name}")

        # Pose
        M = ensure_4x4(f.get("transform_matrix"))
        R_cv, t = to_colmap_from_nerf_c2w(M)  # convert NeRF c2w -> COLMAP c2w
        R_w2c, t_w2c = c2w_to_w2c(R_cv, t)    # invert to COLMAP w2c
        q = mat3_to_hamilton_quat(R_w2c)

        records.append({
            "name": img_name,
            "w": int(round(w)), "h": int(round(h)),
            "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
            "q": q, "t": t_w2c
        })

    # Deduplicate cameras by (model, w,h,fx,fy,cx,cy)
    cam_key_to_id = OrderedDict()
    next_cam_id = 1
    for r in records:
        key = (model, r["w"], r["h"], r["fx"], r["fy"], r["cx"], r["cy"])
        if key not in cam_key_to_id:
            cam_key_to_id[key] = next_cam_id
            next_cam_id += 1
        r["camera_id"] = cam_key_to_id[key]

    # Assign image ids
    for i, r in enumerate(records, start=1):
        r["image_id"] = i

    # Write cameras.txt
    cam_path = os.path.join(out_dir, "cameras.txt")
    with open(cam_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: {}\n".format(len(cam_key_to_id)))
        for key, cam_id in cam_key_to_id.items():
            mdl, w, h, fx, fy, cx, cy = key
            if mdl.upper() == "PINHOLE":
                params = [fx, fy, cx, cy]
            elif mdl.upper() == "OPENCV":
                # If you really have distortion, replace zeros with your k1 k2 p1 p2.
                params = [fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0]
            else:
                raise ValueError(f"Unsupported camera model: {mdl}")
            f.write(f"{cam_id} {mdl} {w} {h} {fmt(*params)}\n")

    # Write images.txt
    img_path = os.path.join(out_dir, "images.txt")
    with open(img_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write("# Number of images: {}, mean observations per image: 0\n".format(len(records)))
        for r in records:
            qw, qx, qy, qz = r["q"]
            tx, ty, tz = r["t"]
            line = f"{r['image_id']} {fmt(qw, qx, qy, qz, tx, ty, tz)} {r['camera_id']} {r['name']}\n"
            f.write(line)
            f.write("\n")  # empty 2D observations line (allowed/expected for known-poses workflow)

    # Write points3D.txt (empty)
    pts_path = os.path.join(out_dir, "points3D.txt")
    with open(pts_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")
  
    return parent_folder


def run_colmap_frozen_poses(metadata_path: Path, workdir: Path, out_path: Path, cleanup: bool = True, sparse_only: bool = False) -> Tuple[float, float]: 
    """
    This function is a crude way to run the colmap pipeline with given extrinsics. 
    The extrinsics and images must be in the "data" directory. 
    The extrinsics are translated into the proper format, then colmap is run step for step.
    """
    text_model = workdir / Path("colmap")
    db = workdir / "database.db"
    sparse = workdir / "sparse_triangulated"
    dense = workdir / "dense"
    ensure_dir(workdir)

    images = build_and_write_colmap_text(metadata_path, out_dir=text_model, model="PINHOLE")
    sparse_ply = out_path / "ply.ply"
    fused_ply = out_path / "fused.ply"

    # Checkups 
    if not images.exists():
        print(f"[ERR] Images dir not found: {images}", file=sys.stderr)
        sys.exit(2)
    for fname in ["cameras.txt", "images.txt", "points3D.txt"]:
        if not (text_model / fname).exists():
            print(f"[ERR] Missing {fname} in text-model folder: {text_model}", file=sys.stderr)
            sys.exit(2)


    # Remove files that should not be there 
    remove(db) 
    remove(sparse)
    remove(dense)

    t_start = perf_counter()
    # Feature extraction
    run([
        "colmap", "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(images),
        "--SiftExtraction.use_gpu", "1" # was *sift_gpu_flag if I want that back 
    ])


    # Matching
    run([
        "colmap", "exhaustive_matcher", # If it takes to long change to "sequential_matcher"
        "--database_path", str(db),
        "--SiftMatching.use_gpu", "1" # Again remove this if not on GPU 
    ])
    
    # Triangulate using poses (keeps extrinsics fixed)
    ensure_dir(sparse) # Directory has to exist 
    run([
        "colmap", "point_triangulator",
        "--database_path", str(db),
        "--image_path", str(images),
        "--input_path", str(text_model),
        "--output_path", str(sparse),
    ])

    # Export sparse PLY
    run([
        "colmap", "model_converter",
        "--input_path", str(sparse),
        "--output_path", str(sparse_ply),
        "--output_type", "PLY",
    ])
    t_sparse_done = perf_counter()

    ######
    # Now the sparse could is actually finished
    ######

    if sparse_only : return (t_sparse_done - t_start, 0.)

    # Undistort for dense
    run([
        "colmap", "image_undistorter",
        "--image_path", str(images),
        "--input_path", str(sparse),
        "--output_path", str(dense),
    ])

    # PatchMatch stereo
    run([
        "colmap", "patch_match_stereo",
        "--workspace_path", str(dense),
        # "--PatchMatchStereo.max_image_size", str(3200), # Have not played with any of these settings 
        # "--PatchMatchStereo.geom_consistency", True,
        # "--PatchMatchStereo.gpu_index", str(0),
        # "--PatchMatchStereo.num_iterations", str(8),
        # "--PatchMatchStereo.num_samples", str(2),
    ])
    
    # Fusion
    run([
        "colmap", "stereo_fusion",
        "--workspace_path", str(dense),
        "--output_path", str(fused_ply),
        ])
    
    t_dense_done = perf_counter()

    # Cleanup if given 
    if cleanup : remove(workdir) 

    return (t_sparse_done - t_start, t_dense_done - t_start)