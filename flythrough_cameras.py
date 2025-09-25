import numpy as np
import json
from pathlib import Path


def _mat3_to_quat(R):
    # robust 3x3 -> (w,x,y,z)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    else:
        i = np.argmax([R[0,0], R[1,1], R[2,2]])
        if i == 0:
            s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)

def _quat_to_mat3(q):
    w, x, y, z = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=np.float64)

def _slerp(q0, q1, t):
    # q0,q1: unit quats (w,x,y,z)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + t*(q1 - q0)
        return q / np.linalg.norm(q)
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin0 = np.sin(theta0)
    theta = theta0 * t
    s0 = np.sin(theta0 - theta) / sin0
    s1 = np.sin(theta) / sin0
    return s0*q0 + s1*q1

def _pose_slerp(M0, M1, t):
    """Interpolate two NeRF-style c2w 4x4s: SLERP(R) + LERP(t)."""
    R0, R1 = M0[:3,:3], M1[:3,:3]
    q0, q1 = _mat3_to_quat(R0), _mat3_to_quat(R1)
    q  = _slerp(q0, q1, t)
    R  = _quat_to_mat3(q)
    p0, p1 = M0[:3,3], M1[:3,3]
    p  = (1.0 - t) * p0 + t * p1
    M  = np.eye(4, dtype=np.float64)
    M[:3,:3] = R
    M[:3, 3] = p
    return M

def _load_training_c2w_from_transforms(scene_path):
    """Return list of NeRF-style 4x4 c2w from transforms.json in scene_path."""
    tf = Path(scene_path) / "transforms.json"
    data = json.loads(tf.read_text())
    c2ws = []
    for fr in data["frames"]:
        M = np.array(fr["transform_matrix"], dtype=np.float64)
        # ensure bottom row
        if M.shape == (3,4):
            M = np.vstack([M, np.array([0,0,0,1], dtype=np.float64)])
        c2ws.append(M)
    return c2ws

def intermediate_poses_between_training_views(scene_path, n_between=2):
    """Build two (or n_between) poses between each consecutive training view."""
    c2ws = _load_training_c2w_from_transforms(scene_path)
    Ts = []
    if len(c2ws) < 2:
        return Ts
    # e.g., t = [1/3, 2/3] when n_between=2
    # ts = [(k+1)/(n_between+1) for k in range(n_between)]
    ts = [(k)/(n_between) for k in range(n_between)]
    for i in range(len(c2ws)-1):
        M0, M1 = c2ws[i], c2ws[i+1]
        for t in ts:
            Ts.append(_pose_slerp(M0, M1, t))
    return Ts 