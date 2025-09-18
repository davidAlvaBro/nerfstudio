import shutil
import subprocess
import json 
from pathlib import Path

# Running suff in command line 
def run(cmd, cwd=None):
    print("\n$ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=cwd)

# File management 
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def remove(p: Path):
    if p.exists() and p.is_dir(): 
        shutil.rmtree(p)
    elif p.exists(): 
        p.unlink()

def load_transforms_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    # Nerfstudio / Instant-NGP style:
    # global intrinsics: fl_x, fl_y, cx, cy, w, h (optional)
    # per-frame: file_path, transform_matrix (c2w)
    return data
