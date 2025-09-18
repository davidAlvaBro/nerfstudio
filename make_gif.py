import imageio
from pathlib import Path

def make_gif(path: Path, out_path: Path): 
    filenames = [p for p in path.iterdir()]
    filenames = sorted(filenames)
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(out_path, images)

if __name__ == "__main__": 
    out_path = Path("gifs/gsplat_exp.gif")
    path = Path("temp/images/train")
    make_gif(path=path, out_path=out_path)