import imageio
from pathlib import Path

# Kill later 
import imageio.v2 as iio  

def downsample_gif(in_path: str, out_path: str):
    in_path, out_path = Path(in_path), Path(out_path)

    # Read frames (as numpy arrays) and collect per-frame durations if available
    frames = []
    durations = []

    reader = iio.get_reader(in_path)
    try:
        n = reader.get_length()
    except Exception:
        # Some GIFs don't report length; fall back to iterating until StopIteration
        n = None

    def frame_duration(idx):
        # Try to get per-frame duration (ms). Fallback to global duration or 100ms.
        try:
            meta = reader.get_meta_data(index=idx)  # per-frame meta (if supported)
            d_ms = meta.get("duration", None)
        except Exception:
            meta = reader.get_meta_data()  # global meta
            d_ms = meta.get("duration", None)
        return (d_ms / 1000.0) if d_ms else 0.1  # imageio expects seconds

    if n is not None and n >= 0:
        for i in range(n):
            if i % 2 == 0:  # keep every other frame
                frames.append(reader.get_data(i))
                durations.append(frame_duration(i))
    else:
        i = 0
        while True:
            try:
                frame = reader.get_data(i)
                if i % 2 == 0:
                    frames.append(frame)
                    durations.append(frame_duration(i))
                i += 1
            except IndexError:
                break
    reader.close()

    if not frames:
        raise ValueError("No frames read from input GIF.")

    # Preserve loop behavior if present
    try:
        global_meta = iio.get_reader(in_path).get_meta_data()
        loop = global_meta.get("loop", 0)  # 0 = loop forever
    except Exception:
        loop = 0

    # Write the new GIF. durations can be a list (seconds per frame).
    iio.mimsave(out_path, frames, duration=durations, loop=loop) 

def make_gif(path: Path, out_path: Path): 
    filenames = [p for p in path.iterdir()]
    filenames = sorted(filenames)
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(out_path, images)#, fps=12, codec="libx264")

if __name__ == "__main__": 
    # out_path = Path("gifs/gsplat_sparse_fly_around.gif")
    out_path = Path("gifs/gsplat_with_poses.gif")
    path = Path("dataset2/rendered/renders")
    # make_gif(path=path, out_path=out_path)
    downsample_gif(in_path=out_path, out_path=out_path)
    # downsample_gif("input.gif", "output_every_other.gif")
