# # save as export_tb_images.py or paste into a cell
# import os, io
# from pathlib import Path
# from PIL import Image
# from tensorboard.backend.event_processing import event_accumulator

# def export_tb_images(run_dir: str | Path, out_dir: str | Path, tag_filter: str | None = None):
#     """Find TensorBoard event files under `run_dir` and export all logged images to PNGs."""
#     run_dir = Path(run_dir)
#     out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

#     # Find directories that contain TB event files
#     event_dirs = set()
#     for root, _, files in os.walk(run_dir):
#         if any(f.startswith("events.out.tfevents") for f in files):
#             event_dirs.add(root)

#     if not event_dirs:
#         print(f"No TensorBoard event files found under {run_dir}")
#         return

#     for ed in sorted(event_dirs):
#         print(f"[events] {ed}")
#         ea = event_accumulator.EventAccumulator(ed, size_guidance={'images': 10**6})
#         ea.Reload()

#         img_tags = ea.Tags().get('images', [])
#         if tag_filter:
#             img_tags = [t for t in img_tags if tag_filter in t]
#         if not img_tags:
#             print("  (no image tags here)"); continue

#         for tag in img_tags:
#             events = ea.Images(tag)
#             print(f"  tag: {tag}  (frames={len(events)})")
#             safe_tag = tag.replace("/", "_")
#             for ev in events:
#                 img = Image.open(io.BytesIO(ev.encoded_image_string))
#                 # step is the training step for this frame
#                 fname = f"{safe_tag}_step{ev.step:06d}.png"
#                 img.save(out_dir / fname)

# # Example usage:
# export_tb_images("temp/demo_gsplat/splatfacto/nah", "temp/viewer_snaps", tag_filter="Eval")
# export_tb_images_all.py
import os, io, re, json
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tensorboard.backend.event_processing import event_accumulator

def _safe(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")

def _index_from_tag(tag: str) -> int | None:
    """
    Try to extract an image index from tags like:
    'EvalImages/image_0005', 'Eval/image-12', 'image_7', etc.
    Returns None if no trailing integer is found.
    """
    m = re.search(r'(\d+)\D*$', tag)
    return int(m.group(1)) if m else None

def export_tb_images(
    run_dir: str | Path,
    out_dir: str | Path,
    tag_prefix: str | None = "Eval",     # only export image tags containing this
    group_by: str = "step",              # "step" -> step_<n>/image_XXXX.png ; "tag" -> tagFolder/step_<n>.png
    keep_latest: bool = True,            # keep the last occurrence for (tag, step)
    index_to_name: dict[int, str] | None = None,  # optional mapping {idx: "nice_name"}
):
    """
    Export all TensorBoard logged images to PNGs, grouping either by training step or by tag.
    Useful when Nerfstudio renders *all* eval images at each eval step.
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) find all event dirs under run_dir
    event_dirs: set[str] = set()
    for root, _, files in os.walk(run_dir):
        if any(f.startswith("events.out.tfevents") for f in files):
            event_dirs.add(root)

    if not event_dirs:
        print(f"No TensorBoard event files under {run_dir}")
        return

    # 2) collect images keyed by (tag, step) — keep the latest wall_time if duplicates
    records: dict[tuple[str,int], tuple[float, bytes]] = {}
    for ed in sorted(event_dirs):
        ea = event_accumulator.EventAccumulator(ed, size_guidance={'images': 10**6})
        ea.Reload()
        for tag in ea.Tags().get('images', []):
            if tag_prefix and tag_prefix not in tag:
                continue
            for ev in ea.Images(tag):
                key = (tag, ev.step)
                if keep_latest:
                    # wall_time grows across runs; keep the last we see
                    old = records.get(key)
                    if (old is None) or (ev.wall_time >= old[0]):
                        records[key] = (ev.wall_time, ev.encoded_image_string)
                else:
                    # first occurrence wins
                    records.setdefault(key, (ev.wall_time, ev.encoded_image_string))

    if not records:
        print("No matching image tags found.")
        return

    # 3) materialize to disk
    #    group by step or tag for cleaner folder structure
    by_step: defaultdict[int, list[tuple[str, bytes]]] = defaultdict(list)
    by_tag: defaultdict[str, list[tuple[int, bytes]]] = defaultdict(list)

    for (tag, step), (_, enc) in records.items():
        if group_by == "step":
            by_step[step].append((tag, enc))
        else:
            by_tag[tag].append((step, enc))

    if group_by == "step":
        for step in sorted(by_step.keys()):
            step_dir = out_dir / f"step_{step:06d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            # sort frames by parsed index if present, else by tag
            frames = by_step[step]
            frames.sort(key=lambda t: (_index_from_tag(t[0]) if _index_from_tag(t[0]) is not None else 1_000_000, t[0]))
            for tag, enc in frames:
                idx = _index_from_tag(tag)
                base = index_to_name.get(idx, f"image_{idx:04d}") if (index_to_name and idx is not None) else f"{_safe(tag)}"
                img = Image.open(io.BytesIO(enc))
                img.save(step_dir / f"{base}.png")
    else:
        # group_by == "tag": one folder per tag, frames ordered by step
        for tag, items in by_tag.items():
            tag_dir = out_dir / _safe(tag)
            tag_dir.mkdir(parents=True, exist_ok=True)
            for step, enc in sorted(items, key=lambda p: p[0]):
                img = Image.open(io.BytesIO(enc))
                img.save(tag_dir / f"step_{step:06d}.png")

    # 4) optional: write a small manifest to help you match frames
    manifest = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "group_by": group_by,
        "tags_exported": sorted({t for (t, _) in records.keys()}),
        "steps": sorted({s for (_, s) in records.keys()}),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Exported {len(records)} images to {out_dir}")

export_tb_images("temp/demo_gsplat/splatfacto/nah", "temp/viewer_snaps")