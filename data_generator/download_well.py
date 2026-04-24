import os, sys
from huggingface_hub import snapshot_download

BASE = "/scratch/user/u.kt348068/PDE_data/Well"
DATASETS = [
    "gray_scott_reaction_diffusion",
    "rayleigh_benard",
    "shear_flow",
]

for ds in DATASETS:
    dest = os.path.join(BASE, ds)
    os.makedirs(dest, exist_ok=True)
    print(f"\n{'='*60}", flush=True)
    print(f"Downloading polymathic-ai/{ds} → {dest}", flush=True)
    print(f"{'='*60}", flush=True)
    try:
        snapshot_download(
            repo_id=f"polymathic-ai/{ds}",
            repo_type="dataset",
            local_dir=dest,
            max_workers=8,
            ignore_patterns=["*.gitattributes"],
        )
        print(f"✓ Done: {ds}", flush=True)
    except Exception as e:
        print(f"✗ Failed: {ds} — {e}", flush=True)
        sys.exit(1)

print("\nAll downloads complete.", flush=True)
