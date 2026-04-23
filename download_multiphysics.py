from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Indulge-Bai/Multiphysics_Bench",
    repo_type="dataset",
    local_dir="/scratch/user/u.kt348068/PDE_data/Multiphysices_bench",
)
print("Done.")
