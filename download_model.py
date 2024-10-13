from huggingface_hub import snapshot_download

# Download the model snapshot from Hugging Face
snapshot_download(
    repo_id="michaelfeil/ct2fast-nllb-200-3.3B", 
    local_dir="models/ct2fast-nllb-200-3.3B",
    local_dir_use_symlinks=False,
    token="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"
)

print("Model snapshot downloaded successfully.")
