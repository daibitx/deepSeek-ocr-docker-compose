from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="deepseek-ai/DeepSeek-OCR",
    cache_dir="/path/to/local/cache",
    resume_download=True
)
