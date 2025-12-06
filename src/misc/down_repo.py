from huggingface_hub import snapshot_download

snapshot_download(repo_id="deepcs233/Visual-CoT", repo_type="dataset", local_dir="data_viscot")
