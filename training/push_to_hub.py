from huggingface_hub import create_repo, get_full_repo_name, upload_folder

output_dir = "/root/distil-whisper/corpus/output/hallucination/zhen_csasr_k2d_whisper_comp"
hub_model_id = "zhen_csasr_k2d_whisper_comp"
repo_name = hub_model_id
create_repo(repo_name, exist_ok=True)

upload_folder(
    folder_path=output_dir,
    repo_id="andybi7676/zhen_csasr_k2d_whisper_comp",
    repo_type="model",
    commit_message=f"Initial commit of {hub_model_id} model.",
)