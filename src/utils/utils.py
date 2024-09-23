import os
import torch
from pathlib import Path
from datetime import date
from argparse import Namespace
from huggingface_hub import HfApi, HfFolder

def print_gpu_is_used() -> None:
    """ Print banner to show if gpu is used. """
    # Check if GPU available
    if torch.cuda.is_available():
        print("\n###################################################")
        print("Using GPU for training.")
        print("###################################################\n")
    else:
        print("GPU not available, using CPU instead.")


def get_session_name_and_ouput_dir(args: Namespace, config_env: dict[str, str]) -> tuple[str, Path, Path | None, Path | None] :
    """ Return session_name and output_dir computed by args. """

    today = date.today().strftime("%Y_%m_%d")
    model_size = args.model_name[args.model_name.find("-")+1:]
    freeze = 'unfreeze' if args.no_freeze else 'freeze'
    test_data = "prova_" if args.test_data else ""
    training_type = "_monolabel" if args.training_type == "monolabel" else ""
    session_name = f"{args.new_model_name}-{model_size}-{today}-{test_data}batch-size{args.batch_size}_{freeze}{training_type}"
    output_dir = Path(config_env["MODEL_PATH"], session_name)

    resume_from_checkpoint, latest_checkpoint = None, None
    if args.resume:
        session_name = config_env["MODEL_NAME"]
        output_dir = Path(config_env["MODEL_PATH"], session_name)
        checkpoints = [f for f in output_dir.iterdir() if f.name.startswith('checkpoint')]
        if len(checkpoints) != 0:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            resume_from_checkpoint = latest_checkpoint
        
    output_dir.mkdir(exist_ok=True, parents=True)

    return session_name, output_dir, resume_from_checkpoint, latest_checkpoint


def send_data_to_hugging_face(session_name: str, output_dir: Path) -> None:
    """ Send files to hugging face."""
    
    # Send data to huggingface.
    token = HfFolder.get_token()
    hf_api = HfApi(token=token)
    try:
        username = hf_api.whoami()["name"]
    except:
        print("User not found with hugging face token provide.")
        return
    
    repo_id = f"{username}/{session_name}"
    try:
        repo_url = hf_api.create_repo(token=token, repo_id=repo_id, private=False, exist_ok=True)
        print(f"Repository URL: {repo_url}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        raise

    all_files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f)) and f != "model.safetensors"]

    for filename in all_files:
        file_path = os.path.join(output_dir, filename)
        hf_api.upload_file(
            token=token,
            path_or_fileobj=file_path,
            path_in_repo=filename,
            repo_id=repo_id,
            commit_message=f"Upload {filename}"
        )

    print(f"All files successfully uploaded to the Hub: {repo_url}")

