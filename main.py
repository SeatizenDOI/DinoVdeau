import torch
from accelerate import Accelerator
from config.arguments import get_args
from data.data_loading import load_datasets
from data.data_preprocessing import create_datasets
from model.model_setup import setup_model
from utils.training import setup_trainer
from utils.evaluation import evaluate_and_save
from utils.utils import generate_model_card, save_hyperparameters_to_config
from codecarbon import EmissionsTracker
from huggingface_hub import HfApi, HfFolder, HFSummaryWriter
import os
from datetime import date
import time
import numpy as np
import json

def bytes_to_gb(memory_in_bytes):
    return memory_in_bytes / (1024 ** 3)

def print_gpu_memory():
    free_mem, total_mem = torch.cuda.mem_get_info()
    print(f"GPU Memory - Free: {bytes_to_gb(free_mem):.2f} GB, Total: {bytes_to_gb(total_mem):.2f} GB")
    print(torch.cuda.mem_get_info())
    #print(torch.cuda.memory_summary())

def main():
    args = get_args()

    with open("config.json", 'r') as file:
        config_env = json.load(file)

    accelerator = Accelerator()
    
    # Clear GPU at beginning 
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        print("\n###################################################")
        print("Using GPU for training.")
        print(f"Training is using {accelerator.state.num_processes} GPU(s).")
        print("###################################################\n")
    else:
        print("GPU not available, using CPU instead.")

    today = date.today().strftime("%Y_%m_%d")
    start_time = time.time()
    session_name = f"{args.model_name.replace('facebook/', '')}-{today}-prova_batch-size{args.batch_size}_epochs{args.num_train_epochs}_{'freeze' if args.freeze_flag else 'unfreeze'}"
    output_dir = os.path.join(config_env["MODEL_PATH"], session_name)

    HfFolder.save_token(config_env["HUGGINGFACE_TOKEN"])
    _ = HFSummaryWriter(repo_id=session_name, logdir=os.path.join(output_dir, "runs"), commit_every=5)

    # Load dataset
    train_df, val_df, test_df = load_datasets(config_env["ANNOTATION_PATH"])
    classes_names = train_df.columns[1:].tolist()
    classes_nb = list(np.arange(len(classes_names)))
    id2label = {int(classes_nb[i]): classes_names[i] for i in range(len(classes_nb))}
    label2id = {v: k for k, v in id2label.items()}

    ds = create_datasets(config_env["ANNOTATION_PATH"], args, config_env["IMG_PATH"], output_dir)

    model = setup_model(args, classes_names, id2label, label2id)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_learning_rate, weight_decay=args.weight_decay)

    # Clear GPU cache before major allocation
    torch.cuda.empty_cache()

    model, optimizer, ds["train"], ds["validation"] = accelerator.prepare(
        model, optimizer, ds["train"], ds["validation"]
    )

    trainer = setup_trainer(args, model, ds, output_dir)

    tracker = EmissionsTracker(log_level="WARNING", save_to_file=False)
    tracker.start()
    count = 0
    max_count = 3
    training_complete = 0

    print("info : Training model...\n")

    while count < max_count and training_complete == 0:
        try:
            train_results = trainer.train()
            training_complete = 1
        except RuntimeError as e:
            count += 1
            if "CUDA out of memory" in str(e):
                print("CUDA out of memory error caught. Reducing batch size and trying again.")
                args.batch_size = max(1, args.batch_size // 2)  # Reduce batch size
                print(f"New batch size: {args.batch_size}")
                trainer.args.per_device_train_batch_size = args.batch_size
                trainer.args.per_device_eval_batch_size = args.batch_size
                #train_results = trainer.train()
            else:
                raise e

    end_time = time.time()
    print(f"Total training time: {end_time - start_time} seconds")
    # Clear GPU cache after training
    torch.cuda.empty_cache()
    print("info : Saving model...\n")
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    print("info : Evaluating model on test set...\n")
    evaluate_and_save(trainer, ds, accelerator)
    emissions = tracker.stop()

    emissions_data = {
        'emissions': emissions,
        'source': "Code Carbon",
        'training_type': "fine-tuning",
        'geographical_location': "Brest, France",
        'hardware_used': "NVIDIA Tesla V100 PCIe 32 Go"
    }

    hyperparameters = {
        'initial_learning_rate': args.initial_learning_rate,
        'train_batch_size': args.batch_size,
        'eval_batch_size': args.batch_size,
        'optimizer': {'type': 'Adam'},
        'lr_scheduler_type': {'type': 'ReduceLROnPlateau'},
        'patience_lr_scheduler': args.patience_lr_scheduler,
        'factor_lr_scheduler': args.factor_lr_scheduler,
        'weight_decay': args.weight_decay,
        'early_stopping_patience': args.early_stopping_patience,
        'freeze_encoder': args.freeze_flag,
        'data_augmentation':args.data_aug_flag,
        'num_epochs': args.num_train_epochs,
        'emissions_data': emissions_data
    }
    save_hyperparameters_to_config(output_dir, hyperparameters)

    counts_path = os.path.join(config_env["ANNOTATION_PATH"], "count_df.csv")
    data_paths = [
        os.path.join(output_dir, 'train_results.json'),
        os.path.join(output_dir, 'test_results.json'),
        os.path.join(output_dir, 'trainer_state.json'),
        os.path.join(output_dir, 'all_results.json'),
        os.path.join(output_dir, 'config.json'),
        os.path.join(output_dir, 'transforms.json')
    ]
    print("info : Generating model card...\n")
    generate_model_card(data_paths, counts_path, output_dir)


    hf_api = HfApi()
    token = HfFolder.get_token()

    repo_name = session_name
    username = "lombardata"
    repo_id = f"{username}/{repo_name}"
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

if __name__ == "__main__":
    main()
