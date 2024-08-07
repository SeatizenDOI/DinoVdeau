import os
import json
import time
import torch
import numpy as np
from datetime import date
from codecarbon import EmissionsTracker
from huggingface_hub import HfApi, HfFolder, HFSummaryWriter
from PIL import ImageFile as PILImageFile
PILImageFile.LOAD_TRUNCATED_IMAGES = True

from config.arguments import get_args
from data.data_loading import load_datasets
from data.data_preprocessing import create_datasets
from model.model_setup import setup_model
from utils.training import setup_trainer
from utils.evaluation import evaluate_and_save
from utils.utils import generate_model_card, save_hyperparameters_to_config


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
    # Clear GPU at beginning 
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        print("\n###################################################")
        print("Using GPU for training.")
        print("###################################################\n")
    else:
        print("GPU not available, using CPU instead.")

    today = date.today().strftime("%Y_%m_%d")
    start_time = time.time()        
    session_name = f"{args.model_name.replace('facebook/dinov2', 'DinoVdeau')}-{today}-batch-size{args.batch_size}_epochs{args.epochs}_{'freeze' if args.freeze_flag else 'unfreeze'}"
    if args.test_data_flag :
        session_name = session_name.replace("batch-size", "prova_batch-size")
    output_dir = os.path.join(config_env["MODEL_PATH"], session_name)

    # Resume training logic
    resume_from_checkpoint = None
    if args.resume:
        session_name = config_env["MODEL_NAME"]
        output_dir = os.path.join(config_env["MODEL_PATH"], session_name)
        checkpoints = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith('checkpoint')]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            resume_from_checkpoint = latest_checkpoint
            
    print("\ninfo : Model name is \n", session_name)

    if not os.path.exists(output_dir):
        # Safely create the output directory
        try:
            os.makedirs(output_dir, exist_ok=True)
        except FileExistsError:
            pass
    if args.enable_web:
        HfFolder.save_token(config_env["HUGGINGFACE_TOKEN"])
        logger = HFSummaryWriter(repo_id=session_name, logdir = os.path.join(output_dir, "runs"), commit_every=5)

    # Load dataset
    train_df, val_df, test_df = load_datasets(config_env["ANNOTATION_PATH"], args.test_data_flag)
    classes_names = train_df.columns[1:].tolist()
    classes_nb = list(np.arange(len(classes_names)))
    id2label = {int(classes_nb[i]): classes_names[i] for i in range(len(classes_nb))}
    label2id = {v: k for k, v in id2label.items()}
    # Set up dataset
    ds, dummy_feature_extractor = create_datasets(config_env["ANNOTATION_PATH"], args, config_env["IMG_PATH"], output_dir)
    # Set up model
    model = setup_model(args, classes_names, id2label, label2id)
    # Clear GPU cache before major allocation
    torch.cuda.empty_cache()
    # Set up the Trainer object
    trainer = setup_trainer(args, model, ds, dummy_feature_extractor, output_dir)
    if args.enable_web:
        # Track carbon emissions
        tracker = EmissionsTracker(log_level="WARNING", save_to_file=False)
        tracker.start()
    # Start training
    print("\ninfo : Training model...\n")
    if args.resume:
        print("\ninfo : Resuming training from checkpoint \n", latest_checkpoint)
        train_results =  trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else :
        train_results = trainer.train()

    end_time = time.time()
    print(f"Total training time: {end_time - start_time} seconds")
    # Clear GPU cache after training
    torch.cuda.empty_cache()
    #print("\ninfo : Saving model...\n")
    #trainer.save_model()
    #trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    print("\ninfo : Evaluating model on test set...\n")
    evaluate_and_save(args, trainer, ds)

    if args.enable_web:

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
            'num_epochs': args.epochs,
            'emissions_data': emissions_data
        }

    else :

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
            'num_epochs': args.epochs
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

    if args.enable_web:
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
