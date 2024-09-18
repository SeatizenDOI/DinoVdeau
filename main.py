import json
import time
import argparse
from pathlib import Path
from codecarbon import EmissionsTracker
from huggingface_hub import HfFolder, HFSummaryWriter

from src.utils.training import setup_trainer
from src.data.data_loading import generate_labels
from src.utils.evaluation import evaluate_and_save
from src.data.data_preprocessing import create_datasets
from src.model.model_setup import setup_model, get_training_type_from_args
from src.utils.model_card_generator import generate_model_card, save_hyperparameters_to_config
from src.utils.utils import print_gpu_is_used, send_data_to_hugging_face, get_session_name_and_ouput_dir
    
def get_args():
    parser = argparse.ArgumentParser(description="DINOv2 Image Classification Training Script")
    
    # Training parameters.
    parser.add_argument('-is', '--image_size', type=int, default=518, help='Image size for both dimensions')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--initial_learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for the optimizer')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--patience_lr_scheduler', type=int, default=5, help='Patience for learning rate scheduler')
    parser.add_argument('--factor_lr_scheduler', type=float, default=0.1, help='Factor for learning rate scheduler')

    # Trainings options.
    parser.add_argument('--no_freeze', action="store_true", help='Flag to unfreeze model backbone')
    parser.add_argument('--no_data_aug', action="store_true", help='Flag to disable data augmentation')
    parser.add_argument('--test_data', action="store_true", help='Flag to test the model on a small subset of data')
    parser.add_argument('--resume', action="store_true", help='Flag to resume training from the last checkpoint')

    # Model options.
    parser.add_argument('--model_name', type=str, default="facebook/dinov2-large", help="Model name or path")
    parser.add_argument('--new_model_name', type=str, default="Aina", help="New model name")
    parser.add_argument('-tt', '--training_type', type=str, default="multilabel", help="Choose your training type. Can be multilabel or monolabel.")
    parser.add_argument('--no_custom_head', action="store_true", help='Flag to use linear layer instead of custom head')

    # Global options.
    parser.add_argument('--disable_web', action="store_true", help='Flag to disable the connection to the web')
    parser.add_argument('--config_path', default="config.json", help="Path to config.json file.")

    
    return parser.parse_args()


def main(args):

    # -- Load and parse arguments.

    # Load config json.
    config_path = Path(args.config_path)
    if not config_path.exists() or not config_path.is_file():
        print(f"Config file not found for path {config_path}")
        return

    with open(config_path, 'r') as file:
        config_env = json.load(file)

    print_gpu_is_used()

    start_time = time.time()

    # Create new model name.
    session_name, output_dir, resume_from_checkpoint, latest_checkpoint = get_session_name_and_ouput_dir(args, config_env)

    print("\ninfo : Model name is ", session_name)

    # Load Huggingface token.
    if not args.disable_web:
        HfFolder.save_token(config_env["HUGGINGFACE_TOKEN"])
        logger = HFSummaryWriter(repo_id=session_name, logdir=str(Path(output_dir, "runs")), commit_every=5)
    
    # Set up dataset.
    ds, dummy_feature_extractor, train_df = create_datasets(config_env["ANNOTATION_PATH"], args, config_env["IMG_PATH"], output_dir)
    classes_names, id2label, label2id = generate_labels(train_df)
    
    # Setup model.
    model = setup_model(args, classes_names, id2label, label2id)

    # Setup trainer.
    trainer = setup_trainer(args, model, ds, dummy_feature_extractor, output_dir)
    
    if not args.disable_web:
        # Track carbon emissions
        tracker = EmissionsTracker(log_level="WARNING", save_to_file=False, allow_multiple_runs=True)
        tracker.start()
    
    # Start training.
    print("\ninfo : Training model...\n")
    if args.resume:
        print("\ninfo : Resuming training from checkpoint \n", latest_checkpoint)
        train_results =  trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else :
        train_results = trainer.train()

    print(f"Total training time: {time.time() - start_time} seconds")

    print("\ninfo : Saving model...\n")
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    
    print("\ninfo : Evaluating model on test set...\n")
    evaluate_and_save(args, trainer, ds)

    # Save hyperparameters.
    emissions = None #tracker.stop() if not args.disable_web else None
    save_hyperparameters_to_config(output_dir, args, emissions)

    # Generate model card.
    counts_path = Path(config_env["ANNOTATION_PATH"], "count_df.csv")
    files = ['train_results.json', 'test_results.json', 'trainer_state.json', 'all_results.json', 'config.json', 'transforms.json']
    data_paths = [Path(output_dir, file) for file in files]
    
    print("info : Generating model card...\n")
    generate_model_card(data_paths, counts_path, output_dir, args)

    # Send data to hugging face if needed.
    if args.disable_web: return 
    send_data_to_hugging_face(session_name, output_dir)


if __name__ == "__main__":
    args = get_args()
    main(args)
