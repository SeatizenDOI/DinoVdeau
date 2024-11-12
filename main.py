import time
from pathlib import Path
from argparse import ArgumentParser, Namespace
from huggingface_hub import HfFolder, HFSummaryWriter

from src.utils.enums import ClassificationType, LabelType
from src.utils.utils import print_gpu_is_used, get_config_env
from src.utils.evaluation import evaluate_and_save, generate_threshold
from src.utils.model_card_generator import generate_model_card, save_hyperparameters_to_config

from src.data.DatasetManager import DatasetManager

from src.model.HuggingModelManager import HuggingModelManager

from src.trainer.training import setup_trainer
from src.trainer.f1perclass import generate_f1_per_class

def get_args() -> Namespace:
    parser = ArgumentParser(description="DINOv2 Image Classification Training Script")
    
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
    parser.add_argument('--model_name', type=str, default="facebook/dinov2-large", help="Model name to fine-tune on hugging-face.")
    parser.add_argument('--new_model_name', type=str, default="DinoVdeau", help="New model name")
    parser.add_argument('-tt', '--training_type', type=str, default="multilabel", help="Choose your training type. Can be multilabel or monolabel.")
    parser.add_argument('--no_custom_head', action="store_true", help='Flag to use linear layer instead of custom head')

    # Global options.
    parser.add_argument('--disable_web', action="store_true", help='Flag to disable the connection to the web')
    parser.add_argument('--config_path', default="config.json", help="Path to config.json file.")

    
    return parser.parse_args()


def main(args: Namespace) -> None:

    # -- Load and parse arguments.

    # Load config json.
    config_env = get_config_env(args.config_path)

    print_gpu_is_used()
    start_time = time.time()
        
    # Setup dataset.
    datasetManager = DatasetManager(args, config_env["ANNOTATION_PATH"])

    # Setup model.
    modelManager = HuggingModelManager(args, datasetManager.label_type)
    modelManager.setup_model_dir()


    datasetManager.create_datasets(config_env["IMG_PATH"], modelManager.output_dir)
    modelManager.setup_model(datasetManager.classes_names, datasetManager.id2label, datasetManager.label2id)

    print("\ninfo : Model name is ", modelManager.model_name)

    # Load Huggingface token.
    if not args.disable_web:
        HfFolder.save_token(config_env["HUGGINGFACE_TOKEN"])
        logger = HFSummaryWriter(repo_id=modelManager.model_name, logdir=str(Path(modelManager.output_dir, "runs")), commit_every=5)

    # Setup trainer.
    trainer = setup_trainer(modelManager, datasetManager)
    
    # Start training.
    print("\ninfo : Training model...\n")
    if args.resume:
        print("\ninfo : Resuming training from checkpoint \n", modelManager.latest_checkpoint)
        train_results = trainer.train(resume_from_checkpoint=modelManager.resume_from_checkpoint)
    else :
        train_results = trainer.train()

    print(f"Total training time: {time.time() - start_time} seconds")

    print("\ninfo : Saving model...\n")
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    
    print("\ninfo : Evaluating model on test set...\n")
    evaluate_and_save(args, trainer, datasetManager.prepared_ds["test"])

    # Generate f1 score per class based on target scale.
    if modelManager.training_type == ClassificationType.MULTILABEL and datasetManager.label_type == LabelType.BIN:
        
        # Generate threshold file.
        print("\ninfo : Create threshold file on val set...\n")
        thresholds = generate_threshold(trainer, datasetManager.prepared_ds["validation"], modelManager.output_dir, datasetManager.classes_names)
        
        print("\ninfo : Generate f1 score per class based on target scale...\n")
        generate_f1_per_class(modelManager, datasetManager, thresholds)

    # Save hyperparameters.
    save_hyperparameters_to_config(modelManager.output_dir, args)

    # Generate model card.
    files = ['train_results.json', 'test_results.json', 'trainer_state.json', 'all_results.json', 'config.json', 'transforms.json', 'test_f1_per_class.json']
    data_paths = [Path(modelManager.output_dir, file) for file in files]
    
    print("info : Generating model card...\n")
    generate_model_card(data_paths, modelManager, datasetManager)

    # Send data to hugging face if needed.
    if args.disable_web: return 
    modelManager.send_data_to_hugging_face()

if __name__ == "__main__":
    args = get_args()
    main(args)
