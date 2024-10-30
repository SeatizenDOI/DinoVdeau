import json
import torch
import datasets
import tokenizers
import transformers
from pathlib import Path
from ..utils.enums import LabelType
from ..data.DatasetManager import DatasetManager
from ..model.HuggingModelManager import HuggingModelManager

def format_training_results_to_markdown(trainer_state: dict, label_type: LabelType) -> str:
    training_logs = trainer_state.get("log_history", [])

    markdown_table = "Epoch | Validation Loss | Accuracy | F1 Macro | F1 Micro | Learning Rate\n"
    if label_type == LabelType.PROBS:
        markdown_table = "Epoch | Validation Loss | MAE | RMSE | R2 | Learning Rate\n"

    markdown_table += "--- | --- | --- | --- | --- | ---\n"

    
    seen_epochs = set()

    for log in training_logs:
        epoch = log.get("epoch", "N/A")
        epoch = int(epoch)  # Ensure epoch is displayed as an integer
        if epoch in seen_epochs:
            continue  # Skip this log if the epoch has already been added
        seen_epochs.add(epoch)

        validation_loss = log.get("eval_loss", "N/A")
        validation_accuracy_or_mae = log.get("eval_accuracy", "N/A") if label_type == LabelType.BIN else log.get("eval_mae", "N/A")
        eval_f1_micro_or_rmse = log.get("eval_f1_micro", "N/A") if label_type == LabelType.BIN else log.get("eval_rmse", "N/A")
        eval_f1_macro_or_r2 = log.get("eval_f1_macro", "N/A") if label_type == LabelType.BIN else log.get("eval_r2", "N/A")
        learning_rate = log.get("learning_rate", "N/A")
        markdown_table += f"{epoch} | {validation_loss} | {validation_accuracy_or_mae} | {eval_f1_micro_or_rmse} | {eval_f1_macro_or_r2} | {learning_rate}\n"
    
    return markdown_table


def extract_test_results(test_results: dict, label_type: LabelType, test_f1_per_class: dict) -> str:
    
    markdown = f"\n- Loss: {test_results.get('eval_loss', test_results.get('test_loss', 0.0)):.4f}"
    markdown += f"\n- F1 Micro: {test_results.get('eval_f1_micro', test_results.get('test_f1_micro', 0.0)):.4f}"
    markdown += f"\n- F1 Macro: {test_results.get('eval_f1_macro', test_results.get('test_f1_macro', 0.0)):.4f}"
    markdown += f"\n- Accuracy: {test_results.get('eval_accuracy', test_results.get('test_accuracy', 0.0)):.4f}"

    if label_type == LabelType.PROBS:

        markdown += f"\n- RMSE: {test_results.get('eval_rmse', test_results.get('test_rmse', 0.0)):.4f}"
        markdown += f"\n- MAE: {test_results.get('eval_mae', test_results.get('test_mae', 0.0)):.4f}"
        markdown += f"\n- R2: {test_results.get('eval_r2', test_results.get('test_r2', 0.0)):.4f}"


    if len(test_f1_per_class) > 0:
        
        markdown += "\n\n| Class | F1 per class |\n|----------|-------|\n"
    
        # Populate rows
        for key, value in test_f1_per_class.items():
            markdown += f"| {key} | {value:.4f} |\n"
    
    return markdown


def save_hyperparameters_to_config(output_dir: Path, args) -> None:

    # Regroup and save hyperparameters
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
        'freeze_encoder': not(args.no_freeze),
        'data_augmentation':not(args.no_data_aug),
        'num_epochs': args.epochs
    }
    
    # Load hyperparameters.
    config_path, config = Path(output_dir, 'config.json'), {}
    if Path.exists(config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)

    # Save hyperparameters.
    config.update(hyperparameters)
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)

    print("Updated configuration saved to config.json")


def generate_model_card(data_paths: list[Path], modelManager: HuggingModelManager, datasetManager: DatasetManager) -> None:

    data = {}
    for data_path in data_paths:
        if not data_path.exists(): 
            data[data_path.stem] = {}
            continue
        with open(data_path, 'r') as file:
            data[data_path.stem] = json.load(file)
  
    markdown_training_results = format_training_results_to_markdown(data["trainer_state"], datasetManager.label_type)
    test_results_metrics_markdown = extract_test_results(data["test_results"], datasetManager.label_type, data["test_f1_per_class"])
    markdown_counts = datasetManager.counts_df.to_markdown(index=False)
    transforms_markdown = format_transforms_to_markdown(data["transforms"])
    if data["config"].get('data_augmentation') == False :
        transforms_markdown = "No augmentation"
    hyperparameters_markdown = format_hyperparameters_to_markdown(data["config"], data["all_results"])
    framework_versions_markdown = format_framework_versions_to_markdown()  

    markdown_content = f"""
---
language:
- eng
license: CC0-1.0
tags:
- {modelManager.args.training_type}-image-classification
- {modelManager.args.training_type}
- generated_from_trainer
base_model: {modelManager.model_name}
model-index:
- name: {modelManager.output_dir.name}
  results: []
---

{modelManager.args.new_model_name} is a fine-tuned version of [{modelManager.model_name}](https://huggingface.co/{modelManager.model_name}). It achieves the following results on the test set:

{test_results_metrics_markdown}

---

# Model description
{modelManager.args.new_model_name} is a model built on top of {modelManager.model_name} model for underwater multilabel image classification.The classification head is a combination of linear, ReLU, batch normalization, and dropout layers.
\nThe source code for training the model can be found in this [Git repository](https://github.com/SeatizenDOI/DinoVdeau).

- **Developed by:** [lombardata](https://huggingface.co/lombardata), credits to [César Leblanc](https://huggingface.co/CesarLeblanc) and [Victor Illien](https://huggingface.co/groderg)

---

# Intended uses & limitations
You can use the raw model for classify diverse marine species, encompassing coral morphotypes classes taken from the Global Coral Reef Monitoring Network (GCRMN), habitats classes and seagrass species.

---

# Training and evaluation data
Details on the {'' if datasetManager.label_type == LabelType.BIN else 'estimated'} number of images for each class are given in the following table:
{markdown_counts}

---

# Training procedure

## Training hyperparameters
{hyperparameters_markdown}

## Data Augmentation
Data were augmented using the following transformations :
{transforms_markdown}

## Training results
{markdown_training_results}

---

# Framework Versions
{framework_versions_markdown}
"""

    output_filename = "README.md"
    with open(Path(modelManager.output_dir, output_filename), 'w') as file:
        file.write(markdown_content)

    print(f"Model card generated and saved to {output_filename} in the directory {modelManager.output_dir}")


def format_transforms_to_markdown(transforms_dict):
    transforms_markdown = "\n"
    for key, value in transforms_dict.items():
        transforms_markdown += f"{key.replace('_', ' ').title()}\n"
        for item in value:
            probability = item.get('probability', 'No additional parameters')
            if isinstance(probability, float):
                probability = f"probability={probability:.2f}"
            transforms_markdown += f"- **{item['operation']}**: {probability}\n"
        transforms_markdown += "\n"
    return transforms_markdown


def format_hyperparameters_to_markdown(config, all_results):
    epoch = all_results.get("epoch", None)
    if epoch == None:
        epoch = config.get('num_epochs', 'Not specified')

    markdown = "\n"
    markdown += "The following hyperparameters were used during training:\n\n"
    markdown += f"- **Number of Epochs**: {epoch}\n"
    markdown += f"- **Learning Rate**: {config.get('initial_learning_rate', 'Not specified')}\n"
    markdown += f"- **Train Batch Size**: {config.get('train_batch_size', 'Not specified')}\n"
    markdown += f"- **Eval Batch Size**: {config.get('eval_batch_size', 'Not specified')}\n"
    markdown += f"- **Optimizer**: {config.get('optimizer', {}).get('type', 'Not specified')}\n"
    markdown += f"- **LR Scheduler Type**: {config.get('lr_scheduler_type', {}).get('type', 'Not specified')} with a patience of {config.get('patience_lr_scheduler', 'Not specified')} epochs and a factor of {config.get('factor_lr_scheduler', 'Not specified')}\n"
    markdown += f"- **Freeze Encoder**: {'Yes' if config.get('freeze_encoder', True) else 'No'}\n"
    markdown += f"- **Data Augmentation**: {'Yes' if config.get('data_augmentation', True) else 'No'}\n"
    return markdown


def format_framework_versions_to_markdown():
    transformers_version = transformers.__version__
    pytorch_version = torch.__version__
    datasets_version = datasets.__version__
    tokenizers_version = tokenizers.__version__

    markdown = "\n"
    markdown += f"- **Transformers**: {transformers_version}\n"
    markdown += f"- **Pytorch**: {pytorch_version}\n"
    markdown += f"- **Datasets**: {datasets_version}\n"
    markdown += f"- **Tokenizers**: {tokenizers_version}\n"
    return markdown
