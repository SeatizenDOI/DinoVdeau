import json
import torch
import datasets
import tokenizers
import transformers
import pandas as pd
from pathlib import Path

def format_training_results_to_markdown(trainer_state):
    training_logs = trainer_state.get("log_history", [])
    markdown_table = "Epoch | Validation Loss | Accuracy | F1 Macro | F1 Micro | Learning Rate\n"
    markdown_table += "--- | --- | --- | --- | --- | ---\n"
    
    seen_epochs = set()

    for log in training_logs:
        epoch = log.get("epoch", "N/A")
        epoch = int(epoch)  # Ensure epoch is displayed as an integer
        if epoch in seen_epochs:
            continue  # Skip this log if the epoch has already been added
        seen_epochs.add(epoch)
        training_loss = log.get("loss", "N/A")
        validation_loss = log.get("eval_loss", "N/A")
        validation_accuracy = log.get("eval_accuracy", "N/A")
        eval_f1_micro = log.get("eval_f1_micro", "N/A")
        eval_f1_macro = log.get("eval_f1_macro", "N/A")
        learning_rate = log.get("learning_rate", "N/A")
        markdown_table += f"{epoch} | {validation_loss} | {validation_accuracy} | {eval_f1_micro} | {eval_f1_macro} | {learning_rate}\n"
    
    return markdown_table

def extract_test_results(test_results):
    eval_loss = test_results.get('eval_loss', 0)
    f1_micro = test_results.get('eval_f1_micro', 0)
    f1_macro = test_results.get('eval_f1_macro', 0)
    roc_auc = test_results.get('eval_roc_auc', 0)
    accuracy = test_results.get('eval_accuracy', 0)
    return eval_loss, f1_micro, f1_macro, roc_auc, accuracy

def save_hyperparameters_to_config(output_dir: Path, args, emissions: float | None):

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

    if emissions != None:
        hyperparameters['emissions_data'] = {
            'emissions': emissions,
            'source': "Code Carbon",
            'training_type': "fine-tuning",
            'geographical_location': "Brest, France",
            'hardware_used': "NVIDIA Tesla V100 PCIe 32 Go"
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

def generate_model_card(data_paths: list[Path], counts_path: Path, output_dir: Path):

    data = {}
    for data_path in data_paths:
        with open(data_path, 'r') as file:
            data[data_path.stem] = json.load(file)
  
    model_name = data["config"].get('_name_or_path', 'Unknown model')  

    markdown_training_results = format_training_results_to_markdown(data["trainer_state"])
    eval_loss, f1_micro, f1_macro, roc_auc, accuracy = extract_test_results(data["test_results"])
    markdown_counts = format_counts_to_markdown(counts_path)
    transforms_markdown = format_transforms_to_markdown(data["transforms"])
    if data["config"].get('data_augmentation') == False :
        transforms_markdown = "No augmentation"
    hyperparameters_markdown = format_hyperparameters_to_markdown(data["config"])
    carbon_footprint_markdown = format_carbon_footprint_to_markdown(data["config"])
    framework_versions_markdown = format_framework_versions_to_markdown()  

    markdown_content = f"""
---
language:
- eng
license: wtfpl
tags:
- multilabel-image-classification
- multilabel
- generated_from_trainer
base_model: {model_name}
model-index:
- name: {output_dir.name}
  results: []
---

DinoVd'eau is a fine-tuned version of [{model_name}](https://huggingface.co/{model_name}). It achieves the following results on the test set:

- Loss: {eval_loss:.4f}
- F1 Micro: {f1_micro:.4f}
- F1 Macro: {f1_macro:.4f}
- Roc Auc: {roc_auc:.4f}
- Accuracy: {accuracy:.4f}

---

# Model description
DinoVd'eau is a model built on top of dinov2 model for underwater multilabel image classification.The classification head is a combination of linear, ReLU, batch normalization, and dropout layers.
\nThe source code for training the model can be found in this [Git repository](https://github.com/SeatizenDOI/DinoVdeau).

- **Developed by:** [lombardata](https://huggingface.co/lombardata), credits to [César Leblanc](https://huggingface.co/CesarLeblanc) and [Victor Illien](https://huggingface.co/groderg)

---

# Intended uses & limitations
You can use the raw model for classify diverse marine species, encompassing coral morphotypes classes taken from the Global Coral Reef Monitoring Network (GCRMN), habitats classes and seagrass species.

---

# Training and evaluation data
Details on the number of images for each class are given in the following table:
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

# CO2 Emissions
{carbon_footprint_markdown}

---

# Framework Versions
{framework_versions_markdown}
"""

    output_filename = "README.md"
    with open(Path(output_dir, output_filename), 'w') as file:
        file.write(markdown_content)

    print(f"Model card generated and saved to {output_filename} in the directory {output_dir}")

def format_counts_to_markdown(counts_path):
    counts_df = pd.read_csv(counts_path)
    counts_df.rename(columns={counts_df.columns[0]: "Class"}, inplace=True)
    markdown_table = counts_df.to_markdown(index=False)
    return markdown_table

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

def format_hyperparameters_to_markdown(config):
    markdown = "\n"
    markdown += "The following hyperparameters were used during training:\n\n"
    markdown += f"- **Number of Epochs**: {config.get('num_epochs', 'Not specified')}\n"
    markdown += f"- **Learning Rate**: {config.get('initial_learning_rate', 'Not specified')}\n"
    markdown += f"- **Train Batch Size**: {config.get('train_batch_size', 'Not specified')}\n"
    markdown += f"- **Eval Batch Size**: {config.get('eval_batch_size', 'Not specified')}\n"
    markdown += f"- **Optimizer**: {config.get('optimizer', {}).get('type', 'Not specified')}\n"
    markdown += f"- **LR Scheduler Type**: {config.get('lr_scheduler_type', {}).get('type', 'Not specified')} with a patience of {config.get('patience_lr_scheduler', 'Not specified')} epochs and a factor of {config.get('factor_lr_scheduler', 'Not specified')}\n"
    markdown += f"- **Freeze Encoder**: {'Yes' if config.get('freeze_encoder', True) else 'No'}\n"
    markdown += f"- **Data Augmentation**: {'Yes' if config.get('data_augmentation', True) else 'No'}\n"
    return markdown

def format_carbon_footprint_to_markdown(config):
    markdown = "\n"
    if 'emissions_data' in config:
        emissions_data = config['emissions_data']
        markdown += "The estimated CO2 emissions for training this model are documented below:\n\n"
        markdown += f"- **Emissions**: {emissions_data.get('emissions', 'Not specified')} grams of CO2\n"
        markdown += f"- **Source**: {emissions_data.get('source', 'Not specified')}\n"
        markdown += f"- **Training Type**: {emissions_data.get('training_type', 'Not specified')}\n"
        markdown += f"- **Geographical Location**: {emissions_data.get('geographical_location', 'Not specified')}\n"
        markdown += f"- **Hardware Used**: {emissions_data.get('hardware_used', 'Not specified')}\n"
    else:
        markdown += "No carbon footprint data available.\n"
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
