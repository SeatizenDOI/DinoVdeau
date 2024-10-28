import enum
import json
import torch.nn as nn
from pathlib import Path
from argparse import Namespace
from transformers import AutoConfig, AutoModelForImageClassification

class ClassificationType(enum.Enum):
    MULTILABEL = "multi_label_classification"
    MONOLABEL = "single_label_classification"


def get_training_type_from_args(args: Namespace) -> ClassificationType:
    """ Return ClassificationType object from args. """
    if args.training_type == "multilabel": return ClassificationType.MULTILABEL
    elif args.training_type == "monolabel": return ClassificationType.MONOLABEL
    else:
        raise NameError(f"Argument provide for training type not found: {args.training_type}.") 


def create_head(num_features: int, number_classes: int, dropout_prob: float = 0.5, activation_func=nn.ReLU) -> nn.Sequential:
    features_lst = [num_features, num_features // 2, num_features // 4]
    layers = []
    for in_f, out_f in zip(features_lst[:-1], features_lst[1:]):
        layers.append(nn.Linear(in_f, out_f))
        layers.append(activation_func())
        layers.append(nn.BatchNorm1d(out_f))
        if dropout_prob != 0: layers.append(nn.Dropout(dropout_prob))
    layers.append(nn.Linear(features_lst[-1], number_classes))
    return nn.Sequential(*layers)


def setup_model(args: Namespace, label_names: list, id2label: dict, label2id: dict):

    training_type = get_training_type_from_args(args)

    model_config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        problem_type=training_type.value,
        image_size=args.image_size
    )
    # Get hidden_size number from model
    hidden_size = 1024 # Default value if not found.
    if hasattr(model_config, "hidden_size"):
        hidden_size = getattr(model_config, "hidden_size")
    if hasattr(model_config, "hidden_sizes"):
        hidden_size = getattr(model_config, "hidden_sizes")[-1]

    if not args.disable_web:
        model = AutoModelForImageClassification.from_pretrained(args.model_name, config=model_config, ignore_mismatched_sizes=True)
    else:
        # Load the model from a local directory if web is disabled
        config_path = Path(args.config_path)
        if not config_path.exists() or not config_path.is_file():
            raise NameError(f"Config file not found for path {config_path}")
            
        with open(config_path, 'r') as file:
            config_env: dict[str, str] = json.load(file)

        model_name = config_env["LOCAL_MODEL_PATH"] if config_env["LOCAL_MODEL_PATH"] != '' else args.model_name

        model = AutoModelForImageClassification.from_pretrained(model_name, config=model_config, ignore_mismatched_sizes=True)

    if not(args.no_custom_head):
        model.classifier = create_head(hidden_size * 2, model_config.num_labels)

    if not args.no_freeze:
        model_name = args.model_name.split("/")[1].split("-")[0] # Extract dinov2 from facebook/dinov2-large
        for name, param in model.named_parameters():
            if name.startswith(model_name):
                param.requires_grad = False

    return model
