import enum
import torch.nn as nn
from transformers import AutoConfig, AutoModelForImageClassification, Dinov2ForImageClassification
import json

class ClassificationType(enum.Enum):
    MULTILABEL = "multi_label_classification"
    MONOLABEL = "single_label_classification"


def get_training_type_from_args(args) -> ClassificationType:
    """ Return ClassificationType object from args. """
    if args.training_type == "multilabel": return ClassificationType.MULTILABEL
    elif args.training_type == "monolabel": return ClassificationType.MONOLABEL
    else:
        raise NameError(f"Argument provide for training type not found: {args.training_type}.") 

def create_head(num_features, number_classes, dropout_prob=0.5, activation_func=nn.ReLU):
    features_lst = [num_features, num_features // 2, num_features // 4]
    layers = []
    for in_f, out_f in zip(features_lst[:-1], features_lst[1:]):
        layers.append(nn.Linear(in_f, out_f))
        layers.append(activation_func())
        layers.append(nn.BatchNorm1d(out_f))
        if dropout_prob != 0: layers.append(nn.Dropout(dropout_prob))
    layers.append(nn.Linear(features_lst[-1], number_classes))
    return nn.Sequential(*layers)

def setup_model(args, label_names, id2label, label2id, classification_type: ClassificationType):
       
    model_config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        problem_type=classification_type.value,
        image_size=args.image_size
    )

    if not args.disable_web:
        model = AutoModelForImageClassification.from_pretrained(args.model_name, config=model_config, ignore_mismatched_sizes=True)
        model.classifier = create_head(model_config.hidden_size * 2, model_config.num_labels)

    else:
        # Load the model from a local directory if web is disabled
        with open("config.json", 'r') as file:
            config_env = json.load(file)
        model = AutoModelForImageClassification.from_pretrained(config_env["LOCAL_MODEL_PATH"], config=model_config, ignore_mismatched_sizes=True)
        model.classifier = create_head(model_config.hidden_size * 2, model_config.num_labels)

    if not args.no_freeze:
        model_name = args.model_name.split("/")[1].split("-")[0] # Extract dinov2 from facebook/dinov2-large
        for name, param in model.named_parameters():
            if name.startswith(model_name):
                param.requires_grad = False

    return model
