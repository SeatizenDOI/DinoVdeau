import torch.nn as nn
from transformers import Dinov2Config, Dinov2ForImageClassification

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

class NewheadDinov2ForImageClassification(Dinov2ForImageClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = create_head(config.hidden_size * 2, config.num_labels)

def setup_model(args, label_names, id2label, label2id):
    num_labels = len(label_names)
    model_config = Dinov2Config.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
        image_size=args.image_size
    )
    
    model = NewheadDinov2ForImageClassification(model_config)
    model = model.from_pretrained(args.model_name, config=model_config, ignore_mismatched_sizes=True)

    if args.freeze_flag:
        for name, param in model.named_parameters():
            if name.startswith("dinov2"):
                param.requires_grad = False

    return model
