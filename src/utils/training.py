import torch
import numpy as np
from typing import Dict
from pathlib import Path
from argparse import Namespace
from datasets import DatasetDict
from sklearn.metrics import f1_score, accuracy_score
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, EvalPrediction

from ..model.model_setup import ClassificationType, get_training_type_from_args


class MyTrainer(Trainer):
    def log(self, logs: Dict[str, float]) -> None:
        # Ensure evaluations are logged only at the end of each epoch
        logs["learning_rate"] = self._get_learning_rate()
        super().log(logs)


def collate_fn_monolabel(batch: list) -> dict:
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor(np.array([np.array(x['label']) for x in batch]))
    }


def collate_fn_multilabel(batch: list) -> dict:
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor(np.array([np.array(x['label'], dtype=np.float32) for x in batch]))
    }


def compute_metrics_monolabel(p: EvalPrediction) -> dict:
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    softmax = torch.nn.Softmax(dim=1)
    probs = softmax(torch.Tensor(preds))
    y_pred = torch.argmax(probs, dim=1).numpy()
    y_true = p.label_ids

    return {
        'f1_micro': f1_score(y_true=y_true, y_pred=y_pred, average='micro'), 
        'f1_macro': f1_score(y_true=y_true, y_pred=y_pred, average='macro'),
        'accuracy': accuracy_score(y_true, y_pred)
    }


def compute_metrics_multilabel(p: EvalPrediction) -> dict:
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    threshold = 0.5
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(preds))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = p.label_ids
    
    return {
        'f1_micro': f1_score(y_true=y_true, y_pred=y_pred, average='micro'), 
        'f1_macro': f1_score(y_true=y_true, y_pred=y_pred, average='macro'),
        'accuracy': accuracy_score(y_true, y_pred)
    }


def get_compute_metrics_for_training_type(training_type: ClassificationType):
    return compute_metrics_multilabel if training_type == ClassificationType.MULTILABEL else compute_metrics_monolabel


def get_collate_fn_from_training_type(training_type: ClassificationType):
    return collate_fn_multilabel if training_type == ClassificationType.MULTILABEL else collate_fn_monolabel


def setup_trainer(args: Namespace, model, ds: DatasetDict, dummy_feature_extractor, output_dir: Path) -> MyTrainer:
    
    training_type = get_training_type_from_args(args)

    training_args = TrainingArguments(
        output_dir = output_dir,  
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        num_train_epochs = args.epochs,
        learning_rate = args.initial_learning_rate,
        weight_decay = args.weight_decay,
        load_best_model_at_end = True,
        eval_strategy = "epoch",
        save_strategy = "epoch",
        logging_dir = Path(output_dir, "logs"),
        metric_for_best_model = "eval_loss",
        greater_is_better = False,
        save_total_limit = 1,
        remove_unused_columns = False,
        push_to_hub = not(args.disable_web),
        report_to = 'tensorboard',
        fp16 = True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.factor_lr_scheduler, patience=args.patience_lr_scheduler)
    early_stop = EarlyStoppingCallback(early_stopping_patience = args.early_stopping_patience)
    compute_metrics = get_compute_metrics_for_training_type(training_type)
    collate_fn = get_collate_fn_from_training_type(training_type)

    trainer = MyTrainer(
        model = model,
        args = training_args,
        data_collator = collate_fn,
        train_dataset = ds["train"],
        eval_dataset = ds["validation"],
        tokenizer = dummy_feature_extractor,
        callbacks = [early_stop],
        compute_metrics = compute_metrics,
        optimizers = (optimizer, lr_scheduler)
    )

    return trainer
