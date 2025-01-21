import torch
from pathlib import Path
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from .tools import *
from .TrainerMetrics import TrainerMetrics
from ..data.DatasetManager import DatasetManager
from ..model.HuggingModelManager import HuggingModelManager

class MyTrainer(Trainer):
    def log(self, logs: dict[str, float]) -> None:
        # Ensure evaluations are logged only at the end of each epoch
        logs["learning_rate"] = self._get_learning_rate()
        super().log(logs)


def setup_trainer(modelManager: HuggingModelManager, datasetManager: DatasetManager) -> MyTrainer:

    training_args = TrainingArguments(
        output_dir = modelManager.output_dir,  
        per_device_train_batch_size = modelManager.args.batch_size,
        per_device_eval_batch_size = modelManager.args.batch_size,
        num_train_epochs = modelManager.args.epochs,
        learning_rate = modelManager.args.initial_learning_rate,
        weight_decay = modelManager.args.weight_decay,
        load_best_model_at_end = True,
        eval_strategy = "epoch",
        save_strategy = "epoch",
        logging_dir = Path(modelManager.output_dir, "logs"),
        metric_for_best_model = "eval_loss",
        greater_is_better = False,
        save_total_limit = 1,
        remove_unused_columns = False,
        push_to_hub = not(modelManager.args.disable_web),
        report_to = 'tensorboard',
        fp16 = True,
    )

    optimizer = torch.optim.Adam(
        modelManager.model.parameters(), 
        lr=modelManager.args.initial_learning_rate, 
        weight_decay=modelManager.args.weight_decay
    )
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=modelManager.args.factor_lr_scheduler,
        patience=modelManager.args.patience_lr_scheduler
    )

    early_stop = EarlyStoppingCallback(early_stopping_patience = modelManager.args.early_stopping_patience)
    trainingMetrics = TrainerMetrics(modelManager, datasetManager)

    trainer = MyTrainer(
        model = modelManager.model,
        args = training_args,
        data_collator = trainingMetrics.get_collate_fn_from_training_type(),
        train_dataset = datasetManager.prepared_ds["train"],
        eval_dataset = datasetManager.prepared_ds["validation"],
        processing_class=datasetManager.dummy_feature_extractor,
        callbacks = [early_stop],
        compute_metrics = trainingMetrics.get_compute_metrics_for_training_type(),
        optimizers = (optimizer, lr_scheduler)
    )

    return trainer
