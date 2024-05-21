import torch
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, EvalPrediction
from typing import Callable, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import os

# add learning rate to the training output
class MyTrainer(Trainer):
    def log(self, logs: Dict[str, float]) -> None:
        # Ensure evaluations are logged only at the end of each epoch
        #if "epoch" in logs:
        logs["learning_rate"] = self._get_learning_rate()
        super().log(logs)


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor(np.array([np.array(x['label'], dtype=np.float32) for x in batch]))
    }

def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {'f1_micro': f1_micro_average, 'f1_macro': f1_macro_average, 'roc_auc': roc_auc, 'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(preds, p.label_ids)
    return result
    
def setup_trainer(args, model, ds, output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,  
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.initial_learning_rate,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, "logs"),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        remove_unused_columns=False,
        push_to_hub=True,
        report_to='tensorboard',
        fp16=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.factor_lr_scheduler, patience=args.patience_lr_scheduler)
    early_stop = EarlyStoppingCallback(early_stopping_patience = args.early_stopping_patience)

    trainer = MyTrainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    callbacks=[early_stop],
    compute_metrics=compute_metrics,
    optimizers=(optimizer, lr_scheduler)
    )

    return trainer
