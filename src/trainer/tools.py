import torch
import numpy as np
from transformers import EvalPrediction
from sklearn.metrics import f1_score, accuracy_score

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