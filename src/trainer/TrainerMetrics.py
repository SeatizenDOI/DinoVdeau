import torch
import numpy as np
from transformers import EvalPrediction
from sklearn.metrics import f1_score, accuracy_score, root_mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

from ..data.DatasetManager import DatasetManager
from ..utils.enums import ClassificationType, LabelType
from ..model.HuggingModelManager import HuggingModelManager
from .tools import collate_fn_multilabel, collate_fn_monolabel, compute_metrics_monolabel


DEFAULT_THRESHOLD = 0.5

class TrainerMetrics:

    def __init__(self, modelManager: HuggingModelManager, datasetManager: DatasetManager, thresholds: dict | None = None) -> None:
        self.thresholds = thresholds
        self.modelManager = modelManager
        self.datasetManager = datasetManager

        self.is_f1 = isinstance(self.thresholds, dict)

    
    def get_collate_fn_from_training_type(self):
        if self.modelManager.training_type == ClassificationType.MULTILABEL:
            return collate_fn_multilabel

        return collate_fn_monolabel


    def get_compute_metrics_for_training_type(self):
        if self.modelManager.training_type == ClassificationType.MONOLABEL:
            return compute_metrics_monolabel
        
        if self.datasetManager.label_type == LabelType.BIN:
            return self.compute_metrics_multilabel_bin
        return self.compute_metrics_multilabel_probs


    def compute_metrics_multilabel_bin(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        return self.compute_metrics_with_thresholds(preds, p.label_ids)


    def compute_metrics_multilabel_probs(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = self.continuous_metrics(preds, p.label_ids)
        result_binary = self.compute_metrics_with_thresholds(preds, p.label_ids) if self.is_f1 else {}
        # Merging both dictionaries
        merged_result = {**result, **result_binary}
        # Convert all ndarray elements to lists
        for key, value in merged_result.items():
            if isinstance(value, np.ndarray):
                merged_result[key] = value.tolist()
        return merged_result


    def continuous_metrics(self, predictions, labels):
        sigmoid = torch.nn.Sigmoid()
        y_pred = sigmoid(torch.Tensor(predictions))
        y_true = labels

        metrics = {
            'rmse': root_mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred),
        }

        if self.is_f1:
            metrics['mse_per_class'] = root_mean_squared_error(y_true, y_pred, multioutput = "raw_values")

        return metrics
    

    def compute_metrics_with_thresholds(self, predictions, labels):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))

        # Initialize binary predictions array with zeros
        y_pred = np.zeros(probs.shape)
        y_true = np.zeros(probs.shape) if self.datasetManager.label_type == LabelType.PROBS else labels
        
        if isinstance(self.thresholds, dict):
            # next, use threshold to turn them into integer predictions
            for i, class_name in enumerate(self.thresholds.keys()):
                threshold = self.thresholds[class_name]
                y_pred[:, i] = np.where(probs[:, i] >= threshold, 1, 0)
                if self.datasetManager.label_type == LabelType.PROBS:
                   y_true[:, i] = np.where(labels[:, i] >= threshold, 1, 0) 
        else:
            y_pred[np.where(probs >= DEFAULT_THRESHOLD)] = 1


        metrics = {
            # Calculate metrics globally by counting the total true positives, false negatives and false positives.
            'f1_micro': f1_score(y_true=y_true, y_pred=y_pred, average='micro'),
            # Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
            'f1_macro': f1_score(y_true=y_true, y_pred=y_pred, average='macro'),            
            # In multilabel classification, the function returns the subset accuracy. 
            # If the entire set of predicted labels for a sample strictly match with the true set of labels, 
            # then the subset accuracy is 1.0; otherwise it is 0.0.
            'accuracy': accuracy_score(y_true, y_pred)
        }

        if self.is_f1:
            metrics['f1_per_class'] = f1_score(y_true=y_true, y_pred=y_pred, average=None)

        return metrics 
