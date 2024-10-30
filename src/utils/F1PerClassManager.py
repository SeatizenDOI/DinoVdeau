import json
import torch
import numpy as np
from enum import Enum
from typing import Dict
from pathlib import Path
from datasets import Dataset
from argparse import Namespace
from abc import ABC, abstractmethod
from transformers import EvalPrediction, TrainingArguments, Trainer
from sklearn.metrics import f1_score, accuracy_score, root_mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

# Enum to map target_scale.
class TargetScale(Enum):
    FINE_SCALE = "fine_scale"
    MEDIUM_SCALE = "medium_scale"
    LARGE_SCALE = "large_scale"


def parse_target_scale_from_input(args: Namespace) -> TargetScale:
    """ Parse target scale from input. """
    if args.target_scale == "fine_scale":
        return TargetScale.FINE_SCALE
    elif args.target_scale == "medium_scale":
        return TargetScale.MEDIUM_SCALE
    elif args.target_scale == "large_scale":
        return TargetScale.LARGE_SCALE
    else:
        raise NameError("Target scale not found.")


class F1PerClassManager:
    """ Manager to choose the correct F1PerClass based on target_scale. """

    def __init__(self, target_scale: TargetScale, thresholds: dict, classes_names: list, id2label: dict) -> None:
        self.f1_score_manager: F1PerClassBase | None = None

        if target_scale == TargetScale.FINE_SCALE:
            self.f1_score_manager = F1PerClassFineScale(thresholds, classes_names, id2label)
        elif target_scale == TargetScale.MEDIUM_SCALE:
            self.f1_score_manager = F1PerClassMediumScale(thresholds, classes_names, id2label)
        # TODO implement LargeScale
    

    def generate(self, ds: Dataset, output_dir: Path, model) -> None:
        """ Generate f1 score per class in one file. """
        if self.f1_score_manager == None:
            raise NameError("Cannot create F1 score manager.")

        training_args = TrainingArguments(
            output_dir=output_dir,
            remove_unused_columns=False
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=self.f1_score_manager.collate_fn,
            compute_metrics=self.f1_score_manager.compute_metrics
        )

        predictions, labels, sklearn_metrics = trainer.predict(ds)

        self.f1_score_manager.create_f1_score_per_class(sklearn_metrics, output_dir)


class F1PerClassBase(ABC):
    """ Base class to ensure all methods are instantiate. """

    @abstractmethod
    def compute_metrics(self, p: EvalPrediction):
        pass

    @abstractmethod
    def create_f1_score_per_class(self, metrics: Dict[str, float], output_dir: Path):
        pass

    def collate_fn(self, batch):
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            # convert array of np.arrays to unique np array and then convert it to tensor, in order to solve the following warning
            # "Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single 
            # numpy.ndarray with numpy.array() before converting to a tensor."
            'labels': torch.tensor(np.array([np.array(x['label'], dtype=np.float32) for x in batch]))
        }


class F1PerClassFineScale(F1PerClassBase):
    """ F1PerClass for fine scale target. """

    def __init__(self, thresholds: dict, classes_names: list, id2label: dict) -> None:
        self.thresholds = thresholds
        self.classes_names = classes_names
        self.id2label = id2label
        super().__init__()


    def multi_label_metrics(self, predictions, labels):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # Initialize binary predictions array with zeros
        y_pred = np.zeros(probs.shape)
        
        # next, use threshold to turn them into integer predictions
        for i, class_name in enumerate(self.thresholds.keys()):
            threshold = self.thresholds[class_name]
            y_pred[:, i] = np.where(probs[:, i] >= threshold, 1, 0)
        # finally, compute metrics
        y_true = labels

        return {
            # Calculate metrics globally by counting the total true positives, false negatives and false positives.
            'f1_micro': f1_score(y_true=y_true, y_pred=y_pred, average='micro'),
            # Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
            'f1_macro': f1_score(y_true=y_true, y_pred=y_pred, average='macro'),
            #  If None, the scores for each class are returned.
            'f1_per_class': f1_score(y_true=y_true, y_pred=y_pred, average=None),
            # In multilabel classification, the function returns the subset accuracy. 
            # If the entire set of predicted labels for a sample strictly match with the true set of labels, 
            # then the subset accuracy is 1.0; otherwise it is 0.0.
            'accuracy': accuracy_score(y_true, y_pred)
        }


    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = self.multi_label_metrics(preds, p.label_ids)
        return result


    def create_f1_score_per_class(self, metrics: Dict[str, float], output_dir: Path):
        test_f1_per_class_dict_path = Path(output_dir, "test_f1_per_class.json")
        f1_scores_dict = dict(zip(self.classes_names, metrics["test_f1_per_class"]))

        with open(test_f1_per_class_dict_path, "w") as outfile: 
            json.dump(f1_scores_dict, outfile)
        
    
class F1PerClassMediumScale(F1PerClassBase):
    """ F1PerClass for medium scale target. """

    def __init__(self, thresholds: dict, classes_names: list, id2label: dict) -> None:
        self.thresholds = thresholds
        self.classes_names = classes_names
        self.id2label = id2label
        super().__init__()
    

    def continuous_metrics(self, predictions, labels):
        sigmoid = torch.nn.Sigmoid()
        y_pred = sigmoid(torch.Tensor(predictions))
        y_true = labels

        mse = root_mean_squared_error(y_true, y_pred)
        return {
            'mse': mse,
            'mse_per_class': root_mean_squared_error(y_true, y_pred, multioutput = "raw_values"),
            'rmse': np.sqrt(mse),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred),
        }
    

    def compute_metrics_with_thresholds(self, predictions, probability_annotations):

        # Transform probabilities into binary values based on thresholds
        binary_annotations, binary_predictions = [], []

        for i in range(len(probability_annotations)):
            binary_annotation, binary_prediction = [], []

            for j in range(len(probability_annotations[i])):
                binary_annotation.append(1 if probability_annotations[i][j] > self.thresholds[self.id2label[j]] else 0)
                binary_prediction.append(1 if predictions[i][j] > self.thresholds[self.id2label[j]] else 0)

            binary_annotations.append(binary_annotation)
            binary_predictions.append(binary_prediction)

        # Return results
        return {
            # Calculate metrics globally by counting the total true positives, false negatives and false positives.
            'f1_micro': f1_score(y_true=binary_annotations, y_pred=binary_predictions, average='micro'),
            # Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
            'f1_macro': f1_score(y_true=binary_annotations, y_pred=binary_predictions, average='macro'),
            #  If None, the scores for each class are returned.
            'f1_per_class': f1_score(y_true=binary_annotations, y_pred=binary_predictions, average=None),
            # In multilabel classification, the function returns the subset accuracy. 
            # If the entire set of predicted labels for a sample strictly match with the true set of labels, 
            # then the subset accuracy is 1.0; otherwise it is 0.0.
            'accuracy': accuracy_score(binary_annotations, binary_predictions)
        }
    
    
    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = self.continuous_metrics(preds, p.label_ids)
        result_binary = self.compute_metrics_with_thresholds(preds, p.label_ids)
        # Merging both dictionaries
        merged_result = {**result, **result_binary}
        # Convert all ndarray elements to lists
        for key, value in merged_result.items():
            if isinstance(value, np.ndarray):
                merged_result[key] = value.tolist()
        return merged_result
    

    def create_f1_score_per_class(self, metrics: Dict[str, float], output_dir: Path):
        test_f1_per_class_dict_path = Path(output_dir, "test_f1_per_class.json")
        f1_scores_dict = dict(zip(self.classes_names, metrics["test_f1_per_class"]))
        test_results_dict_path = Path(output_dir, "test_results.json")
        print(metrics)
        with open(test_f1_per_class_dict_path, "w") as outfile: 
            json.dump(f1_scores_dict, outfile)
        with open(test_results_dict_path, "w") as outfile: 
            json.dump(dict(metrics), outfile)