import json
import numpy as np
from pathlib import Path
from datetime import date
from datasets import Dataset
from argparse import Namespace
from sklearn.metrics import f1_score

from ..utils.training import MyTrainer

def evaluate_and_save(args: Namespace, trainer: MyTrainer, ds_test: Dataset) -> None:
    metrics = trainer.evaluate(ds_test)

    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)

    if args.disable_web: return

    today = date.today().strftime("%Y_%m_%d")
    try:
        trainer.push_to_hub(commit_message=f"Evaluation on the test set completed on {today}.")
    except Exception as e:
            print(f"Error while pushing to Hub: {e}")


def sigmoid(_outputs):
    return 1.0 / (1.0 + np.exp(-_outputs))


def logits_to_probs(predictions: np.ndarray) -> list:
    return [sigmoid(pred) for pred in predictions]


def calculate_scores(y_true, y_probs):

    best_precision, best_threshold = 0.0, 0.0

    for threshold in np.arange(0, 1, 0.001):
        y_pred = [1 if prob > threshold else 0 for prob in y_probs]
        precision = f1_score(y_true = y_true, y_pred = y_pred, zero_division=0.0)
       
        if precision > best_precision:
            best_precision, best_threshold = precision, threshold

    return best_precision, best_threshold


def generate_threshold(trainer: MyTrainer, ds_val: Dataset, output_dir: Path, classes_names: list) -> dict:

    predictions, labels, metrics = trainer.predict(ds_val)

    probabilities = logits_to_probs(predictions)

    # compute thresholds for each class on the base of the f1 score
    vec_best_threshold = []
    for i in range(0, len(classes_names)) :
        probs_current_class = [prob[i] for prob in probabilities]   
        label_current_class = [label[i] for label in labels]

        best_precision, best_threshold = calculate_scores(label_current_class, probs_current_class)
        vec_best_threshold.append(np.round(best_threshold, 3))

    threshold_dict_path = Path(output_dir, "threshold.json")
    threshold_dict = dict(zip(classes_names, vec_best_threshold))

    with open(threshold_dict_path, "w") as outfile: 
        json.dump(threshold_dict, outfile)
    
    return threshold_dict