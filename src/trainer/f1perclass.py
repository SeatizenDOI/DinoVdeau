import json
from pathlib import Path
from transformers import TrainingArguments, Trainer

from ..utils.enums import LabelType
from .TrainerMetrics import TrainerMetrics
from ..data.DatasetManager import DatasetManager
from ..model.HuggingModelManager import HuggingModelManager


def generate_f1_per_class(modelManager: HuggingModelManager, datasetManager: DatasetManager, threshold: dict) -> None:
    
    training_args = TrainingArguments(
        output_dir=modelManager.output_dir,
        remove_unused_columns=False
    )

    trainerMetrics = TrainerMetrics(modelManager, datasetManager, threshold)

    trainer = Trainer(
        model=modelManager.model,
        args=training_args,
        data_collator=trainerMetrics.get_collate_fn_from_training_type(),
        compute_metrics=trainerMetrics.get_compute_metrics_for_training_type()
    )

    predictions, labels, metrics = trainer.predict(datasetManager.prepared_ds["test"])

    test_f1_per_class_dict_path = Path(modelManager.output_dir, "test_f1_per_class.json")
    f1_scores_dict = dict(zip(datasetManager.classes_names, metrics["test_f1_per_class"]))
    with open(test_f1_per_class_dict_path, "w") as outfile: 
        json.dump(f1_scores_dict, outfile)
    
    if datasetManager.label_type == LabelType.PROBS:
        test_results_dict_path = Path(modelManager.output_dir, "test_results.json")
        with open(test_results_dict_path, "w") as outfile: 
            json.dump(dict(metrics), outfile)