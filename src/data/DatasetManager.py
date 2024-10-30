import numpy as np
import kornia as K
import pandas as pd
import torch.nn as nn
from pathlib import Path
from argparse import Namespace
from transformers import AutoImageProcessor
from datasets import Dataset, DatasetDict, Image as DatasetsImage


from PIL import ImageFile as PILImageFile
PILImageFile.LOAD_TRUNCATED_IMAGES = True

from ..utils.enums import LabelType, get_training_type_from_args

from .data_preprocessing import save_transforms_to_json, PreProcess, load_and_preprocess_df


class DatasetManager:
    def __init__(self, args: Namespace, df_folder: str) -> None:
        
        self.args = args
        self.df_folder = Path(df_folder)
        self.train_df, self.val_df, self.test_df, self.counts_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.classes_names, self.id2label, self.label2id = [], {}, {}
        self.prepared_ds: DatasetDict | None = None
        self.dummy_feature_extractor: AutoImageProcessor | None = None

        self.load_datasets(args.test_data)

        self.label_type = self.is_bin_or_probs()


    def load_datasets(self, test_data_flag: bool) -> None:
        """ From a path folder, find and load train, test, val file into DataFrame. """

        for filepath in self.df_folder.iterdir():
            if "train" in filepath.name:
                train_df_path = filepath
            elif "val" in filepath.name:
                val_df_path = filepath
            elif "test" in filepath.name:
                test_df_path = filepath

        assert train_df_path is not None, "Training dataset not found."
        assert val_df_path is not None, "Validation dataset not found."
        assert test_df_path is not None, "Test dataset not found."

        self.train_df = pd.read_csv(train_df_path)
        self.val_df = pd.read_csv(val_df_path)
        self.test_df = pd.read_csv(test_df_path)

        if test_data_flag :
            print("info : Working on a small dataset for test purposes...\n")
            N = 10
            self.train_df = self.train_df.iloc[:N]
            self.val_df = self.val_df.iloc[:N]
            self.test_df = self.test_df.iloc[:N]
    

    def generate_labels(self) -> tuple[list, dict, dict]:
        """ Extract classes_names and map label - id """
        
        self.classes_names = self.train_df.columns[1:].tolist()
        classes_nb = list(np.arange(len(self.classes_names)))
        self.id2label = { int(classes_nb[i]): self.classes_names[i] for i in range(len(classes_nb)) }
        self.label2id = { v: k for k, v in self.id2label.items()}


    def generate_count_df(self) -> None:
        threshold = 0.5
        train_counts = self.train_df.drop(columns=["FileName"]).apply(lambda col: col.map(lambda x: 1 if x >= threshold else 0)).sum().reset_index()
        train_counts.columns = ["Class", "train"]
        test_counts = self.test_df.drop(columns=["FileName"]).apply(lambda col: col.map(lambda x: 1 if x >= threshold else 0)).sum().reset_index()
        test_counts.columns = ["Class", "test"]
        val_counts = self.val_df.drop(columns=["FileName"]).apply(lambda col: col.map(lambda x: 1 if x >= threshold else 0)).sum().reset_index()
        val_counts.columns = ["Class", "val"]

        self.counts_df = train_counts.merge(test_counts, on="Class").merge(val_counts, on="Class")
        self.counts_df["Total"] = self.counts_df["train"] + self.counts_df["test"] + self.counts_df["val"]


    def create_datasets(self, img_path: str, output_dir: Path):
        
        self.generate_labels()
        self.generate_count_df()

        classification_type = get_training_type_from_args(self.args)

        train_data = load_and_preprocess_df(self.train_df, img_path, classification_type)
        val_data = load_and_preprocess_df(self.val_df, img_path, classification_type)
        test_data = load_and_preprocess_df(self.test_df, img_path, classification_type)

        ds_train = Dataset.from_list(train_data).cast_column("image", DatasetsImage())
        ds_val = Dataset.from_list(val_data).cast_column("image", DatasetsImage())
        ds_test = Dataset.from_list(test_data).cast_column("image", DatasetsImage())

        ds = DatasetDict({
            "train": ds_train,
            "validation": ds_val,
            "test": ds_test
        })    

        self.dummy_feature_extractor = AutoImageProcessor.from_pretrained(
            self.args.model_name,
            size={"height": self.args.image_size, "width": self.args.image_size},
            do_center_crop=False, 
            do_resize=True, 
            do_rescale=True, 
            do_normalize=True,
        )    

        train_transforms = nn.Sequential(
            PreProcess(),
            K.augmentation.Resize(size=(self.args.image_size, self.args.image_size)),
            K.augmentation.RandomHorizontalFlip(p=0.25),
            K.augmentation.RandomVerticalFlip(p=0.25),
            K.augmentation.ColorJiggle(p=0.25),
            K.augmentation.RandomPerspective(p=0.25),
            K.augmentation.Normalize(mean=self.dummy_feature_extractor.image_mean, std=self.dummy_feature_extractor.image_std),
        )

        val_transforms = nn.Sequential(
            PreProcess(),
            K.augmentation.Resize(size=(self.args.image_size, self.args.image_size)),
            K.augmentation.Normalize(mean=self.dummy_feature_extractor.image_mean, std=self.dummy_feature_extractor.image_std),
        )   

        tansforms_path = Path(output_dir, 'transforms.json')
        save_transforms_to_json(train_transforms, val_transforms, tansforms_path)    
        
        def preprocess_train(example_batch):
            example_batch["pixel_values"] = [train_transforms(image).squeeze() for image in example_batch["image"]]
            return example_batch

        def preprocess_val(example_batch):
            example_batch["pixel_values"] = [val_transforms(image).squeeze() for image in example_batch["image"]]
            return example_batch

        self.prepared_ds = ds
        self.prepared_ds["train"] = ds["train"].with_transform(preprocess_train)
        self.prepared_ds["validation"] = ds["validation"].with_transform(preprocess_val)
        self.prepared_ds["test"] = ds["test"].with_transform(preprocess_val)
    
    
    def is_bin_or_probs(self) -> LabelType :
        
        # From the input csv file, we get the first column after FileName and check if any of this value is a probability.
        is_probabilistic = any(value not in [0, 1] for value in self.val_df[list(self.val_df)[1]].to_list())

        return LabelType.PROBS if is_probabilistic else LabelType.BIN