import os
import json
import kornia as K
import numpy as np
from transformers import AutoImageProcessor
from datasets import Dataset, DatasetDict, Image as DatasetsImage

import torch
import torch.nn as nn

from PIL import Image as PILImage
from PIL import ImageFile as PILImageFile
PILImageFile.LOAD_TRUNCATED_IMAGES = True

from .data_loading import load_datasets

class PreProcess(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, x: PILImage) -> torch.Tensor:
        x_tmp: np.ndarray = np.array(x)
        x_out: torch.Tensor = K.image_to_tensor(x_tmp, keepdim=True)
        return x_out.float() / 255.0


def load_and_preprocess_df(df, img_path, img_col_name="FileName"):
    df.loc[:, df.columns != img_col_name] = df.loc[:, df.columns != img_col_name].astype(float)
    img_names = df[img_col_name].values.tolist()
    img_paths = [os.path.join(img_path, name) for name in img_names]
    labels = df.drop(columns=[img_col_name]).values.tolist()
    return {"image": img_paths, "label": labels, "image_name": img_names}

def transform_to_dict(transforms):
    transform_list = []
    for transform in transforms:
        transform_details = {
            'operation': type(transform).__name__
        }
        if hasattr(transform, 'p'):
            transform_details['probability'] = transform.p
        if hasattr(transform, 'size'):
            transform_details['size'] = transform.size if isinstance(transform.size, (int, float, tuple, list)) else str(transform.size)

        transform_list.append(transform_details)
    return transform_list

def save_transforms_to_json(train_transforms, val_transforms, filename):

    transforms_dict = {
        'train_transforms': transform_to_dict(train_transforms),
        'val_transforms': transform_to_dict(val_transforms)
    }
    with open(filename, 'w') as f:
        json.dump(transforms_dict, f, indent=4)

def create_datasets(df_folder, args, img_path, output_dir):

    train_df, val_df, test_df = load_datasets(df_folder, args.test_data)

    train_data = load_and_preprocess_df(train_df, img_path)
    val_data = load_and_preprocess_df(val_df, img_path)
    test_data = load_and_preprocess_df(test_df, img_path)
    print(train_data)
    ds_train = Dataset.from_dict(train_data).cast_column("image", DatasetsImage())
    ds_val = Dataset.from_dict(val_data).cast_column("image", DatasetsImage())
    ds_test = Dataset.from_dict(test_data).cast_column("image", DatasetsImage())
    print(ds_train)

    ds = DatasetDict({
        "train": ds_train,
        "validation": ds_val,
        "test": ds_test
    })    

    dummy_feature_extractor = AutoImageProcessor.from_pretrained(
        args.model_name,
        size={"height": args.image_size, "width": args.image_size},
        do_center_crop=False, 
        do_resize=True, 
        do_rescale=True, 
        do_normalize=True
    )    

    train_transforms = nn.Sequential(
        PreProcess(),
        K.augmentation.Resize(size=(args.image_size, args.image_size)),
        K.augmentation.RandomHorizontalFlip(p=0.25),
        K.augmentation.RandomVerticalFlip(p=0.25),
        K.augmentation.ColorJiggle(p=0.25),
        K.augmentation.RandomPerspective(p=0.25),
        K.augmentation.Normalize(mean=dummy_feature_extractor.image_mean, std=dummy_feature_extractor.image_std),
    )

    val_transforms = nn.Sequential(
        PreProcess(),
        K.augmentation.Resize(size=(args.image_size, args.image_size)),
        K.augmentation.Normalize(mean=dummy_feature_extractor.image_mean, std=dummy_feature_extractor.image_std),
    )   

    tansforms_path = os.path.join(output_dir, 'transforms.json')
    save_transforms_to_json(train_transforms, val_transforms, tansforms_path)    
    
    def preprocess_train(example_batch):
        example_batch["pixel_values"] = [train_transforms(image).squeeze() for image in example_batch["image"]]
        return example_batch

    def preprocess_val(example_batch):
        example_batch["pixel_values"] = [val_transforms(image).squeeze() for image in example_batch["image"]]
        return example_batch

    prepared_ds = ds
    prepared_ds["train"] = ds["train"].with_transform(preprocess_train)
    prepared_ds["validation"] = ds["validation"].with_transform(preprocess_val)
    prepared_ds["test"] = ds["test"].with_transform(preprocess_val)

    return prepared_ds, dummy_feature_extractor, train_df
