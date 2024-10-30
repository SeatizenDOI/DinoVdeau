import os
import json
import kornia as K
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn

from PIL import Image as PILImage
from PIL import ImageFile as PILImageFile
PILImageFile.LOAD_TRUNCATED_IMAGES = True

from ..utils.enums import ClassificationType

class PreProcess(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, x: PILImage) -> torch.Tensor:
        x_tmp: np.ndarray = np.array(x)
        x_out: torch.Tensor = K.image_to_tensor(x_tmp, keepdim=True)
        return x_out.float() / 255.0


def load_and_preprocess_df(df: pd.DataFrame, img_path: str, training_type: ClassificationType, img_col_name: str="FileName") -> list[dict]:
    df.loc[:, df.columns != img_col_name] = df.loc[:, df.columns != img_col_name].astype(float)
    img_names = df[img_col_name].values.tolist()
    img_paths = [os.path.join(img_path, name) for name in img_names]
    labels = df.drop(columns=[img_col_name]).values.tolist()
    
    if training_type == ClassificationType.MONOLABEL:
        labels = [label.index(max(label)) for label in labels]
    
    return [{"image": img_path, "label": label} for img_path, label in zip(img_paths, labels)]


def transform_to_dict(transforms: nn.Sequential) -> list:
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



def save_transforms_to_json(train_transforms: nn.Sequential, val_transforms: nn.Sequential, filename: Path) -> None:

    transforms_dict = {
        'train_transforms': transform_to_dict(train_transforms),
        'val_transforms': transform_to_dict(val_transforms)
    }
    with open(filename, 'w') as f:
        json.dump(transforms_dict, f, indent=4)