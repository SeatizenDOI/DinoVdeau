import os
import numpy as np
import pandas as pd

def load_datasets(df_folder: str, test_data_flag: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ From a path folder, find and load train, test, val file into DataFrame. """
    train_df_path, val_df_path, test_df_path = None, None, None

    for filename in os.listdir(df_folder):
        filepath = os.path.join(df_folder, filename)
        if "train" in filename:
            train_df_path = filepath
        elif "val" in filename:
            val_df_path = filepath
        elif "test" in filename:
            test_df_path = filepath

    assert train_df_path is not None, "Training dataset not found."
    assert val_df_path is not None, "Validation dataset not found."
    assert test_df_path is not None, "Test dataset not found."

    train_df = pd.read_csv(train_df_path)
    val_df = pd.read_csv(val_df_path)
    test_df = pd.read_csv(test_df_path)

    if test_data_flag :
        print("info : Working on a small dataset for test purposes...\n")
        N = 50
        train_df = train_df.iloc[:N]
        val_df = val_df.iloc[:N]
        test_df = test_df.iloc[:N]
    
    return train_df, val_df, test_df

def generate_labels(train_df: pd.DataFrame) -> tuple[list, dict, dict]:
    """ Extract classes_names and map label - id """
    
    classes_names = train_df.columns[1:].tolist()
    classes_nb = list(np.arange(len(classes_names)))
    id2label = { int(classes_nb[i]): classes_names[i] for i in range(len(classes_nb)) }
    label2id = { v: k for k, v in id2label.items()}

    return classes_names, id2label, label2id