import os
import pandas as pd

def load_datasets(df_folder):
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

    # N = 50
    # train_df = train_df.iloc[:N]
    # val_df = val_df.iloc[:N]
    # test_df = test_df.iloc[:N]
    
    return train_df, val_df, test_df
