import pandas as pd
from pathlib import Path



def main():

    template_folder = Path("/home/bioeos/Documents/Bioeos/dataset/2024/annotations")
    output_folder = Path("/home/bioeos/Documents/Bioeos/dataset/2024/annotations_50")
    for file in template_folder.iterdir():
        ext = file.name.split("_")[0]
        if not ext in ["train", "test", "val"]: continue

        df = pd.read_csv(file)
        df = df[:50]
        df[list(df)[2:]] = 0
        df[list(df)[1]] = 1

        df.to_csv(Path(output_folder, f"{ext}.csv"), index=False)



if __name__ == "__main__":
    main()