import pandas as pd
from pathlib import Path
import random


def collect_dataset_files(save: bool = False, path_to_save: Path | None = None):
    path_to_dataset = Path("data/raw/Sound3")
    data = pd.read_excel("./data/dataset.ods", engine="odf").dropna()

    rows = []
    for row in data.itertuples(index=False, name=None):
        directory = row[0]
        label = row[1]
        folder_path = path_to_dataset / directory
        for file_path in folder_path.iterdir():
            if file_path.is_file():
                rows.append([str(file_path.resolve()), bool(label)])

    df = pd.DataFrame(rows, columns=["Path", "Label"])

    if save:
        save_path = path_to_save if path_to_save else Path(".")
        df.to_parquet(save_path / "dataset.parquet", index=False)
    return df


def create_splits(
    train_percent=0.8,
    val_percent=0.1,
    test_percent=0.1,
    seed=90,
    fixed_test_speakers: list[str]
    | None = None,  # список фиксированных дикторов для теста
):
    df = pd.read_parquet("data/processed/dataset.parquet")

    def get_speaker(path: str):
        parts = path.split("/")
        for part in parts:
            if part.isdigit():
                return part
        return None

    df["Speaker"] = df["Path"].apply(get_speaker)
    speakers = df["Speaker"].dropna().unique().tolist()

    random.seed(seed)
    random.shuffle(speakers)
    
    if fixed_test_speakers:
        remaining_speakers = [s for s in speakers if s not in fixed_test_speakers]
        test_speakers = fixed_test_speakers
    else:
        n_test = int(len(speakers) * test_percent)
        test_speakers = speakers[:n_test]
        remaining_speakers = speakers[n_test:]

    n_train = int(
        len(remaining_speakers) * train_percent / (train_percent + val_percent)
    )
    train_speakers = remaining_speakers[:n_train]
    val_speakers = remaining_speakers[n_train:]

    assert set(train_speakers).isdisjoint(val_speakers)
    assert set(val_speakers).isdisjoint(test_speakers)
    assert set(train_speakers).isdisjoint(test_speakers)

    train_df = df[df["Speaker"].isin(train_speakers)]
    val_df = df[df["Speaker"].isin(val_speakers)]
    test_df = df[df["Speaker"].isin(test_speakers)]

    train_df.to_parquet("data/processed/train_dataset.parquet", index=False)
    val_df.to_parquet("data/processed/val_dataset.parquet", index=False)
    test_df.to_parquet("data/processed/test_dataset.parquet", index=False)

    print(
        f"✅ Train: {len(train_df)} файлов, дикторов: {len(train_speakers)}, дикторы: {train_speakers}"
    )
    print(
        f"✅ Val:   {len(val_df)} файлов, дикторов: {len(val_speakers)}, дикторы: {val_speakers}"
    )
    print(
        f"✅ Test:  {len(test_df)} файлов, дикторов: {len(test_speakers)}, дикторы: {test_speakers}"
    )


if __name__ == "__main__":
    collect_dataset_files(True, Path("/workspaces/super-duper-dollop/data/processed"))
    create_splits(fixed_test_speakers=["29", "62"])
