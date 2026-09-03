import pandas as pd
from pathlib import Path
import random

INPUT_CSV = "fundus-metadata.csv"
RANDOM_SEED = 42

# load csv
df = pd.read_csv(INPUT_CSV)

# filter and rename columns
df = df[["fundus", "types"]].rename(
    columns={
        "fundus": "image_name",
        "types": "target"
    }
)

# remove leading slash from image_name column
df["image_name"] = df["image_name"].str.lstrip("/")


# delete samples with -1 label
df = df[df["target"] != -1].reset_index(drop=True)

print(f"number of remaining samples: {len(df)}")


def split_train_valid_test(df, test_size, random_state, stratify=None):
    """Shuffle and split a DataFrame deterministically without sklearn."""
    if stratify is not None:
        groups = df.groupby(stratify.name)
        indices = []
        for _, group in groups:
            idx = group.index.to_list()
            random.Random(random_state).shuffle(idx)
            indices.extend(idx)
        shuffled = df.loc[indices].reset_index(drop=True)
    else:
        shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    n_test = round(len(shuffled) * test_size)
    test = shuffled.iloc[:n_test]
    train = shuffled.iloc[n_test:]
    return train, test


# 80 % train, 20 % rest
train_df, temp_df = split_train_valid_test(
    df,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=df["target"]
)

# split rest into 50 % valid, 50 % test -> 10 % valid, 10 % test
valid_df, test_df = split_train_valid_test(
    temp_df,
    test_size=0.5,
    random_state=RANDOM_SEED,
    stratify=temp_df["target"]
)

# safe
output_dir = Path(INPUT_CSV).parent

train_df.to_csv(output_dir / "fundus-train.csv", index=False)
valid_df.to_csv(output_dir / "fundus-valid.csv", index=False)
test_df.to_csv(output_dir / "fundus-test.csv", index=False)

# print statistics
print("dataset size:")
print(f"train: {len(train_df)} ({len(train_df)/len(df):.1%})")
print(f"valid: {len(valid_df)} ({len(valid_df)/len(df):.1%})")
print(f"test : {len(test_df)} ({len(test_df)/len(df):.1%})")

print("class distribution:")
print("train:")
print(train_df["target"].value_counts(normalize=True).sort_index())

print("valid:")
print(valid_df["target"].value_counts(normalize=True).sort_index())

print("test:")
print(test_df["target"].value_counts(normalize=True).sort_index())