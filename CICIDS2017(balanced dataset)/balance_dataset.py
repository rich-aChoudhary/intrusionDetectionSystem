import pandas as pd
from preprocess import load_and_preprocess

dataset = load_and_preprocess()

print("\nBalancing dataset...")

# separate classes
benign = dataset[dataset[' Label'] == 0]
attack = dataset[dataset[' Label'] == 1]

print("\nOriginal counts:")
print("Benign:", len(benign))
print("Attack:", len(attack))

# undersample benign to match attack
benign_sample = benign.sample(len(attack), random_state=42)

balanced_dataset = pd.concat([benign_sample, attack])

# shuffle dataset
balanced_dataset = balanced_dataset.sample(frac=1, random_state=42)

print("\nBalanced dataset counts:")
print(balanced_dataset[' Label'].value_counts())

print("\nBalanced dataset shape:", balanced_dataset.shape)

# save balanced dataset
balanced_dataset.to_csv("balanced_dataset.csv", index=False)

print("\nBalanced dataset saved as balanced_dataset.csv")