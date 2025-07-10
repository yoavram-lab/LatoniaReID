import pandas as pd
import numpy as np

# Load data
fname = "bina_photos.csv"
print(f"Loading data from {fname}")
df = pd.read_csv(fname)
print(f"Total number of images: {df.shape[0]}")
# Find label counts
label_counts = df['label'].value_counts()
n_labels = len(label_counts)
print(f"Number of labels: {n_labels}")

# Labels with >3 images
many_imgs_labels = label_counts[label_counts > 3].index.tolist()
# Labels with ≤3 images
few_imgs_labels = label_counts[label_counts <= 3].index.tolist()

# Split many_imgs_labels into 75% train labels and 25% "split" labels
many_imgs_labels = np.array(many_imgs_labels)
np.random.seed(42)
np.random.shuffle(many_imgs_labels)
split_point = int(np.ceil(len(many_imgs_labels) * 0.75))
many_train_labels = many_imgs_labels[:split_point]
many_split_labels = many_imgs_labels[split_point:]

# All images for 75% of labels go to train
many_imgs_train = df[df['label'].isin(many_train_labels)]
# For the remaining 25% of labels, split images half/half within each label
split_imgs = df[df['label'].isin(many_split_labels)]
split_imgs_train = []
split_imgs_val = []
for label, group in split_imgs.groupby('label'):
    idx = group.sample(frac=1, random_state=42).index  # shuffle
    split = int(np.ceil(len(group) / 2))
    split_imgs_train.append(group.loc[idx[:split]])
    split_imgs_val.append(group.loc[idx[split:]])

split_imgs_train = pd.concat(split_imgs_train) if split_imgs_train else pd.DataFrame(columns=df.columns)
split_imgs_val = pd.concat(split_imgs_val) if split_imgs_val else pd.DataFrame(columns=df.columns)

# For labels with ≤3 images, split at label level: 75% train, 25% val
few_imgs_labels = np.array(few_imgs_labels)
np.random.seed(42)
np.random.shuffle(few_imgs_labels)
split = int(np.ceil(len(few_imgs_labels) * 0.75))
few_labels_train = few_imgs_labels[:split]
few_labels_val = few_imgs_labels[split:]

few_imgs_train = df[df['label'].isin(few_labels_train)]
few_imgs_val = df[df['label'].isin(few_labels_val)]

# Concatenate final splits
train_df = pd.concat([many_imgs_train, split_imgs_train, few_imgs_train]).sample(frac=1, random_state=42).reset_index(drop=True)
val_df = pd.concat([split_imgs_val, few_imgs_val]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Number of images in train set: {train_df.shape[0]}, number of labels: {len(train_df['label'].unique())}")
print(f"Number of images in val set: {val_df.shape[0]}, number of labels: {len(val_df['label'].unique())}")

# Save to CSV
train_df.to_csv(f"{fname.split('.')[0]}_train.csv", index=False)
val_df.to_csv(f"{fname.split('.')[0]}_validation.csv", index=False)