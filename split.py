import pandas as pd
import numpy as np
from tabulate import tabulate
import click

@click.command()
@click.argument('fname', type=click.Path(exists=True, file_okay=True))
@click.option('--p', type=float, default=0.2, help='Proportion of labels to sample for validation set.')
@click.option('--seed', type=int, default=1337, help='Random seed for reproducibility.')
@click.option('--no-split', is_flag=True, help='If set, do not perform the split and just print statistics.')
def main(fname, p, seed, no_split):
    print(f"Loading data from {fname}")
    df = pd.read_csv(fname)
    df['date'] = [p.split('/')[1] for p in df['rel_path']]
    # how many images?
    print(f"Total number of images: {df.shape[0]}")

    # how many unique labels?
    label_counts = df['label'].value_counts()
    n_labels = len(label_counts)
    print(f"Number of labels: {n_labels}")
   
    # how many images per label?
    hist, bins = np.histogram(label_counts, bins=np.arange(1, label_counts.max() + 2))
    median = np.median(label_counts)
    print(f"Histogram of label counts (median {median})")
    print(tabulate([['# images']+bins.tolist(), ['# labels'] + hist.tolist()], tablefmt='rounded_outline'))

    # how many unique dates?
    date_counts = df['date'].value_counts()
    n_dates = len(date_counts)
    print(f"Number of dates: {n_dates}")
    print(f"Dates: {date_counts.index.min()} to {date_counts.index.max()}")

    # how many dates per label?
    label_dates = df.groupby('label')['date'].nunique()
    hist, bins = np.histogram(label_dates, bins=np.arange(1, label_dates.max() + 2))
    median = np.median(label_dates)
    print(f"Histogram of date-label counts (median {median})")
    print(tabulate([['# dates']+bins.tolist(), ['# labels']+hist.tolist()], tablefmt='rounded_outline'))

    # how many images per date-label combination?
    date_label_counts = df.groupby(['date', 'label']).size()
    print(f"Number of date-label combinations: {date_label_counts.shape[0]}")
    hist, bins = np.histogram(date_label_counts, bins=np.arange(1, date_label_counts.max() + 2))
    median = np.median(date_label_counts)
    low, high = np.percentile(date_label_counts, [5, 95])
    print(f"Histogram of date-label counts (median {median}, 90% CI {low:.2f}-{high:.2f})")
    print(tabulate([['# images']+bins.tolist(), ['# date-label'] + hist.tolist()], tablefmt='rounded_outline'))

    # sample p=20% label to validation set so that the number of dates per label is preserved
    train_labels = []
    val_labels = []
    rng = np.random.default_rng(seed=seed)
        
    for n_dates in label_dates.unique():
        labels_with_n_dates = label_dates[label_dates == n_dates].index.tolist()
        size = int(len(labels_with_n_dates) * p)
        size += 1 if len(labels_with_n_dates) > 4 else 0
        rng.shuffle(labels_with_n_dates)
        val_labels += labels_with_n_dates[:size]
        train_labels += labels_with_n_dates[size:]

    assert len(train_labels) + len(val_labels) == n_labels, (len(train_labels) , len(val_labels) , n_labels)
    assert len(set(val_labels) & set(train_labels)) == 0, set(val_labels) & set(train_labels)

    train_df = df[df['label'].isin(train_labels)].reset_index(drop=True)
    val_df = df[df['label'].isin(val_labels)].reset_index(drop=True)
    assert train_df.shape[0] + val_df.shape[0] == df.shape[0], (train_df.shape[0], val_df.shape[0], df.shape[0])


    train_fname = fname.replace('.csv', '_train.csv')
    if not no_split: train_df.to_csv(train_fname, index=False)
    print(f"Saved training set with {train_df.shape[0]} images and {train_df['label'].nunique()} labels to {train_fname}")
    val_fname =fname.replace('.csv', '_validation.csv')
    if not no_split: val_df.to_csv(val_fname, index=False)
    print(f"Saved validation set with {val_df.shape[0]} images and {val_df['label'].nunique()} labels to {val_fname}")

if __name__ == "__main__":
    main()