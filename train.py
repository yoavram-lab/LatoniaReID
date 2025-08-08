# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # Workaround for OpenMP initialization error on macOS
import click
from pathlib import Path
from datetime import datetime
import csv
import shutil
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning.samplers import MPerClassSampler
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
import wandb

from image_transform import ZoomCenterCrop
from evaluate import evaluate, embed
from models import get_model, load_checkpoint, save_checkpoint
import config

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

device = "cuda:1" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
if device.startswith("cuda"):
    torch.set_float32_matmul_precision('medium')
    torch.cuda.memory._set_allocator_settings("expandable_segments:True,max_split_size_mb:128")
    torch.backends.cudnn.benchmark = True  # speed up convolutions when input sizes are fixed


class DataFrameDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = df['rel_path'].tolist() # bbox/2019-5/215/IMGP0147.jpg
        try:
            self.dates = df['date'].tolist() # 2019-5
        except KeyError:
            self.dates = [p.split('/')[1] for p in self.paths]  # extract date from path
            #self.dates = [i for i, _ in enumerate(self.paths)]  # use index as date - different "date" per image
        self.labels = df['label'].tolist() # labels are encoded from ind (215 in path above) 
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx]
    
def get_train_transform(size=440):
    return transforms.Compose([
        transforms.RandomRotation(5), 
        ZoomCenterCrop(zoom=2.0),
        transforms.RandomResizedCrop(size, scale=(0.8,1.0), interpolation=transforms.InterpolationMode.BICUBIC),        
        # transforms.Resize((size, size)), # not needed, RandomResizedCrop already resizes
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        # transforms.RandomErasing(p=0.25, scale=(0.02,0.2), value='random')
    ])

def train_dataloader(train_dataset, m, batch_size, num_workers):
    sampler = MPerClassSampler(
        train_dataset.labels, 
        m=m, 
        length_before_new_iter=len(train_dataset.labels)
    )
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True          # for faster data transfer to GPU
    )

def get_loss_func(num_classes, embedding_size, margin, scale, sub_centers):
    return losses.SubCenterArcFaceLoss(
        num_classes=num_classes, 
        embedding_size=embedding_size, 
        margin=margin, 
        sub_centers=sub_centers,
        scale=scale
    )

def freeze_bn(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()                                   # no running stats update
        module.weight.requires_grad_(False)             # freeze γ
        module.bias.requires_grad_(False)               # freeze β


def train(model, loss_func, train_loader, optimizer, loss_optimizer, epoch):
    model.train()
    model.apply(freeze_bn)
    losses = []
    pbar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", total=len(train_loader), leave=False)
    for batch_idx, (data, labels) in pbar:
        # When using DataParallel, leave `data` on CPU so it can be scattered to GPUs.
        # Only move `labels` to the main device for the loss computation.
        if isinstance(model, torch.nn.DataParallel):
            labels = labels.to(device, non_blocking=True)
        else:
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss_optimizer.zero_grad()
        with torch.amp.autocast(device):
            embeddings = model(data)
            loss = loss_func(embeddings, labels)
        loss.backward()
        optimizer.step()
        loss_optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix({"loss": losses[-1]})
        
    avg_loss = np.mean(losses)
    return avg_loss


@click.command()
@click.option("--train_csv", type=str, required=True)
@click.option("--val_csv", type=str, required=False, default=None)
@click.option("--backbone_name", default="miewid-msv3")
@click.option("--checkpoint", type=str, default=None, help="Path to a checkpoint to resume training.")
@click.option("--m", type=int, default=1, help="Number of samples per class in each batch.")
@click.option("--batch_size", type=int, default=24, help="Total batch size (must be divisible by m).")
@click.option("--epochs", type=int, default=500)
@click.option("--lr_backbone", type=float, default=2e-5)
@click.option("--lr_head", type=float, default=1e-3)
@click.option("--eval_interval", type=int, default=5)
@click.option("--num_workers", type=int, default=4)
@click.option("--dataparallel/--no-dataparallel", default=False, help="Enable DataParallel for multi-GPU training.")
@click.option("--margin", type=float, default=0.5, help="ArcFace margin parameter.")
@click.option("--scale", type=float, default=64.0, help="ArcFace scale parameter.")
@click.option("--sub_centers", type=int, default=1, help="Number of sub-centers for SubCenterArcFaceLoss.")
@click.option("--early_stopping/--no-early-stopping", default=True, help="Enable early stopping based on mAP@R.")
@click.option("--load_optimizer/--no-load-optimizer", default=False, help="Load optimizers and schedulers from checkpoint.")
def main(train_csv, val_csv, backbone_name, checkpoint, m, batch_size, epochs, lr_backbone, lr_head, eval_interval, num_workers, dataparallel, margin, scale, sub_centers, early_stopping, load_optimizer):
    assert batch_size % m == 0, "Batch size must be divisible by m (number of positive samples per class)."
    print(f"Starting training with backbone {backbone_name}, checkpoint {checkpoint}, m={m}, batch_size={batch_size}, epochs={epochs}, lr_backbone={lr_backbone}, lr_head={lr_head}")
    
    backbone_tag = backbone_name.split("/")[-1]
    ckpt_base = f"{config.CHECKPOINTS_FOLDER}/{backbone_tag}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    Path(ckpt_base).mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints, CSVs, and metrics will be saved to {ckpt_base}")

    # Copy train/val CSV file into the checkpoint directory
    shutil.copy2(train_csv, Path(ckpt_base) / Path(train_csv).name)
    shutil.copy2(val_csv, Path(ckpt_base) / Path(val_csv).name)
    
    # create a CSV file to log results
    csv_file = open(f'{ckpt_base}/metrics.csv', mode='w', newline='')
    csv_writer = csv.writer(csv_file)

    # load train data
    train_df = pd.read_csv(train_csv)
    remap_labels = {lbl:i for i, lbl in enumerate(train_df['label'].unique())} 
    train_df['label'] = [remap_labels[lbl] for lbl in train_df['label']] 
    num_classes = train_df['label'].max() + 1 

    # load validation data
    if val_csv is None: 
        val_csv = train_csv 
    val_df = pd.read_csv(val_csv)

    # load model and loss function
    model, preprocess, model_name = get_model(backbone_name)
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad = True
    
    if hasattr(model, 'final'):
        model_output_size = model.final.in_features
    elif hasattr(model, 'num_features'):
        model_output_size = model.num_features
    else:
        raise ValueError(f"Cannot determine model {model_name} output size.")    

    loss_func = get_loss_func(
        num_classes=num_classes,
        embedding_size=model_output_size,
        margin=margin,
        scale=scale, 
        sub_centers=sub_centers
    ).to(device)

    try: # get size from preprocess
        size = preprocess.transforms[0].size[0]  
    except TypeError:
        size = preprocess.transforms[0].size 
    train_transforms = get_train_transform(size=size) 
    val_transforms = preprocess
    val_transforms.transforms.insert(0, ZoomCenterCrop(zoom=2.0))
    train_dataset = DataFrameDataset(train_df, transform=train_transforms)
    train_loader = train_dataloader(train_dataset, m, batch_size, num_workers=num_workers)
    val_dataset = DataFrameDataset(val_df, transform=val_transforms)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_backbone, weight_decay=1e-4)
    loss_optimizer = torch.optim.AdamW(loss_func.parameters(), lr=lr_head, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(loss_optimizer, T_max=epochs, eta_min=1e-6)

    start_epoch = 1
    if checkpoint is not None:
        print(f"Loading checkpoint from {checkpoint}...")
        if load_optimizer:
            loaded_epoch = load_checkpoint(checkpoint, model, loss_func, optimizer, loss_optimizer, scheduler, loss_scheduler, map_location=device)
        else:
            loaded_epoch = load_checkpoint(checkpoint, model, loss_func, map_location=device)
        start_epoch += loaded_epoch
        print(f"Resuming training from epoch {start_epoch}")

    # Enable simple multi-GPU training if flag is set
    if dataparallel and device == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: total={total_params:,} | trainable={trainable_params:,}")  

    wandb_config = click.get_current_context().params
    wandb_config['start_epoch'] = start_epoch
    wandb_config['checkpoint'] = checkpoint
    wandb_config['ckpt_base'] = ckpt_base
    wandb_config['model_name'] = model_name
    wandb_config['device'] = device
    wandb_run = wandb.init(
        project="LatoniaReID",
        config=wandb_config
    )
    
    tic = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    best_epoch = 0
    best_metric = 0
    print(f"Starting training for {epochs} epochs at {tic}.")
    for epoch in range(start_epoch, start_epoch + epochs):
        loss = train(model, loss_func, train_loader, optimizer, loss_optimizer, epoch)        
        scheduler.step()
        loss_scheduler.step()

        if epoch % eval_interval == 0 or epoch == start_epoch or epoch == start_epoch + epochs:
            # evaluation
            embeddings = embed(model, val_dataset, device)
            similarity_func = CosineSimilarity()
            similarity_matrix = similarity_func(embeddings, embeddings)
            metrics = evaluate(similarity_matrix, val_dataset)
            if epoch == start_epoch: 
                # print header
                print("{:<6} {:<12} {}".format("epoch", "train_loss", " ".join([f"{k:<15}" for k in metrics.keys()])))
                csv_writer.writerow(['epoch', 'train_loss', *metrics.keys()])                
            if epoch > 50 and metrics['mAP@R'] > best_metric:  # mAP@R: high is better
                best_metric = metrics['mAP@R']
                best_epoch = epoch
                print(f"New best metric: {best_metric:.6f} at epoch {best_epoch}")
                # Save the best model checkpoint
                ckpt_path = f'{ckpt_base}/best_model.ckpt'
                save_checkpoint(ckpt_path, model, loss_func, optimizer, loss_optimizer, scheduler, loss_scheduler, epoch)
                print(f"Saved model checkpoint to {ckpt_path}")
                wandb_run.log_artifact(ckpt_path, type='model')            
            # print metrics
            print("{:<6} {:<12.6f} {}".format(epoch, loss, " ".join([f"{v:<15.6f}" for v in metrics.values()])), flush=True)
            csv_writer.writerow([epoch, loss, *metrics.values()])
            csv_file.flush()
            wandb_run.log({'train_loss': loss}, step=epoch, commit=False)
            wandb_run.log(metrics, step=epoch, commit=True)
            # early stopping based on mAP@R
            if early_stopping and (
                epoch == 30 and metrics["mAP@R"] < 0.12 or 
                epoch == 60 and metrics["mAP@R"] < 0.24 ):
                print(f"Early stopping at epoch {epoch} with mAP@R {metrics['mAP@R']:.6f}")
                break
    
    toc = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Training completed at {toc}.")
    # Save model checkpoint
    ckpt_path = f'{ckpt_base}/final_model.ckpt'
    save_checkpoint(ckpt_path, model, loss_func, optimizer, loss_optimizer, scheduler, loss_scheduler, epoch)
    print(f"Saved final model checkpoint to {ckpt_path}")
    wandb_run.log_artifact(ckpt_path, type='model')
    wandb_run.finish()

if __name__ == "__main__":
    main()