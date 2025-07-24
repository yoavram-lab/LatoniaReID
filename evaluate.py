import timm
from torchvision import transforms
import click
import pandas as pd
from transformers import AutoModel
from huggingface_hub import hf_hub_download
import torch

from training_miewid import embed, evaluate, DataFrameDataset, val_transform, load_ckpt
import os
import numpy as np

device = 'cpu'

def get_model(model_name):
    if model_name.startswith('MegaDescriptor'):
        model, preprocess, model_name = get_mega_model(model_name)
    elif model_name == 'miewid-msv3':
        model, preprocess, model_name = get_miewid_model()
    else:
        raise ValueError("No model specified or model not recognized.")
    return model_name,model,preprocess

def get_mega_model(mega_model_name):
    if mega_model_name.startswith('MegaDescriptor'):
        model = timm.create_model(f"hf-hub:BVRA/{mega_model_name}", pretrained=True)
        # model = AutoModel.from_pretrained(f"BVRA/{mega_model_name}", trust_remote_code=True)
        # model(imgs).pooler_output to get embeddings of dim 1536

        img_size = int(mega_model_name.split('-')[-1])
        preprocess = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(), 
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    elif mega_model_name == 'MegaDescriptor-EfficientNetB3':
        ckpt = hf_hub_download(f"BVRA/{mega_model_name}", "pytorch_model.bin")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        # drop missing classifier keys (we're using the encoder only)
        state = {k: v for k, v in state.items() if not k.startswith('classifier.')}
        model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0, global_pool='avg')
        model.load_state_dict(state, strict=False)

        preprocess = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)), 
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    model = model.to(device)
    model = model.eval()
    return model, preprocess, mega_model_name

def get_miewid_model():
    model = AutoModel.from_pretrained('conservationxlabs/miewid-msv3', trust_remote_code=True).to(device)
    model = model.eval()
    preprocess = val_transform()

    return model, preprocess, "miewid-msv3"

def get_embeddings(val_csv, model_name, model, val_dataset, device):
    cache_file = f"results/{model_name}_{os.path.basename(val_csv)}.npz"
    if os.path.exists(cache_file):
        embeddings = np.load(cache_file)["embeddings"]
        print(f"Loaded embeddings from {cache_file}")
    else:
        embeddings = embed(model.to(device), val_dataset, device=device)
        np.savez_compressed(cache_file, embeddings=embeddings.to('cpu').numpy())
        print(f"Saved embeddings to {cache_file}")
    return embeddings

@click.command()
@click.argument('model_name', type=str)
@click.option('--val_csv', type=str, default='bina_photos_validation.csv')
@click.option('--checkpoint', type=str, default=None, help='Path to the model checkpoint')
@click.option('--device', type=str, default='cpu', help='Device to run the model on (e.g., cpu, cuda)')
def main(model_name, val_csv, checkpoint, device):
    print(f"Evaluating {model_name} on {val_csv}...")
    
    model_name, model, preprocess = get_model(model_name)

    val_df = pd.read_csv(val_csv)
    val_dataset = DataFrameDataset(val_df, transform=preprocess)

    if checkpoint is not None:
        print(f"Loading checkpoint from {checkpoint}...")
        load_ckpt(checkpoint, model, map_location=device)
        model_name += '_' + checkpoint.split('/')[1]
    
    embeddings = get_embeddings(val_csv, model_name, model, val_dataset, device)
    if device != 'cpu':
        embeddings = torch.tensor(embeddings).to(device)
    metrics = evaluate(embeddings, val_dataset)
     
    print("{}".format(" ".join([f"{k:<15}" for k in metrics.keys()])))
    print("{}".format(" ".join([f"{v:<15.6f}" for v in metrics.values()])), flush=True)

if __name__ == "__main__":
    main()
