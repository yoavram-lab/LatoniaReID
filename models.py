from transformers import AutoModel
from train import val_transform
from torchvision import transforms
import timm
import torch
from transformers import AutoModel
from huggingface_hub import hf_hub_download


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
    model = model.eval()
    return model, preprocess, mega_model_name

def get_miewid_model():
    model = AutoModel.from_pretrained('conservationxlabs/miewid-msv3', trust_remote_code=True)
    model = model.eval()
    preprocess = val_transform()

    return model, preprocess, "miewid-msv3"