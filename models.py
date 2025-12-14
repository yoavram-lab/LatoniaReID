import torch
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def get_model(model_name):
    if model_name.lower().startswith('aliked'):
        return get_aliked_model(model_name)
    elif model_name.lower().startswith('sift'):
        return get_sift_model(model_name)
    elif model_name.lower().startswith('megadescriptor'):
        return get_mega_model(model_name)
    elif model_name.lower() == 'miewid-msv3':
        return get_miewid_model()
    elif model_name.lower() == 'efficientnetv2':
        return get_efficientnet_model()
    else:
        raise ValueError("No model specified or model not recognized.")
    

def get_aliked_model(model_name=None):
    from lightglue import ALIKED

    if not model_name is None and '-' in model_name:
        max_num_keypoints = int(model_name.split('-')[1])        
    else:
        max_num_keypoints = 1432 # Default value
    print(f"Creating ALIKED model with max_num_keypoints={max_num_keypoints}")
    model = ALIKED(max_num_keypoints=max_num_keypoints, detection_threshold=0.01)
    preprocess = transforms.Compose([
        transforms.ToTensor(),        
    ])
    model = model.eval()
    def call(tensor):
        if tensor.ndim == 4:
            assert tensor.shape[0] == 1, "ALIKED model only supports batch size of 1"
            tensor = tensor.squeeze(0)
        return model.extract(tensor)
    model._call_impl = call
    return model, preprocess, 'aliked'

def get_sift_model(model_name=None):
    from lightglue import SIFT
    from torchvision.transforms import functional as F
    from torchvision.transforms import InterpolationMode

    max_dim = 1024
    def resize_to_max_dim(img):
        """Downscale so the longest edge is at most `max_dim`."""
        w, h = img.size
        scale = max_dim / max(h, w)
        if scale < 1.0:
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))
            return F.resize(img, (new_h, new_w), interpolation=InterpolationMode.BILINEAR)
        return img
    
    if not model_name is None and '-' in model_name:
        max_num_keypoints = int(model_name.split('-')[1])        
    else:
        max_num_keypoints = 1432 # Default value
    print(f"Creating SIFT model with max_num_keypoints={max_num_keypoints}")
    model = SIFT(max_num_keypoints=1432)
    preprocess = transforms.Compose([
        transforms.Lambda(resize_to_max_dim),
        transforms.ToTensor(),
    ])
    model = model.eval()

    def call(tensor):
        if tensor.ndim == 4:
            assert tensor.shape[0] == 1, "SIFT model only supports batch size of 1"
            tensor = tensor.squeeze(0)
        return model.extract(tensor)

    model._call_impl = call
    return model, preprocess, 'sift'


def get_efficientnet_model(model_name=None):
    import timm
    model = timm.create_model('efficientnetv2_rw_m.agc_in1k', pretrained=True, num_classes=0) # num_classes=0 for feature extraction
    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config, is_training=False)
    model = model.eval()
    return model, preprocess, 'efficientnetv2'

def get_mega_model(mega_model_name):
    if mega_model_name.startswith('MegaDescriptor'):
        import timm
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
        from huggingface_hub import hf_hub_download
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

def get_miewid_model(model_name=None):
    from transformers import AutoModel
    model = AutoModel.from_pretrained('conservationxlabs/miewid-msv3', trust_remote_code=True)
    model = model.eval()
    preprocess = transforms.Compose([
        transforms.Resize((440, 440)),            
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])

    return model, preprocess, "miewid-msv3"

def save_checkpoint(ckpt_path, model, loss_func, optimizer, loss_optimizer, scheduler, loss_scheduler, epoch):
    model = model.module if isinstance(model, torch.nn.DataParallel) else model
    state = {
        'model': model.state_dict(), 
        'loss_func': loss_func.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss_optimizer': loss_optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss_scheduler": loss_scheduler.state_dict(),
        'epoch': epoch
    }
    torch.save(state, ckpt_path)

def load_checkpoint(checkpoint_path, model, loss_func=None, optimizer=None, loss_optimizer=None, scheduler=None, loss_scheduler=None, map_location=None):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint['model'])
    if loss_func is not None:
        loss_func.load_state_dict(checkpoint['loss_func'])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if loss_optimizer is not None and "loss_optimizer" in checkpoint:
        loss_optimizer.load_state_dict(checkpoint['loss_optimizer'])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    if loss_scheduler is not None and "loss_scheduler" in checkpoint:
        loss_scheduler.load_state_dict(checkpoint['loss_scheduler'])
    return checkpoint.get('epoch', None)
