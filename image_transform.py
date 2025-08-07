import torch.nn.functional as F

class ZoomCenterCrop:
    def __init__(self, zoom=1.0):
        self.zoom = zoom

    def __call__(self, img):
        w, h = img.size
        m = int(min(h, w) / self.zoom)
        left = (w - m) // 2
        top = (h - m) // 2
        return img.crop((left, top, left + m, top + m))
    
if __name__ == "__main__":
    import click
    import torch
    from torchvision import transforms
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    import os

    # Helper to unnormalize for plotting
    def unnormalize(tensor, mean, std):
            for t, m, s in zip(tensor, mean, std):
                t.mul_(s).add_(m)
            return tensor
    
    def tranform_img(img_path, patch_size=44, zoom=1.0):
        img = Image.open(img_path).convert("RGB")
        ts = transforms.Compose([
            transforms.RandomRotation(5), 
            ZoomCenterCrop(zoom),            
            transforms.Resize((440, 440)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),             
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02,0.2), value='random')
        ])
        img_transformed = ts(img)

        # Prepare images for plotting
        img_np = np.array(img)
        img_trans_np = unnormalize(img_transformed.clone(), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img_trans_np = img_trans_np.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        return img_np, img_trans_np
   
    @click.command()
    @click.argument('img_path', type=click.Path(exists=True, file_okay=False, dir_okay=True))
    @click.option('--patch_size', default=44, help='Size of the patches to shuffle')
    @click.option('--zoom', default=1.0, help='Zoom factor (>1 for zoom in, <1 for zoom out)')
    def main(img_path, patch_size, zoom):
        # img_path is a folder
        img_files = [os.path.join(img_path, f) for f in os.listdir(img_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        img_files.sort()
        origs, trans = [], []
        for f in img_files:
            img_np, img_trans_np = tranform_img(f, patch_size, zoom)
            origs.append(img_np)
            trans.append(img_trans_np)
        n = len(origs)
        fig, axs = plt.subplots(2, n, figsize=(4*n, 8))
        for i in range(n):
            axs[0, i].imshow(origs[i])
            axs[0, i].axis("off")
            axs[1, i].imshow(trans[i])
            axs[1, i].axis("off")
        plt.tight_layout()
        plt.savefig("tmp.png")
        print("Saved to tmp.png")

    main()

