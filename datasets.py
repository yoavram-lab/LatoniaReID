from PIL import Image
from torch.utils.data import Dataset


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
