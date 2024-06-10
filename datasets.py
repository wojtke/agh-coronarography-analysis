import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np

class SimpleSeriesSelectionDataset(Dataset):
    def __init__(self, images, seq_len=5, split=None):
        self.images = images
        self.seq_len = seq_len

        self.index = self.__construct_index(split)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=None),
            transforms.Normalize([0.5], [0.5])   
        ])

    def __construct_index(self, split):
        good_ids = pd.read_csv("data/series_selection_accepted.csv", delimiter=";").IMAGE_ID
        bad_ids = pd.read_csv("data/series_selection_rejected.csv", delimiter=";").IMAGE_ID
        good_ids = pd.unique(good_ids)
        bad_ids = pd.unique(bad_ids)
        
        index = list(zip(
            list(good_ids) + list(bad_ids),
            len(good_ids)*[1] + len(bad_ids)*[0]
        ))
        index = [i for i in index if i[0] in self.images]
        index = [i for i in index if self.images[i[0]].ndim == 3]
        index = [i for i in index if self.images[i[0]].shape[0]>=self.seq_len]

        test_img_ids = set(pd.read_csv("testimgs.csv").ID)
        if split == "train":
            index = [i for i in index if i[0] not in test_img_ids]
        elif split == "test":
            index = [i for i in index if i[0] in test_img_ids]
        
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx): 
        img_id, label = self.index[idx]
        img = self.images[img_id]
        random_frame_idx = np.random.choice(range(len(img)), self.seq_len, replace=False).astype(int)
        img = img[random_frame_idx, :, :].transpose(1,2,0)
        img = self.transform(img).unsqueeze(1)
        label = torch.tensor(label)
        return img, label


class SimpleFrameSelectionDataset:  
    def __init__(self, images: dict, split=None):
        self.images = images
        self.index = self.__construct_index(split)
                    
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize([0.5], [0.5])   
        ])

    def __construct_index(self, split):
        keys = 2 * list(self.images.keys())
        labels = [0] * len(self.images) + [1] * len(self.images)

        df = pd.read_csv("data/framerejectionfeedback.csv")
        n_frames = list(df["REJECTEDFRAME"]) + list(df["SELECTEDFRAME"])

        index = [*zip(keys, n_frames, labels)]
        index = [i for i in index if self.images[i[0]].shape[0]>i[1]]

        test_img_ids = set(pd.read_csv("testimgs.csv").ID)
        if split == "train":
            index = [i for i in index if i[0] not in test_img_ids]
        elif split == "test":
            index = [i for i in index if i[0] in test_img_ids]

        return index
        
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        key, n_frame, label = self.index[idx]
        img = self.images[key][n_frame]
        img = self.transform(img)
        label = torch.tensor(label)
        return img, label


class DominanceDataset(Dataset):
    def __init__(self, images: dict, split=None):
        self.images = images
        self.index = self.__construct_index(split)
                    
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize([0.5], [0.5])   
        ])

    def __construct_index(self, split):
        df_im = pd.read_csv("data/image.csv").set_index("ID")[["REPRESENTATIVEFRAME", "STUDY_ID"]]
        df_im = df_im[df_im.index.isin(self.images)]
        
        df_st = pd.read_csv("data/study.csv").set_index("ID")[["DOMINANCE"]]
        df_st = df_st.dropna(subset="DOMINANCE")
        df_im = df_im.join(df_st, how="inner", on="STUDY_ID")
        
        
        df_if = pd.read_csv("data/imagefeedback.csv").set_index("IMAGE_ID")[["SIDE"]]
        df_im = df_im.join(df_if, how="inner")
        
        df_im = df_im.reset_index()
        
        left = df_im[df_im["SIDE"] == 0].groupby("STUDY_ID")["ID"].apply(list)
        right = df_im[df_im["SIDE"] == 1].groupby("STUDY_ID")["ID"].apply(list)
        
        left_f = df_im[df_im["SIDE"] == 0].groupby("STUDY_ID")["REPRESENTATIVEFRAME"].apply(list)
        right_f = df_im[df_im["SIDE"] == 1].groupby("STUDY_ID")["REPRESENTATIVEFRAME"].apply(list)
        
        df = pd.DataFrame({
            "l": left, "r": right, "l_f": left_f, "r_f": right_f, "label": df_st["DOMINANCE"]
        })
        
        df = df.dropna()

        index = [(i, row) for (i, row) in df.iterrows()]
        test_study_ids = set(pd.read_csv("teststudies.csv").ID)
        if split == "train":
            index = [i for i in index if i[0] not in test_study_ids]
        elif split == "test":
            index = [i for i in index if i[0] in test_study_ids]

        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        study_id, data = self.index[idx]

        i_left = np.random.randint(0, len(data["l"]))
        i_right = np.random.randint(0, len(data["r"]))
        
        img_left = self.images[data["l"][i_left]][data["l_f"][i_left]]
        img_right = self.images[data["r"][i_right]][data["r_f"][i_right]]
        
        img_left, img_right= self.transform(img_left), self.transform(img_right)
        label = torch.tensor(data["label"]).long()
        return img_left, img_right, label
        
        

        
        
        
    
    
