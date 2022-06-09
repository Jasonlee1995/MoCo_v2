import random, os

from PIL import Image, ImageFilter
from torchvision import transforms
from torch.utils.data import Dataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]



''' For Pre-train '''

class GaussianBlur():
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def pretrain_transform():
    transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                    transforms.RandomGrayscale(p=0.2),
                                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                                    transforms.RandomHorizontalFlip(), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return transform



class PretrainDB(Dataset):
    def __init__(self, data_dir):
        self.imgs = []
        self.transform = pretrain_transform()
        
        for label in sorted(os.listdir(data_dir)):
            if label.isnumeric():
                label_dir = os.path.join(data_dir, label)
                for img in sorted(os.listdir(label_dir)):
                    if img.endswith('.JPEG'):
                        img_dir = os.path.join(label_dir, img)
                        self.imgs.append(img_dir)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert('RGB')
        img_q, img_k = self.transform(img), self.transform(img)
        return img_q, img_k

    def __len__(self):
        return len(self.imgs)
    
    
    
''' For Linear Probing '''

def train_transform():
    transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return transform
    

def val_transform():
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return transform


class FinetuneDB(Dataset):
    def __init__(self, data_dir, transform=None):
        self.imgs = []
        self.labels = []
        self.transform = transform
        
        for label in sorted(os.listdir(data_dir)):
            if label.isnumeric():
                label_dir = os.path.join(data_dir, label)
                for img in sorted(os.listdir(label_dir)):
                    if img.endswith('.JPEG'):
                        img_dir = os.path.join(label_dir, img)
                        self.imgs.append(img_dir)
                        self.labels.append(int(label))

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
            
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.imgs)