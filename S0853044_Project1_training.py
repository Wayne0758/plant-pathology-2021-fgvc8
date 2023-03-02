#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import PIL
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import skimage.io as io


# In[2]:


BATCH = 16
LR = 0.0001
IM_SIZE = 299

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path = './plant-pathology-2021-fgvc8/'
TRAIN_DIR = path + 'train_images/'
TEST_DIR = path + 'test_images/'


# In[3]:


TRAIN_DATA_FILE = os.path.join(path, 'train.csv')
def read_image_labels():
    """
    """
    df = pd.read_csv(TRAIN_DATA_FILE).set_index('image')
    return df


# In[4]:


train_df = read_image_labels().sample(
    frac=1.0, 
    random_state=42
)

train_df.head()


# In[5]:


print(len(train_df))


# In[6]:


from typing import List, Dict
def get_single_labels(unique_labels) -> List[str]:
    """Splitting multi-labels and returning a list of classes"""
    single_labels = []
    
    for label in unique_labels:
        single_labels += label.split()
        
    single_labels = set(single_labels)
    return list(single_labels)


# In[7]:


def get_one_hot_encoded_labels(dataset_df) -> pd.DataFrame:
    """
    """
    df = dataset_df.copy()
    
    unique_labels = df.labels.unique()
    column_names = get_single_labels(unique_labels)
    
    df[column_names] = 0        
    
    # one-hot-encoding
    for label in unique_labels:                
        label_indices = df[df['labels'] == label].index
        splited_labels = label.split()
        df.loc[label_indices, splited_labels] = 1
    
    return df


# In[8]:


tr_df = get_one_hot_encoded_labels(train_df)
tr_df


# In[41]:


folders = dict({
        'data': "./plant-pathology-2021-fgvc8",
        'train': './plant-pathology-2021-fgvc8/train_images',
        'val': './plant-pathology-2021-fgvc8/train_images',
        'test':  os.path.join("./plant-pathology-2021-fgvc8", 'test_images')
    })


# In[50]:


def get_image(image_id, kind='train'):
    """Loads an image from file
    """
    fname = os.path.join(Config.folders[kind], image_id)
    return PIL.Image.open(fname)
data=pd.DataFrame(columns=['labels'])
list1=[]
for image_id, label in zip(train_df.index, train_df.labels):
    if label not in list1:
        series=pd.Series({'labels':label},name=image_id)
        data=data.append(series)
        list1.append(label)
def visualize_images(image_ids, labels, nrows=1, ncols=4, kind='train', image_transform=None):
    """
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 8))
    for image_id, label, ax in zip(image_ids, labels, axes.flatten()):
        fname = os.path.join(folders[kind], image_id)
        image = np.array(PIL.Image.open(fname))

        list1.append(label)
        if image_transform:
            image = transform = A.Compose(
                [t for t in image_transform.transforms if not isinstance(t, (
                    A.Normalize, 
                    ToTensorV2
                ))])(image=image)['image']

        io.imshow(image, ax=ax)

        ax.set_title(f"Class: {label}", fontsize=12)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        del image
        
    plt.show()
visualize_images(data.index, data.labels, nrows=3, ncols=4)


# In[9]:


# !!! Just for speed up training
from sklearn.model_selection import train_test_split
CLASSES = [
        'rust', 
        'complex', 
        'healthy', 
        'powdery_mildew', 
        'scab', 
        'frog_eye_leaf_spot'
    ]
X_Train, X_Valid, Y_Train, Y_Valid = train_test_split(
    pd.Series(train_df.index), 
    np.array(tr_df[CLASSES]),  
    test_size=0.2, 
    random_state=42
)
X_Train.head()


# In[10]:


train_transform = A.Compose([
            A.RandomResizedCrop(height=IM_SIZE, width=IM_SIZE),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])
val_transform = A.Compose([
            A.Resize(height=IM_SIZE, width=IM_SIZE),
            A.Normalize(),
            ToTensorV2(),
        ])


# In[11]:





# In[12]:


def get_image(image_id, kind='train'):
    """Loads an image from file
    """
    fname = os.path.join(folders[kind], image_id)
    return Image.open(fname)


# In[13]:


from scipy.stats import bernoulli
from torch.utils.data import Dataset

class PlantDataset(Dataset):
    """
    """
    def __init__(self, 
                 image_ids, 
                 targets,
                 transform=None, 
                 target_transform=None, 
                 kind='train'):
        self.image_ids = image_ids
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.kind = kind
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # load and transform image
        img = np.array(get_image(self.image_ids.iloc[idx], kind=self.kind))
        
        if self.transform:
            img = self.transform(image=img)['image']
        
        # get image target 
        target = self.targets[idx]
        if self.target_transform:
            target = self.target_transform(target)
        
        return img, target


# In[14]:


trainset = PlantDataset(X_Train, Y_Train, transform=train_transform, kind='train')
trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True)
validset = PlantDataset(X_Valid, Y_Valid, transform=val_transform, kind='val')
validloader = DataLoader(validset, batch_size=BATCH, shuffle=False)


# In[15]:


model = torchvision.models.inception_v3(pretrained=True)
model.aux_logits=False
model.fc = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 6),
            nn.Sigmoid()
        )
model = model.to(DEVICE)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# In[16]:


class MetricMonitor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = []
        self.scores = []
        self.metrics = dict({
            'loss': self.losses,
            'f1': self.scores
        })

    def update(self, metric_name, value):
        self.metrics[metric_name] += [value]


# In[17]:


monitor = MetricMonitor()


# In[18]:


print(torch.__version__)
print(torch.version.cuda)


# In[19]:


import torch
from torchmetrics.classification import BinaryF1Score
best_f1score=0
first_epochs=0
last_epochs=20
checkpoint = {
        "model": model.state_dict(),
        'optimizer':optimizer.state_dict()
    }
#checkpointmodel = torch.load(
#    "./plant-pathology-2021-fgvc8/resnet18_adapted_bestmodel/resnet18_adapted_epoch{}.pth".format(first_epochs))
#model.load_state_dict(checkpointmodel['model'])
#optimizer.load_state_dict(checkpointmodel['optimizer'])
for epoch in range(first_epochs,last_epochs):
    tr_loss = 0.0
    f1 = BinaryF1Score(threshold=0.4).to(DEVICE)
    f1score = 0

    model = model.train()

    for i, (images, labels) in enumerate(trainloader):        
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)       
        pred = model(images.float())
        loss = criterion(pred.float(), labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tr_loss += loss.detach().item()
        f1score += f1(pred ,labels)
    
    model.eval()
    print('Train - Epoch: %d | Loss: %.4f | F1: %.4f'%(epoch+1, tr_loss / i, f1score / i))
    monitor.update('loss', tr_loss / i)
    monitor.update('f1', f1score / i)

    
    tr_loss = 0.0
    f1 = BinaryF1Score(threshold=0.4).to(DEVICE)
    f1score = 0

    for i, (images, labels) in enumerate(validloader):        
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)       
        pred = model(images.float()) 
        loss = criterion(pred.float(), labels.float())
        
        tr_loss += loss.detach().item()
        f1score += f1(pred ,labels)
    
    model.eval()
    print('Valid - Epoch: %d | Loss: %.4f | F1: %.4f'%(epoch+1, tr_loss / i, f1score / i))
    monitor.update('loss', tr_loss / i)
    monitor.update('f1', f1score / i)
    if f1score / i > best_f1score:
        path = "./plant-pathology-2021-fgvc8/inception_v3_bestmodel/inception_v3_bestmodel_epoch{}.pth".format(epoch+1)
        torch.save(checkpoint, path)
        best_f1score = f1score / i
    if (epoch+1) %20==0:
        path = "./plant-pathology-2021-fgvc8/inception_v3_bestmodel/inception_v3_epoch{}.pth".format(epoch+1)
        torch.save(checkpoint, path)

