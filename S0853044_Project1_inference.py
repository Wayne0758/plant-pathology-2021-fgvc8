#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
from PIL import Image
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2


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


from typing import List, Dict
def get_single_labels(unique_labels) -> List[str]:
    """Splitting multi-labels and returning a list of classes"""
    single_labels = []
    
    for label in unique_labels:
        single_labels += label.split()
        
    single_labels = set(single_labels)
    return list(single_labels)


# In[6]:


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


# In[7]:


tr_df = get_one_hot_encoded_labels(train_df)
tr_df.head()


# In[8]:


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


# In[9]:


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


# In[10]:


folders = dict({
        'data': "./plant-pathology-2021-fgvc8",
        'train': './plant-pathology-2021-fgvc8/train_images',
        'val': './plant-pathology-2021-fgvc8/train_images',
        'test':  os.path.join("./plant-pathology-2021-fgvc8", 'test_images')
    })


# In[11]:


def get_image(image_id, kind='train'):
    """Loads an image from file
    """
    fname = os.path.join(folders[kind], image_id)
    return Image.open(fname)


# In[12]:


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


# In[13]:


validset = PlantDataset(X_Valid, Y_Valid, transform=val_transform, kind='val')
validloader = DataLoader(validset, batch_size=BATCH, shuffle=False)


# In[14]:


model = torchvision.models.inception_v3(pretrained=True)
model.aux_logits=False
model.fc = nn.Sequential(
            nn.Linear(2048, 6),
            nn.Sigmoid()
            
        )
model = model.to(DEVICE)


# In[15]:


checkpoint = torch.load("./plant-pathology-2021-fgvc8/inception_v3_bestmodel/inception_v3_bestmodel_epoch20.pth")
model.load_state_dict(checkpoint['model'])
### now you can evaluate it
model.eval()


# In[16]:


import matplotlib.pyplot as plt
y_true = np.empty(shape=(0, 6), dtype=np.int)
y_pred_proba = np.empty(shape=(0, 6), dtype=np.int)
model.eval()
for BATCH, (X, y) in enumerate(validloader):
    X = X.to(DEVICE)
    y = y.to(DEVICE).detach().cpu().numpy()
    pred = model(X).detach().cpu().numpy()
    
    y_true = np.vstack((y_true, y))
    y_pred_proba = np.vstack((y_pred_proba, pred))


# In[17]:


from sklearn.metrics import multilabel_confusion_matrix

def plot_confusion_matrix(
    y_test, 
    y_pred_proba, 
    threshold=0.4, 
    label_names=CLASSES
)-> None:
    """
    """
    y_pred = np.where(y_pred_proba > threshold, 1, 0)
    c_matrices = multilabel_confusion_matrix(y_test, y_pred)
    
    cmap = plt.get_cmap('Blues')
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

    for cm, label, ax in zip(c_matrices, label_names, axes.flatten()):
        sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap=cmap);

        ax.set_xlabel('Predicted labels');
        ax.set_ylabel('True labels'); 
        ax.set_title(f'{label}');

    plt.tight_layout()    
    plt.show()


# In[18]:


import seaborn as sns


# In[19]:


plot_confusion_matrix(y_true, y_pred_proba)


# In[20]:


from torchmetrics.classification import BinaryF1Score
f1 = BinaryF1Score(threshold=0.4)
y_pred = torch.as_tensor(np.where(y_pred_proba > 0.4, 1, 0))
y_true = torch.as_tensor(y_true)
f1 = f1(y_pred ,y_true).numpy()

pd.DataFrame({
    'name': ['F1'],
    'sorce': [f1]
}).set_index('name')


# In[21]:


def save_submission(model):
    """
    """
    image_ids = pd.read_csv(os.path.join(path, 'sample_submission.csv'))
    
    dataset = PlantDataset(
        image_ids['image'], 
        image_ids['labels'], 
        transform=val_transform, 
        kind='test'
    )
    
    loader = DataLoader(dataset)

    for idx, (X, _) in enumerate(loader):
        X = X.float().to(DEVICE)
        y_pred = torch.argmax(model(X), dim=1).detach().cpu().numpy()

        pred_labels = ' '.join([CLASSES[i] for i in y_pred]).strip()
        image_ids.iloc[idx]['labels'] = pred_labels
    
    # save data frame as csv
    image_ids.set_index('image', inplace=True)
    image_ids.to_csv(os.path.join('./', 'submission.csv'))
    
    return image_ids


# In[22]:


save_submission(model) 


# In[ ]:




