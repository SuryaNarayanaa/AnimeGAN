#!/usr/bin/env python
# coding: utf-8

# In[1]:



# # Imports

# In[3]:


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import opendatasets as od
import os
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tqdm import tqdm


# In[ ]:





# # Dataset  

# In[4]:


od.download("https://www.kaggle.com/datasets/splcher/animefacedataset")


# In[5]:


DATADIR = os.path.join(os.getcwd(), "animefacedataset")
IMAGE_DIR = os.path.join(DATADIR, "images")
stats = ((.5,.5,.5),(.5,.5,.5))


# In[6]:


train_ds  = ImageFolder(DATADIR, transform=transforms.Compose(
    [transforms.Resize(64),
     transforms.CenterCrop(64),
     transforms.ToTensor(),
     transforms.Normalize(*stats)]
))


# # Data loader

# In[157]:


batch_size = 400


# In[158]:


train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)


# # CUDA

# In[159]:


def to_device(data ,device):
  if isinstance(data , (list , tuple)):
    return [to_device(x ,device) for x in data]
  return data.to(device , non_blocking=True)

class DeviceDataLoader():

  def __init__(self, dl ,device):
    self.dl  = dl
    self.device  =  device

  def __iter__(self):
    for b in self.dl:
      yield to_device(b ,self.device)

  def __len__(self):
    return len(self.dl)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# In[160]:


train_dl_cuda = DeviceDataLoader(train_dl , device)

train_dl_cuda.device


# # VISUALISATION

# In[161]:


def denorm(img_tensors):
  return img_tensors * stats[1][0] + stats[0][0]

def show_images(images , nmax=64):
  fig,ax = plt.subplots(figsize = (8,8))
  ax.set_xticks([])
  ax.set_yticks([])
  ax.imshow(make_grid(denorm(images.detach().cpu()[:nmax]), nrow=8).permute(1,2,0))
  plt.show()

def show_batch(dl):
  for images , _ in dl:
    show_images(images )
    break



# In[162]:


show_batch(train_dl_cuda)


# # DISCRIMINATOR

# In[194]:


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)



# GENERATOR

# In[195]:


class Generator(nn.Module):
    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


# In[196]:


latent_size = 256
generator = Generator(latent_size).to(device)
discriminator = Discriminator().to(device)


# In[197]:


xb  = torch.randn(batch_size , latent_size ,1,1)

fake_images = generator(xb.to(device))
print(fake_images.shape,xb.shape)
show_images(fake_images)


# # Train Discriminator

# In[198]:


def train_discriminator(real_images, opt_d):
    opt_d.zero_grad()
    batch_size = real_images.size(0)  # Get batch size from real_images

    real_preds = discriminator(real_images)
    real_targets = torch.ones(batch_size, 1, device=device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()

    latent = torch.randn(batch_size, latent_size, 1, 1).to(device)
    fake_images = generator(latent)

    fake_targets = torch.zeros(batch_size, 1, device=device)
    fake_preds = discriminator(fake_images)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    

    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score


# # Train Generator

# In[199]:


def train_genrator(opt_g):

    opt_g.zero_grad()

    latent = torch.randn(batch_size , latent_size ,1,1).to(device)
    fake_images = generator(latent)

    fake_preds = discriminator(fake_images)
    fake_targets = torch.ones(batch_size , 1 , device=device)
    loss = F.binary_cross_entropy(fake_preds , fake_targets)

    loss.backward()
    opt_g.step()

    return  loss.item()


# 

# # Save batch of Images
# 

# In[200]:


save_image_dir = "generated-3"
os.makedirs(save_image_dir, exist_ok=True)


# In[201]:


def save_generated_image(index  , latent_tensors , show=True):
  fake_images = generator(latent_tensors)
  fake_fname = "generated-images-{0:0=4d}.png".format(index)
  save_image(denorm(fake_images), os.path.join(save_image_dir, fake_fname), nrow=20)
  if show:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(fake_images.cpu().detach(), nrow=20).permute(1, 2, 0))
    plt.show()


# In[202]:


fixed_latent = torch.randn(batch_size, latent_size, 1, 1, device=device)


# In[207]:


save_generated_image(0, fixed_latent)


# In[208]:


def fit(epochs, lr, start_idx =1):

    torch.cuda.empty_cache()

    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):

        for real_images , _ in tqdm(train_dl_cuda):
            loss_d , real_score , fake_score =train_discriminator(real_images , opt_d)
            loss_g = train_genrator(opt_g)

        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch+1, epochs, loss_g, loss_d, real_score, fake_score))

        save_generated_image(epoch+start_idx, fixed_latent, show=False)

    return losses_g, losses_d, real_scores, fake_scores


# In[210]:


lr = 0.0001
epochs  =150


# In[ ]:




history = []
# In[52]:


history+= fit(epochs , lr)


# In[152]:


history+= fit(epochs , lr)


# In[211]:


history+= fit(epochs , lr)


# In[212]:


torch.save(generator.state_dict(), "generator-3.pth")


# In[216]:


history


# In[ ]:




