import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from stable_diffusion_model import load_vae_diffusion_model, StableDiffusion, load_model, load_model_from_checkpoint
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import os
import torchvision.datasets as dset
import pandas as pd
import ast
import torchvision.utils as vutils
from torchvision.utils import save_image

from model_config_v0 import device, image_size, latent_size, n_epochs, batch_size, lr, num_timesteps, \
    save_checkpoint_interval, lambda_cons, max_lambda_cons, epochs_to_max_lambda, workers, \
    data_path, sd_path, dataroot_train, dataroot_test, gx_train_path, gx_test_path
from utils import load_gene_expression, plot_losses, load_image, load_pair_gx_image

result_path = 'result'

model = load_model('/home/anlt69/Desktop/ST_SIM/SD_model_7Nov2024/SD_7Nov2024/Results/stable_diffusion_model_checkpoint_epoch_500.pth',
                                in_channels=3,
                                latent_dim=4,
                                image_size=256,
                                diffusion_timesteps=1000,
                                device=device)

model.to(device)
model.eval()

gene_expression_val = load_gene_expression(gx_test_path)

with torch.no_grad():
    # fake_images = []
    # for i in random_indices:
    gene_expression_fixed = np.array([gene_expression_val[i] for i in range(0,500)])
    tensor_gene = torch.tensor(gene_expression_fixed, dtype=torch.float32, device=device).view(500, -1).unsqueeze(1)
    # gene_embeddings = gene_pooling_layer(tensor_gene).unsqueeze(1) # (len(random_indices), 2048) -> (len(random_indices), 1, 512)
    sampled_latents = model.sample(tensor_gene, latent_size=latent_size, batch_size=500, guidance_scale=10, device=device)
    fake_images = model.decode(sampled_latents).squeeze(1)

for i, img_tensor in enumerate(fake_images):
    file_name = os.path.join(result_path, f"fake_image_{i+1}.png")
    vutils.save_image(img_tensor, file_name)
    print(f"Saved {file_name}")
