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
from stable_diffusion_model import StableDiffusion
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
    random_indices, data_path, sd_path, dataroot_train, dataroot_test, gx_train_path, gx_test_path, result_path, diversity_weight

from utils import load_gene_expression, plot_losses, load_image

# Configure logging
log_file = f'{result_path}/training.log'
if os.path.exists(log_file):
    os.remove(log_file)
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

losses_mse = []
losses_div = []
losses_cons = []
losses_total = []

def diversity_loss(latents, use_cosine=False):
    batch_size = latents.size(0)
    latents_flat = latents.view(batch_size, -1)

    if use_cosine:
        latents_norm = F.normalize(latents_flat, p=2, dim=1)
        similarity = torch.mm(latents_norm, latents_norm.t())
    else:
        similarity = torch.mm(latents_flat, latents_flat.t())

    similarity = similarity - torch.eye(batch_size, device=latents.device)
    return similarity.sum() / (batch_size * (batch_size - 1))

def train(model, device, dataloader, optimizer, scheduler, epoch, iters, diversity_weight, epoch_loss, num_batches, current_lambda_cons):
    model.train()
    gene_expression_train = load_gene_expression(gx_train_path)

    for i, data in enumerate(dataloader, 0):
        data = data[0].to(device)
        latents = model.encode(data)

        gene_expression_batch = gene_expression_train[i * batch_size:(i + 1) * batch_size]
        tensor_gene = torch.tensor(gene_expression_batch, dtype=torch.float32, device=device).view(batch_size, -1).unsqueeze(1)

        # Add noise
        timesteps = torch.randint(0, num_timesteps, (latents.shape[0],), device=device).long()
        noisy_latents, noise = model.noise_scheduler.add_noise(latents, timesteps)

        # Predict noise
        noise_pred = model(noisy_latents, timesteps, tensor_gene)
        mse_loss = F.mse_loss(noise_pred, noise)
        div_loss = diversity_loss(noisy_latents, use_cosine=True)

        # Calculate latents after denoising
        alpha_t = model.noise_scheduler.alphas[timesteps][:, None, None, None]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        predicted_latents = (noisy_latents - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        cons_loss = F.mse_loss(predicted_latents, latents)

        # Combination loss
        total_loss = mse_loss + diversity_weight * div_loss + cons_loss * current_lambda_cons
        epoch_loss += total_loss.item()
        num_batches += 1

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Dynamically adjust the weight of diversity loss
        if epoch % 10 == 0:
            diversity_weight = min(diversity_weight * 1.05, 0.1)

        # Log and plot losses
        if i % 100 == 0:
            print('[Epoch %d/%d][Batch %d/%d]\tMSE_Loss: %.4f\tDiversity_Loss: %.4f\tConsistency_Loss: %.4f\tTotal_Loss: %.4f\n'
                  % (epoch, n_epochs, i, len(dataloader),
                     mse_loss.item(), div_loss.item(), cons_loss.item(), total_loss.item()))

            logging.info(
                "[Epoch %d/%d] [Batch %d/%d] [MSE_Loss: %.4f] [Diversity_Loss: %.4f] [Consistency_Loss: %.4f] [Total_Loss: %.4f]"
                % (epoch, n_epochs, i, len(dataloader),
                   mse_loss.item(), div_loss.item(), cons_loss.item(), total_loss.item())
            )

            losses_mse.append(mse_loss.item())
            losses_div.append(div_loss.item())
            losses_cons.append(cons_loss.item())
            losses_total.append(total_loss.item())

            plot_losses(losses_mse, losses_div, losses_cons, losses_total, os.path.join(result_path, 'losses.png'))

    iters += 1

def run():
    model = StableDiffusion(in_channels=3, latent_dim=4, image_size=256, diffusion_timesteps=1000, device=device)
    model.to(device)

    # Create data loaders
    train_dataloader = load_image(dataroot_train, image_size, batch_size, workers)

    # Set optimizer and scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=1e-4, epochs=n_epochs, steps_per_epoch=len(train_dataloader))

    # Create result_dir
    os.makedirs(result_path, exist_ok=True)

    iters = 0
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Updated consistency loss weights
        current_lambda_cons = min(lambda_cons * (epoch + 1) / epochs_to_max_lambda, max_lambda_cons)

        # Train model
        train(model, device, train_dataloader, optimizer, scheduler, epoch, iters, diversity_weight, epoch_loss, num_batches, current_lambda_cons)

        if epoch % 20 == 0:
            model.eval()

            image_fixed = [f'{dataroot_test}/test/{i}.jpg' for i in random_indices]
            image_tensors = []
            for img_path in image_fixed:
                img = Image.open(img_path)
                img_tensor = ToTensor()(img)
                image_tensors.append(img_tensor)
            real_images = torch.stack(image_tensors).to(device)

            gene_expression_val = load_gene_expression(gx_test_path)

            # Generate validation image
            with torch.no_grad():
                gene_expression_fixed = np.array([gene_expression_val[i] for i in random_indices])
                tensor_gene = torch.tensor(gene_expression_fixed, dtype=torch.float32, device=device).view(len(random_indices), -1).unsqueeze(1)
                sampled_latents = model.sample(tensor_gene, latent_size=latent_size, batch_size=len(random_indices), guidance_scale=5, device=device)
                fake_images = model.decode(sampled_latents).squeeze(1)

                if not os.path.exists(f'{result_path}/generated_image'):
                    os.makedirs(f'{result_path}/generated_image')

            stacked_images = torch.stack((real_images[:len(random_indices)], fake_images[:len(random_indices)]), dim=1)
            mixed_images = stacked_images.view(-1, *real_images.shape[1:])
            mixed_images = F.interpolate(mixed_images, size=(256, 256), mode='bilinear', align_corners=False)
            grid = vutils.make_grid(mixed_images, padding=2, normalize=True, scale_each=True)
            save_image(grid, f'{result_path}/generated_image/generated_img_epoch_{epoch}.png')

        # Save params in checkpoint
        if (epoch + 1) % save_checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': epoch_loss,
            }, f'{result_path}/stable_diffusion_model_checkpoint_epoch_{epoch+1}.pth')

    torch.save(model.state_dict(), f'{result_path}/stable_diffusion_model_final.pth')

if __name__ == "__main__":
    run()