# Defining the Training Function
import ast
from math import ceil
from typing import List
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import torchvision.datasets as dset
from torchvision import datasets, transforms

from model_config_v0 import dataroot_train, image_size, batch_size, workers

class GeneExpressionImageDataset(Dataset):
    def __init__(self, gene_expressions, image_paths, transform=None):
        """
        Args:
            gene_expressions (list or array): Array-like list of gene expression vectors.
            image_paths (list): List of file paths to corresponding images.
            transform (callable, optional): Transform to apply to images.
        """
        self.gene_expressions = gene_expressions
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.gene_expressions)

    def __getitem__(self, idx):
        gene_expression = self.gene_expressions[idx]

        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)

        gene_expression = torch.tensor(gene_expression, dtype=torch.float32)

        return gene_expression, image

def load_gene_expression(data_path='data.csv'):
    df = pd.read_csv(data_path)
    gene_expression = df['gene_expression']
    new_gene_expression = []

    for i in range(len(gene_expression)):
        try:
            row = ast.literal_eval(gene_expression[i])
            new_gene_expression.append(row)
        except Exception as e:
            print('error : ', e)
    gene_expression = np.array(new_gene_expression)

    return gene_expression


def load_image(dataroot, image_size, batch_size, workers):
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=workers, drop_last=True)
    return dataloader

def load_pair_gx_image(gene_expression: List[List[float]], image_paths: List[str]):
    dataset= GeneExpressionImageDataset(gene_expressions=gene_expression, image_paths=image_paths,
                                        transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))
    dataloader = DataLoader(dataset, batch_size=batch_size,
                         shuffle=True, num_workers=workers, drop_last=True)
    return dataloader


def plot_losses(losses_mse, losses_div, losses_cons, losses_total, filename):
    # 创建 2x2 的子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 绘制 MSE 损失
    axes[0, 0].plot(losses_mse, label='MSE Loss', color='blue')
    axes[0, 0].set_title("MSE Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 绘制 DIV 损失
    axes[0, 1].plot(losses_div, label='DIV Loss', color='orange')
    axes[0, 1].set_title("DIV Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 绘制 CONS 损失
    axes[1, 0].plot(losses_cons, label='CONS Loss', color='green')
    axes[1, 0].set_title("CONS Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 绘制 TOTAL 损失
    axes[1, 1].plot(losses_total, label='Total Loss', color='red')
    axes[1, 1].set_title("Total Loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 保存图像
    plt.savefig(filename)
    plt.close(fig)  # 关闭图像以释放内存