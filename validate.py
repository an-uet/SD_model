import torch
from torchvision.transforms import ToTensor
from PIL import Image
from stable_diffusion_model import StableDiffusion
from torchvision.utils import save_image
from utils import load_gene_expression

# Load the model on CPU
vae_path = 'vae_model.pth'
diffusion_path = '/home/anlt69/Desktop/ST_SIM/SD_model_7Nov2024/SD_7Nov2024/Results/stable_diffusion_model_checkpoint_epoch_250.pth'
model = StableDiffusion(in_channels=3, latent_dim=4, image_size=256, diffusion_timesteps=1000, device='cpu')
model.load_vae(vae_path)
model.load_diffusion(diffusion_path)
model.to('cpu')
model.eval()

# Load gene expression data
gene_expression_data = load_gene_expression('data/gene_expression/data_validation.csv')
gene_expression_tensor = torch.tensor(gene_expression_data, dtype=torch.float32).to('cpu').unsqueeze(1)

# Load and preprocess validation images
image_paths = ['data/image/spots_validation/validation/' + str(i) + '.jpg' for i in range(500)]
image_tensors = []
for img_path in image_paths:
    img = Image.open(img_path)
    img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')  # Convert to tensor and add batch dimension
    image_tensors.append(img_tensor)

# Perform predictions
with torch.no_grad():
    for i, img_tensor in enumerate(image_tensors):
        latents = model.encode(img_tensor)
        sampled_latents = model.sample(gene_expression_tensor, latent_size=64, batch_size=1, guidance_scale=3.0, device='cpu')
        fake_image = model.decode(sampled_latents).squeeze(0)  # Remove batch dimension

        # Save or display the fake_image as needed
        save_image(fake_image, f'result/generated_image_{i}.png')