import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = 256

latent_size = 32 

n_epochs = 500

batch_size = 128

lr = 2e-5

num_timesteps = 1000

save_checkpoint_interval = 50

diversity_weight = 100

max_div_weight = 500

lambda_cons = 0.1  

max_lambda_cons = 1.0  

epochs_to_max_lambda = n_epochs  

workers = 2

random_indices = [609, 852, 296, 856, 86, 689, 62, 44, 527, 113, 285, 283, 660, 741, 293, 608,
                 280, 294, 201, 36, 326, 670, 795, 796, 331, 920, 703, 569, 406, 15, 177, 715]
# random_indices = [609]
# random_indices = [609, 852, 296, 856, 86, 689, 62, 44]

data_path = '/data/sunpengyu/Task_STsim/Data/Data_24Oct2024'
sd_path = '/data/sunpengyu/Task_STsim/Model/STsim_Diffusion'
result_path = '/data/sunpengyu/Task_STsim/Results/SD_9Nov2024_1'
dataroot_train = f'{data_path}/image/spots_train'
dataroot_test = f'{data_path}/image/spots_test'
gx_train_path = f'{data_path}/gene_expression/data_train.csv'
gx_test_path = f'{data_path}/gene_expression/data_test.csv'
