
# %%
import torch
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.datasets.Nii_Gz_Dataset_3D_customPath import Dataset_NiiGz_3D_customPath
from unet import UNet3D
from src.losses.LossFunctions import DiceBCELoss
from src.utils.Experiment import Experiment
from src.agents.Agent_CL_UNet import CLUNetAgent
import time
import os
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

config = [{
    # 'img_path': r"/media/aranem_locale/AR_subs_exps/WACV_2025_NCAdapt/WACV_2025_raw_data/Task89_mHeartA/imagesTr/",
    # 'label_path': r"/media/aranem_locale/AR_subs_exps/WACV_2025_NCAdapt/WACV_2025_raw_data/Task89_mHeartA/labelsTr/",
    # 'name': r'UNet_Heart89_agem', #12 or 13, 54 opt,

    'img_path': r"/media/aranem_locale/AR_subs_exps/WACV_2025_NCAdapt/WACV_2025_raw_data/Task88_mHeartB/imagesTr/",
    'label_path': r"/media/aranem_locale/AR_subs_exps/WACV_2025_NCAdapt/WACV_2025_raw_data/Task88_mHeartB/labelsTr/",
    'name': r'UNet_Heart89_88_agem', #12 or 13, 54 opt, 
    'pretrained':  r'UNet_Heart89_agem',

    'device':"cuda:0",
    # Learning rate
    'lr': 1e-4,
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    # Training config
    'save_interval': 250,
    'evaluate_interval': 501,
    'n_epoch': 500,
    'batch_size': 1,
    # Data
    'input_size': [(64, 64, 3), (256, 256, 12)] ,   #[(16, 16, 12), (64, 64, 48)] #[(9, 12, 10), (36, 48, 40)]
    'data_split': [0.8, 0, 0.2]
}
]

dataset = Dataset_NiiGz_3D(store=True)
device = torch.device(config[0]['device'])
ca = UNet3D(in_channels=1, padding=1, out_classes=1, num_encoding_blocks = 3).to(device)
agent = CLUNetAgent(ca)
exp = Experiment(config, dataset, ca, agent)
exp.agent.cl_m = 'AGem'
dataset.set_experiment(exp)
exp.set_model_state('train')
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

loss_function = DiceBCELoss()
exp.agent.loss_ = loss_function

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Number of parameters
print(sum(p.numel() for p in ca.parameters() if p.requires_grad))

agent.train(data_loader, loss_function)
agent.end_task_AGem(data_loader)


### EVAL DATASETS
#agent.getAverageDiceScore(pseudo_ensemble=False)
print("--------------- TESTING 89 ---------------")
Heart99_test = Dataset_NiiGz_3D_customPath(resize=True, size=(256, 256, 12), imagePath=r"/media/aranem_locale/AR_subs_exps/WACV_2025_NCAdapt/WACV_2025_raw_data/Task89_mHeartA/imagesTs", labelPath=r"/media/aranem_locale/AR_subs_exps/WACV_2025_NCAdapt/WACV_2025_raw_data/Task89_mHeartA/labelsTs")
Heart99_test.exp = exp
agent.getAverageDiceScore(pseudo_ensemble=False, dataset=Heart99_test)
agent.getAverageDiceScore(pseudo_ensemble=False, dataset=Heart99_test)

print("--------------- TESTING 88 ---------------")
Heart99_test = Dataset_NiiGz_3D_customPath(resize=True, size=(256, 256, 12), imagePath=r"/media/aranem_locale/AR_subs_exps/WACV_2025_NCAdapt/WACV_2025_raw_data/Task88_mHeartB/imagesTs", labelPath=r"/media/aranem_locale/AR_subs_exps/WACV_2025_NCAdapt/WACV_2025_raw_data/Task88_mHeartB/labelsTs")
Heart99_test.exp = exp
agent.getAverageDiceScore(pseudo_ensemble=False, dataset=Heart99_test)
agent.getAverageDiceScore(pseudo_ensemble=False, dataset=Heart99_test)


start_time = time.perf_counter()
#agent.getAverageDiceScore(pseudo_ensemble=False)
end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"The function took {elapsed_time} seconds to execute.")
