
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
    # 'img_path': r"/media/aranem_locale/AR_subs_exps/WACV_2025/WACV_2025_raw_data/Task99_HarP/imagesTr/",
    # 'label_path': r"/media/aranem_locale/AR_subs_exps/WACV_2025/WACV_2025_raw_data/Task99_HarP/labelsTr/",
    # 'name': r'UNet_Hyp99_ewc', #12 or 13, 54 opt,

    # 'img_path': r"/media/aranem_locale/AR_subs_exps/WACV_2025/WACV_2025_raw_data/Task98_Dryad/imagesTr/",
    # 'label_path': r"/media/aranem_locale/AR_subs_exps/WACV_2025/WACV_2025_raw_data/Task98_Dryad/labelsTr/",
    # 'name': r'UNet_Hyp99_98_ewc', #12 or 13, 54 opt, 
    # 'pretrained':  r'UNet_Hyp99_ewc',

    'img_path': r"/media/aranem_locale/AR_subs_exps/WACV_2025/WACV_2025_raw_data/Task97_DecathHip/imagesTr/",
    'label_path': r"/media/aranem_locale/AR_subs_exps/WACV_2025/WACV_2025_raw_data/Task97_DecathHip/labelsTr/",
    'name': r'UNet_Hyp99_98_97_ewc', #12 or 13, 54 opt, 
    'pretrained':  r'UNet_Hyp99_98_ewc',

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
    # Data  (64, 64, 48)
    'input_size': [(16, 16, 12), (64, 64, 48)] ,    #[(9, 12, 10), (36, 48, 40)]
    'data_split': [0.8, 0, 0.2]
}
]

dataset = Dataset_NiiGz_3D(store=True)
device = torch.device(config[0]['device'])
ca = UNet3D(in_channels=1, padding=1, out_classes=1).to(device)
agent = CLUNetAgent(ca)
exp = Experiment(config, dataset, ca, agent)
dataset.set_experiment(exp)
exp.set_model_state('train')
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

loss_function = DiceBCELoss() 

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Number of parameters
print(sum(p.numel() for p in ca.parameters() if p.requires_grad))

fisher, params = agent.load_fisher_params_ewc()
agent._initialize_ewc(fisher, params, '0.4')
agent.train(data_loader, loss_function)
agent.after_train_ewc('97', data_loader, loss_function)

### EVAL DATASETS
#agent.getAverageDiceScore(pseudo_ensemble=False)
print("--------------- TESTING HYP 99 ---------------")
hyp99_test = Dataset_NiiGz_3D_customPath(resize=True, size=(64, 64, 48), imagePath=r"/media/aranem_locale/AR_subs_exps/WACV_2025/WACV_2025_raw_data/Task99_HarP/imagesTs", labelPath=r"/media/aranem_locale/AR_subs_exps/WACV_2025/WACV_2025_raw_data/Task99_HarP/labelsTs")
hyp99_test.exp = exp
agent.getAverageDiceScore(pseudo_ensemble=False, dataset=hyp99_test)
agent.getAverageDiceScore(pseudo_ensemble=False, dataset=hyp99_test)

print("--------------- TESTING HYP 98 ---------------")
hyp99_test = Dataset_NiiGz_3D_customPath(resize=True, size=(64, 64, 48), imagePath=r"/media/aranem_locale/AR_subs_exps/WACV_2025/WACV_2025_raw_data/Task98_Dryad/imagesTs", labelPath=r"/media/aranem_locale/AR_subs_exps/WACV_2025/WACV_2025_raw_data/Task98_Dryad/labelsTs")
hyp99_test.exp = exp
agent.getAverageDiceScore(pseudo_ensemble=False, dataset=hyp99_test)
agent.getAverageDiceScore(pseudo_ensemble=False, dataset=hyp99_test)

print("--------------- TESTING HYP 97 ---------------")
hyp99_test = Dataset_NiiGz_3D_customPath(resize=True, size=(64, 64, 48), imagePath=r"/media/aranem_locale/AR_subs_exps/WACV_2025/WACV_2025_raw_data/Task97_DecathHip/imagesTr", labelPath=r"/media/aranem_locale/AR_subs_exps/WACV_2025/WACV_2025_raw_data/Task97_DecathHip/labelsTr")

hyp99_test.exp = exp
agent.getAverageDiceScore(pseudo_ensemble=False, dataset=hyp99_test)
agent.getAverageDiceScore(pseudo_ensemble=False, dataset=hyp99_test)



start_time = time.perf_counter()
#agent.getAverageDiceScore(pseudo_ensemble=False)
end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"The function took {elapsed_time} seconds to execute.")
