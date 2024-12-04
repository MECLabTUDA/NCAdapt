
# %%
import torch
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.datasets.Nii_Gz_Dataset_3D_customPath import Dataset_NiiGz_3D_customPath
from src.models.Model_M3DNCA import M3DNCA_CL_new_module_dict_new_p0s
from src.losses.LossFunctions import DiceFocalLoss
from src.utils.Experiment import Experiment
from src.agents.Agent_M3DNCA_Simple import M3DNCAAgent
import time
import os
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

config = [{
    # 'img_path': r"/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_raw_data/Task99_HarP/imagesTr/",
    # 'label_path': r"/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_raw_data/Task99_HarP/labelsTr/",
    # 'name': r'M3D_NCA_Hyp99_CL_adjust_dict_new_p0s_rwalk', #12 or 13, 54 opt,

    'img_path': r"/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_raw_data/Task98_Dryad/imagesTr/",
    'label_path': r"/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_raw_data/Task98_Dryad/labelsTr/",
    'name': r'M3D_NCA_Hyp99_98_CL_adjust_dict_new_p0s_rwalk', #12 or 13, 54 opt, 
    'pretrained':  r'M3D_NCA_Hyp99_CL_adjust_dict_new_p0s_rwalk',

    # 'img_path': r"/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_raw_data/Task97_DecathHip/imagesTr/",
    # 'label_path': r"/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_raw_data/Task97_DecathHip/labelsTr/",
    # 'name': r'M3D_NCA_Hyp99_98_97_CL_new_p0s_rwalk', #12 or 13, 54 opt, 
    # 'pretrained':  r'M3D_NCA_Hyp99_98_CL_adjust_dict_new_p0s_rwalk',

    'device':"cuda:0",
    'unlock_CPU': True,
    # Optimizer
    'lr': 16e-4,
    'lr_gamma': 0.9999,#0.9999,
    'betas': (0.9, 0.99),
    # Training
    'save_interval': 50,
    'evaluate_interval': 50,
    'n_epoch': 500,
    'batch_duplication': 1,
    # Model
    'channel_n': 16,        # Number of CA state channels
    'inference_steps': [5, 10],
    'cell_fire_rate': 0.5,
    'batch_size': 20,
    'input_channels': 1,
    'output_channels': 1,
    'hidden_size': 64,
    'train_model':1,
    # Data
    'input_size': [(16, 16, 12), (64, 64, 48)] ,    #[(9, 12, 10), (36, 48, 40)]
    'scale_factor': 4,
    'data_split': [0.8, 0, 0.2], 
    'keep_original_scale': False,
    'rescale': True,
    'nr_modules': 3,
}
]

dataset = Dataset_NiiGz_3D(store=True)
device = torch.device(config[0]['device'])
ca1 = M3DNCA_CL_new_module_dict_new_p0s(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], kernel_size=7, input_channels=config[0]['input_channels'], output_channels=config[0]['output_channels'], levels=2, scale_factor=4, steps=20, nr_modules=config[0]['nr_modules']).to(device)
ca1._freeze_model_full()
ca1._un_freeze_modules('1')
# Select now, which model is used during train:
ca1._use_module('1')

ca = ca1
agent = M3DNCAAgent(ca)
exp = Experiment(config, dataset, ca, agent)
dataset.set_experiment(exp)
exp.set_model_state('train')
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

loss_function = DiceFocalLoss() 

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
fisher, params, scores = agent.load_fisher_params_scores_rwalk()
agent._initalize_rwalk(fisher, params, scores, 0.4, '98')
agent.train(data_loader, loss_function)
agent.after_train_rwalk('98', data_loader, loss_function)

### EVAL DATASETS
# Use the first head for first task, second head for second and third task as we haven't trained on the third one yet
#agent.getAverageDiceScore(pseudo_ensemble=False)
print("--------------- TESTING HYP 99 ---------------")
hyp99_test = Dataset_NiiGz_3D_customPath(resize=True, size=(64, 64, 48), imagePath=r"/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_raw_data/Task99_HarP/imagesTs", labelPath=r"/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_raw_data/Task99_HarP/labelsTs")
hyp99_test.exp = exp
agent.model._use_module('0')
agent.getAverageDiceScore(pseudo_ensemble=False, dataset=hyp99_test)
agent.getAverageDiceScore(pseudo_ensemble=False, dataset=hyp99_test)

print("--------------- TESTING HYP 98 ---------------")
hyp99_test = Dataset_NiiGz_3D_customPath(resize=True, size=(64, 64, 48), imagePath=r"/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_raw_data/Task98_Dryad/imagesTs", labelPath=r"/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_raw_data/Task98_Dryad/labelsTs")
hyp99_test.exp = exp
agent.model._use_module('1')
agent.getAverageDiceScore(pseudo_ensemble=False, dataset=hyp99_test)
agent.getAverageDiceScore(pseudo_ensemble=False, dataset=hyp99_test)

print("--------------- TESTING HYP 97 ---------------")
hyp99_test = Dataset_NiiGz_3D_customPath(resize=True, size=(36, 48, 40), imagePath=r"/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_raw_data/Task97_DecathHip/imagesTr", labelPath=r"/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_raw_data/Task97_DecathHip/labelsTr")
hyp99_test.exp = exp
agent.model._use_module('1')
agent.getAverageDiceScore(pseudo_ensemble=False, dataset=hyp99_test)
agent.getAverageDiceScore(pseudo_ensemble=False, dataset=hyp99_test)



start_time = time.perf_counter()
#agent.getAverageDiceScore(pseudo_ensemble=False)
end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"The function took {elapsed_time} seconds to execute.")
