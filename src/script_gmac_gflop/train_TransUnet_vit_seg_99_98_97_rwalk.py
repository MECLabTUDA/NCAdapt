
# %%
from unet import UNet3D
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.datasets.Nii_Gz_Dataset_3D_customPath import Dataset_NiiGz_3D_customPath
from src.utils.Experiment import Experiment
import torch
from src.losses.LossFunctions import DiceBCELoss
from src.agents.Agent_TransUNet import TransUNetAgent
import time
import numpy as np
import argparse
from src.models.vit_seg_modeling import VisionTransformer as ViT_seg
from src.models.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from ptflops import get_model_complexity_info
import re

config = [{
    'img_path': r"/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_raw_data/Task97_DecathHip/imagesTr/",
    'label_path': r"/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_raw_data/Task97_DecathHip/labelsTr/",
    'name': r'TransUNet_Hyp99_98_97_rwalk', #12 or 13, 54 opt, 
    'pretrained':  r'TransUNet_Hyp99_98_rwalk',
    'device':"cuda:0",
    # Learning rate
    'lr': 1e-4,
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    # Training config
    'save_interval': 50,
    'evaluate_interval': 25,
    'n_epoch': 250,
    'batch_size': 16,
    # Data
    'input_size': (64, 64), # (384, 384)
    'data_split': [0.8, 0, 0.2], 

}]

# Define Experiment
dataset = Dataset_NiiGz_3D(slice=2)
device = torch.device(config[0]['device'])
#ca = TransUNet(img_dim=320, in_channels=1, out_channels=128, head_num=4, mlp_dim=512, block_num=8, patch_dim=16, class_num=1).to(device) #in_channels=1, padding=1, out_classes=1

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=64, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()

config_vit = CONFIGS_ViT_seg[args.vit_name]
config_vit.n_classes = args.num_classes
config_vit.n_skip = args.n_skip
if args.vit_name.find('R50') != -1:
    config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
net.load_from(weights=np.load(r"/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_raw_data/TransUNet_weights/imagenet21k_R50+ViT-B_16.npz"))#config_vit.pretrained_path))

# Load TransUNet weights

agent = TransUNetAgent(net)
exp = Experiment(config, dataset, net, agent)
exp.set_model_state('train')
dataset.set_experiment(exp)
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))
loss_function = DiceBCELoss() 

# Number of parameters
print("Nr. Params.: ", sum(p.numel() for p in net.parameters() if p.requires_grad))

fisher, params, scores = agent.load_fisher_params_scores_rwalk()
agent._initalize_rwalk(fisher, params, scores, 0.4, '97')
# agent.train(data_loader, loss_function)

########################################################
# -- To calculate GMAC and GFLOP -- #
macs, params = get_model_complexity_info(net, (1, 64, 64), as_strings=True,
                                         print_per_layer_stat=True, verbose=True)
# Extract the numerical value
flops = eval(re.findall(r'([\d.]+)', macs)[0])*2

# Extract the unit
flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]
print('Computational complexity: {:<8}'.format(macs))
print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
print('Number of parameters: {:<8}'.format(params))
raise
########################################################

agent.after_train_rwalk('97', data_loader, loss_function)

start_time = time.perf_counter()
### EVAL DATASETS
#agent.getAverageDiceScore(pseudo_ensemble=False)
print("--------------- TESTING HYP 99 ---------------")
hyp99_test = Dataset_NiiGz_3D_customPath(resize=True, slice=2, size=(64, 64, 48), imagePath=r"/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_raw_data/Task99_HarP/imagesTs", labelPath=r"/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_raw_data/Task99_HarP/labelsTs")
hyp99_test.exp = exp
agent.getAverageDiceScore(pseudo_ensemble=False, dataset=hyp99_test)
agent.getAverageDiceScore(pseudo_ensemble=False, dataset=hyp99_test)

print("--------------- TESTING HYP 98 ---------------")
hyp99_test = Dataset_NiiGz_3D_customPath(resize=True, slice=2, size=(64, 64, 48), imagePath=r"/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_raw_data/Task98_Dryad/imagesTs", labelPath=r"/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_raw_data/Task98_Dryad/labelsTs")
hyp99_test.exp = exp
agent.getAverageDiceScore(pseudo_ensemble=False, dataset=hyp99_test)
agent.getAverageDiceScore(pseudo_ensemble=False, dataset=hyp99_test)

print("--------------- TESTING HYP 97 ---------------")
hyp99_test = Dataset_NiiGz_3D_customPath(resize=True, slice=2, size=(64, 64, 48), imagePath=r"/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_raw_data/Task97_DecathHip/imagesTr", labelPath=r"/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_raw_data/Task97_DecathHip/labelsTr")
hyp99_test.exp = exp
agent.getAverageDiceScore(pseudo_ensemble=False, dataset=hyp99_test)
agent.getAverageDiceScore(pseudo_ensemble=False, dataset=hyp99_test)


# agent.getAverageDiceScore(pseudo_ensemble=False)
end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"The function took {elapsed_time} seconds to execute.")


