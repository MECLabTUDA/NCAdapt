import nibabel as nib
import numpy as np
import os
import torch
import torch.optim as optim
from src.utils.helper import convert_image, merge_img_label_gt
from src.losses.LossFunctions import DiceLoss
from src.utils.Experiment import Experiment
import seaborn as sns
import math
from matplotlib import figure
from torch.utils.data import DataLoader
from tqdm import tqdm
from codecarbon import track_emissions
import torch.nn as nn

class BaseAgent():
    """Base class for all agents. Handles basic training and only needs to be adapted if special use cases are necessary.
    
    .. note:: In many cases only the data preparation and outputs need to be changed."""
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def set_exp(self, exp: Experiment) -> None:
        r"""Set experiment of agent and initialize.
            #Args
                exp (Experiment): Experiment class"""
        self.exp = exp
        self.initialize()

    def initialize(self):
        r"""Initialize agent with optimizers and schedulers
        """
        self.device = torch.device(self.exp.get_from_config('device'))
        self.batch_size = self.exp.get_from_config('batch_size')
        # If stacked NCAs
        if isinstance(self.model, list):
            self.optimizer = []
            self.scheduler = []
            for m in range(len(self.model)):
                self.optimizer.append(optim.Adam(self.model[m].parameters(), lr=self.exp.get_from_config('lr'), betas=self.exp.get_from_config('betas')))
                #print("SGD")
                #self.optimizer.append(optim.AdamW(self.model[m].parameters(), lr=self.exp.get_from_config('lr'), betas=self.exp.get_from_config('betas')))
                #self.optimizer.append(optim.SGD(self.model[m].parameters(), lr=self.exp.get_from_config('lr')))
                self.scheduler.append(optim.lr_scheduler.ExponentialLR(self.optimizer[m], self.exp.get_from_config('lr_gamma')))
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.exp.get_from_config('lr'), betas=self.exp.get_from_config('betas'))
            #self.optimizer = optim.SGD(self.model.parameters(), lr=self.exp.get_from_config('lr'))
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.exp.get_from_config('lr_gamma'))

    def printIntermediateResults(self, loss: torch.Tensor, epoch: int) -> None:
        r"""Prints intermediate results of training and adds it to tensorboard
            #Args 
                loss (torch)
                epoch (int) 
        """
        print(epoch, "loss =", loss.item())
        self.exp.save_model()
        self.exp.write_scalar('Loss/train', loss, epoch)

    def prepare_data(self, data: list, eval: bool = False) -> list:
        r"""If any data preparation needs to be done do it here. 
            #Args
                data ([]): The data to be processed.
                eval (Bool): Whether or not its for evaluation. 
        """
        return data

    def get_outputs(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Get the output of the model.
            #Args 
                data (torch): The data to be passed to the model.
        """
        return self.model(data)

    def initialize_epoch(self) -> None:
        r"""Everything that should happen once before each epoch should be defined here.
        """
        return

    def conclude_epoch(self) -> None:
        r"""Everything that should happen once after each epoch should be defined here.
        """
        return

    def batch_step(self, data: tuple, loss_f: torch.nn.Module, gradient_norm: bool = False) -> dict:
        r"""Execute a single batch training step
            #Args
                data (tensor, tensor): inputs, targets
                loss_f (torch.nn.Module): loss function
            #Returns:
                loss item
        """
        data = self.prepare_data(data)
        outputs, targets = self.get_outputs(data)
        self.optimizer.zero_grad()
        loss = 0
        loss_ret = {}
        #print(outputs.shape, targets.shape)
        if len(outputs.shape) == 5:
            for m in range(targets.shape[-1]):
                loss_loc = loss_f(outputs[..., m], targets[...])
                loss = loss + loss_loc
                loss_ret[m] = loss_loc.item()
        else:
            for m in range(targets.shape[-1]):
                if 1 in targets[..., m]:
                    loss_loc = loss_f(outputs[..., m], targets[..., m])
                    loss = loss + loss_loc
                    loss_ret[m] = loss_loc.item()

        try:
            loss += self.exp.agent.penalty_weight * self.exp.agent.penalty()
            # Method exists and was used.  
        except AttributeError:
            # Method does not exist; What now?
            print("No penalty method for this experiment/agent.")

        if loss != 0:
            loss.backward()

            if gradient_norm:
                max_norm = 1.0
                # Gradient normalization
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                # Calculate scaling factor and scale gradients if necessary
                scale_factor = max_norm / (total_norm + 1e-6)
                if scale_factor < 1:
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad.data.mul_(scale_factor)

            nn.utils.clip_grad.clip_grad_value_(self.exp.agent.model.parameters(), 1)
            self.optimizer.step()
            self.scheduler.step()

            try:
                self.after_loss(self.exp.config['lr'])
            except AttributeError:
                # Method does not exist; What now?
                print("No after loss function for this experiment/agent.")
                
        return loss_ret

    def intermediate_results(self, epoch: int, loss_log: list) -> None:
        r"""Write intermediate results to tensorboard
            #Args
                epoch (int): Current epoch
                los_log ([loss]): Array of losses
        """
        for key in loss_log.keys():
            if len(loss_log[key]) != 0:
                average_loss = sum(loss_log[key]) / len(loss_log[key])
            else:
                average_loss = 0
            print(epoch, "loss =", average_loss)
            self.exp.write_scalar('Loss/train/' + str(key), average_loss, epoch)

    def plot_results_byPatient(self, loss_log: dict) -> figure:
        r"""Plot losses in a per patient fashion with seaborn to display in tensorboard.
            #Args
                loss_log ({name: loss}: Dictionary of losses
        """
        print(loss_log)
        sns.set_theme()
        plot = sns.scatterplot(x=loss_log.keys(), y=loss_log.values())
        plot.set(ylim=(0, 1))
        plot = plot.get_figure()
        return plot

    def intermediate_evaluation(self, dataloader, epoch: int) -> None:
        r"""Do an intermediate evluation during training 
            .. todo:: Make variable for more evaluation scores (Maybe pass list of metrics)
            #Args
                dataset (Dataset)
                epoch (int)
        """
        diceLoss = DiceLoss(useSigmoid=True)
        loss_log = self.test(diceLoss)
        if loss_log is not None:
            for key in loss_log.keys():
                img_plot = self.plot_results_byPatient(loss_log[key])
                self.exp.write_figure('Patient/dice/mask' + str(key), img_plot, epoch)
                if len(loss_log[key]) > 0:
                    self.exp.write_scalar('Dice/test/mask' + str(key), sum(loss_log[key].values())/len(loss_log[key]), epoch)
                    self.exp.write_histogram('Dice/test/byPatient/mask' + str(key), np.fromiter(loss_log[key].values(), dtype=float), epoch)
        param_lst = []
        # TODO: ADD AGAIN 
        #for param in self.model.parameters():
        #    param_lst.extend(np.fromiter(param.flatten(), dtype=float))
        #self.exp.write_histogram('Model/weights', np.fromiter(param_lst, dtype=float), epoch)

    def getAverageDiceScore(self, useSigmoid: bool = True, tag: str = "", pseudo_ensemble: bool = False, dataset = None) -> dict:
        r"""Get the average Dice test score.
            #Returns:
                return (float): Average Dice score of test set. """
        diceLoss = DiceLoss(useSigmoid=useSigmoid)
        loss_log = self.test(diceLoss, save_img=None, pseudo_ensemble=pseudo_ensemble, dataset = dataset, inference=True)

        return loss_log

    def predictOnPath(self, path: str, useSigmoid: bool = True, tag: str = "", pseudo_ensemble: bool = False) -> dict:
        r"""Get the average Dice test score.
            #Returns:
                return (float): Average Dice score of test set. """
        diceLoss = DiceLoss(useSigmoid=useSigmoid)
        loss_log = self.test(diceLoss, save_img=[], pseudo_ensemble=pseudo_ensemble)

        return loss_log

    def save_state(self, model_path: str) -> None:
        r"""Save state of current model
        """
        os.makedirs(model_path, exist_ok=True)
        # self.model.save(os.path.join(model_path, 'model.pth'))
        torch.save(self.model.state_dict(), os.path.join(model_path, 'model.pth'))
        torch.save(self.optimizer.state_dict(), os.path.join(model_path, 'optimizer.pth'))
        torch.save(self.scheduler.state_dict(), os.path.join(model_path, 'scheduler.pth'))

    def load_state(self, model_path: str) -> None:
        r"""Load state of current model
        """
        # self.model.load(os.path.join(model_path, 'model.pth'))
        self.model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth')))
        self.optimizer.load_state_dict(torch.load(os.path.join(model_path, 'optimizer.pth')))
        self.scheduler.load_state_dict(torch.load(os.path.join(model_path, 'scheduler.pth')))

    @track_emissions()
    def train(self, dataloader: DataLoader, loss_f: torch.Tensor) -> None:
        r"""Execute training of model
            #Args
                dataloader (Dataloader): contains training data
                loss_f (nn.Model): The loss for training"""
        dataloader_ = iter(dataloader)   # NOTE: remove if we go back to for i, data loop!
        for epoch in range(self.exp.currentStep, self.exp.get_max_steps()+1):
            print("Epoch: " + str(epoch))
            self.exp.set_model_state('train')
            loss_log = {}
            for m in range(self.output_channels):
                loss_log[m] = []
            self.initialize_epoch()
            print('Dataset size: ' + str(len(dataloader)))
            for _ in tqdm(range(10)):
                try:
                    data = next(dataloader_)
                except StopIteration:
                    dataloader_ = iter(dataloader)
                    data = next(dataloader_)
            # for i, data in enumerate(tqdm(dataloader)):
                loss_item = self.batch_step(data, loss_f)

                for key in loss_item.keys():
                    if isinstance(loss_item[key], float):
                        loss_log[key].append(loss_item[key])
                    else:
                        loss_log[key].append(loss_item[key].detach())
            self.intermediate_results(epoch, loss_log)
            if epoch % self.exp.get_from_config('evaluate_interval') == 0:
                print("Evaluate model")
                self.intermediate_evaluation(dataloader, epoch)
            #if epoch % self.exp.get_from_config('ood_interval') == 0:
            #    print("Evaluate model in OOD cases")
            #    self.ood_evaluation(epoch=epoch)
            if epoch % self.exp.get_from_config('save_interval') == 0:
                print("Model saved")
                self.save_state(os.path.join(self.exp.get_from_config('model_path'), 'models', 'epoch_' + str(self.exp.currentStep)))
            self.conclude_epoch()
            self.exp.increase_epoch()

    def prepare_image_for_display(self, image: torch.Tensor) -> torch.Tensor:
        r"""Prepare an image to be displayed in tensorboard. Since images need to be in a specific format these modifications these can be done here.
            #Args
                image (torch): The image to be processed for display. 
        """
        return image


    #def ood_evaluation(self, ood_cases=["random_noise", "random_spike", "random_anitrosopy"], epoch=0):
    #    print("OOD EVALUATION")
    #    dataset_train = self.exp.dataset
    #    diceLoss = DiceLoss(useSigmoid=True)
    #    for augmentation in ood_cases:
    #        dataset_eval = Nii_Gz_Dataset(aug_type=augmentation)
    #        self.exp.dataset = dataset_eval
    #        loss_log = self.test(diceLoss, tag='ood/' + str(augmentation) + '/')
    #        for key in loss_log.keys():
    #            self.exp.write_scalar('ood/Dice/' + str(key) + ", " + str(augmentation), sum(loss_log[key].values())/len(loss_log[key]), epoch)
    #            self.exp.write_histogram('ood/Dice/' + str(key) + ", " + str(augmentation) + '/byPatient', np.fromiter(loss_log[key].values(), dtype=float), epoch)
    #    self.exp.dataset = dataset_train


    def labelVariance(self, images: torch.Tensor, median: torch.Tensor, img_mri: torch.Tensor, img_id: str, targets: torch.Tensor) -> None:
        r"""Calculate variance over all predictions
            #Args
                images (torch): The inferences
                median: The median of all inferences
                img_mri: The mri image
                img_id: The id of the image
                targets: The target segmentation
        """
        mean = np.sum(images, axis=0) / images.shape[0]
        stdd = 0
        for id in range(images.shape[0]):
            img = images[id] - mean
            img = np.power(img, 2)
            stdd = stdd + img
        stdd = stdd / images.shape[0]
        stdd = np.sqrt(stdd)

        print("NQM Score: ", np.sum(stdd) / np.sum(median))

        # Save files refactor
        if False:
        # if True:
            nib_save = np.expand_dims(img_mri[0, ..., 0], axis=-1) 
            nib_save = nib.Nifti1Image(nib_save , np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 4, 0), (0, 0, 0, 1))), nib.Nifti1Header()) #np.array(((0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1)))
            nib.save(nib_save, os.path.join("/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_evaluation/stds/head_2", str(img_id) + "_image.nii.gz"))
            # nib.save(nib_save, os.path.join("path", str(img_id) + "_image.nii.gz"))
            
            nib_save = np.expand_dims(targets[0, ..., 0], axis=-1) 
            nib_save = nib.Nifti1Image(nib_save , np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 4, 0), (0, 0, 0, 1))), nib.Nifti1Header()) #np.array(((0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1)))
            nib.save(nib_save, os.path.join("/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_evaluation/stds/head_2", str(img_id) + "_gt.nii.gz"))
            # nib.save(nib_save, os.path.join("path", str(img_id) + "_gt.nii.gz"))

            nib_save = np.expand_dims(stdd[0, ..., 0], axis=-1) 
            nib_save = nib.Nifti1Image(nib_save , np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 4, 0), (0, 0, 0, 1))), nib.Nifti1Header()) #np.array(((0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1)))
            nib.save(nib_save, os.path.join("/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_evaluation/stds/head_2", str(img_id) + "_variance.nii.gz"))
            # nib.save(nib_save, os.path.join("path", str(img_id) + "_variance.nii.gz"))

            nib_save = np.expand_dims(mean[0, ..., 0], axis=-1) 
            nib_save[nib_save > 0.5] = 1 
            nib_save[nib_save != 1] = 0
            nib_save = nib.Nifti1Image(nib_save , np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 4, 0), (0, 0, 0, 1))), nib.Nifti1Header()) #np.array(((0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1)))
            nib.save(nib_save, os.path.join("/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_evaluation/stds/head_2", str(img_id) + "_label.nii.gz"))
            # nib.save(nib_save, os.path.join("path", str(img_id) + "_label.nii.gz"))
        
            f = open(os.path.join("/media/aranem_locale/AR_subs_exps/MICCAI_2024/MICCAI_2024_evaluation/stds/head_2", str(img_id) + "_score.txt"), "a")
            # f = open(os.path.join("path", str(img_id) + "_score.txt"), "a")
            f.write(str(np.sum(stdd) / np.sum(median)))
            f.close()

        return

    @staticmethod
    def standard_deviation(loss_log: dict) -> float:
        r"""Calculate the standard deviation
            #Args
                loss_log: losses
        """
        mean = sum(loss_log.values())/len(loss_log)
        stdd = 0
        for e in loss_log.values():
            stdd = stdd + pow(e - mean, 2)
        stdd = stdd / len(loss_log)
        stdd = math.sqrt(stdd)
        return stdd

    def test(self, loss_f: torch.nn.Module, save_img: list = None, tag: str = 'test/img/', pseudo_ensemble: bool = False, **kwargs):
        raise NotImplementedError
    
        r"""Evaluate model on testdata by merging it into 3d volumes first
            TODO: Clean up code and write nicer. Replace fixed images for saving in tensorboard.
            #Args
                dataset (Dataset)
                loss_f (torch.nn.Module)
                steps (int): Number of steps to do for inference
        """
        with torch.no_grad():
            # Prepare dataset for testing
            dataset = self.exp.dataset
            self.exp.set_model_state('test')
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
            # Prepare arrays
            patient_id, patient_3d_image, patient_3d_label, average_loss, patient_count = None, None, None, 0, 0
            patient_real_Img = None
            loss_log = {}
            for m in range(self.output_channels):
                loss_log[m] = {}
            if save_img == None:
                save_img = []#1, 2, 3, 4, 5, 32, 45, 89, 357, 53, 122, 267, 97, 389]

            # For each data sample
            for i, data in enumerate(dataloader):
                data = self.prepare_data(data, eval=True)
                data_id, inputs, _ = data['id'], data['image'], data['label']
                outputs, targets = self.get_outputs(data, full_img=True, tag="0")

                if isinstance(data_id, str):
                    _, id, slice = dataset.__getname__(data_id).split('_')
                else:
                    print("DATA_ID", data_id)
                    text = str(data_id[0]).split('_')
                    if len(text) == 3:
                        _, id, slice = text
                    else:
                        id = data_id[0]
                        slice = None


                # Run inference 10 times to create a pseudo ensemble
                if pseudo_ensemble: # 5 + 5 times
                    outputs2, _ = self.get_outputs(data, full_img=True, tag="1")
                    outputs3, _ = self.get_outputs(data, full_img=True, tag="2")
                    outputs4, _ = self.get_outputs(data, full_img=True, tag="3")
                    outputs5, _ = self.get_outputs(data, full_img=True, tag="4")
                    if True: 
                        outputs6, _ = self.get_outputs(data, full_img=True, tag="5")
                        outputs7, _ = self.get_outputs(data, full_img=True, tag="6")
                        outputs8, _ = self.get_outputs(data, full_img=True, tag="7")
                        outputs9, _ = self.get_outputs(data, full_img=True, tag="8")
                        outputs10, _ = self.get_outputs(data, full_img=True, tag="9")
                        stack = torch.stack([outputs, outputs2, outputs3, outputs4, outputs5, outputs6, outputs7, outputs8, outputs9, outputs10], dim=0)
                        
                        # Calculate median
                        outputs, _ = torch.median(stack, dim=0)
                        self.labelVariance(torch.sigmoid(stack).detach().cpu().numpy(), torch.sigmoid(outputs).detach().cpu().numpy(), inputs.detach().cpu().numpy(), id, targets.detach().cpu().numpy() )

                    else:
                        outputs, _ = torch.median(torch.stack([outputs, outputs2, outputs3, outputs4, outputs5], dim=0), dim=0)

                # --------------- 2D ---------------------
                if dataset.slice is not None:
                    # If next patient
                    if id != patient_id and patient_id != None:
                        out = str(patient_id) + ", "
                        for m in range(patient_3d_label.shape[3]):
                            if(1 in np.unique(patient_3d_label[...,m].detach().cpu().numpy())):
                                loss_log[m][patient_id] = 1 - loss_f(patient_3d_image[...,m], patient_3d_label[...,m], smooth = 0).item() #,, mask = patient_3d_label[...,4].bool()

                                if math.isnan(loss_log[m][patient_id]):
                                    loss_log[m][patient_id] = 0
                                out = out + str(loss_log[m][patient_id]) + ", "
                            else:
                                out = out + " , "
                        print(out)
                        patient_id, patient_3d_image, patient_3d_label = id, None, None
                    # If first slice of volume
                    if patient_3d_image == None:
                        patient_id = id
                        patient_3d_image = outputs.detach().cpu()
                        patient_3d_label = targets.detach().cpu()
                        patient_real_Img = inputs.detach().cpu()
                    else:
                        patient_3d_image = torch.vstack((patient_3d_image, outputs.detach().cpu()))
                        patient_3d_label = torch.vstack((patient_3d_label, targets.detach().cpu()))
                        patient_real_Img = torch.vstack((patient_real_Img, inputs.detach().cpu()))
                    # Add image to tensorboard
                    if i in save_img: 
                        self.exp.write_img(str(tag) + str(patient_id) + "_" + str(len(patient_3d_image)),
                                           merge_img_label_gt(np.squeeze(inputs.detach().cpu().numpy()), 
                                                              torch.sigmoid(outputs).detach().cpu().numpy(), 
                                                              targets.detach().cpu().numpy()), 
                                           self.exp.currentStep)
                                           
                        #self.exp.write_img(str(tag) + str(patient_id) + "_" + str(len(patient_3d_image)), 
                        #convert_image(self.prepare_image_for_display(inputs.detach().cpu()).numpy(), 
                        #self.prepare_image_for_display(outputs.detach().cpu()).numpy(), 
                        #self.prepare_image_for_display(targets.detach().cpu()).numpy(), 
                        #encode_image=False), self.exp.currentStep)
                # --------------------------------- 3D ----------------------------
                else: 
                    patient_3d_image = outputs.detach().cpu()
                    patient_3d_label = targets.detach().cpu()
                    patient_3d_real_Img = inputs.detach().cpu()
                    patient_id = id
                    print(patient_id)

                    print(patient_3d_image.shape,patient_3d_label.shape )
                    for m in range(patient_3d_image.shape[-1]):
                        loss_log[m][patient_id] = 1 - loss_f(patient_3d_image[...,m], patient_3d_label[...,m], smooth = 0).item()
                        print(",",loss_log[m][patient_id])
                        # Add image to tensorboard
                        if False: 
                            if len(patient_3d_label.shape) == 4:
                                patient_3d_label = patient_3d_label.unsqueeze(dim=-1)
                            middle_slice = int(patient_3d_real_Img.shape[3] /2)
                            self.exp.write_img(str(tag) + str(patient_id) + "_" + str(len(patient_3d_image)),
                                            merge_img_label_gt(patient_3d_real_Img[:,:,:,middle_slice:middle_slice+1,0].numpy(), torch.sigmoid(patient_3d_image[:,:,:,middle_slice:middle_slice+1,m]).numpy(), patient_3d_label[:,:,:,middle_slice:middle_slice+1,m].numpy()), 
                                            self.exp.currentStep)
                            #self.exp.write_img(str(tag) + str(patient_id) + "_" + str(len(patient_3d_image)), 
                            #convert_image(self.prepare_image_for_display(patient_3d_real_Img[:,:,:,5:6,:].detach().cpu()).numpy(), 
                            #self.prepare_image_for_display(patient_3d_image[:,:,:,5:6,:].detach().cpu()).numpy(), 
                            #self.prepare_image_for_display(patient_3d_label[:,:,:,5:6,:].detach().cpu()).numpy(), 
                            #encode_image=False), self.exp.currentStep)

                            # REFACTOR: Save predictions
                            if False:
                                label_out = torch.sigmoid(patient_3d_image[0, ...])
                                nib_save = nib.Nifti1Image(label_out  , np.array(((0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1))), nib.Nifti1Header())
                                nib.save(nib_save, os.path.join("path", str(len(loss_log[0])) + ".nii.gz"))

                                nib_save = nib.Nifti1Image(torch.sigmoid(patient_3d_real_Img[0, ...])  , np.array(((0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1))), nib.Nifti1Header())
                                nib.save(nib_save, os.path.join("path", str(len(loss_log[0])) + "_real.nii.gz"))

                                nib_save = nib.Nifti1Image(patient_3d_label[0, ...]  , np.array(((0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1))), nib.Nifti1Header())
                                nib.save(nib_save, os.path.join("path", str(len(loss_log[0])) + "_ground.nii.gz"))

            # If 2D
            if dataset.slice is not None:
                out = str(patient_id) + ", "
                for m in range(patient_3d_label.shape[-1]):
                    if(1 in np.unique(patient_3d_label[...,m].detach().cpu().numpy())):
                        loss_log[m][patient_id] = 1 - loss_f(patient_3d_image[...,m], patient_3d_label[...,m], smooth = 0).item() 
                        out = out + str(loss_log[m][patient_id]) + ", "
                    else:
                        out = out + " , "
                print(out)
            # Print dice score per label
            for key in loss_log.keys():
                if len(loss_log[key]) > 0:
                    print("Average Dice Loss 3d: " + str(key) + ", " + str(sum(loss_log[key].values())/len(loss_log[key])))
                    print("Standard Deviation 3d: " + str(key) + ", " + str(self.standard_deviation(loss_log[key])))

            self.exp.set_model_state('train')
            return loss_log