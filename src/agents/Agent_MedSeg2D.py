import torch
from src.agents.Agent import BaseAgent
from src.utils.helper import convert_image, merge_img_label_gt, merge_img_label_gt_simplified
import numpy as np
import math, os
import SimpleITK as sitk

class Agent_MedSeg2D(BaseAgent):
    @torch.no_grad()
    def test(self, loss_f: torch.nn.Module, save_img: list = None, tag: str = 'test/img/', pseudo_ensemble: bool = False, dataset=None, inference=False, **kwargs):
        r"""Evaluate model on testdata by merging it into 3d volumes first
            TODO: Clean up code and write nicer. Replace fixed images for saving in tensorboard.
            #Args
                dataset (Dataset)
                loss_f (torch.nn.Module)
                steps (int): Number of steps to do for inference
        """
        # Prepare dataset for testing
        if dataset is None:
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
            save_img = [1, 2, 3, 4, 5, 32, 45, 89, 357, 53, 122, 267, 97, 389]

        # For each data sample
        for i, data in enumerate(dataloader):
            data = self.prepare_data(data, eval=True)
            data_id, inputs, _ = data['id'], data['image'], data['label']
            outputs, targets = self.get_outputs(data, full_img=True, tag="0")

            if isinstance(data_id, str):
                _, id, slice = dataset.__getname__(data_id).split('_')
            else:
                text = str(data_id[0]).split('_')
                if len(text) == 3:
                    _, id, slice = text
                else:
                    id = data_id[0]
                    slice = None

            # --------------- 2D ---------------------
            # If next patient
            if id != patient_id and patient_id != None:
                out = str(patient_id) + ", "
                for m in range(patient_3d_label.shape[3]):
                    if(1 in np.unique(patient_3d_label[...,m].detach().cpu().numpy())):
                        loss_log[m][patient_id] = 1 - loss_f(patient_3d_image[...,m], patient_3d_label[...,m], smooth = 0).item() #,, mask = patient_3d_label[...,4].bool()

                        task_ = dataset.images_path.split(os.sep)[-2]
                        if inference:
                            out_ = os.path.join(self.exp.config['model_path'], 'inference', task_, id)
                        else:
                            out_ = os.path.join(self.exp.config['model_path'], 'validate', task_, id)
                        os.makedirs(out_, exist_ok=True)
                        sitk.WriteImage(sitk.GetImageFromArray(torch.sigmoid(patient_3d_image[..., 0]).numpy().round().astype(float)), os.path.join(out_, 'pred_seg.nii.gz'))
                        sitk.WriteImage(sitk.GetImageFromArray(patient_real_Img.squeeze().numpy()), os.path.join(out_, 'img.nii.gz'))
                        sitk.WriteImage(sitk.GetImageFromArray(patient_3d_label.squeeze().numpy()), os.path.join(out_, 'seg_gt.nii.gz'))

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
                                merge_img_label_gt_simplified(patient_real_Img[0:1, ...].transpose(1,3), torch.sigmoid(patient_3d_image[0:1, ...]), patient_3d_label[0:1, ...]),
                                #merge_img_label_gt(patient_3d_real_Img[:,:,:,middle_slice:middle_slice+1,0].numpy(), torch.sigmoid(patient_3d_image[:,:,:,middle_slice:middle_slice+1,m]).numpy(), patient_3d_label[:,:,:,middle_slice:middle_slice+1,m].numpy()), 
                                self.exp.currentStep)
                #self.exp.write_img(str(tag) + str(patient_id) + "_" + str(len(patient_3d_image)),
                #                    merge_img_label_gt(np.squeeze(inputs.detach().cpu().numpy()), 
                #                                        torch.sigmoid(outputs).detach().cpu().numpy(), 
                #                                        targets.detach().cpu().numpy()), 
                #                    self.exp.currentStep)

        # If 2D
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
                average = sum(loss_log[key].values())/len(loss_log[key])
                print("Average Dice Loss 3d: " + str(key) + ", " + str(average))
                print("Standard Deviation 3d: " + str(key) + ", " + str(self.standard_deviation(loss_log[key])))
                self.exp.write_scalar('Loss/test/' + str(key), average, self.exp.currentStep)

        self.exp.set_model_state('train')
        return loss_log