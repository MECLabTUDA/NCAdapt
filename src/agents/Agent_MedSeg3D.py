import torch
from src.agents.Agent import BaseAgent
from src.utils.helper import convert_image, merge_img_label_gt, merge_img_label_gt_simplified
import numpy as np
import math, os, time
import SimpleITK as sitk

class Agent_MedSeg3D(BaseAgent):
    def test(self, loss_f: torch.nn.Module, save_img: list = None, tag: str = 'test/img/', pseudo_ensemble: bool = False, dataset=None, inference=False, **kwargs):
        r"""Evaluate model on testdata by merging it into 3d volumes first
            TODO: Clean up code and write nicer. Replace fixed images for saving in tensorboard.
            #Args
                dataset (Dataset)
                loss_f (torch.nn.Module)
                steps (int): Number of steps to do for inference
        """
        with torch.no_grad():
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
                save_img = []#1, 2, 3, 4, 5, 32, 45, 89, 357, 53, 122, 267, 97, 389]

            # For each data sample
            for i, data in enumerate(dataloader):
                start_time = time.perf_counter()

                data = self.prepare_data(data, eval=True)
                data_id, inputs, *_ = data['id'], data['image'], data['label']
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

                patient_3d_image = outputs.detach().cpu()
                patient_3d_label = targets.detach().cpu()
                patient_3d_real_Img = inputs.detach().cpu()
                patient_id = id
                #print(patient_id)

                task_ = dataset.images_path.split(os.sep)[-2]
                if inference:
                    out_ = os.path.join(self.exp.config['model_path'], 'inference', task_, id)
                else:
                    out_ = os.path.join(self.exp.config['model_path'], 'validate', task_, id)
                os.makedirs(out_, exist_ok=True)

                sitk.WriteImage(sitk.GetImageFromArray(torch.sigmoid(patient_3d_image[0, ...]).squeeze().numpy().round().astype(float)), os.path.join(out_, 'pred_seg.nii.gz'))
                sitk.WriteImage(sitk.GetImageFromArray(patient_3d_real_Img.squeeze().numpy()), os.path.join(out_, 'img.nii.gz'))
                sitk.WriteImage(sitk.GetImageFromArray(patient_3d_label.squeeze().numpy()), os.path.join(out_, 'seg_gt.nii.gz'))

                for m in range(patient_3d_image.shape[-1]):
                    loss_log[m][patient_id] = 1 - loss_f(patient_3d_image[...,m], patient_3d_label[...,m], smooth = 0).item()
                    print(",",loss_log[m][patient_id])
                    # Add image to tensorboard
                    if True: 
                        if len(patient_3d_label.shape) == 4:
                            patient_3d_label = patient_3d_label.unsqueeze(dim=-1)
                        middle_slice = int(patient_3d_real_Img.shape[3] /2)
                        #print(patient_3d_real_Img.shape, patient_3d_image.shape, patient_3d_label.shape)
                        self.exp.write_img(str(tag) + str(patient_id) + "_" + str(len(patient_3d_image)),
                                        merge_img_label_gt_simplified(patient_3d_real_Img, patient_3d_image, patient_3d_label),
                                        #merge_img_label_gt(patient_3d_real_Img[:,:,:,middle_slice:middle_slice+1,0].numpy(), torch.sigmoid(patient_3d_image[:,:,:,middle_slice:middle_slice+1,m]).numpy(), patient_3d_label[:,:,:,middle_slice:middle_slice+1,m].numpy()), 
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

                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f"The prediction of one image {elapsed_time} seconds to execute.")

            # Print dice score per label
            for key in loss_log.keys():
                if len(loss_log[key]) > 0:
                    print("Average Dice Loss 3d: " + str(key) + ", " + str(sum(loss_log[key].values())/len(loss_log[key])))
                    print("Standard Deviation 3d: " + str(key) + ", " + str(self.standard_deviation(loss_log[key])))

            self.exp.set_model_state('train')
            return loss_log