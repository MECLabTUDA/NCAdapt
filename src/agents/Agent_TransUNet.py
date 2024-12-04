import torch
from src.agents.Agent_UNet import UNetAgent
import torch.nn.functional as F
import random
from src.losses.LossFunctions import EWCLoss, RWalkLoss, DiceLoss
from src.agents.Agent_MedSeg3D import Agent_MedSeg3D
from torch.utils.data import DataLoader
from src.utils.ProjectConfiguration import ProjectConfiguration as pc
from tqdm import tqdm
import os
import numpy as np

import pickle

EPSILON = 1e-8
ALPHA = 0.9

def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a

def write_pickle(obj, file: str, mode: str = 'wb') -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)


"""
Schema for EWC:
    For training stage 0:
        - agent = M3DNCAAgent(model) --> initialize Agent with model
        - dataset.set_experiment(exp) --> set experiment for dataset and initialize agent
        - agent.train() --> train model
        - agent.after_train_ewc(task, dataloader, loss_f) --> calculate Fisher and params for EWC

    For training stage > 0:
        - agent = M3DNCAAgent(model) --> initialize Agent with model
        - dataset.set_experiment(exp) --> set experiment for dataset and initialize agent
        - fisher, params = agent.load_fisher_params_ewc() --> load Fisher and params from previous task
        - agent._initialize_ewc(fisher, params, ewc_lambda) --> initialize EWC
        - agent.train() --> train model
        - agent.after_train_ewc(task, dataloader, loss_f) --> calculate fisher and params for EWC

Schema for RWalk:
    For training stage 0:
        - agent = M3DNCAAgent(model) --> initialize Agent with model
        - dataset.set_experiment(exp) --> set experiment for dataset and initialize agent
        - agent.before_train_rwalk(task) --> initialize RWalk for very first task as fisher is calculated on the fly
        - agent.train() --> train model
        - agent.after_train_rwalk(task, dataloader, loss_f) --> calculate fisher, params and scores for RWalk
    
    For training stage > 0:
        - agent = M3DNCAAgent(model) --> initialize Agent with model
        - dataset.set_experiment(exp) --> set experiment for dataset and initialize agent
        - fisher, params, scores = agent.load_fisher_params_scores_rwalk() --> load Fisher, params and scores from previous task
        - agent._initalize_rwalk(fisher, params, scores, rwalk_lambda, task) --> initialize RWalk
        - agent.train() --> train model
        - agent.after_train_rwalk(task, dataloader, loss_f) --> calculate fisher, params and scores for RWalk
"""

class TransUNetAgent(UNetAgent):
    """Base agent for training TransUNet models
    """
    def initialize(self):
        super().initialize()
        self.use_ewc_loss = False
        self.use_rwalk_loss = False
        self.use_rwalk = False
        self.prev_param = None
        self.fisher_dict, self.params_dict, self.scores = {}, {}, {}

    def _initialize_ewc(self, fisher_dict: dict, params_dict: dict, ewc_lambda: float):
        r"""Initialize EWC after first task is trained and after training after_train_ewc is called.
        """
        if self.use_rwalk_loss:
            assert True, "RWalk is already initialized, cannot initialize EWC at the same time."
        self.fisher_dict = fisher_dict
        self.params_dict = params_dict
        self.ewc_lambda = ewc_lambda
        self.use_ewc_loss = True
        self._ewc_loss = EWCLoss(ewc_lambda, fisher_dict, params_dict)

    def _initalize_rwalk(self, fisher_dict: dict, params_dict: dict, scores: dict, rwalk_lambda: float, task: str):
        r"""Initialize RWalk after first task is trained and after training after_train_rwalk is called.
        """
        if self.use_ewc_loss:
            assert True, "EWC is already initialized, cannot initialize RWalk at the same time."
        self.fisher_dict = fisher_dict
        self.params_dict = params_dict
        self.scores = scores
        self.rwalk_lambda = rwalk_lambda
        self.use_rwalk_loss = True
        self.prev_param = None
        self._rwalk_loss = RWalkLoss(rwalk_lambda, fisher_dict, params_dict, scores)
        self.task = task
        self.fisher_dict[task] = {n: torch.zeros_like(p, device=p.device, requires_grad=False) for n, p in self.model.named_parameters() if p.requires_grad}
        self.params_dict[task] = dict()
        self.scores[task] = {n: torch.zeros_like(p, device=p.device, requires_grad=False) for n, p in self.model.named_parameters() if p.requires_grad}

    def batch_step(self, data: tuple, loss_f: torch.nn.Module, gradient_norm: bool = False, make_steps=True) -> dict:
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
            if make_steps:
                self.optimizer.step()
                self.scheduler.step()
        return loss_ret
    
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

        if self.use_rwalk:
            # -- Update the importance score using distance in Riemannian Manifold -- #
            if self.prev_param is not None:
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        # -- Get parameter difference from old param and current param t -- #
                        delta = param.grad.detach() * (self.prev_param[name].to(param.device) - param.detach())
                        delta = delta.to(param.device)
                        # -- Calculate score denominator -- #
                        den = 0.5 * self.fisher_dict[self.task][name] * (param.detach() - self.prev_param[name].to(param.device)).pow(
                            2).to(param.device) + EPSILON
                        # -- Score: delat(L) / 0.5*F_t*delta(param)^2 --> only positive or zero values -- #
                        scores_ = (delta / den)
                        scores_[scores_ < 0] = 0  # Ensure no negative values
                        # -- Update the scores -- #
                        self.scores[self.task][name] += scores_

            # -- Update the prev params -- #
            if epoch != 0:
                self.prev_param = {k: torch.clone(v).detach().cpu() for k, v in self.model.named_parameters() if
                              v.grad is not None}

            # -- Update the fisher values -- #
            for name, param in self.model.named_parameters():
                # -- F_t = alpha * F_t + (1-alpha) * F_t-1
                if param.grad is not None:
                    f_t = param.grad.data.clone().pow(2).to(param.device)
                    f_to = self.fisher_dict[self.task][name] if self.fisher_dict[self.task][name] is not None else torch.tensor([0],
                                                                                                            device=param.device)
                    self.fisher_dict[self.task][name] = (ALPHA * f_t) + ((1 - ALPHA) * f_to)

            for name, param in self.model.named_parameters():
                # -- Update the params dict -- #
                self.params_dict[self.task][name] = param.data.clone()

            model_dir = os.path.join(self.exp.get_from_config('model_path'))
            write_pickle(self.fisher_dict, os.path.join(model_dir, 'fisher.pkl'))
            write_pickle(self.params_dict, os.path.join(model_dir, 'params.pkl'))
            write_pickle(self.scores, os.path.join(model_dir, 'scores.pkl'))

        # TODO: ADD AGAIN 
        #for param in self.model.parameters():
        #    param_lst.extend(np.fromiter(param.flatten(), dtype=float))
        #self.exp.write_histogram('Model/weights', np.fromiter(param_lst, dtype=float), epoch)
            
    def before_train_rwalk(self, task):
        r"""Call this before starting the first task for RWalk training
        """
        self.task = task
        self.fisher_dict[task] = {n: torch.zeros_like(p, device=p.device, requires_grad=False) for n, p in self.model.named_parameters() if p.requires_grad}
        self.params_dict[task] = dict()
        self.scores[task] = {n: torch.zeros_like(p, device=p.device, requires_grad=False) for n, p in self.model.named_parameters() if p.requires_grad}
        self.use_rwalk = True

    def after_train_ewc(self, task, dataloader: DataLoader, loss_f: torch.Tensor):
        r"""Call this after train for EWC
        """
        self.fisher_dict[task] = {}
        self.params_dict[task] = {}
        # ewc specific code
        self.optimizer.zero_grad()
        dataloader_ = iter(dataloader)   # NOTE: remove if we go back to for i, data loop!
        
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
            loss_item = self.batch_step(data, loss_f, make_steps=False) # Make NO optimization
            for key in loss_item.keys():
                if isinstance(loss_item[key], float):
                    loss_log[key].append(loss_item[key])
                else:
                    loss_log[key].append(loss_item[key].detach())
            
            # backpropagate but NO optimization!
            self.optimizer.zero_grad()

        # -- Set fisher and params in current fold from last iteration --> final model parameters -- #
        for name, param in self.model.named_parameters():
            # -- Update the fisher and params dict -- #
            if param.grad is None:
                self.fisher_dict[task][name] = torch.tensor([1], device=param.device)
            else:
                self.fisher_dict[task][name] = param.grad.data.clone().pow(2)
            self.params_dict[task][name] = param.data.clone()

        model_path = os.path.join(self.exp.get_from_config('model_path'))
        write_pickle(self.fisher_dict, os.path.join(model_path, 'fisher.pkl'))
        write_pickle(self.params_dict, os.path.join(model_path, 'params.pkl'))

    def load_fisher_params_ewc(self):
        pretrained_path = os.path.join(pc.STUDY_PATH, 'Experiments', self.exp.config['pretrained'] + "_" + self.exp.projectConfig['description'])
        # model_path = os.path.join(self.exp.get_from_config('model_path'))
        self.fisher_dict = load_pickle(os.path.join(pretrained_path, 'fisher.pkl'))
        self.params_dict = load_pickle(os.path.join(pretrained_path, 'params.pkl'))
        return self.fisher_dict, self.params_dict

    def after_train_rwalk(self, task, dataloader: DataLoader, loss_f: torch.Tensor):
        r"""Call this after train for RWalk
        """
        # rwalk specific code
        self.optimizer.zero_grad()
        dataloader_ = iter(dataloader)   # NOTE: remove if we go back to for i, data loop!
        
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
            loss_item = self.batch_step(data, loss_f, make_steps=False) # Make NO optimization
            for key in loss_item.keys():
                if isinstance(loss_item[key], float):
                    loss_log[key].append(loss_item[key])
                else:
                    loss_log[key].append(loss_item[key].detach())
            
            # backpropagate but NO optimization!
            self.optimizer.zero_grad()

        # -- Update the importance score one last time once finished training using distance in Riemannian Manifold -- #
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # -- Get parameter difference from old param and current param t -- #
                delta = param.grad.detach() * (self.prev_param[name].to(param.device) - param.detach())
                delta = delta.to(param.device)
                # -- Calculate score denominator -- #
                den = 0.5 * self.fisher_dict[task][name] * (param.detach() - self.prev_param[name].to(param.device)).pow(2).to(param.device) + EPSILON
                # -- Score: delat(L) / 0.5*F_t*delta(param)^2 --> only positive or zero values -- #
                scores_ = (delta / den)
                scores_[scores_ < 0] = 0  # Ensure no negative values
                # -- Update the scores -- #
                self.scores[task][name] += scores_
            else:
                self.scores[task][name] = torch.tensor([0], device=param.device)

        # -- Store params -- #
        for name, param in self.model.named_parameters():
            # -- Update the params dict -- #
            self.params_dict[task][name] = param.data.clone()

        # -- Update the fisher values -- #
        for name, param in self.model.named_parameters():
            # -- F_t = alpha * F_t + (1-alpha) * F_t-1
            if param.grad is not None:
                f_t = param.grad.data.clone().pow(2).to(param.device)
                f_to = self.fisher_dict[task][name] if self.fisher_dict[task][name] is not None else torch.tensor([0],
                                                                                                        device=param.device)
                self.fisher_dict[task][name] = (ALPHA * f_t) + ((1 - ALPHA) * f_to)

        # -- Normalize the fisher values to be in range 0 to 1 -- #
        values = [torch.max(val) for val in self.scores[task].values()]  # --> only for the current task of course
        minim, maxim = min(values), max(values)
        for k, v in self.fisher_dict[task].items():
            self.fisher_dict[task][k] = (v - minim) / (maxim - minim + EPSILON)

        # -- Normalize the score values to be in range 0 to 1 -- #
        values = [torch.max(val) for val in self.scores[task].values()]  # --> only for the current task of course
        minim, maxim = min(values), max(values)

        if len([x for x in self.scores.keys() if x != task]) > 0:
            # -- Average current and previous scores -- #
            prev_scores = {k: v.clone() for k, v in self.scores[list(self.scores.keys())[-1]].items()}
            for k, v in self.scores[task].items():
                # -- Normalize the score -- #
                curr_score_norm = (v - minim) / (maxim - minim + EPSILON)
                # -- Average the score to alleviate rigidity due to the accumulating sum of the scores otherwise -- #
                self.scores[task][k] = 0.5 * (prev_scores[k] + curr_score_norm)
        else:
            # -- Only average current scores -- #
            for k, v in self.scores[task].items():
                # -- Normalize and scale the score so that division does not have an effect -- #
                curr_score_norm = (v - minim) / (maxim - minim + EPSILON)
                self.scores[task][k] = 2 * curr_score_norm

        model_path = os.path.join(self.exp.get_from_config('model_path'))
        write_pickle(self.fisher_dict, os.path.join(model_path, 'fisher.pkl'))
        write_pickle(self.params_dict, os.path.join(model_path, 'params.pkl'))
        write_pickle(self.scores, os.path.join(model_path, 'scores.pkl'))

    def load_fisher_params_scores_rwalk(self):
        pretrained_path = os.path.join(pc.STUDY_PATH, 'Experiments', self.exp.config['pretrained'] + "_" + self.exp.projectConfig['description'])
        self.fisher_dict = load_pickle(os.path.join(pretrained_path, 'fisher.pkl'))
        self.params_dict = load_pickle(os.path.join(pretrained_path, 'params.pkl'))
        self.scores = load_pickle(os.path.join(pretrained_path, 'scores.pkl'))
        return self.fisher_dict, self.params_dict, self.scores

    #def batch_step(self, data: tuple, loss_f: torch.nn.Module, gradient_norm: bool = False) -> dict:
    #    return super().batch_step(data, loss_f, gradient_norm)