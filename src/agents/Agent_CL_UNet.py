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

"""
Implementations of most baselines (except EWC and RWalk) extracted from:
https://github.com/NeurAI-Lab/DUCA/tree/main/models
"""

EPSILON = 1e-8
ALPHA = 0.9

def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a

def write_pickle(obj, file: str, mode: str = 'wb') -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)

def store_grad(params, grads, grad_dims):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[:count + 1])
            grads[begin: end].copy_(param.grad.data.view(-1))
        count += 1

def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger

def overwrite_grad(params, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = sum(grad_dims[:count + 1])
            this_grad = newgrad[begin: end].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        count += 1

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

Schema for SI: https://arxiv.org/pdf/1703.04200
    For training stage 0:
        - agent = M3DNCAAgent(model) --> initialize Agent with model
        - dataset.set_experiment(exp) --> set experiment for dataset and initialize agent
        - agent.train() --> train model
        - agent.end_task_SI()
    
    For training stage > 0:
        - agent = M3DNCAAgent(model) --> initialize Agent with model
        - dataset.set_experiment(exp) --> set experiment for dataset and initialize agent
        - agent.start_task_SI() --> initialize SI
        - agent.train() --> train model
        - agent.end_task_SI()

Schema for DER (Dark Experience): https://proceedings.neurips.cc/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf
    For training stage 0:
        - agent = M3DNCAAgent(model) --> initialize Agent with model
        - dataset.set_experiment(exp) --> set experiment for dataset and initialize agent
        - agent.train() --> train model
        - agent.end_task_DER()
    
    For training stage > 0:
        - agent = M3DNCAAgent(model) --> initialize Agent with model
        - dataset.set_experiment(exp) --> set experiment for dataset and initialize agent
        - agent.train() --> train model
        - agent.end_task_DER()

Schema for FDR (Function Distance Regularization): https://arxiv.org/pdf/1805.08289
    For training stage 0:
        - agent = M3DNCAAgent(model) --> initialize Agent with model
        - dataset.set_experiment(exp) --> set experiment for dataset and initialize agent
        - agent.train() --> train model
        - agent.end_task_FDR()
    
    For training stage > 0:
        - agent = M3DNCAAgent(model) --> initialize Agent with model
        - dataset.set_experiment(exp) --> set experiment for dataset and initialize agent
        - agent.train() --> train model
        - agent.end_task_FDR()

Schema for AGem (Averaged GEM): https://arxiv.org/pdf/1812.00420
    For training stage 0:
        - agent = M3DNCAAgent(model) --> initialize Agent with model
        - dataset.set_experiment(exp) --> set experiment for dataset and initialize agent
        - agent.train() --> train model
        - agent.end_task_AGem()
    
    For training stage > 0:
        - agent = M3DNCAAgent(model) --> initialize Agent with model
        - dataset.set_experiment(exp) --> set experiment for dataset and initialize agent
        - agent.train() --> train model
        - agent.end_task_AGem()
"""

class CLUNetAgent(UNetAgent):
    """Base agent for training UNet models
    """
    def initialize(self):
        super().initialize()
        self.use_ewc_loss = False
        self.use_rwalk_loss = False
        self.use_rwalk = False
        self.prev_param = None
        self.big_omega = None
        self.checkpoint = None
        self.small_omega = 0
        self.penalty_weight = 0.1
        self.der_alpha = 0.4
        self.cl_m = None
        self.buffer = Buffer()
        self.fdrsoft = torch.nn.Softmax(dim=1)
        self.fdrlogsoft = torch.nn.LogSoftmax(dim=1)
        self.loss_ = None
        # AGem:
        self.grad_dims = []
        for param in self.model.parameters():
            self.grad_dims.append(param.data.numel())
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.device)
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.device)

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

    def penalty(self):
        r"""Calculate penalty for different CL methods --> This is added to losss BEFORE loss.backward
        """
        penalty = 0
        if self.cl_m == 'SI':
            if self.checkpoint is None: # <-- First task, no penalty here..
                return penalty
            if self.big_omega is None:
                return torch.tensor(0.0).to(self.device)
            else:
                penalty = (self.big_omega * ((self.get_params() - self.checkpoint) ** 2)).sum()
                return penalty
        if self.cl_m == "DER":
            if self.buffer.len() > 0:
                buf_inputs, buf_logits = self.buffer.get_data()
                buf_outputs = self.model(buf_inputs)
                penalty = self.der_alpha * F.mse_loss(buf_outputs, buf_logits)
                return penalty
        if self.cl_m == "FDR":
            if self.buffer.len() > 0:
                buf_inputs, buf_logits = self.buffer.get_data()
                buf_outputs = self.model(buf_inputs)
                penalty = torch.norm(self.fdrsoft(buf_outputs) - self.fdrsoft(buf_logits), 2, 1).mean()
                return penalty
        if self.cl_m == "AGem":
            if self.buffer.len() > 0:
                store_grad(self.model.parameters, self.grad_xy, self.grad_dims)
                buf_inputs, buf_labels = self.buffer.get_data()
                self.model.zero_grad()
                buf_outputs = self.model.forward(buf_inputs)
                penalty = self.loss_(buf_outputs, buf_labels)
        return penalty

    def after_loss(self, lr):
        r"""Calculate penalty and update small omega for SI
        """
        if self.cl_m == 'SI':
            self.small_omega += lr * self.get_grads().data ** 2
        if self.cl_m == 'AGem':
            store_grad(self.model.parameters, self.grad_er, self.grad_dims)

            dot_prod = torch.dot(self.grad_xy, self.grad_er)
            if dot_prod.item() < 0:
                g_tilde = project(gxy=self.grad_xy, ger=self.grad_er)
                overwrite_grad(self.model.parameters, g_tilde, self.grad_dims)
            else:
                overwrite_grad(self.model.parameters, self.grad_xy, self.grad_dims)
        
    def start_task_SI(self):
        self.checkpoint = self.get_params().data.clone().to(self.device)

    def end_task_SI(self):
        # big omega calculation step
        if self.big_omega is None:
            self.big_omega = torch.zeros_like(self.get_params()).to(self.device)
        if self.checkpoint is not None:
            self.big_omega += self.small_omega / ((self.get_params().data - self.checkpoint) ** 2)

        # store parameters checkpoint and reset small_omega
        self.checkpoint = self.get_params().data.clone().to(self.device)
        self.small_omega = 0

    def end_task_DER(self, dataloader):
        for _ in range(int(len(dataloader)*0.2)):
            dataloader_ = iter(dataloader)
            data = next(dataloader_)
            data = self.prepare_data(data)
            x = data['image']
            logits = self.model(x)
            self.buffer.add_data((x, logits))    # Buffer holds (data, preds) pairs
        print("Added 20% of dataset to buffer.")

    def end_task_FDR(self, dataloader):
        with torch.no_grad():
            self.end_task_DER(dataloader)

    def end_task_AGem(self, dataloader):
        for _ in range(int(len(dataloader)*0.2)):
            dataloader_ = iter(dataloader)
            data = next(dataloader_)
            data = self.prepare_data(data)
            x = data['image']
            y = data['label']
            self.buffer.add_data((x, y))    # Buffer holds (data, preds) pairs
        print("Added 20% of dataset to buffer.")

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.model.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for pp in list(self.model.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)


class Buffer:
    def __init__(self):
        """
        Initialize the DataBuffer with a specified buffer size.
        """
        self.buffer = []

    def add_data(self, data_point):
        """
        Add a data point to the buffer.

        :param data_point: The data point to be added to the buffer.
        """
        self.buffer.append(data_point)
        
    def get_data(self):
        """
        Retrieve a data point from the buffer.

        :return: The data point at the specified index.
        """
        data, logits = random.choice(self.buffer)
        return data, logits
        
    def len(self):
        """
        Get the current size of the buffer.

        :return: The number of data points currently in the buffer.
        """
        return int(len(self.buffer))
    
    def empty(self):
        self.buffer = []