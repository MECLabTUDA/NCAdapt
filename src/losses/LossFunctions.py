import torch
import torch.nn.functional as F

# -- From: https://github.com/MECLabTUDA/Lifelong-nnUNet/blob/f22c01eebc2b0c542ca0a40ac722f90aab05fc54/nnunet_ext/training/loss_functions/deep_supervision.py#L86 -- #
class RWalkLoss(torch.nn.Module):
    """
    RWalk Loss.
    """
    def __init__(self, ewc_lambda=0.4, fisher=dict(), params=dict(), parameter_importance=dict()):
        self.ewc_lambda = float(ewc_lambda)
        self.tasks = list(fisher.keys())[:-1]   # <-- Current task is already in there not like simple EWC!
        self.fisher = fisher
        self.params = params
        self.parameter_importance = parameter_importance
        super(RWalkLoss, self).__init__()

    def forward(self, network_params):
        # -- Update the network_params -- #
        loss = 0
        for task in list(self.fisher.keys())[:-1]:
            for name, param in network_params: # Get named parameters of the current model
                # -- Extract corresponding fisher and param values -- #
                param_ = param
                fisher_value = self.fisher[task][name]
                param_value = self.params[task][name]
                importance = self.parameter_importance[task][name]
                
                # -- loss = loss_{t} + ewc_lambda * \sum_{i} (F_{i} + S(param_{i})) * (param_{i} - param_{t-1, i})**2 -- #
                loss += self.ewc_lambda * ((fisher_value + importance) * (param_ - param_value).pow(2)).sum()
        return loss

# -- From: https://github.com/MECLabTUDA/Lifelong-nnUNet/blob/f22c01eebc2b0c542ca0a40ac722f90aab05fc54/nnunet_ext/training/loss_functions/deep_supervision.py#L15 -- #
class EWCLoss(torch.nn.Module):
    """
    EWC Loss.
    """
    def __init__(self, ewc_lambda=0.4, fisher=dict(), params=dict()):
        self.ewc_lambda = float(ewc_lambda)
        self.fisher = fisher
        self.params = params
        super(EWCLoss, self).__init__()

    def forward(self, network_params):
        # -- Update the network_params -- #
        loss = 0
        for task in self.fisher.keys():
            for name, param in network_params: # Get named parameters of the current model
                # -- Extract corresponding fisher and param values -- #
                # fisher_value = self.fisher[task][name]
                # param_value = self.params[task][name]
                param_ = param
                fisher_value = self.fisher[task][name]
                param_value = self.params[task][name]
                # loss = to_cuda(loss, gpu_id=param.get_device())
                
                # -- loss = loss_{t} + ewc_lambda/2 * \sum_{i} F_{i}(param_{i} - param_{t-1, i})**2 -- #
                loss += self.ewc_lambda/2 * (fisher_value * (param_ - param_value).pow(2)).sum()
        return loss

class DiceLoss(torch.nn.Module):
    r"""Dice Loss
    """
    def __init__(self, useSigmoid: bool = True) -> None:
        r"""Initialisation method of DiceLoss
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        super(DiceLoss, self).__init__()

    def forward(self, input: torch.tensor, target: torch.tensor, smooth: float = 1) -> torch.Tensor:
        r"""Forward function
            #Args:
                input: input array
                target: target array
                smooth: Smoothing value
        """
        if self.useSigmoid:
            input = torch.sigmoid(input)  
        input = torch.flatten(input)
        target = torch.flatten(target)
        intersection = (input * target).sum()
        dice = (2.*intersection + smooth)/(input.sum() + target.sum() + smooth)

        return 1 - dice

class DiceLoss_mask(torch.nn.Module):
    r"""Dice Loss mask, that only calculates on masked values
    """
    def __init__(self, useSigmoid = True) -> None:
        r"""Initialisation method of DiceLoss mask
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        super(DiceLoss_mask, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None, smooth: float=1) -> torch.Tensor:
        r"""Forward function
            #Args:
                input: input array
                target: target array
                smooth: Smoothing value
                mask: The mask which defines which values to consider
        """
        if self.useSigmoid:
            input = torch.sigmoid(input)  
        input = torch.flatten(input)
        target = torch.flatten(target)
        mask = torch.flatten(mask)

        input = input[~mask]  
        target = target[~mask]  
        intersection = (input * target).sum()
        dice = (2.*intersection + smooth)/(input.sum() + target.sum() + smooth)

        return 1 - dice

class DiceBCELoss(torch.nn.Module):
    r"""Dice BCE Loss
    """
    def __init__(self, useSigmoid: bool = True) -> None:
        r"""Initialisation method of DiceBCELoss
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        super(DiceBCELoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, smooth: float = 1) -> torch.Tensor:
        r"""Forward function
            #Args:
                input: input array
                target: target array
                smooth: Smoothing value
        """
        input = torch.sigmoid(input)       
        input = torch.flatten(input) 
        target = torch.flatten(target)
        
        intersection = (input * target).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(input.sum() + target.sum() + smooth)  
        BCE = torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class BCELoss(torch.nn.Module):
    r"""BCE Loss
    """
    def __init__(self, useSigmoid: bool = True) -> None:
        r"""Initialisation method of DiceBCELoss
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        super(BCELoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, smooth: float = 1) -> torch.Tensor:
        r"""Forward function
            #Args:
                input: input array
                target: target array
                smooth: Smoothing value
        """
        input = torch.sigmoid(input)       
        input = torch.flatten(input) 
        target = torch.flatten(target)

        BCE = torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')
        return BCE

class FocalLoss(torch.nn.Module):
    r"""Focal Loss
    """
    def __init__(self, gamma: float = 2, eps: float = 1e-7) -> None:
        r"""Initialisation method of DiceBCELoss
            #Args:
                gamma
                eps
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Forward function
            #Args:
                input: input array
                target: target array
        """
        input = torch.sigmoid(input)
        input = torch.flatten(input)
        target = torch.flatten(target)

        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss_bce = torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')
        loss = loss_bce * (1 - logit) ** self.gamma  # focal loss
        loss = loss.mean()
        return loss

class DiceFocalLoss(FocalLoss):
    r"""Dice Focal Loss
    """
    def __init__(self, gamma: float = 2, eps: float = 1e-7):
        r"""Initialisation method of DiceBCELoss
            #Args:
                gamma
                eps
        """
        super(DiceFocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Forward function
            #Args:
                input: input array
                target: target array
        """
        input = torch.sigmoid(input)
        input = torch.flatten(input)
        target = torch.flatten(target)

        intersection = (input * target).sum()
        dice_loss = 1 - (2.*intersection + 1.)/(input.sum() + target.sum() + 1.)

        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss_bce = torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')
        focal = loss_bce * (1 - logit) ** self.gamma  # focal loss
        dice_focal = focal.mean() + dice_loss
        return dice_focal