import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.modelio import LoadableModel, store_config_args
 
class BasicNCA3D(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1, init_method="standard", kernel_size=7, groups=False, **kwargs):
        r"""Init function
            #Args:
                channel_n: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                input_channels: number of input channels
                init_method: Weight initialisation function
                kernel_size: defines kernel input size
                groups: if channels in input should be interconnected
        """
        super(BasicNCA3D, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.input_channels = input_channels

        # One Input
        self.fc0 = nn.Linear(channel_n*2, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        self.padding = int((kernel_size-1) / 2)
        self.kernel_size = kernel_size

        self.p0 = nn.Conv3d(channel_n, channel_n, kernel_size=kernel_size, stride=1, padding=self.padding, padding_mode="reflect", groups=channel_n)
        self.bn = torch.nn.BatchNorm3d(hidden_size, track_running_stats=False)
        
        with torch.no_grad():
            self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fire_rate = fire_rate
        self.hidden_size = hidden_size
        self.to(self.device)

    def perceive(self, x):
        r"""Perceptive function, combines learnt conv outputs with the identity of the cell
            #Args:
                x: image
        """
        y1 = self.p0(x)
        y = torch.cat((x,y1),1)
        return y

    def update(self, x_in, fire_rate):
        r"""Update function runs same nca rule on each cell of an image with a random activation
            #Args:
                x_in: image
                fire_rate: random activation of cells
        """
        x = x_in.transpose(1,4)
        dx = self.perceive(x)
        dx = dx.transpose(1,4)
        dx = self.fc0(dx)
        dx = dx.transpose(1,4)
        dx = self.bn(dx)
        dx = dx.transpose(1,4)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),1])>fire_rate
        #stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),dx.size(4)])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,4)

        x = x.transpose(1,4)

        return x

    def forward(self, x, steps=10, fire_rate=0.5):
        r"""Forward function applies update function s times leaving input channels unchanged
            #Args:
                x: image
                steps: number of steps to run update
                fire_rate: random activation rate of each cell
        """
        for step in range(steps):
            x2 = self.update(x, fire_rate).clone() #[...,3:][...,3:]
            x = torch.concat((x[...,0:self.input_channels], x2[...,self.input_channels:]), 4)
        return x
   
class BasicNCA3D_CL_simple(BasicNCA3D):
    def __init__(self, *args, **kwargs):
        super(BasicNCA3D_CL_simple, self).__init__(*args, **kwargs)
        self.adjust_0 = nn.Conv3d(2*self.channel_n, 2*self.channel_n, kernel_size=1, stride=1, padding=0, groups=2*self.channel_n)
        self.adjust_1 = nn.Conv3d(self.hidden_size, self.hidden_size, kernel_size=1, stride=1, padding=0, groups=self.hidden_size)

    def update(self, x_in, fire_rate):
        r"""Update function runs same nca rule on each cell of an image with a random activation
            #Args:
                x_in: image
                fire_rate: random activation of cells
        """
        x = x_in.transpose(1,4)
        dx = self.perceive(x)
        dx = self.adjust_0(dx)
        dx = dx.transpose(1,4)
        dx = self.fc0(dx)
        dx = dx.transpose(1,4)
        dx = self.bn(dx)
        dx = dx.transpose(1,4)
        dx = F.relu(dx)
        dx = dx.transpose(1,4)
        dx = self.adjust_1(dx)
        dx = dx.transpose(1,4)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),1])>fire_rate
        #stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),dx.size(4)])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,4)

        x = x.transpose(1,4)

        return x
    
class BasicNCA3D_CL_two_adjusts(BasicNCA3D):
    def __init__(self, *args, **kwargs):
        super(BasicNCA3D_CL_two_adjusts, self).__init__(*args, **kwargs)
        self.adjusts = nn.ModuleDict()
        self.use_module = '0' # has to be updated during training where necessary!
        self.register = 0
        for _ in range(kwargs['nr_modules']):
            self._new_adjust()

    def _new_adjust(self):
        self.adjusts[str(self.register)] = nn.ModuleDict({
            "adjust_0": nn.Conv3d(2*self.channel_n, 2*self.channel_n, kernel_size=1, stride=1, padding=0, groups=2*self.channel_n),
            "adjust_1": nn.Conv3d(self.hidden_size, self.hidden_size, kernel_size=1, stride=1, padding=0, groups=self.hidden_size)
        })
        self.register += 1

    def update(self, x_in, fire_rate):
        r"""Update function runs same nca rule on each cell of an image with a random activation
            #Args:
                x_in: image
                fire_rate: random activation of cells
        """
        x = x_in.transpose(1,4)
        dx = self.perceive(x)
        dx = self.adjusts[str(self.use_module)]['adjust_0'](dx)
        dx = dx.transpose(1,4)
        dx = self.fc0(dx)
        dx = dx.transpose(1,4)
        dx = self.bn(dx)
        dx = dx.transpose(1,4)
        dx = F.relu(dx)
        dx = dx.transpose(1,4)
        dx = self.adjusts[str(self.use_module)]['adjust_1'](dx)
        dx = dx.transpose(1,4)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),1])>fire_rate
        #stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),dx.size(4)])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,4)

        x = x.transpose(1,4)

        return x
   
class BasicNCA3D_CL_two_adjusts_and_conv_adjust(BasicNCA3D):
    def __init__(self, *args, **kwargs):
        super(BasicNCA3D_CL_two_adjusts_and_conv_adjust, self).__init__(*args, **kwargs)
        self.adjusts = nn.ModuleDict()
        self.p0s = nn.ModuleDict()
        self.use_module = '0' # has to be updated during training where necessary!
        self.register = 0
        for _ in range(kwargs['nr_modules']):
            self._new_adjust()

    def _new_adjust(self):
        self.adjusts[str(self.register)] = nn.ModuleDict({
            "adjust_0": nn.Conv3d(2*self.channel_n, 2*self.channel_n, kernel_size=1, stride=1, padding=0, groups=2*self.channel_n),
            "adjust_1": nn.Conv3d(self.hidden_size, self.hidden_size, kernel_size=1, stride=1, padding=0, groups=self.hidden_size)
        })
        self.p0s[str(self.register)] = nn.Conv3d(self.channel_n, self.channel_n, kernel_size=self.kernel_size, stride=1, padding=self.padding, padding_mode="reflect", groups=self.channel_n)
        self.register += 1

    def perceive(self, x):
        r"""Perceptive function, combines learnt conv outputs with the identity of the cell
            #Args:
                x: image
        """
        y1 = self.p0s[str(self.use_module)](x)
        y = torch.cat((x,y1),1)
        return y
    
    def update(self, x_in, fire_rate):
        r"""Update function runs same nca rule on each cell of an image with a random activation
            #Args:
                x_in: image
                fire_rate: random activation of cells
        """
        x = x_in.transpose(1,4)
        dx = self.perceive(x)
        dx = self.adjusts[str(self.use_module)]['adjust_0'](dx)
        dx = dx.transpose(1,4)
        dx = self.fc0(dx)
        dx = dx.transpose(1,4)
        dx = self.bn(dx)
        dx = dx.transpose(1,4)
        dx = F.relu(dx)
        dx = dx.transpose(1,4)
        dx = self.adjusts[str(self.use_module)]['adjust_1'](dx)
        dx = dx.transpose(1,4)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),1])>fire_rate
        #stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),dx.size(4)])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,4)

        x = x.transpose(1,4)

        return x