# Random conv
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

class GradlessGCReplayNonlinBlock(nn.Module):
    def __init__(self, out_channel=3, in_channel=3, kernel_sizes=[1, 3], layer_id=0, use_act=True, 
                 requires_grad=False, distribution='kaiming_normal', **kwargs):
        super(GradlessGCReplayNonlinBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]
        self.kernel_sizes = kernel_sizes
        
        self.layer_id = layer_id
        self.use_act = use_act
        self.requires_grad = requires_grad
        self.distribution = distribution
        assert requires_grad == False

    def _initialize_weights(self, kernel_size):
        
        nb = self.current_batch_size
        weight_shape = [self.out_channel * nb, self.in_channel, kernel_size, kernel_size]
        
        if self.distribution == 'kaiming_normal':
            # kaiming 
            fan = nn.init._calculate_correct_fan(torch.zeros(weight_shape), 'fan_in')
            gain = nn.init.calculate_gain('leaky_relu', 0.2)
            std = gain / math.sqrt(fan)
            weight = torch.randn(weight_shape) * std
        elif self.distribution == 'kaiming_uniform':
            # kaiming 
            fan = nn.init._calculate_correct_fan(torch.zeros(weight_shape), 'fan_in')
            gain = nn.init.calculate_gain('leaky_relu', 0.2)
            bound = gain * math.sqrt(3.0 / fan)
            weight = torch.rand(weight_shape) * 2 * bound - bound
        else:
            
            weight = torch.randn(weight_shape)
        
        return weight

    def forward(self, x_in):
        # random kernel size
        k = random.choice(self.kernel_sizes)
        self.current_batch_size = x_in.shape[0]
        
        nb, nc, nx, ny = x_in.shape

        
        ker = self._initialize_weights(k).to(x_in.device)
        
        # Bias
        if hasattr(self, 'rand_bias') and self.rand_bias:
            fan_in = nc * k * k
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            shift = torch.rand([self.out_channel * nb, 1, 1], device=x_in.device) * 2 * bound - bound
        else:
            shift = torch.randn([self.out_channel * nb, 1, 1], device=x_in.device) * 0.1

        x_in_reshaped = x_in.view(1, nb * nc, nx, ny)
        x_conv = F.conv2d(x_in_reshaped, ker, stride=1, padding=k//2, dilation=1, groups=nb)
        x_conv = x_conv + shift

        #print(f"x_in_reshaped shape: {x_in_reshaped.shape}")
        #print(f"ker shape: {ker.shape}")
        #print(f"nb: {nb}, nc: {nc}, self.out_channel: {self.out_channel}")
        #print(f"Expected groups: {nb}, given groups: {nb}")

        
        if self.use_act:
            x_conv = F.leaky_relu(x_conv, 0.2)

        x_conv = x_conv.view(nb, self.out_channel, nx, ny)
        
        return x_conv


class GINGroupConv(nn.Module):
    def __init__(self, out_channel, in_channel, interm_channel=16, kernel_sizes=[1, 3], n_layer=4, 
                 out_norm='frob', distribution='kaiming_normal', **kwargs):
        super(GINGroupConv, self).__init__()
        
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]
        self.kernel_sizes = kernel_sizes
        
        self.n_layer = n_layer
        self.layers = []
        self.out_norm = out_norm
        self.out_channel = out_channel
        self.distribution = distribution

        self.layers.append(
            GradlessGCReplayNonlinBlock(
                out_channel=interm_channel, 
                in_channel=in_channel, 
                kernel_sizes=kernel_sizes,
                layer_id=0,
                distribution=distribution
            )
        )
        
        for ii in range(n_layer - 2):
            self.layers.append(
                GradlessGCReplayNonlinBlock(
                    out_channel=interm_channel, 
                    in_channel=interm_channel, 
                    kernel_sizes=kernel_sizes,
                    layer_id=ii + 1,
                    distribution=distribution
                )
            )
    
        self.layers.append(
            GradlessGCReplayNonlinBlock(
                out_channel=out_channel, 
                in_channel=interm_channel, 
                kernel_sizes=kernel_sizes,
                layer_id=n_layer - 1, 
                use_act=False,
                distribution=distribution
            )
        )

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x_in):
        if isinstance(x_in, list):
            x_in = torch.cat(x_in, dim=0)

        nb, nc, nx, ny = x_in.shape

        alphas = torch.rand(nb, device=x_in.device)[:, None, None, None]
        alphas = alphas.repeat(1, nc, 1, 1)

        x = self.layers[0](x_in)
        for blk in self.layers[1:]:
            x = blk(x)

        #mixed = alphas * x + (1.0 - alphas) * x_in
        mixed = x

        if self.out_norm == 'frob':
            _in_frob = torch.norm(x_in.view(nb, nc, -1), dim=(-1, -2), p='fro', keepdim=False)
            _in_frob = _in_frob[:, None, None, None].repeat(1, nc, 1, 1)
            _self_frob = torch.norm(mixed.view(nb, self.out_channel, -1), dim=(-1,-2), p='fro', keepdim=False)
            _self_frob = _self_frob[:, None, None, None].repeat(1, self.out_channel, 1, 1)
            mixed = mixed * (1.0 / (_self_frob + 1e-5)) * _in_frob

        return mixed


class RandConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, interm_channel=16, kernel_sizes=[1, 3, 5], n_layer=4,
                 mixing=False, identity_prob=0.0, distribution='kaiming_normal'):
        super(RandConv2D, self).__init__()
        
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]
            
        self.mixing = mixing
        self.identity_prob = identity_prob
        self.distribution = distribution
        
        self.gin_conv = GINGroupConv(
            out_channel=out_channels,
            in_channel=in_channels, 
            interm_channel = interm_channel,
            kernel_sizes=kernel_sizes,
            n_layer = n_layer,
            distribution=distribution
        )
        
        if self.mixing:
            self.alpha = random.random()

    def forward(self, input):
        if self.identity_prob > 0 and torch.rand(1) < self.identity_prob:
            return input

        output = self.gin_conv(input)
        
        if self.mixing:
            output = (self.alpha * output + (1 - self.alpha) * input)

        return output

    def randomize(self):
        if self.mixing:
            self.alpha = random.random()



####################### 3D  ########################

class GradlessGCReplayNonlinBlock3D(nn.Module):
    def __init__(self, out_channel=3, in_channel=3, kernel_sizes=[1, 3], layer_id=0, use_act=True,
                 requires_grad=False, distribution='kaiming_normal', **kwargs):
        super(GradlessGCReplayNonlinBlock3D, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]
        self.kernel_sizes = kernel_sizes

        self.layer_id = layer_id
        self.use_act = use_act
        self.requires_grad = requires_grad
        self.distribution = distribution
        assert requires_grad == False

    def _initialize_weights(self, kernel_size):
        nb = self.current_batch_size
        #nc = self.in_channel
        oc = self.out_channel
        
        # Adjusted weight shape for proper 3D group convolution
        weight_shape = [oc * nb, self.in_channel, kernel_size, kernel_size, kernel_size]

        if self.distribution == 'kaiming_normal':
            fan = nn.init._calculate_correct_fan(torch.zeros(weight_shape), 'fan_in')
            gain = nn.init.calculate_gain('leaky_relu', 0.2)
            std = gain / math.sqrt(fan)
            weight = torch.randn(weight_shape) * std
        elif self.distribution == 'kaiming_uniform':
            fan = nn.init._calculate_correct_fan(torch.zeros(weight_shape), 'fan_in')
            gain = nn.init.calculate_gain('leaky_relu', 0.2)
            bound = gain * math.sqrt(3.0 / fan)
            weight = torch.rand(weight_shape) * 2 * bound - bound
        else:
            weight = torch.randn(weight_shape)

        return weight

    def forward(self, x_in):
        k = random.choice(self.kernel_sizes)  # randomly select kernel size
        self.current_batch_size = x_in.shape[0]
        
        nb, nc, nd, nx, ny = x_in.shape  # batch, channels, depth, height, width
        
        # Initialize kernel weights
        ker = self._initialize_weights(k).to(x_in.device)
        
        # Correct bias (shift)
        shift = torch.randn([nb, self.out_channel, nd, nx, ny], device=x_in.device) * 0.1

        # Reshape input for grouped convolution
        x_in_reshaped = x_in.view(1, nb * nc, nd, nx, ny)
        #print(f"x_in_reshaped shape: {x_in_reshaped.shape}")
        #print(f"ker shape: {ker.shape}")
        
        # Perform grouped convolution with proper groups
        x_conv = F.conv3d(
            x_in_reshaped, ker, stride=1, padding=k // 2, dilation=1, groups=nb
        )
        x_conv = x_conv.view(nb, self.out_channel, nd, nx, ny)  # Reshape to match batch size and output channels

        # Ensure the shift can broadcast properly
        x_conv = x_conv + shift

        # Apply activation function
        if self.use_act:
            x_conv = F.leaky_relu(x_conv, 0.2)

        return x_conv


class GINGroupConv3D(nn.Module):
    def __init__(self, in_channel, out_channel, interm_channel=16, kernel_sizes=[1, 3, 5], n_layer=4,
                 out_norm='frob', distribution='kaiming_normal', **kwargs):
        super(GINGroupConv3D, self).__init__()

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]
        self.kernel_sizes = kernel_sizes

        self.n_layer = n_layer
        self.layers = []
        self.out_norm = out_norm
        self.out_channel = out_channel
        self.distribution = distribution

        self.layers.append(
            GradlessGCReplayNonlinBlock3D(
                out_channel=interm_channel,
                in_channel=in_channel,
                kernel_sizes=kernel_sizes,
                layer_id=0,
                distribution=distribution
            )
        )

        for ii in range(n_layer - 2):
            self.layers.append(
                GradlessGCReplayNonlinBlock3D(
                    out_channel=interm_channel,
                    in_channel=interm_channel,
                    kernel_sizes=kernel_sizes,
                    layer_id=ii + 1,
                    distribution=distribution
                )
            )

        self.layers.append(
            GradlessGCReplayNonlinBlock3D(
                out_channel=out_channel,
                in_channel=interm_channel,
                kernel_sizes=kernel_sizes,
                layer_id=n_layer - 1,
                use_act=False,
                distribution=distribution
            )
        )

        self.layers = nn.ModuleList(self.layers)


    def forward(self, x_in):
        if isinstance(x_in, list):
            x_in = torch.cat(x_in, dim=0)

        nb, nc, nd, nx, ny = x_in.shape

        # Random alpha per sample, broadcast to (nb, 1, 1, 1, 1)
        alphas = torch.rand(nb, device=x_in.device)[:, None, None, None, None]
        alphas = alphas.repeat(1, nc, 1, 1, 1)

        #print('#############################################')
        #print(x_in.shape)
        #print(self.layers)
        #print('#############################################')

        x = self.layers[0](x_in)
        for blk in self.layers[1:]:
            x = blk(x)

        #mixed = alphas * x + (1.0 - alphas) * x_in
        mixed = x

        if self.out_norm == 'frob':
            # Frobenius norm over channels and all spatial dimensions (depth, height, width)
            _in_frob = torch.norm(x_in.view(nb, nc, -1), dim=(-1, -2), p='fro', keepdim=False)  # shape: (nb,)
            _in_frob = _in_frob[:, None, None, None, None].repeat(1, nc, 1, 1, 1)

            _self_frob = torch.norm(mixed.view(nb, self.out_channel, -1), dim=(-1, -2), p='fro', keepdim=False)
            _self_frob = _self_frob[:, None, None, None, None].repeat(1, self.out_channel, 1, 1, 1)

            mixed = mixed * (1.0 / (_self_frob + 1e-5)) * _in_frob

        return mixed


class RandConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, interm_channel=16, kernel_sizes=[1, 3, 5], n_layer=4,
                 mixing=False, identity_prob=0.0, distribution='kaiming_normal'):
        super(RandConv3D, self).__init__()

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]

        self.mixing = mixing
        self.identity_prob = identity_prob
        self.distribution = distribution

        self.gin_conv3d = GINGroupConv3D(
            in_channel=in_channels,
            out_channel=out_channels,
            interm_channel = interm_channel,
            kernel_sizes=kernel_sizes,
            n_layer = n_layer,
            distribution=distribution
        )



        if self.mixing:
            self.alpha = random.random()

    def forward(self, input):
        if self.identity_prob > 0 and torch.rand(1) < self.identity_prob:
            return input

        output = self.gin_conv3d(input)

        if self.mixing:
            output = (self.alpha * output + (1 - self.alpha) * input)

        return output

    def randomize(self):
        if self.mixing:
            self.alpha = random.random()
