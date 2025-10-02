import torch
import torch.nn as nn
import torch.distributions


from utils.smooth_clip import softclip
from utils.latent_distribution import LatentData, LogScaleNormal
from models.utils.blocks import SelfAttentionBlock, ConvBlock, DeconvBlock

from supn_base.supn_data import SUPNData, get_num_off_diag_weights, get_num_cross_channel_weights
from supn_base.supn_distribution import SUPN

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) implementation in PyTorch.
    """
    def __init__(self, 
                 image_size=64,
                 latent_dim=64, 
                 local_connection_dist=5,
                 num_channels = 3,
                 use_attention=False, 
                 use_group_norm=True,
                 init_decoder_var=1,
                 activation = nn.SiLU()) -> None:
        """
        Initialize the VAE model.
        
        Args:
            latent_dim (int): Dimension of the latent space. 
            local_connection_dist (int): Local connection distance for the SUPN distribution.
            use_attention (bool): Whether to use self-attention.
            init_decoder_var (float): Initial variance for the decoder.
        """
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.local_connection_dist = local_connection_dist
        self.num_channels = num_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.use_attention = use_attention
        self.use_group_norm = use_group_norm
        self.init_decoder_logvar = torch.log(init_decoder_var)
        self.activation = activation

        self.num_off_diag_weights = get_num_off_diag_weights(self.local_connection_dist)
        self.num_channel_weights = get_num_cross_channel_weights(self.num_channels)
        self.base_pars = get_VAE_pars(self.image_size,self.num_off_diag_weights,self.num_channel_weights,num_channels=self.num_channels)


        # Encoder layers
        self.encoder_conv_layers = []
        for i, layer in enumerate(self.base_pars['encoder']):
            self.encoder_conv_layers.append(ConvBlock(in_dim = layer['in_ch'], 
                                                 out_dim = layer['out_ch'], 
                                                 kernel_size = layer['kernel'], 
                                                 stride = layer['stride'], 
                                                 padding = layer['padding'],
                                                 use_group_norm = self.use_group_norm))
            # Add cross attention after second and fourth conv layers
            if i in self.base_pars['attention_layers_enc'] and use_attention:
                self.encoder_conv_layers.append(SelfAttentionBlock(layer['out_ch']))
        
        # Latent projections
        self.fc_mu = nn.Linear(self.base_pars['bottleneck_size'], latent_dim)
        self.fc_logvar = nn.Linear(self.base_pars['bottleneck_size'], latent_dim)


        # Decoder mean layers
        self.decoder_mean_deconv_layers = []
        for i, layer in enumerate(self.base_pars['decoder_mean']):
            end = True if i == len(self.base_pars['decoder_mean']) - 1 else False
            self.decoder_mean_deconv_layers.append(DeconvBlock(layer['in_ch'],
                                                   layer['out_ch'],
                                                   kernel_size = layer['kernel'],
                                                   stride = layer['stride'],
                                                   padding = layer['padding'],
                                                   use_group_norm=self.use_group_norm,
                                                   end = end))
            # Add cross attention after second and fourth deconv layers
            if i in self.base_pars['attention_layers_dec'] and use_attention:
                self.decoder_mean_deconv_layers.append(SelfAttentionBlock(layer['out_ch']))

        self.fc_zmu = nn.Linear(latent_dim, self.base_pars['bottleneck_size'])


        # Decoder cholesky layers
        self.decoder_chol_deconv_layers = []
        for i, layer in enumerate(self.base_pars['decoder_chol']):
            self.decoder_chol_deconv_layers.append(DeconvBlock(layer['in_ch'],
                                                   layer['out_ch'],
                                                   kernel_size = layer['kernel'],
                                                   stride = layer['stride'],
                                                   padding = layer['padding'],
                                                   use_group_norm=self.use_group_norm,
                                                   end = end))
            # Add cross attention after second and fourth deconv layers
            if i in self.base_pars['attention_layers_dec'] and use_attention:
                self.decoder_chol_deconv_layers.append(SelfAttentionBlock(layer['out_ch']))

        self.fc_zchol = nn.Linear(latent_dim, self.base_pars['bottleneck_size'])

        # separate parameters into module dicts so that they can be optimized separately
        self.params = nn.ModuleDict({
            'encoder': nn.ModuleList([layer for layer in self.encoder_conv_layers]+[self.fc_logvar, self.fc_mu]),
            'decoder_mean': nn.ModuleList([layer for layer in self.decoder_mean_deconv_layers]+[self.fc_zmu]),
            'decoder_chol': nn.ModuleList([layer for layer in self.decoder_chol_deconv_layers]+[self.fc_zchol])})

        self.ones_scaling = torch.load(f'scaling{self.image_size}.pt',map_location=self.device)
        self.scaling = nn.Parameter(torch.ones(1,1,self.image_size,self.image_size))
        self.min = nn.Parameter(torch.tensor(-5.0))
        self.max = nn.Parameter(torch.tensor(5.0))

    def encode(self, 
               x: torch.tensor) -> LogScaleNormal:
        """
        Encode the input into the latent space.
        
        Args:
            x (torch.Tensor): (Batch of) Input tensor.
        
        Returns:
            LogScaleNormal: A LogScaleNormal object containing the latent distribution.
        """
        if x.ndim == 5:
            x = x.unsqueeze(1)
        h = x
        for layer in self.encoder_conv_layers:
            h = layer(h)
        h = h.view(h.size(0), -1)
        latent_data = LatentData(self.fc_mu(h), self.fc_logvar(h))
        return torch.distributions.LogScaleNormal(latent_data)
    

    def decode(self, 
               z: torch.tensor,
               data_mode: bool = False) -> SUPNData:
        """
        Decode the latent vector into the reconstructed input.
        
        Args:
            z (torch.Tensor): Latent vector.
        
        Returns:
            SUPNData: A SUPNData object containing the decoded SUPN distribution.
        """
        h = self.fc_zmu(z)
        h = h.view([h.size(0)]+self.base_pars['bottleneck_shape'])
        for layer in self.decoder_mean_deconv_layers:
            h = layer(h)

        hc = self.fc_zchol(z)
        hc = hc.view([hc.size(0)]+self.base_pars['bottleneck_shape'])
        for layer in self.decoder_chol_deconv_layers:
            hc = layer(hc)

        h_diag = hc[:,:self.num_channels].reshape([h.shape[0],self.num_channels,h.shape[2],h.shape[3]]) 


        h_chol = hc[:,self.num_channels:-self.num_channel_weights].reshape([h.shape[0],get_num_off_diag_weights(self.local_connection_dist)*self.num_channels,h.shape[2],h.shape[3]])
        h_chan = hc[:,-self.num_channel_weights:].reshape([h.shape[0],self.num_channel_weights,h.shape[2],h.shape[3]])

        mean = h
        #add scaling 
        log_diag = softclip(h_diag*self.scaling/(self.ones_scaling+1e-2)*0.001 - self.init_decoder_logvar)
        off_diag_neighbours = h_chol*self.scaling/(self.ones_scaling+1e-2)*0.001
        off_diag_channels   = h_chan*self.scaling/(self.ones_scaling+1e-2)*0.001
        if data_mode:
            return SUPNData(mean, log_diag, off_diag_neighbours, local_connection_dist=self.local_connection_dist, cross_ch=off_diag_channels)
        else:
            return torch.distributions.SUPN(SUPNData(mean, log_diag, off_diag_neighbours, local_connection_dist=self.local_connection_dist, cross_ch=off_diag_channels))
        #return SUPNData(torch.tanh(h), torch.tanh(hv), torch.tanh(hc), self.local_connection_dist)

    def forward(self, 
                x: torch.tensor) -> tuple[SUPN, LogScaleNormal]:
        """
        Forward pass through the VAE.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            supn_dist (SUPN): Sampled SUPN distribution object for the given input.
            latent_dist (LogScaleNormal): Latent distribution object for the given input.
        """
        latent_dist = self.encode(x)
        z = latent_dist.rsample()
        supn_dist = self.decode(z)
        return supn_dist, latent_dist



def get_VAE_pars(image_size,num_off_diag_weights, num_channels_weights,num_channels):
    """
    Get VAE architecture parameters with cross attention layers.
    Uses 5x5 initial kernel for 64x64 and 7x7 for larger resolutions.
    
    Args:
        image_size (int): Input image size (64, 128, 256, or 512)
    
    Returns:
        dict: Dictionary containing VAE architecture parameters
    """
    base_pars = {
        64: {
            'encoder': [
                {'in_ch': num_channels, 'out_ch': 32, 'kernel': 5, 'stride': 2, 'padding': 1},
                {'in_ch': 32, 'out_ch': 64, 'kernel': 3, 'stride': 2, 'padding': 1},
                {'in_ch': 64, 'out_ch': 128, 'kernel': 3, 'stride': 2, 'padding': 1},
                {'in_ch': 128, 'out_ch': 256, 'kernel': 3, 'stride': 2, 'padding': 1}
            ],
            'decoder_mean': [
                {'in_ch': 256, 'out_ch': 128, 'kernel': 3, 'stride': 2, 'padding': 1},
                {'in_ch': 128, 'out_ch': 64, 'kernel': 3, 'stride': 2, 'padding': 1},
                {'in_ch': 64, 'out_ch': 32, 'kernel': 3, 'stride': 2, 'padding': 1},
                {'in_ch': 32, 'out_ch': num_channels, 'kernel': 3, 'stride': 2, 'padding': 1}
            ],
            'decoder_chol': [
                {'in_ch': 256, 'out_ch': 128, 'kernel': 3, 'stride': 2, 'padding': 1},
                {'in_ch': 128, 'out_ch': 64, 'kernel': 3, 'stride': 2, 'padding': 1},
                {'in_ch': 64, 'out_ch': 32, 'kernel': 3, 'stride': 2, 'padding': 1},
                {'in_ch': 32, 'out_ch': num_off_diag_weights*num_channels + num_channels_weights + num_channels, 'kernel': 3, 'stride': 2, 'padding': 1}
            ],
            'attention_layers_enc': [1, 3],
            'attention_layers_dec': [0, 2],
            'bottleneck_shape': [256, 4, 4],
            'bottleneck_size': 256*4*4
        },
        128: {
            'encoder': [
                {'in_ch': num_channels, 'out_ch': 32, 'kernel': 9, 'stride': 2, 'padding': 1},
                {'in_ch': 32, 'out_ch': 64, 'kernel': 3, 'stride': 2, 'padding': 1},
                {'in_ch': 64, 'out_ch': 128, 'kernel': 3, 'stride': 2, 'padding': 1},
                {'in_ch': 128, 'out_ch': 256, 'kernel': 3, 'stride': 2, 'padding': 1},
                {'in_ch': 256, 'out_ch': 256, 'kernel': 3, 'stride': 2, 'padding': 1}
            ],
            'decoder_mean': [
                {'in_ch': 256, 'out_ch': 256, 'kernel': 3, 'stride': 2, 'padding': 1},
                {'in_ch': 256, 'out_ch': 128, 'kernel': 3, 'stride': 2, 'padding': 1},
                {'in_ch': 128, 'out_ch': 64, 'kernel': 3, 'stride': 2, 'padding': 1},
                {'in_ch': 64, 'out_ch': 32, 'kernel': 3, 'stride': 2, 'padding': 1},
                {'in_ch': 32, 'out_ch': num_channels, 'kernel': 9, 'stride': 2, 'padding': 4}
            ],
            'decoder_chol': [
                {'in_ch': 256, 'out_ch': 256, 'kernel': 3, 'stride': 2, 'padding': 1},
                {'in_ch': 256, 'out_ch': 128, 'kernel': 3, 'stride': 2, 'padding': 1},
                {'in_ch': 128, 'out_ch': 64, 'kernel': 3, 'stride': 2, 'padding': 1},
                {'in_ch': 64, 'out_ch': 32, 'kernel': 3, 'stride': 2, 'padding': 1},
                {'in_ch': 32, 'out_ch': num_off_diag_weights*num_channels + num_channels_weights + num_channels, 'kernel': 9, 'stride': 2, 'padding': 4}
            ],
            'attention_layers_enc': [1, 3],
            'attention_layers_dec': [0, 2],
            'bottleneck_shape': [256, 4, 4],
            'bottleneck_size': 256*4*4
        }
    }
    
    # Create identical structure for larger sizes, maintaining 7x7 initial kernel
    base_pars[256] = dict(base_pars[128])
    base_pars[256]['bottleneck_shape'] = [512, 8, 8]
    base_pars[256]['bottleneck_size'] = 512 * 8 * 8
    
    base_pars[512] = dict(base_pars[128])
    base_pars[512]['bottleneck_shape'] = [512, 16, 16]
    base_pars[512]['bottleneck_size'] = 512 * 16 * 16
    
    return base_pars[image_size]