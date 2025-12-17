
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def prod(t):
    p = 1
    for v in t:
        p *= v
    return p

class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(True))
            last = h
        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(last, latent_dim)
        self.fc_logvar = nn.Linear(last, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim, use_sigmoid=False):
        super().__init__()
        layers = []
        last = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(True))
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)
        self.use_sigmoid = use_sigmoid

    def forward(self, z):
        output = self.net(z)
        return torch.sigmoid(output) if self.use_sigmoid else output


def log_normal_diag(x, mu, logvar):
    # x, mu, logvar: (..., D)
    # returns (...,) log-density
    # add small epsilon for numerical stability when converting logvar -> var
    var = torch.exp(logvar) + 1e-6
    return -0.5 * ( (math.log(2 * math.pi) + torch.log(var)) + ((x - mu) ** 2) / var ).sum(dim=-1)


class GMMVAE(nn.Module):
    """A small VAE with a learnable GMM prior in latent space.

    - Encoder/decoder are small MLPs suitable for CIFAR-scale inputs.
    - Prior is a mixture of diagonal Gaussians with learnable pi, mu_c, logvar_c.
    """

    def __init__(self, input_shape, embedding_dim=32, num_classes=10, hidden_enc=(512, 256), hidden_dec=(256, 512)):
        super(GMMVAE, self).__init__()
        if isinstance(input_shape, int):
            input_dim = input_shape
        else:
            input_dim = prod(input_shape)
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.latent_dim = embedding_dim
        self.num_components = num_classes

        self.encoder = SimpleEncoder(input_dim, hidden_enc, embedding_dim)
        self.decoder = SimpleDecoder(embedding_dim, hidden_dec, input_dim)

        # GMM prior params
        # mixture logits (unnormalized)
        self.pi_logits = nn.Parameter(torch.zeros(self.num_components))
        # component means and logvars
        self.comp_mu = nn.Parameter(torch.randn(self.num_components, self.latent_dim) * 0.01)
        self.comp_logvar = nn.Parameter(torch.zeros(self.num_components, self.latent_dim))

    def encode(self, x):
        # x: (B, C, H, W) or (B, D)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # clamp logvar to avoid extreme std values that cause NaNs/Infs
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return {
            'recon': recon,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }

    def compute_loss(self, x, outputs, recon_loss_type='mse', beta=1.0):
        """Returns a dict with total loss and components.

        recon_loss_type: 'mse' or 'bce'
        beta: weight for KL term (beta-VAE). Use beta < 1 to prioritize reconstruction.
              beta=0 → pure autoencoder, beta=1 → standard VAE
        """
        recon = outputs['recon']
        mu = outputs['mu']
        logvar = outputs['logvar']
        z = outputs['z']

        if x.dim() > 2:
            x_flat = x.view(x.size(0), -1)
        else:
            x_flat = x

        # Use per-element mean to keep reconstruction loss magnitude stable
        if recon_loss_type == 'bce':
            recon_loss = F.binary_cross_entropy(recon, x_flat, reduction='mean')
        else:
            recon_loss = F.mse_loss(recon, x_flat, reduction='mean')

        # log q(z|x)
        log_qzx = log_normal_diag(z, mu, logvar)  # (B,)

        # log p(z) = log sum_c pi_c N(z|mu_c, var_c)
        log_pi = F.log_softmax(self.pi_logits, dim=0)  # (K,)
        # compute log N(z|mu_c,var_c) for each component
        # z: (B, D); comp_mu: (K, D) -> we want (B, K)
        z_exp = z.unsqueeze(1)  # (B,1,D)
        comp_mu = self.comp_mu.unsqueeze(0)  # (1,K,D)
        # stabilize component variances too
        comp_logvar = self.comp_logvar.unsqueeze(0)  # (1,K,D)
        comp_var = torch.exp(comp_logvar) + 1e-6
        log_nk = -0.5 * ( (math.log(2 * math.pi) + torch.log(comp_var)) + ((z_exp - comp_mu) ** 2) / comp_var ).sum(dim=-1)  # (B,K)
        log_components = log_pi.unsqueeze(0) + log_nk  # (B,K)
        log_pz = torch.logsumexp(log_components, dim=1)  # (B,)

        kl = (log_qzx - log_pz).mean()

        loss = recon_loss + beta * kl

        # responsibilities (soft assignments) for monitoring
        gamma = F.softmax(log_components, dim=1)  # (B,K)

        return {
            'loss': loss,
            'recon_loss': recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss,
            'kl': kl.item() if isinstance(kl, torch.Tensor) else kl,
            'gamma': gamma,
            'log_pz': log_pz
        }

    def sample_from_prior(self, num_samples=16):
        # sample component according to pi, then sample from component Gaussian
        with torch.no_grad():
            pi = F.softmax(self.pi_logits, dim=0)
            comps = torch.multinomial(pi, num_samples=num_samples, replacement=True)
            mu = self.comp_mu[comps]
            logvar = self.comp_logvar[comps]
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            x = self.decode(z)
            if self.input_dim is not None:
                return x.view(num_samples, *self.input_shape)
            return x


class VaDE(GMMVAE):
    """VaDE: Variational Deep Embedding.

    For this simple implementation we reuse GMMVAE behavior. VaDE typically
    pretrains a VAE and fits a GMM on latent means; here we expose the same
    interface but the training script can choose to initialize the GMM from
    a fitted sklearn GaussianMixture if desired.
    """

    def __init__(self, input_shape, embedding_dim=32, num_classes=10, **kwargs):
        super().__init__(input_shape, embedding_dim, num_classes, **kwargs)


class ResBlock(nn.Module):
    """Residual block for conv encoder/decoder."""
    def __init__(self, channels, use_batchnorm=True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class ConvEncoder(nn.Module):
    """Conv encoder with optional residual blocks for better reconstruction.

    Assumes input_shape = (C, H, W) and that H and W are divisible by 2**len(channels).
    """
    def __init__(self, input_shape, channels=(32, 64), latent_dim=32, kernel_size=3, use_batchnorm=True, use_residual=False, num_residual_blocks=1, conv_strides=None):
        super().__init__()
        kernel_size = min(input_shape[-1], input_shape[-2], kernel_size) # Hack
        C, H, W = input_shape
        self.use_residual = use_residual
        # Use per-layer stride if provided, else default to 2
        if conv_strides is None:
            conv_strides = [2] * len(channels)
        assert len(conv_strides) == len(channels), "conv_strides must match conv_channels length"
        # Build encoder layers
        layers = []
        in_ch = C
        self.spatial_sizes = []
        for idx, (ch, stride) in enumerate(zip(channels, conv_strides)):
            layers.append(nn.Conv2d(in_ch, ch, kernel_size=kernel_size, stride=stride, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.ReLU(True))
            if use_residual:
                for _ in range(num_residual_blocks):
                    layers.append(ResBlock(ch, use_batchnorm))
            in_ch = ch
        self.conv = nn.Sequential(*layers)
        # compute spatial size after convs
        Hf, Wf = H, W
        for stride in conv_strides:
            Hf = (Hf + 2*1 - 1*(kernel_size-1) - 1) // stride + 1
            Wf = (Wf + 2*1 - 1*(kernel_size-1) - 1) // stride + 1
        self.flatten_dim = in_ch * Hf * Wf
        # FC hidden dimension - keep reasonable size
        fc_hidden = max(256, latent_dim * 4)
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, fc_hidden),
            nn.ReLU(True)
        )
        self.fc_mu = nn.Linear(fc_hidden, latent_dim)
        self.fc_logvar = nn.Linear(fc_hidden, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        return self.fc_mu(h), self.fc_logvar(h)


class ConvDecoder(nn.Module):
    """Conv transpose decoder with optional residual blocks for better reconstruction.

    Decoder outputs a flattened vector of size (B, C*H*W) to be compatible with
    the existing compute_loss which flattens inputs.
    """
    def __init__(self, input_shape, channels=(32, 64), latent_dim=32, kernel_size=3, use_sigmoid=False, use_batchnorm=True, use_residual=False, num_residual_blocks=1, conv_strides=None):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.use_residual = use_residual
        C, H, W = input_shape
        n_down = len(channels)
        # Use per-layer stride if provided, else default to 2
        if conv_strides is None:
            conv_strides = [2] * len(channels)
        assert len(conv_strides) == len(channels), "conv_strides must match conv_channels length"
        # Compute spatial size after convs (reverse order)
        Hf, Wf = H, W
        for stride in conv_strides:
            Hf = (Hf + 2*1 - 1*(kernel_size-1) - 1) // stride + 1
            Wf = (Wf + 2*1 - 1*(kernel_size-1) - 1) // stride + 1
        self.Hf = Hf
        self.Wf = Wf
        self.out_ch = channels[-1]
        # FC layer to project latent to feature map size
        self.fc_hidden_dim = self.out_ch * Hf * Wf
        fc_hidden = max(256, latent_dim * 4)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, fc_hidden),
            nn.ReLU(True),
            nn.Linear(fc_hidden, self.fc_hidden_dim),
            nn.ReLU(True)
        )
        # Build decoder layers (reverse of encoder)
        layers = []
        in_ch = self.out_ch
        for i in range(len(channels) - 1, 0, -1):
            out_ch = channels[i - 1]
            stride = conv_strides[i]  # reverse order, but keep stride for each layer
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=1, output_padding=stride-1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(True))
            if use_residual:
                for _ in range(num_residual_blocks):
                    layers.append(ResBlock(out_ch, use_batchnorm))
            in_ch = out_ch
        # Final upsampling layer to original image
        stride = conv_strides[0]
        layers.append(nn.ConvTranspose2d(in_ch, C, kernel_size=kernel_size, stride=stride, padding=1, output_padding=stride-1))
        self.deconv = nn.Sequential(*layers)
        self.input_shape = input_shape

    def forward(self, z):
        # z: (B, latent_dim)
        B = z.size(0)
        C, H, W = self.input_shape
        
        # Project to feature map
        h = self.fc(z)
        h = h.view(B, self.out_ch, self.Hf, self.Wf)
        
        # Upsample through deconv layers
        x = self.deconv(h)
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        return x.view(B, -1)


class GMMVAE_CNN(GMMVAE):
    """GMMVAE variant that uses ConvEncoder/ConvDecoder.

    It keeps the same training & loss interface as `GMMVAE`.
    """
    def __init__(self, input_shape, embedding_dim=32, num_classes=10, conv_channels=(32, 64), use_residual=False, num_residual_blocks=1, conv_strides=None):
        # bypass parent init that creates MLPs; we'll call nn.Module.__init__ via super and then set fields
        super(GMMVAE_CNN, self).__init__(input_shape, embedding_dim, num_classes)
        # replace encoder/decoder with conv versions
        self.encoder = ConvEncoder(input_shape, channels=conv_channels, latent_dim=embedding_dim, 
                                   use_batchnorm=True, use_residual=use_residual, num_residual_blocks=num_residual_blocks, conv_strides=conv_strides)
        self.decoder = ConvDecoder(input_shape, channels=conv_channels, latent_dim=embedding_dim,
                    use_sigmoid=True, use_batchnorm=False, use_residual=use_residual, num_residual_blocks=num_residual_blocks, conv_strides=conv_strides)
        
    def encode(self, x):
        # x: (B, C, H, W) - don't flatten for CNN encoder
        mu, logvar = self.encoder(x)
        return mu, logvar

class VaDE_CNN(GMMVAE_CNN):
    def __init__(self, input_shape, embedding_dim=32, num_classes=10, conv_channels=(32, 64)):
        super().__init__(input_shape, embedding_dim, num_classes, conv_channels=conv_channels)


MODELS = {
    'GMMVAE': GMMVAE,
    'VaDE': VaDE,
    'GMMVAE_CNN': GMMVAE_CNN,
    'VaDE_CNN': VaDE_CNN,
}