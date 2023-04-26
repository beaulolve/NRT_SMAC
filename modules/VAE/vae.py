import torch
from torch import nn
from torch.nn import functional as F


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class BetaVAE(nn.Module):

    def __init__(self,
                 device,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dim: int = 128,
                 kld_weight: float = 0.005):

        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.num_iter = 0  # keep track of iterations
        self.kld_weight = kld_weight

        active_func = nn.ReLU()

        self.encoder = nn.Sequential(nn.Linear(in_channels, hidden_dim), active_func, nn.LayerNorm(
            hidden_dim), nn.Linear(hidden_dim, hidden_dim), active_func)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), active_func, nn.LayerNorm(hidden_dim),
                                     nn.Linear(hidden_dim, hidden_dim), active_func, nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, in_channels))
        self.apply(weights_init_)
        self.to(device)

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        """
        result = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        result = self.decoder(z)
        return result

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough to compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        logpi = -F.mse_loss(self.decode(z), input,
                            reduction="none").sum(dim=-1)
        return logpi

    def get_logp(self, input):
        # old is :
        # return self.forward(input)

        z = torch.randn(input.shape[:-1] +
                        (32, self.latent_dim)).to(input.device)
        input_expand = input.unsqueeze(-2).expand(
            z.shape[:-1] + (input.shape[-1],))
        logpi = -F.mse_loss(self.decode(z), input_expand,
                            reduction="none").sum(dim=-1)
        pi = logpi.exp().mean(dim=-1)
        return pi.clamp(min=1e-10).log()

    def get_loss(self, input) -> dict:

        self.num_iter += 1
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        input_hat = self.decode(z)

        recons_loss = F.mse_loss(input_hat, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 +
                              log_var - mu ** 2 - log_var.exp(), dim=-1))
        loss = recons_loss + self.kld_weight * kld_loss

        return loss

        # return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}
