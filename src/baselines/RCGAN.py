import torch
from src.baselines.base import BaseTrainer
from tqdm import tqdm
from torch.nn.functional import one_hot
from os import path as pt
from src.utils import save_obj


class RCGANTrainer(BaseTrainer):
    def __init__(self, D, G, train_dl, config,
                 **kwargs):
        super(RCGANTrainer, self).__init__(
            G=G,
            G_optimizer=torch.optim.Adam(
                G.parameters(), lr=config.lr_G, betas=(0, 0.9)),
            **kwargs
        )

        self.config = config
        self.D_steps_per_G_step = config.D_steps_per_G_step
        self.D = D
        self.D_optimizer = torch.optim.Adam(
            D.parameters(), lr=config.lr_D, betas=(0, 0.9))  # Using TTUR

        self.train_dl = train_dl
        self.reg_param = 0
        self.losses_history

    def fit(self, device):
        self.G.to(device)
        self.D.to(device)

        for i in tqdm(range(self.n_gradient_steps)):
            self.step(device, i)

    def step(self, device, step):
        for i in range(self.D_steps_per_G_step):
            # generate x_fake
            condition = None
            x_real_batch = next(iter(self.train_dl))[0].to(device)
            with torch.no_grad():
                x_fake = self.G(batch_size=self.batch_size,
                                n_lags=self.config.n_lags, condition=condition, device=device)

            D_loss_real, D_loss_fake = self.D_trainstep(
                x_fake, x_real_batch)
            if i == 0:
                self.losses_history['D_loss_fake'].append(D_loss_fake)
                self.losses_history['D_loss_real'].append(D_loss_real)
                self.losses_history['D_loss'].append(D_loss_fake + D_loss_real)
        G_loss = self.G_trainstep(x_real_batch, device, step)

    def G_trainstep(self, x_real, device, step):
        condition = None
        x_fake = self.G(batch_size=self.batch_size,
                        n_lags=self.config.n_lags, condition=condition, device=device)
        self.toggle_grad(self.G, True)
        self.G.train()
        self.G_optimizer.zero_grad()
        d_fake = self.D(x_fake)
        self.D.train()
        G_loss = self.compute_loss(d_fake, 1.)
        G_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.G.parameters(), 10)
        self.losses_history['G_loss'].append(G_loss)
        self.G_optimizer.step()

        return G_loss.item()

    def D_trainstep(self, x_fake, x_real):
        self.toggle_grad(self.D, True)
        self.D.train()
        self.D_optimizer.zero_grad()

        # On real data
        x_real.requires_grad_()
        d_real = self.D(x_real)
        dloss_real = self.compute_loss(d_real, 1.)

        # On fake data
        x_fake.requires_grad_()
        d_fake = self.D(x_fake)
        dloss_fake = self.compute_loss(d_fake, 0.)

        # Compute regularizer on fake / real
        dloss = dloss_fake + dloss_real

        dloss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.D.parameters(), 10)
        # Step discriminator params
        self.D_optimizer.step()

        # Toggle gradient to False
        self.toggle_grad(self.D, False)

        return dloss_real.item(), dloss_fake.item()

    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        loss = torch.nn.BCELoss()(torch.nn.Sigmoid()(d_out), targets)
        return loss

    def save_model_dict(self):
        save_obj(self.G.state_dict(), pt.join(
            self.config.exp_dir, 'generator_state_dict.pt'))
