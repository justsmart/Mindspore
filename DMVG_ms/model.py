# %% [1]import
import mindspore as ms
from mindspore import nn, ops


# %% [2]AE
def Normalize(in_channels):
    # return nn.GroupNorm(num_groups=8, num_channels=in_channels, eps=1e-5, affine=True)
    return nn.BatchNorm1d(in_channels)
    # return nn.BatchNorm2d(in_channels)


class ResBlock1d(nn.Cell):
    def __init__(self, in_channels, out_channels, is_res=False):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.SequentialCell(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=1, pad_mode="pad", padding=2),
            Normalize(out_channels),
            nn.GELU(approximate=False),  # nn.SiLU()
        )
        self.conv2 = nn.SequentialCell(
            nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, pad_mode="pad", padding=2),
            Normalize(out_channels),
            nn.GELU(approximate=False),  # nn.SiLU()
        )

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class ResBlock2d(nn.Cell):
    def __init__(self, in_channels, out_channels, is_res=False):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.SequentialCell(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, pad_mode="pad", padding=1),
            Normalize(out_channels),
            nn.GELU(approximate=False),
        )
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, pad_mode="pad", padding=1),
            Normalize(out_channels),
            nn.GELU(approximate=False),
        )

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class Down1d(nn.Cell):
    def __init__(self, in_channels, out_channels, down):
        super().__init__()
        self.model = nn.SequentialCell(
            ResBlock1d(in_channels, out_channels),
            nn.MaxPool1d(down, stride=down),
        )

    def construct(self, x):
        return self.model(x)


class Down2d(nn.Cell):
    def __init__(self, in_channels, out_channels, down):
        super().__init__()
        self.model = nn.SequentialCell(
            ResBlock2d(in_channels, out_channels),
            nn.MaxPool2d(down, stride=down),
        )

    def construct(self, x):
        return self.model(x)


class Up1d(nn.Cell):
    def __init__(self, in_channels, out_channels, up):
        super().__init__()
        self.model = nn.SequentialCell(
            nn.Conv1dTranspose(in_channels, out_channels, kernel_size=up, stride=up),
            ResBlock1d(out_channels, out_channels),
        )

    def construct(self, x, skip=None):
        if skip != None:
            return self.model(ms.ops.cat((x, skip), 1))
        else:
            return self.model(x)


class Up2d(nn.Cell):
    def __init__(self, in_channels, out_channels, up):
        super().__init__()
        self.model = nn.SequentialCell(
            nn.Conv2dTranspose(in_channels, out_channels, kernel_size=up, stride=up),
            ResBlock2d(out_channels, out_channels),
            ResBlock2d(out_channels, out_channels),
        )

    def construct(self, x, skip=None):
        if skip != None:
            return self.model(ms.ops.cat((x, skip), 1))
        else:
            return self.model(x)


class AE2d(nn.Cell):
    def __init__(self, in_channels, n_feat):
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat

        self.init_conv = ResBlock2d(in_channels, n_feat, is_res=True)
        down = [Down2d(2 ** i * n_feat, 2 ** (i + 1) * n_feat, down=4) for i in range(3)]
        self.down = nn.CellList(down)
        self.down_to_vec = nn.SequentialCell(nn.AvgPool2d(4), nn.Tanh(), )
        self.vec_to_up = nn.SequentialCell(
            nn.Conv2dTranspose(2 ** 3 * n_feat, 2 ** 3 * n_feat, 4, 4), Normalize(2 ** 3 * n_feat),
            nn.ReLU())
        up = [Up2d(2 ** (i + 1) * n_feat, 2 ** i * n_feat, up=4) for i in range(3)]
        self.up = nn.CellList(up)
        self.out = nn.SequentialCell(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, pad_mode="pad", padding=1),
                                     Normalize(n_feat), nn.ReLU(),
                                     nn.Conv2d(n_feat, in_channels, kernel_size=3, stride=1, pad_mode="pad", padding=1),
                                     nn.Sigmoid(),
                                     )

    def construct(self, x):
        z = self.init_conv(x)
        for i in range(3):
            z = self.down[i](z)
        z = self.down_to_vec(z)  # N*512*1*1
        x_rec = self.vec_to_up(z)
        for i in range(3, 0, -1):
            x_rec = self.up[i - 1](x_rec, None)
        x_rec = self.out(x_rec)
        return x_rec, z

    def forward_z(self, x):
        z = self.init_conv(x)
        for i in range(3):
            z = self.down[i](z)
        z = self.down_to_vec(z)
        return z

    def forward_x_rec(self, z):
        x_rec = self.vec_to_up(z)
        for i in range(3, 0, -1):
            x_rec = self.up[i - 1](x_rec, None)
        x_rec = self.out(x_rec)
        return x_rec


def ae_mse_loss(recon_x, x):
    MSE = ms.ops.mse_loss(recon_x, x, reduction='mean')
    # BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='mean')
    return MSE


def ae_bce_loss(recon_x, x):
    # MSE = nn.functional.mse_loss(recon_x, x, reduction='mean')
    BCE = ms.ops.binary_cross_entropy(recon_x, x, reduction='mean')
    return BCE


class VAE2d(nn.Cell):
    def __init__(self, in_channels, n_feat):
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat

        self.init_conv = ResBlock2d(in_channels, n_feat, is_res=True)
        down = [Down2d(2 ** i * n_feat, 2 ** (i + 1) * n_feat, down=4) for i in range(3)]
        self.down = nn.CellList(down)
        self.down_to_vec = nn.SequentialCell(nn.AvgPool2d(4, 4), nn.GELU(approximate=False), )

        self.fc_mu = nn.SequentialCell(nn.Dense(512, 512), nn.ReLU(), nn.Dense(512, 512))
        self.fc_log_var = nn.SequentialCell(nn.Dense(512, 512), nn.ReLU(), nn.Dense(512, 512))
        # self.fc=nn.Dense(512, 512)

        self.vec_to_up = nn.SequentialCell(
            nn.Conv2dTranspose(2 ** 3 * n_feat, 2 ** 3 * n_feat, 4, 4),
            nn.GroupNorm(8, 2 ** 3 * n_feat), nn.ReLU())
        up = [Up2d(2 ** (i + 1) * n_feat, 2 ** i * n_feat, up=4) for i in range(3)]
        self.up = nn.CellList(up)
        self.out = nn.SequentialCell(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, pad_mode="pad", padding=1),
                                     nn.GroupNorm(8, n_feat), nn.ReLU(),
                                     nn.Conv2d(n_feat, in_channels, kernel_size=3, stride=1, pad_mode="pad", padding=1),
                                     nn.Sigmoid(),
                                     )

    def reparameterize(self, mu, log_var):
        std = ms.ops.exp(0.5 * log_var)
        eps = ms.ops.randn_like(std)
        return mu + eps * std

    def construct(self, x):
        z = self.init_conv(x)
        for i in range(3):
            z = self.down[i](z)
        z = self.down_to_vec(z)  # N*512*1*1
        z = z.view(-1, 512)  # N*512
        mu = self.fc_mu(z)
        log_var = self.fc_log_var(z)
        z = self.reparameterize(mu, log_var)
        # z=self.fc(z)
        z = z.view(-1, 512, 1, 1)  # N*512*1*1
        x_rec = self.vec_to_up(z)
        for i in range(3, 0, -1):
            x_rec = self.up[i - 1](x_rec, None)
        x_rec = self.out(x_rec)
        return x_rec, mu, log_var

    def forward_z(self, x):
        z = self.init_conv(x)
        for i in range(3):
            z = self.down[i](z)
        z = self.down_to_vec(z)
        z = z.view(-1, 512)
        mu = self.fc_mu(z)
        log_var = self.fc_log_var(z)
        return mu, log_var

    def forward_x_rec(self, mu, log_var):
        z = self.reparameterize(mu, log_var)
        z = z.view(-1, 512, 1, 1)
        x_rec = self.vec_to_up(z)
        for i in range(3, 0, -1):
            x_rec = self.up[i - 1](x_rec, None)
        x_rec = self.out(x_rec)
        return x_rec


def vae_loss(recon_x, x, mu, log_var):
    BCE = ms.ops.mse_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * ms.ops.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


# %% [3]diffusion
class Embed(nn.Cell):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.SequentialCell(
            nn.Dense(input_dim, emb_dim),
            nn.GELU(approximate=False),
            nn.Dense(emb_dim, emb_dim),
        )

    def construct(self, x):
        return self.model(x.reshape(-1, self.input_dim))


class UNet(nn.Cell):
    def __init__(self, in_channels, n_feat, feature_dim, arch):
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.feature_dim = feature_dim
        self.arch = arch

        self.init_conv = ResBlock1d(in_channels, n_feat, is_res=True)

        down = [Down1d(2 ** i * n_feat, 2 ** (i + 1) * n_feat, down=arch[i]) for i in range(3)]
        self.down = nn.CellList(down)

        self.down_to_vec = nn.SequentialCell(nn.Conv1d(512, 512, 5, 1, pad_mode="pad", padding=2),
                                             nn.AvgPool1d(arch[-1], arch[-1]), nn.GELU(approximate=False))
        self.vec_to_up = nn.SequentialCell(
            nn.Conv1dTranspose(
                feature_dim + 2 ** 3 * n_feat, 2 ** 3 * n_feat, arch[-1], arch[-1]
            ),
            Normalize(2 ** 3 * n_feat), nn.ReLU())

        up = [Up1d(2 ** (i + 2) * n_feat, 2 ** i * n_feat, up=arch[i]) for i in range(3)]
        self.up = nn.CellList(up)

        self.out = nn.SequentialCell(
            nn.Conv1d(2 * n_feat, n_feat, kernel_size=5, stride=1, pad_mode="pad", padding=2),
            Normalize(n_feat), nn.ReLU(),
            nn.Conv1d(n_feat, in_channels, kernel_size=5, stride=1, pad_mode="pad", padding=2)
        )

        temb = [Embed(1, 2 ** (i + 1) * n_feat) for i in range(3)]
        self.temb = nn.CellList(temb)

    def construct(self, x, c, t, context_mask):
        c = c * context_mask
        temb = [self.temb[i](t.astype(ms.float32))[:, :, None] for i in range(3)]

        down = [None] * 4
        down[0] = self.init_conv(x)
        for i in range(3):
            down[i + 1] = self.down[i](down[i])
        vec = self.down_to_vec(down[-1])
        vec = ms.ops.cat([vec, c], 1)

        up = [None] * 4
        up[-1] = self.vec_to_up(vec)
        for i in range(2, -1, -1):
            up[i] = (self.up[i](up[i + 1] + temb[i], down[i + 1]))
        out = self.out(
            ms.ops.cat((up[0], down[0]), 1))
        return out


def ddpm_schedules(beta1, beta2, n_T):
    '''
    Returns pre-computed schedules for DDPM sampling, training process.
    '''
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * ms.ops.arange(0, n_T + 1, dtype=ms.float32) / n_T + beta1
    sqrt_beta_t = ms.ops.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = ms.ops.log(alpha_t)
    alphabar_t = ms.ops.cumsum(log_alpha_t, 0).exp()

    sqrtab = ms.ops.sqrt(alphabar_t)
    oneover_sqrta = 1 / ms.ops.sqrt(alpha_t)

    sqrtmab = ms.ops.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab_inv": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Cell):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super().__init__()
        self.nn_model = nn_model

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.insert_param_to_cell(k, ms.Parameter(ms.Tensor(v), requires_grad=False))
            # self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss(reduction='mean')

    def construct(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """
        _ts = ms.ops.randint(1, self.n_T + 1, (x.shape[0],))
        noise = ms.ops.randn_like(x)
        xt = self.sqrtab[_ts, None, None] * x + self.sqrtmab[_ts, None, None] * noise
        # This is the xt, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this xt. Loss is what we return.

        # dropout context with some probability
        # 0 represent mask  # ※ 参数与pytorch不同
        context_mask = ms.ops.bernoulli(ms.ops.zeros(x.shape[0]), p=(1 - self.drop_prob))
        context_mask = context_mask.reshape(-1, 1, 1)

        # return MSE between added noise, and our predicted noise
        noise_pre = self.nn_model(xt, c, _ts.view(-1, 1) / self.n_T, context_mask)
        loss = self.loss_mse(noise, noise_pre)
        return loss

    def ddpm_sample(self, c, n_sample, size, guide_w):
        batch = c.shape[0]
        cp = c.copy()
        for i in range(n_sample - 1):
            c = ms.ops.cat((cp, c), 0)
        xt = ms.ops.randn(batch * n_sample, *size)  # x_T ~ N(0, 1), sample initial noise
        # c = c.repeat(2, axis=0)  # double the batch
        cp = c.copy()
        c = ms.ops.cat((cp, c), 0)

        # don't drop context at test time
        context_mask = ms.ops.ones((2 * batch * n_sample, 1, 1), ms.float32)
        context_mask[batch * n_sample:] = 0.  # makes second half of batch context free

        for i in range(self.n_T, 0, -1):
            print(f'\rsampling timestep {i}', end='')
            ts = ms.Tensor([i / self.n_T])
            ts = ts.repeat(2 * batch * n_sample, 0).reshape(2 * batch * n_sample, 1)

            # double batch
            z = ms.ops.randn(xt.shape) if i > 1 else 0
            # xt = xt.repeat(2, 0)
            xp = xt.copy()
            xt = ms.ops.cat((xt, xp), 0)

            # split predictions and compute weighting
            eps = self.nn_model(xt, c, ts, context_mask)
            eps1 = eps[:batch * n_sample]  # context
            eps2 = eps[batch * n_sample:]  # context free
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            xt = xt[:batch * n_sample]
            xt = (
                    self.oneover_sqrta[i] * (xt - eps * self.mab_over_sqrtmab_inv[i])
                    + self.sqrt_beta_t[i] * z
            )
            if i % 100 == 0:
                print(xt)
        return xt
