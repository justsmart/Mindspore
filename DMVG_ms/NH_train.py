from tqdm import tqdm
import argparse
from model import *
from datasets import *
import mindspore as ms
from mindspore import nn


def train_epoch(ep, model, optimizer, train_dataloader, view):
    def forward_fn(x, c):
        _ts = ms.ops.randint(1, n_T + 1, (x.shape[0],))
        noise = ms.ops.randn_like(x, dtype=ms.float32)
        xt = sqrtab[_ts, None, None] * x + sqrtmab[_ts, None, None] * noise
        # This is the xt, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this xt. Loss is what we return.

        # dropout context with some probability
        # 0 represent mask
        context_mask = ms.ops.bernoulli(ms.ops.zeros(x.shape[0]), p=(1 - drop_prob))
        context_mask = context_mask.reshape(-1, 1, 1)

        noise_pre = model(xt, c, _ts.view(-1, 1) / n_T, context_mask)

        loss = loss_fn(noise, noise_pre)
        return loss

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    def train_step(data, condition):
        loss, grad = grad_fn(data, condition)
        optimizer(grad)
        return loss

    loss_ema = None
    pbar = tqdm(train_dataloader)
    for data, condition in pbar:
        data = ms.Tensor(data, dtype=ms.float32)
        condition = ms.Tensor(condition, dtype=ms.float32)
        if view == 2:
            pad = nn.Pad(paddings=[12, 12], mode='constant')
            data = pad(data)
        condition = condition.transpose(0, 2, 1)
        loss = float(train_step(data, condition).asnumpy())
        if loss_ema is None:
            loss_ema = loss
        else:
            loss_ema = 0.95 * loss_ema + 0.05 * loss
        pbar.set_description(f"loss: {loss_ema:.4f}")
    if (ep + 1) % 100 == 0:
        ms.save_checkpoint(unet, save_path)


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='device')
parser.add_argument('--dataname', type=str, default='NH_face')
parser.add_argument('--view', type=int, default=1)
parser.add_argument('--pairedrate', type=float, default=0.1)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--lr_decay_step', type=float, default=10, help='StepLr_Step_size')
parser.add_argument('--lr_decay_gamma', type=float, default=0.9, help='StepLr_Gamma')
args = parser.parse_args()

device = f'cuda:{args.device}'
dataname = args.dataname
view = args.view
pairedrate = args.pairedrate
fold = args.fold

n_epoch = 100
n_T = 1000
betas = (1e-6, 2e-2)  # betas=(1e-4, 2e-2)
drop_prob = 0.1
n_feat = 64
lrate = 1e-4
ms.set_context(device_id=0, device_target="GPU")

# ddpm_schedule
beta1 = betas[0]
beta2 = betas[1]
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

train_dataloader, test_dataloader, configs = get_data(dataname=dataname, view=view, pairedrate=pairedrate, fold=fold)
unet = UNet(in_channels=1, n_feat=n_feat, feature_dim=configs['dim_c'], arch=configs['arch'])

optimizer = nn.Adam(unet.trainable_params(), learning_rate=lrate)
loss_fn = nn.MSELoss(reduction="mean")

save_path = f"./model_view{view}_{n_epoch}.ckpt"

for ep in range(n_epoch):
    print(f'epoch {ep}')
    unet.set_train()
    train_epoch(ep, unet, optimizer, train_dataloader, view=view)
