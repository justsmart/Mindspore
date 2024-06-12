from tqdm import tqdm
import argparse
from model import *
from datasets import *
import mindspore as ms

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='device')
parser.add_argument('--dataname', type=str, default='NH_face')
parser.add_argument('--view', type=int, default=1)
parser.add_argument('--pairedrate', type=float, default=0.1)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--ep', type=int, default=100)
args = parser.parse_args()

device = f'cuda:{args.device}'
dataname = args.dataname
view = args.view
pairedrate = args.pairedrate
fold = args.fold
ep = args.ep

n_epoch = 1
n_T = 1000
lrate = 1e-4
betas = (1e-6, 2e-2)  # betas=(1e-4, 2e-2)
drop_prob = 0.1
n_feat = 64
ms.set_context(device_target="GPU")

train_dataloader, test_dataloader, configs = get_data(dataname=dataname, view=view, pairedrate=pairedrate, fold=fold)
unet = UNet(in_channels=1, n_feat=n_feat, feature_dim=configs['dim_c'], arch=configs['arch'])
unet.set_train(False)
param_dict = ms.load_checkpoint(f"./model_view1_100.ckpt")
param_not_load, _ = ms.load_param_into_net(unet, param_dict)
print(param_not_load)

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

pbar = tqdm(test_dataloader)
out = []
n_sample = 1
guide_w = 1
for x, c in pbar:
    x = ms.Tensor(x, dtype=ms.float32)
    c = ms.Tensor(c, dtype=ms.float32)
    c = c.permute(0, 2, 1)
    batch = c.shape[0]
    cp = c.copy()
    for i in range(n_sample - 1):
        c = ms.ops.cat((cp, c), 0)
    xt = ms.ops.randn(batch * n_sample, *[1, configs['dim_x']])  # x_T ~ N(0, 1), sample initial noise
    cp = c.copy()
    c = ms.ops.cat((cp, c), 0)

    # don't drop context at test time
    context_mask = ms.ops.ones((2 * batch * n_sample, 1, 1), ms.float32)
    context_mask[batch * n_sample:] = 0.  # makes second half of batch context free

    for i in range(n_T, 0, -1):
        print(f'\rsampling timestep {i}', end='')
        ts = ms.Tensor([i / n_T])
        ts = ts.repeat(2 * batch * n_sample, 0).reshape(2 * batch * n_sample, 1)

        # double batch
        z = ms.ops.randn(xt.shape) if i > 1 else 0
        xp = xt.copy()
        xt = ms.ops.cat((xt, xp), 0)

        # split predictions and compute weighting
        eps = unet(xt, c, ts, context_mask)
        eps1 = eps[:batch * n_sample]  # context
        eps2 = eps[batch * n_sample:]  # context free
        eps = (1 + guide_w) * eps1 - guide_w * eps2
        xt = xt[:batch * n_sample]
        xt = (
                oneover_sqrta[i] * (xt - eps * mab_over_sqrtmab_inv[i])
                + sqrt_beta_t[i] * z
        )
        if i % 100 == 0:
            print(xt)
    x_rec = xt
    out.append(x_rec)
    break
out = ms.ops.cat(out, axis=0).numpy()
if view == 0:
    out = out / 100
elif view == 1:
    out = out * 100
elif view == 2:
    out = out[:, :, 12:-12]
    out = (out / 10) ** 2
np.save(f"./ddpm_{dataname}_view{view}_pairedrate{pairedrate}_fold{fold}_ep{ep}.npy",
        out)
