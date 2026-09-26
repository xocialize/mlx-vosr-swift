#!/usr/bin/env python
"""dump_goldens.py — per-sub-op goldens from the PyTorch VOSR2 reference, CPU fp32 (the oracle for S1/S2).

Run with the reference venv from the upstream checkout root (preset/ckpts populated):
    python dump_goldens.py --src <vosr checkout> --image <512² png> --out <dir> [--image1024 <png>]
Every array is saved C-contiguous fp32 in goldens.safetensors; shapes/args in goldens.json.
Nothing here re-implements the model — hooks read the upstream modules.
"""
import argparse, json, os, sys, numpy as np
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
p = argparse.ArgumentParser()
p.add_argument("--src", required=True); p.add_argument("--image", required=True); p.add_argument("--image1024", default="")
p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=42)
a = p.parse_args()
os.chdir(os.path.abspath(a.src)); sys.path.insert(0, os.getcwd())
import torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from safetensors.torch import load_file
from safetensors.numpy import save_file
torch.hub.set_dir("preset/ckpts/torch_cache")
torch.manual_seed(a.seed); torch.set_num_threads(os.cpu_count() or 8)
from inference_vosr_onestep import load_dinov2, get_venc_features, tiled_latent_inference, preprocess_raw_image
from models.lightningdit import LightningDiT
from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
from vosr import VOSR
from tiled_vae import encode_dispatch, decode_dispatch

cfg = json.load(open("preset/ckpts/VOSR2/args.json"))
args = argparse.Namespace(**cfg); args.upscale = 1; args.tile_size = 512; args.tile_overlap = 32
args.vae_tile_size = None; args.vae_tile_overlap = None; args.posterior_mode = True; args.infer_steps = 1
args.auxiliary_time_cond = bool(getattr(args, "auxiliary_time_cond", False))
device = torch.device("cpu")

vae = AutoencoderKLQwenImage2D.from_pretrained(args.ae_path).to(device).eval()
venc = load_dinov2(args, device)
model = LightningDiT(input_size=args.resolution // 8, patch_size=args.patch_size, in_channels=32, out_channels=16,
                     hidden_size=args.dim, depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
                     z_dims=args.enc_dim, encdim_ratio=args.encdim_ratio, auxiliary_time_cond=args.auxiliary_time_cond,
                     use_qknorm=args.use_qknorm, use_swiglu=args.use_swiglu, use_rope=args.use_rope,
                     use_rmsnorm=args.use_rmsnorm, wo_shift=args.wo_shift, num_fused_layers=len(args.layer_dinov2b_list))
sd = load_file("preset/ckpts/VOSR2/checkpoints/ema_model.safetensors")
missing, unexpected = model.load_state_dict(sd, strict=False); assert not missing and not unexpected, (missing, unexpected)
model.to(device).eval(); model.forward = model.forward_flexible
vosr_model = VOSR(time_dist=args.time_dist, cfg_ratio=args.cfg_ratio, cfg_scale=args.cfg_scale, interp_type=args.interp_type,
                  accelerator=type("A", (), {"device": device})(), t_start=args.t_start, t_end=args.t_end, args=args)

G, meta = {}, {"seed": a.seed, "args": cfg, "image": os.path.basename(a.image)}
def put(name, t): 
    arr = np.ascontiguousarray(t.detach().float().cpu().numpy()); G[name] = arr; meta.setdefault("shapes", {})[name] = list(arr.shape)
hooks = []
def hook(mod, name):
    def f(m, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        put(name, o)
    hooks.append(mod.register_forward_hook(f))
# DiT sub-ops
hook(model.x_embedder, "dit.x_embedder"); hook(model.t_embedder, "dit.t_embedder"); hook(model.t_block, "dit.t_block")
hook(model.layer_norm, "dit.layer_norm"); hook(model.mlp_ca, "dit.mlp_ca"); hook(model.final_layer, "dit.final_layer")
for i in (0, 1, 17, 35): hook(model.blocks[i], f"dit.block{i}")
hook(model.blocks[0].attn, "dit.block0.attn"); hook(model.blocks[0].cross_attn, "dit.block0.cross_attn"); hook(model.blocks[0].mlp, "dit.block0.mlp")
hook(model.blocks[0].norm1, "dit.block0.norm1")
# DINOv2
hook(venc.patch_embed, "dino.patch_embed")
for i in (0, 17): hook(venc.blocks[i], f"dino.block{i}")
# VAE
hook(vae.encoder, "vae.encoder"); hook(vae.quant_conv, "vae.quant_conv"); hook(vae.post_quant_conv, "vae.post_quant_conv"); hook(vae.decoder, "vae.decoder")
hook(vae.encoder.down_blocks[0], "vae.enc.down0"); hook(vae.encoder.mid_block, "vae.enc.mid"); hook(vae.decoder.mid_block, "vae.dec.mid"); hook(vae.decoder.up_blocks[0], "vae.dec.up0")

to_tensor = transforms.ToTensor()
img = Image.open(a.image).convert("RGB"); assert img.size == (512, 512), img.size
lq = to_tensor(img).unsqueeze(0).to(device) * 2.0 - 1.0
put("input", lq)
with torch.no_grad():
    # VAE encode (deterministic posterior) + normalised latent
    lq_latent, lm, ls = encode_dispatch(vae, lq, args, device)
    put("vae.latent_raw", vae.encode(lq).latent_dist.mode()); put("vae.latent_norm", lq_latent)
    put("vae.latents_mean", lm); put("vae.latents_inv_std", ls)
    # DINOv2 features (list of one layer)
    raw_image = (0.5 * lq + 0.5) * 255
    put("dino.input448", preprocess_raw_image(raw_image, args))
    z_fea = get_venc_features(venc, lq, args)
    put("dino.z", z_fea[0])
    # DiT one step with injected noise
    rng = np.random.RandomState(a.seed); noise = torch.from_numpy(rng.randn(*lq_latent.shape).astype(np.float32))
    put("dit.noise", noise)
    inp = torch.cat([lq_latent, noise], 1); put("dit.input", inp)
    t1 = torch.ones(1); t0 = torch.zeros(1)
    u = model(inp, t1, t0, z_fea); put("dit.output", u)
    z1 = noise - (1.0 - 0.0) * u; put("dit.latent_out", z1)
    out = decode_dispatch(vae, z1, args, lm, ls, None); put("output", out)
    # VAE round trip of the input latent (decode parity on a non-model latent)
    put("vae.decode_of_input_latent", decode_dispatch(vae, lq_latent, args, lm, ls, None))
for h in hooks: h.remove()

if a.image1024:
    img2 = Image.open(a.image1024).convert("RGB"); assert img2.size == (1024, 1024), img2.size
    lq2 = to_tensor(img2).unsqueeze(0) * 2.0 - 1.0
    put("tiled.input", lq2)
    torch.manual_seed(a.seed)
    with torch.no_grad():
        # capture the shared noise the driver draws (torch.randn_like on the full latent) by re-seeding identically
        lq_latent2, lm2, ls2 = encode_dispatch(vae, lq2, args, device)
        put("tiled.latent_norm", lq_latent2)
        torch.manual_seed(a.seed); noise2 = torch.randn_like(lq_latent2); put("tiled.noise", noise2)
        torch.manual_seed(a.seed)
        out2 = tiled_latent_inference(model, vae, venc, lq2, args, device=device, light_decoder=None)
        put("tiled.output", out2)
    meta["tiled"] = {"tile_size": 512, "tile_overlap": 32, "note": "noise = torch.randn_like(full latent) after torch.manual_seed(seed); driver re-seeded identically"}

os.makedirs(a.out, exist_ok=True)
save_file(G, os.path.join(a.out, "goldens.safetensors")); json.dump(meta, open(os.path.join(a.out, "goldens.json"), "w"), indent=1)
print("goldens:", len(G), "arrays →", a.out); 
for k, v in G.items(): print(f"  {k:32} {list(v.shape)}  |x|max {np.abs(v).max():.4g}")
