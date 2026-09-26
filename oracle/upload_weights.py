#!/usr/bin/env python
"""upload_weights.py — publish the converted VOSR2 weight lanes to mlx-community.

    python upload_weights.py --weights <dir> [--lanes fp16,fp32] [--dry-run]

⚠️ OUTWARD ACTION (public model repos). mlx-community grammar: `<UpstreamRepoName>-<quant>`, one repo per lane:
  mlx-community/VOSR2-fp16  transformer/vae/dinov2 _fp16 + config.json   (the shipping lane; int8 is derived at load)
  mlx-community/VOSR2-fp32  transformer/vae/dinov2 _fp32 + config.json   (the parity / reference lane)
bf16 is not published (rejected tier, 37.6 dB). Each repo gets the model card from MODEL_CARD.md with the lane filled in.
"""
import argparse, os, sys
p = argparse.ArgumentParser(); p.add_argument("--weights", required=True); p.add_argument("--lanes", default="fp16,fp32")
p.add_argument("--dry-run", action="store_true"); a = p.parse_args()
LANES = {"fp16": "mlx-community/VOSR2-fp16", "fp32": "mlx-community/VOSR2-fp32"}
here = os.path.dirname(os.path.abspath(__file__))
card_t = open(os.path.join(here, "MODEL_CARD.md")).read()
for lane in a.lanes.split(","):
    repo = LANES[lane]
    files = [f"transformer_{lane}.safetensors", f"vae_{lane}.safetensors", f"dinov2_{lane}.safetensors", "config.json"]
    for f in files:
        path = os.path.join(a.weights, f); assert os.path.exists(path), path
        print(f"  {repo}: {f:32} {os.path.getsize(path)/1e9:6.2f} GB")
    card = card_t.replace("{{LANE}}", lane).replace("{{REPO}}", repo)
    if a.dry_run: continue
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo)
    for f in files:
        print("uploading", repo, f, flush=True)
        api.upload_file(path_or_fileobj=os.path.join(a.weights, f), path_in_repo=f, repo_id=repo)
    print("done:", f"https://huggingface.co/{repo}", flush=True)
if a.dry_run: print("dry run — nothing uploaded")
