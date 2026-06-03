"""
Checkpoint → Diffusers 纯本地转换（无需网络）

用法:
    python scripts/convert_checkpoint.py -i anything-v5.safetensors -o ./models/anything-v5
"""
import argparse
import json
import os
import shutil
import sys
import warnings

warnings.filterwarnings("ignore")

for m in ["torch", "diffusers", "transformers", "safetensors"]:
    try: __import__(m)
    except ImportError: print(f"缺少依赖: {m}"); sys.exit(1)

import torch
from safetensors.torch import load_file
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    convert_ldm_unet_checkpoint,
    convert_ldm_vae_checkpoint,
    convert_ldm_clip_checkpoint,
)


def convert(input_path: str, output_dir: str, base_dir: str = "./models/stable-diffusion-v1-5"):
    print("=" * 60)
    print("  🔄 Checkpoint → Diffusers（纯本地转换）")
    print("=" * 60)
    print(f"  输入: {input_path}")
    print(f"  输出: {output_dir}")
    print(f"  配置来源: {base_dir}\n")

    if not os.path.exists(input_path):
        print(f"❌ 文件不存在: {input_path}"); return False
    if not os.path.isdir(base_dir):
        print(f"❌ 基础模型目录不存在: {base_dir}"); return False

    print(f"📦 文件大小: {os.path.getsize(input_path)/(1024**3):.2f} GB")

    # ── 1. 加载 checkpoint ──
    print("📥 加载 checkpoint...")
    ckpt = load_file(input_path) if input_path.endswith(".safetensors") else torch.load(input_path, map_location="cpu")
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    print(f"   ✅ {len(ckpt)} 个参数")

    # ── 2. 转换 UNet ──
    print("🔧 转换 UNet...")
    unet_config = json.load(open(os.path.join(base_dir, "unet", "config.json")))
    # diffusers 0.37 转换器需要的额外字段
    for k, v in {
        "class_embed_type": None,
        "num_class_embeds": None,
        "resnet_time_scale_shift": "default",
        "addition_embed_type": None,
        "addition_time_embed_dim": None,
        "projection_class_embeddings_input_dim": None,
        "encoder_hid_dim": None,
        "encoder_hid_dim_type": None,
        "time_embedding_type": "positional",
        "time_embedding_dim": None,
        "time_embedding_act_fn": None,
        "timestep_post_act": None,
        "conv_in_kernel": 3,
        "conv_out_kernel": 3,
        "dual_cross_attention": False,
        "only_cross_attention": False,
        "use_linear_projection": False,
        "upcast_attention": False,
    }.items():
        unet_config.setdefault(k, v)
    unet_ckpt = convert_ldm_unet_checkpoint(ckpt, unet_config)
    unet = UNet2DConditionModel.from_config(unet_config)
    unet.load_state_dict(unet_ckpt, strict=False)
    unet_out = os.path.join(output_dir, "unet")
    unet.save_pretrained(unet_out, safe_serialization=True)
    print(f"   ✅ UNet → {unet_out}")

    # ── 3. 转换 VAE ──
    print("🔧 转换 VAE...")
    vae_config = json.load(open(os.path.join(base_dir, "vae", "config.json")))
    vae_ckpt = convert_ldm_vae_checkpoint(ckpt, vae_config)
    vae = AutoencoderKL.from_config(vae_config)
    vae.load_state_dict(vae_ckpt, strict=False)
    vae_out = os.path.join(output_dir, "vae")
    vae.save_pretrained(vae_out, safe_serialization=True)
    print(f"   ✅ VAE → {vae_out}")

    # ── 4. 转换 CLIP ──
    print("🔧 转换 CLIP Text Encoder...")
    text_enc = convert_ldm_clip_checkpoint(ckpt)  # 直接返回 CLIPTextModel 对象
    text_enc_out = os.path.join(output_dir, "text_encoder")
    text_enc.save_pretrained(text_enc_out, safe_serialization=True)
    print(f"   ✅ CLIP → {text_enc_out}")

    # ── 5. 复制 Tokenizer ──
    print("📋 复制 Tokenizer...")
    tok_out = os.path.join(output_dir, "tokenizer")
    tok_src = os.path.join(base_dir, "tokenizer")
    if os.path.exists(tok_out): shutil.rmtree(tok_out)
    if os.path.isdir(tok_src):
        shutil.copytree(tok_src, tok_out)
    else:
        CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14").save_pretrained(tok_out)
    print(f"   ✅ Tokenizer → {tok_out}")

    # ── 6. 复制 Scheduler ──
    print("📋 复制 Scheduler...")
    sch_out = os.path.join(output_dir, "scheduler")
    if os.path.exists(sch_out): shutil.rmtree(sch_out)
    shutil.copytree(os.path.join(base_dir, "scheduler"), sch_out)
    print(f"   ✅ Scheduler → {sch_out}")

    # ── 7. 清理 ──
    del ckpt, unet_ckpt, vae_ckpt, unet, vae, text_enc
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ── 8. 验证 ──
    print("\n" + "=" * 60)
    print("  🔍 验证输出...")
    checks = [
        ("unet/diffusion_pytorch_model.safetensors", "UNet"),
        ("vae/diffusion_pytorch_model.safetensors", "VAE"),
        ("text_encoder/model.safetensors", "CLIP"),
        ("tokenizer/vocab.json", "Tokenizer"),
        ("scheduler/scheduler_config.json", "Scheduler"),
    ]
    ok = all(os.path.exists(os.path.join(output_dir, f)) for f, _ in checks)
    for f, name in checks:
        print(f"   {'✅' if os.path.exists(os.path.join(output_dir, f)) else '❌'} {name}")

    if ok:
        print(f"\n✅ 转换成功！模型: {output_dir}")
        print(f"\n🚀 推理命令:")
        print(f"   python src/main.py --model_path {output_dir} --prompt \"1girl, masterpiece\" --no_lora")
        print(f"\n🚀 Agent WebUI 设置底模路径: {output_dir}")
    return ok


def main():
    p = argparse.ArgumentParser(description="纯本地 checkpoint → diffusers")
    p.add_argument("-i", "--input", required=True, help=".safetensors / .ckpt 文件")
    p.add_argument("-o", "--output", required=True, help="输出目录")
    p.add_argument("--base", default="./models/stable-diffusion-v1-5", help="SD1.5 配置来源")
    args = p.parse_args()
    sys.exit(0 if convert(args.input, args.output, args.base) else 1)


if __name__ == "__main__":
    main()
