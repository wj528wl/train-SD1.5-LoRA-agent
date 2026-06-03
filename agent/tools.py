"""
Agent 工具层 — 将现有 SD1.5/LoRA 管线封装为 Agent 可调用的标准工具

每个工具返回 dict: {"success": bool, "data": Any, "message": str}
"""
import json
import os
import sys
import glob
import time
from typing import Optional

import torch

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import StableDiffusionPipeline


# ── 全局管线单例（避免重复加载模型） ──
_pipeline: Optional[StableDiffusionPipeline] = None
_pipeline_config: dict = {}
_last_generated_image: Optional[str] = None
_last_generated_prompt: Optional[str] = None

# 默认模型路径（可通过 set_model_path 修改）
_default_model_path: str = "./models/stable-diffusion-v1-5"


def set_model_path(model_path: str) -> dict:
    """切换底模路径（下次生成时生效）"""
    global _default_model_path
    # 拒绝 .safetensors / .ckpt 单文件（必须用转换后的diffusers目录）
    if model_path.endswith(('.safetensors', '.ckpt', '.pt', '.bin')):
        return {
            "success": False, "data": None,
            "message": f"{model_path} 是单文件checkpoint，不能直接用。请用 diffusers 目录路径，例如 ./models/anything-v5（需先用 scripts/convert_checkpoint.py 转换）"
        }
    if not os.path.isdir(model_path):
        return {"success": False, "data": None, "message": f"模型目录不存在: {model_path}"}
    if not os.path.exists(os.path.join(model_path, "unet", "config.json")):
        return {"success": False, "data": None, "message": f"{model_path} 不是有效的diffusers模型目录（缺少 unet/config.json）"}
    _default_model_path = model_path
    return {
        "success": True,
        "data": {"model_path": model_path},
        "message": f"底模已切换为: {model_path}（下次生成生效）",
    }


def get_model_path() -> str:
    """获取当前底模路径"""
    return _default_model_path


def list_available_models() -> dict:
    """扫描 models/ 目录下所有可用的 diffusers 格式模型"""
    models_dir = os.path.join(PROJECT_ROOT, "models")
    if not os.path.exists(models_dir):
        return {"success": True, "data": {"models": [], "current": _default_model_path},
                "message": "models 目录不存在"}

    available = []
    for name in sorted(os.listdir(models_dir)):
        full = os.path.join(models_dir, name)
        if not os.path.isdir(full):
            continue
        unet_dir = os.path.join(full, "unet")
        if os.path.exists(os.path.join(unet_dir, "diffusion_pytorch_model.safetensors")) or \
           os.path.exists(os.path.join(unet_dir, "config.json")):
            available.append({
                "name": name,
                "path": full,
                "is_current": os.path.abspath(full) == os.path.abspath(_default_model_path),
            })

    return {
        "success": True,
        "data": {
            "models": available,
            "current": _default_model_path,
            "count": len(available),
        },
        "message": f"找到 {len(available)} 个可用底模",
    }


def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_pipeline(
    model_path: str = None,
    lora_path: Optional[str] = None,
    lora_strength: float = 1.0,
) -> StableDiffusionPipeline:
    """获取或初始化管线单例"""
    global _pipeline, _pipeline_config

    if model_path is None:
        model_path = _default_model_path

    device = _get_device()
    new_config = {
        "model_path": model_path,
        "lora_path": _pipeline_config.get("lora_path") if lora_path is None else lora_path,
        "lora_strength": _pipeline_config.get("lora_strength", 1.0) if lora_strength == 1.0 and lora_path is None else lora_strength,
    }

    # 模型路径变了就重建管线
    if _pipeline is not None and model_path != _pipeline_config.get("model_path"):
        print(f"[Agent] 底模切换: {_pipeline_config.get('model_path')} → {model_path}")
        _pipeline = None

    if _pipeline is None:
        print(f"[Agent] 正在加载管线 (model={model_path})...")
        dtype = torch.float16 if device == "cuda" else torch.float32
        _pipeline = StableDiffusionPipeline(
            device=device,
            dtype=dtype,
            model_path=model_path,
            lora_path=lora_path,   # 只在首次加载时传入
            lora_strength=lora_strength,
            verbose=False,
        )
        _pipeline_config = new_config
        print("[Agent] 管线就绪。")
    elif lora_path is not None:
        # 只有显式传入 lora_path 时才切换 LoRA（None = 保持现状）
        if lora_path != _pipeline_config.get("lora_path"):
            if _pipeline_config.get("lora_path"):
                _pipeline.unload_lora()
            if lora_path:
                _pipeline.load_lora(lora_path, strength=lora_strength)
            _pipeline_config["lora_path"] = lora_path
            _pipeline_config["lora_strength"] = lora_strength

    return _pipeline


def _resolve_model_path():
    if os.path.exists(_default_model_path):
        return _default_model_path
    default = os.path.join(PROJECT_ROOT, "models", "stable-diffusion-v1-5")
    if os.path.exists(default):
        return default
    return "./models/stable-diffusion-v1-5"


def _resolve_output_dir():
    d = os.path.join(PROJECT_ROOT, "outputs", "agent")
    os.makedirs(d, exist_ok=True)
    return d


def _resolve_lora_dir():
    d = os.path.join(PROJECT_ROOT, "outputs")
    return d


# ══════════════════════════════════════════════════════════════
#  Tool 1: generate_image — 文生图
# ══════════════════════════════════════════════════════════════

def generate_image(
    prompt: str,
    negative_prompt: str = "low quality, blurry, worst quality, bad anatomy, deformed",
    height: int = 512,
    width: int = 512,
    steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = -1,
    lora_path: Optional[str] = None,
    lora_strength: float = 1.0,
) -> dict:
    """
    使用 SD1.5 生成图片

    参数:
        prompt: 正向提示词（英文）
        negative_prompt: 反向提示词
        height: 图片高度（需被8整除）
        width: 图片宽度（需被8整除）
        steps: 推理步数（20-50）
        guidance_scale: CFG引导强度（7.5为默认值）
        seed: 随机种子（-1表示随机）
        lora_path: LoRA模型路径（可选）
        lora_strength: LoRA权重强度（0.0-1.5）
    """
    global _last_generated_image, _last_generated_prompt

    model_path = _resolve_model_path()
    output_dir = _resolve_output_dir()

    # 序列化 lora_path
    lora_path_abs = None
    if lora_path and os.path.exists(lora_path):
        lora_path_abs = lora_path

    try:
        pipeline = _get_pipeline(
            model_path=model_path,
            lora_path=lora_path_abs,
            lora_strength=lora_strength,
        )

        actual_seed = seed if seed >= 0 else int(torch.randint(0, 2**31 - 1, (1,)).item())
        print(f"[Agent] 正在生成图片 (seed={actual_seed})...")
        print(f"[Agent] prompt: {prompt[:120]}...")

        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            seed=actual_seed,
        )

        timestamp = int(time.time())
        filename = f"gen_{timestamp}_{actual_seed}.png"
        output_path = os.path.join(output_dir, filename)
        image.save(output_path)

        _last_generated_image = output_path
        _last_generated_prompt = prompt

        return {
            "success": True,
            "data": {
                "image_path": output_path,
                "seed": actual_seed,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "lora_used": lora_path_abs,
                "lora_strength": lora_strength,
            },
            "message": f"图片已生成: {output_path} (seed={actual_seed})",
        }
    except Exception as e:
        return {"success": False, "data": None, "message": f"生成失败: {str(e)}"}


# ══════════════════════════════════════════════════════════════
#  Tool 2: enhance_prompt — 提示词优化（纯文本，由LLM本身完成更合适）
#  这里提供一个模板生成器作为 fallback
# ══════════════════════════════════════════════════════════════

def enhance_prompt(rough_idea: str, style: str = "anime") -> dict:
    """
    根据简短描述生成高质量提示词模板。
    注意：实际优化由 Agent 的 LLM 大脑完成，此函数提供结构化模板。

    参数:
        rough_idea: 用户的简短描述（中文或英文均可）
        style: 目标风格 (anime/realistic/illustration/concept_art/3d_render)
    """
    style_tags = {
        "anime": "anime style, 2d, cel shading, vibrant colors",
        "realistic": "photorealistic, 8k, highly detailed, professional photography",
        "illustration": "digital illustration, artstation, detailed painting, fantasy art",
        "concept_art": "concept art, character design, trending on artstation, cinematic lighting",
        "3d_render": "3d render, octane render, blender, ray tracing, cgi",
    }

    quality_tags = "best quality, masterpiece, highres, extremely detailed"
    style_tag = style_tags.get(style, style_tags["anime"])
    neg = "low quality, worst quality, blurry, bad anatomy, deformed, ugly, jpeg artifacts"

    return {
        "success": True,
        "data": {
            "rough_idea": rough_idea,
            "style": style,
            "suggested_prompt": f"{rough_idea}, {style_tag}, {quality_tags}",
            "suggested_negative": neg,
        },
        "message": f"已为「{rough_idea[:30]}...」生成 {style} 风格模板",
    }


# ══════════════════════════════════════════════════════════════
#  Tool 3: train_lora — LoRA 训练
# ══════════════════════════════════════════════════════════════

def train_lora(
    data_dir: str,
    trigger_word: str = "sks_style",
    output_name: str = "agent_lora",
    num_epochs: int = 10,
    lora_rank: int = 8,
    learning_rate: float = 1e-4,
    resolution: int = 512,
) -> dict:
    """
    使用图片数据集训练 LoRA 模型

    参数:
        data_dir: 图片文件夹（需包含images/和captions/子目录）
        trigger_word: 触发词（用于激活该LoRA的关键词）
        output_name: 输出模型名
        num_epochs: 训练轮数（5-30）
        lora_rank: LoRA秩（4-32，越大容量越大）
        learning_rate: 学习率
        resolution: 训练分辨率
    """
    from train.config import LoRAConfig
    from train.train_lora import train

    if not os.path.isdir(data_dir):
        return {"success": False, "data": None, "message": f"数据目录不存在: {data_dir}"}

    output_dir = os.path.join(_resolve_output_dir(), "loras", output_name)
    os.makedirs(output_dir, exist_ok=True)

    config = LoRAConfig(
        model_path=_resolve_model_path(),
        data_dir=data_dir,
        output_dir=output_dir,
        device=_get_device(),
        resolution=resolution,
        batch_size=1,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        lora_alpha=lora_rank * 1.0,
        mixed_precision="fp16" if _get_device() == "cuda" else "no",
        save_steps=0,
        logging_steps=5,
        seed=42,
        num_workers=0,
    )

    try:
        print(f"[Agent] 开始训练 LoRA (epochs={num_epochs}, rank={lora_rank})...")
        print(f"[Agent] 数据目录: {data_dir}")
        print(f"[Agent] 触发词: {trigger_word}")
        print(f"[Agent] 输出目录: {output_dir}")
        train(config)

        # 找到输出的 safetensors 文件
        lora_files = glob.glob(os.path.join(output_dir, "*.safetensors"))
        output_path = lora_files[0] if lora_files else os.path.join(output_dir, "final_lora.safetensors")

        return {
            "success": True,
            "data": {
                "lora_path": output_path,
                "trigger_word": trigger_word,
                "output_dir": output_dir,
                "num_epochs": num_epochs,
                "lora_rank": lora_rank,
            },
            "message": f"LoRA 训练完成！模型已保存到 {output_path}。使用时在 prompt 中包含触发词「{trigger_word}」即可激活。",
        }
    except Exception as e:
        return {"success": False, "data": None, "message": f"LoRA 训练失败: {str(e)}"}


# ══════════════════════════════════════════════════════════════
#  Tool 4: list_loras — 列出可用 LoRA
# ══════════════════════════════════════════════════════════════

def list_loras() -> dict:
    """列出所有已训练的 LoRA 模型"""
    lora_base = _resolve_lora_dir()
    safetensors_files = glob.glob(os.path.join(lora_base, "**", "*.safetensors"), recursive=True)

    loras = []
    for f in safetensors_files:
        rel_path = os.path.relpath(f, PROJECT_ROOT)
        size_kb = os.path.getsize(f) / 1024
        loras.append({
            "name": os.path.splitext(os.path.basename(f))[0],
            "path": f,
            "relative_path": rel_path,
            "size_kb": round(size_kb, 1),
        })

    current = _pipeline_config.get("lora_path", None)

    return {
        "success": True,
        "data": {
            "loras": loras,
            "count": len(loras),
            "currently_loaded": current,
        },
        "message": f"找到 {len(loras)} 个 LoRA 模型" + (f"，当前加载: {current}" if current else ""),
    }


# ══════════════════════════════════════════════════════════════
#  Tool 5: switch_lora — 切换/管理 LoRA
# ══════════════════════════════════════════════════════════════

def switch_lora(
    action: str = "load",
    lora_path: Optional[str] = None,
    lora_name: Optional[str] = None,
    strength: float = 1.0,
) -> dict:
    """
    管理 LoRA 模型：加载、卸载、调整强度

    参数:
        action: load / unload / set_strength
        lora_path: LoRA文件路径（与lora_name二选一）
        lora_name: LoRA文件名（自动搜索outputs目录）
        strength: 权重强度
    """
    global _pipeline, _pipeline_config

    if _pipeline is None:
        _get_pipeline()

    try:
        if action == "unload":
            _pipeline.unload_lora()
            _pipeline_config["lora_path"] = None
            return {"success": True, "data": None, "message": "已卸载当前 LoRA"}

        # 解析路径
        resolved_path = lora_path
        if not resolved_path and lora_name:
            lora_base = _resolve_lora_dir()
            candidates = glob.glob(os.path.join(lora_base, "**", f"{lora_name}*"), recursive=True)
            safetensors = [c for c in candidates if c.endswith(".safetensors")]
            if safetensors:
                resolved_path = safetensors[0]
            else:
                return {"success": False, "data": None, "message": f"未找到名为 {lora_name} 的 LoRA"}

        if action == "load":
            if not resolved_path or not os.path.exists(resolved_path):
                return {"success": False, "data": None, "message": f"LoRA 文件不存在: {resolved_path}"}

            print(f"[Agent] 加载 LoRA: {resolved_path}")
            # 确保 pipeline 存在
            if _pipeline is None:
                _get_pipeline()
            result = _pipeline.load_lora(resolved_path, strength=strength)
            _pipeline_config["lora_path"] = resolved_path
            _pipeline_config["lora_strength"] = strength
            return {"success": True, "data": {"loaded": resolved_path, "strength": strength, "layers": result},
                    "message": f"已加载 LoRA (layers={result}, strength={strength})"}

        elif action == "set_strength":
            if _pipeline is None:
                _get_pipeline()
            _pipeline.set_lora_strength(strength)
            _pipeline_config["lora_strength"] = strength
            return {"success": True, "data": {"strength": strength},
                    "message": f"LoRA 强度已调整为: {strength}"}

        else:
            return {"success": False, "data": None, "message": f"未知操作: {action}"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "data": None, "message": f"LoRA 操作失败: {str(e)}"}


# ══════════════════════════════════════════════════════════════
#  Tool 6: get_last_result — 获取上次生成结果
# ══════════════════════════════════════════════════════════════

def get_last_result() -> dict:
    """获取上一次生成的结果信息，供 Agent 分析用"""
    global _last_generated_image, _last_generated_prompt

    if not _last_generated_image:
        return {"success": False, "data": None, "message": "还没有生成过图片"}

    exists = os.path.exists(_last_generated_image)
    return {
        "success": True,
        "data": {
            "image_path": _last_generated_image,
            "prompt": _last_generated_prompt,
            "file_exists": exists,
            "file_size_kb": round(os.path.getsize(_last_generated_image) / 1024, 1) if exists else 0,
        },
        "message": f"上次生成: {_last_generated_image}",
    }


# ══════════════════════════════════════════════════════════════
#  工具注册表
# ══════════════════════════════════════════════════════════════

class ToolsRegistry:
    """工具注册中心 — 提供 OpenAI Function Calling 所需的 JSON Schema"""

    TOOLS = {
        "generate_image": {
            "function": generate_image,
            "schema": {
                "name": "generate_image",
                "description": "使用 Stable Diffusion 1.5 根据文本描述生成图片。当用户要求画图、生成图片、创作图像时调用此工具。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "正向提示词（英文），描述要生成的图像内容。格式：主体+风格+质量标签。例如 'a cat sitting on a sofa, anime style, best quality'",
                        },
                        "negative_prompt": {
                            "type": "string",
                            "description": "反向提示词，描述不希望在图中出现的内容。默认可使用 'low quality, blurry, worst quality, bad anatomy, deformed'",
                        },
                        "height": {
                            "type": "integer",
                            "description": "图片高度，需被8整除。默认512",
                        },
                        "width": {
                            "type": "integer",
                            "description": "图片宽度，需被8整除。默认512",
                        },
                        "steps": {
                            "type": "integer",
                            "description": "推理步数，越多质量越好但越慢。默认50，范围20-50",
                        },
                        "guidance_scale": {
                            "type": "number",
                            "description": "CFG引导强度，值越大越贴合prompt但可能过饱和。默认7.5",
                        },
                        "seed": {
                            "type": "integer",
                            "description": "随机种子，-1表示随机。相同seed可复现相同结果",
                        },
                        "lora_path": {
                            "type": "string",
                            "description": "要使用的LoRA模型文件路径（可选）",
                        },
                        "lora_strength": {
                            "type": "number",
                            "description": "LoRA权重强度，默认1.0。降低可减弱LoRA效果",
                        },
                    },
                    "required": ["prompt"],
                },
            },
        },
        "enhance_prompt": {
            "function": enhance_prompt,
            "schema": {
                "name": "enhance_prompt",
                "description": "将用户的简短描述优化为专业的英文 Stable Diffusion 提示词。在使用 generate_image 之前，建议先调用此工具优化提示词。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "rough_idea": {
                            "type": "string",
                            "description": "用户的原始想法（中文或英文），例如 '一只猫' 或 'a cute cat'",
                        },
                        "style": {
                            "type": "string",
                            "enum": ["anime", "realistic", "illustration", "concept_art", "3d_render"],
                            "description": "目标风格",
                        },
                    },
                    "required": ["rough_idea"],
                },
            },
        },
        "train_lora": {
            "function": train_lora,
            "schema": {
                "name": "train_lora",
                "description": "使用用户提供的图片数据集训练个性化 LoRA 模型。训练完成后可在生成图片时使用。需要提前准备好 data_dir/images/ 和 data_dir/captions/ 目录。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_dir": {
                            "type": "string",
                            "description": "训练数据目录路径，需包含 images/ 和 captions/ 子目录",
                        },
                        "trigger_word": {
                            "type": "string",
                            "description": "触发词，训练后在prompt中包含此词即可激活LoRA。建议使用不常见的词如 'leisai' 或 'mystyle'",
                        },
                        "output_name": {
                            "type": "string",
                            "description": "输出模型名称，默认 'agent_lora'",
                        },
                        "num_epochs": {
                            "type": "integer",
                            "description": "训练轮数，5-30。数据少用多轮，数据多用少轮",
                        },
                        "lora_rank": {
                            "type": "integer",
                            "description": "LoRA秩，4-32。值越大模型容量越大但文件也越大",
                        },
                    },
                    "required": ["data_dir"],
                },
            },
        },
        "list_loras": {
            "function": list_loras,
            "schema": {
                "name": "list_loras",
                "description": "列出所有已训练的 LoRA 模型文件。当用户询问有哪些可用LoRA或想切换风格时调用。",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        "switch_lora": {
            "function": switch_lora,
            "schema": {
                "name": "switch_lora",
                "description": "管理 LoRA 模型：加载、卸载或调整当前LoRA的强度。当用户想切换风格或调整效果时调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["load", "unload", "set_strength"],
                            "description": "操作类型：load加载LoRA，unload卸载LoRA，set_strength调整强度",
                        },
                        "lora_name": {
                            "type": "string",
                            "description": "LoRA文件名（不含路径），用于load操作",
                        },
                        "lora_path": {
                            "type": "string",
                            "description": "LoRA完整路径，用于load操作",
                        },
                        "strength": {
                            "type": "number",
                            "description": "LoRA强度，默认1.0。0.5=减半效果，1.5=加强效果",
                        },
                    },
                    "required": ["action"],
                },
            },
        },
        "get_last_result": {
            "function": get_last_result,
            "schema": {
                "name": "get_last_result",
                "description": "获取上一次生成的图片信息，包含路径和使用的prompt。用于回顾和分析之前的生成结果。",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        "list_models": {
            "function": list_available_models,
            "schema": {
                "name": "list_models",
                "description": "列出所有可用的底模（Stable Diffusion checkpoint）。当用户询问有哪些底模、想切换模型时调用。",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        "switch_model": {
            "function": set_model_path,
            "schema": {
                "name": "switch_model",
                "description": "切换底模。可用的底模只有两个: ./models/stable-diffusion-v1-5 和 ./models/anything-v5。必须是目录路径，不能是.safetensors文件。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_path": {
                            "type": "string",
                            "description": "模型目录路径，只能是 ./models/stable-diffusion-v1-5 或 ./models/anything-v5",
                        },
                    },
                    "required": ["model_path"],
                },
            },
        },
    }

    @classmethod
    def get_schemas(cls) -> list[dict]:
        """返回 OpenAI Function Calling 格式的工具列表"""
        return [
            {"type": "function", "function": t["schema"]}
            for t in cls.TOOLS.values()
        ]

    @classmethod
    def execute(cls, tool_name: str, arguments: dict) -> dict:
        """执行指定工具并返回结果"""
        if tool_name not in cls.TOOLS:
            return {"success": False, "data": None, "message": f"未知工具: {tool_name}"}

        func = cls.TOOLS[tool_name]["function"]
        try:
            return func(**arguments)
        except TypeError as e:
            return {"success": False, "data": None, "message": f"工具参数错误: {str(e)}"}
