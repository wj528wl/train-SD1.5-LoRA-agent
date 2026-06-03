# SD1.5 LoRA 训练与推理项目

一个基于本地 Stable Diffusion 1.5 的全栈 AIGC 项目，支持：

- 本地 SD1.5 推理（官方 diffusers 调度器）
- 基于 UNet 注意力层的 LoRA 训练
- 导出 ComfyUI 兼容的 `.safetensors` 格式 LoRA
- 推理时动态加载/切换/叠加 LoRA
- AIGC Agent：自然语言驱动的智能生图助手（支持本地 Ollama/Qwen）
- 多底模支持（SD1.5 / anything-v5 等任意 SD1.5 系模型）

## 项目特点

- 推理与训练代码分离
- 训练出的 LoRA 直接放入 ComfyUI 使用
- Agent 支持本地 LLM（Ollama + Qwen）完全离线运行
- 一键转换 ComfyUI 单文件 checkpoint → diffusers 格式
- 命令行 + WebUI 双入口

## 项目结构

```text
.
├── src/                     # 推理引擎
│   ├── main.py              # CLI 推理入口
│   ├── pipeline.py          # 完整推理管线
│   ├── clip.py              # CLIP 文本编码器
│   ├── unet.py              # UNet + LoRA 注入管理
│   ├── vae.py               # VAE 编解码
│   └── lora.py              # LoRA 权重加载/注入
│
├── train/                   # 训练引擎
│   ├── main.py              # CLI 训练入口
│   ├── train_lora.py        # 核心训练循环
│   ├── dataset.py           # 图文数据集 / DreamBooth
│   ├── lora.py              # LoRA 层定义与注入
│   ├── save_lora.py         # 导出 safetensors
│   ├── config.py            # 训练配置
│   └── train_utils.py       # 优化器/调度器/工具函数
│
├── agent/                   # AIGC Agent
│   ├── agent.py             # LLM 调度器 + 工具调用
│   ├── tools.py             # 6 大工具封装
│   ├── prompts.py           # 系统提示词
│   ├── webui.py             # Gradio WebUI
│   └── cli.py               # 命令行入口
│
├── scripts/                 # 辅助工具
│   ├── convert_checkpoint.py  # 单文件 checkpoint → diffusers 转换
│   └── test_llm_connection.py # 本地 LLM 连通性测试
│
├── models/                  # 模型权重
├── data/                    # 训练数据
├── outputs/                 # 输出
│
├── download_models.py       # 下载 SD1.5 基础模型
├── test_pipeline.py         # 推理测试
├── test_train_lora.py       # 训练测试
├── requirements.txt         # 公共依赖
├── requirements-cpu.txt     # CPU 环境依赖
└──requirements-cu121.txt   # CUDA 12.1 环境依赖
 
```

---

## Python 版本

推荐：**Python 3.10.6**

---

## 环境安装

本项目将公共依赖和 PyTorch 运行时拆开，方便不同环境一次性安装。

### CUDA 环境

```bash
pip install -r requirements-cu121.txt
```

### CPU 环境

```bash
pip install -r requirements-cpu.txt
```

### 只安装公共依赖

如果你已经自行安装好了 `torch` 和 `torchvision`：

```bash
pip install -r requirements.txt
```


## 一、下载 SD1.5 基础模型

```bash
python download_models.py
```

模型下载到 `./models/stable-diffusion-v1-5/`，包含 tokenizer、text_encoder、unet、vae、scheduler。

---

## 二、数据集格式

支持两种数据组织方式。

### 方式 A：分目录存放（推荐）

```text
data/
├── images/
│   ├── 001.jpg
│   └── 002.jpg
└── captions/
    ├── 001.txt
    └── 002.txt
```

### 方式 B：平铺存放

```text
data/
├── 001.jpg
├── 001.txt
├── 002.jpg
└── 002.txt
```

要求：

- 每张图片对应同名 `.txt` 标注文件
- 支持 `.jpg` `.jpeg` `.png` `.webp`

示例 caption：

```text
leisai, purple hair, green eyes, smile, anime girl
```

---

## 三、训练 LoRA

### 命令示例

```bash
python -m train.main \
  --model_path ./models/stable-diffusion-v1-5 \
  --data_dir ./data \
  --output_dir ./outputs/lora_full \
  --num_epochs 15 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 16
```

### 常用训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | SD 模型路径 | `./models/stable-diffusion-v1-5` |
| `--data_dir` | 训练数据目录 | `./data` |
| `--output_dir` | 输出目录 | `./outputs/lora_full` |
| `--num_epochs` | 训练轮数 | 30 |
| `--batch_size` | 批大小 | 4 |
| `--learning_rate` | 学习率 | 1e-4 |
| `--lora_rank` | LoRA 秩 | 16 |
| `--lora_alpha` | LoRA alpha | 16 |
| `--mixed_precision` | 精度 | fp16 |
| `--gradient_accumulation_steps` | 梯度累积 | 1 |

### 训练输出

```text
outputs/lora_full/final_lora.safetensors        ← 可直接放入 ComfyUI
outputs/lora_full/final_lora_trainer_state.pt   ← 训练状态（可恢复）
```

---

## 四、推理

### 裸模型推理（不加载 LoRA）

```bash
python -m src.main \
  --model_path ./models/anything-v5 \
  --prompt "1girl, purple hair, green eyes, anime style, best quality" \
  --negative_prompt "low quality, blurry, bad anatomy"
```

### 加载 LoRA 推理

```bash
python -m src.main \
  --model_path ./models/anything-v5 \
  --lora_path ./outputs/lora_full/final_lora.safetensors \
  --lora_strength 1.0 \
  --prompt "leisai, purple hair, green eyes, anime girl, best quality"
```

### 常用推理参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--prompt` | 正向提示词 | - |
| `--negative_prompt` | 反向提示词 | `low quality, blurry, bad anatomy` |
| `--model_path` | 模型路径 | `./models/anything-v5` |
| `--lora_path` | LoRA 路径 | - |
| `--lora_strength` | LoRA 强度 | 1.0 |
| `--steps` | 采样步数 | 50 |
| `--guidance_scale` | CFG 引导强度 | 7.5 |
| `--seed` | 随机种子 | 42 |
| `--height / --width` | 图像尺寸 | 512 |
| `--output` | 输出文件名 | `output_lora.png` |
| `--no_lora` | 不加载 LoRA | - |

---

## 五、Smoke Test

### 训练测试

```bash
python test_train_lora.py
```

验证：模型加载 → 数据读取 → 1 轮训练 → safetensors 输出

### 推理测试

```bash
python test_pipeline.py
```

验证：模型加载 → LoRA 加载 → 完整推理 → 图片输出

是否加载 LoRA 可在文件顶部修改 `USE_LORA` 和 `LORA_PATH`。

---

## 六、使用其他底模（anything-v5 等）

ComfyUI 的单文件 `.safetensors` checkpoint 需要先转换为 diffusers 格式：

```bash
python scripts/convert_checkpoint.py -i 你的模型.safetensors -o ./models/模型名
```

转换后在推理命令中指定 `--model_path ./models/模型名` 即可。

---

## 七、AIGC Agent（自然语言生图）

Agent 让你用自然语言驱动 SD 生图和 LoRA 管理，支持本地 LLM 完全离线运行。

### 前置：启动本地 LLM

```bash
# 安装并启动 Ollama
ollama pull qwen2.5:1.5b    # 1.5B 最小可用
ollama pull qwen2.5:3b      # 3B 效果更好
```

### 启动 Agent WebUI

```bash
python agent/webui.py
```

浏览器打开 `http://127.0.0.1:7860`。

在左侧设置面板选择 **「🦙 Ollama + Qwen2.5 1.5B」**，底模路径填 `./models/anything-v5`，即可开始对话。

### 对话示例

```
画一个紫色头发的动漫女孩
加载 final_lora 并画 角色
切换到 anything-v5 底模
用 ./data 训练一个 LoRA，触发词 mystyle（训练Lora时间较长，建议直接使用脚本训练，不使用agent）
```

### 启动 Agent CLI

```bash
python agent/cli.py
```

---

## 八、在 ComfyUI 中使用训练好的 LoRA

训练输出的 `.safetensors` 文件直接放入 ComfyUI：

```text
ComfyUI/models/loras/
```

使用 ComfyUI 自带的 `Load LoRA` 节点加载即可。
