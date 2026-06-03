import os

import torch
from transformers import CLIPTextModel, CLIPTokenizer


LOCAL_TOKENIZER_FILES = (
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
LOCAL_TEXT_ENCODER_FILES = (
    "config.json",
    "model.safetensors",
)


def _resolve_dtype(device, dtype):
    if str(device) == "cpu" and dtype == torch.float16:
        return torch.float32
    return dtype


def _has_required_files(base_dir, required_files):
    return all(os.path.exists(os.path.join(base_dir, name)) for name in required_files)


def _resolve_clip_paths(model_path):
    if not model_path:
        return "openai/clip-vit-large-patch14", "openai/clip-vit-large-patch14", False

    tokenizer_path = os.path.join(model_path, "tokenizer")
    text_encoder_path = os.path.join(model_path, "text_encoder")
    has_tokenizer = _has_required_files(tokenizer_path, LOCAL_TOKENIZER_FILES)
    has_text_encoder = _has_required_files(text_encoder_path, LOCAL_TEXT_ENCODER_FILES)

    if has_tokenizer and has_text_encoder:
        return tokenizer_path, text_encoder_path, True

    return "openai/clip-vit-large-patch14", "openai/clip-vit-large-patch14", False


class CLIPTextEncoder:
    """CLIP tokenizer + text encoder wrapper for SD1.5."""

    def __init__(self, device="cuda", dtype=torch.float16, model_path=None):
        self.device = device
        self.dtype = _resolve_dtype(device, dtype)
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        tokenizer_path, text_encoder_path, local_only = _resolve_clip_paths(model_path)

        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                tokenizer_path,
                local_files_only=local_only,
            )
            self.text_model = CLIPTextModel.from_pretrained(
                text_encoder_path,
                local_files_only=local_only,
            )
        except Exception as exc:
            print(f"Failed to load CLIP tokenizer/text encoder: {exc}")
            print("Tip: run `python download_models.py` to fetch the local SD1.5 files.")
            raise

        self.max_length = self.tokenizer.model_max_length
        self.text_model.to(device=device, dtype=self.dtype)
        self.text_model.eval()

    def _encode_text(self, text):
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            hidden_state = self.text_model(tokens.input_ids.to(self.device)).last_hidden_state
        return hidden_state.to(self.dtype)

    def encode(self, prompt, negative_prompt=""):
        positive_embeds = self._encode_text(prompt)
        negative_embeds = self._encode_text(negative_prompt or "")
        return positive_embeds, negative_embeds
