import os

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def _build_transform(size, center_crop=True, random_flip=False):
    crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
    ops = [
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        crop,
    ]
    if random_flip:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return transforms.Compose(ops)


class ImageTextDataset(Dataset):
    """Image-caption dataset used for standard LoRA training."""

    def __init__(self, data_dir, size=512, center_crop=True):
        self.data_dir = data_dir
        self.size = size
        self.image_files = self._discover_pairs(data_dir)

        if not self.image_files:
            raise ValueError(
                f"No image-caption pairs found under {data_dir}. "
                "Expected either `data_dir/*.jpg + *.txt` or "
                "`data_dir/images/*.jpg` with matching `data_dir/captions/*.txt`."
            )

        print(f"Discovered {len(self.image_files)} image-caption pairs.")
        self.transform = _build_transform(size=size, center_crop=center_crop, random_flip=False)

    def _discover_pairs(self, data_dir):
        image_caption_pairs = []
        images_dir = os.path.join(data_dir, "images")
        captions_dir = os.path.join(data_dir, "captions")

        if os.path.isdir(images_dir) and os.path.isdir(captions_dir):
            search_dir = images_dir
            caption_lookup = lambda file_name: os.path.join(captions_dir, os.path.splitext(file_name)[0] + ".txt")
        else:
            search_dir = data_dir
            caption_lookup = lambda file_name: os.path.splitext(os.path.join(data_dir, file_name))[0] + ".txt"

        if not os.path.isdir(search_dir):
            return []

        for file_name in sorted(os.listdir(search_dir)):
            if not file_name.lower().endswith(IMAGE_EXTENSIONS):
                continue
            image_path = os.path.join(search_dir, file_name)
            caption_path = caption_lookup(file_name)
            if os.path.exists(caption_path):
                image_caption_pairs.append((image_path, caption_path))

        return image_caption_pairs

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path, caption_path = self.image_files[idx]

        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Failed to load image: {image_path}") from exc

        image = self.transform(image)
        with open(caption_path, "r", encoding="utf-8") as handle:
            caption = handle.read().strip()

        if not caption:
            raise ValueError(f"Caption file is empty: {caption_path}")

        return {
            "image": image,
            "caption": caption,
        }


class DreamBoothDataset(Dataset):
    """DreamBooth dataset with optional class prior preservation images."""

    def __init__(self, instance_dir, class_dir=None, size=512, instance_prompt="", class_prompt=""):
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.instance_images = self._load_images(instance_dir)
        self.class_images = self._load_images(class_dir) if class_dir and os.path.exists(class_dir) else []

        if not self.instance_images:
            raise ValueError(f"No training images found in instance_dir: {instance_dir}")

        print(f"Loaded {len(self.instance_images)} instance images.")
        if self.class_images:
            print(f"Loaded {len(self.class_images)} class images.")

        self.transform = _build_transform(size=size, center_crop=True, random_flip=True)

    def _load_images(self, directory):
        if not directory or not os.path.isdir(directory):
            return []

        images = []
        for file_name in sorted(os.listdir(directory)):
            if file_name.lower().endswith(IMAGE_EXTENSIONS):
                images.append(os.path.join(directory, file_name))
        return images

    def __len__(self):
        return len(self.instance_images)

    def __getitem__(self, idx):
        instance_image = Image.open(self.instance_images[idx]).convert("RGB")
        result = {
            "instance_image": self.transform(instance_image),
            "instance_prompt": self.instance_prompt,
        }

        if self.class_images:
            class_idx = idx % len(self.class_images)
            class_image = Image.open(self.class_images[class_idx]).convert("RGB")
            result["class_image"] = self.transform(class_image)
            result["class_prompt"] = self.class_prompt

        return result
