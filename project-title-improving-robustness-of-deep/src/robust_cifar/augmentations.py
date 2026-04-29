from __future__ import annotations

import io
import random
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torchvision.transforms import functional as TF


def _float_parameter(level: float, maxval: float) -> float:
    return float(level) * maxval / 10.0


def _int_parameter(level: float, maxval: int) -> int:
    return int(float(level) * maxval / 10)


def autocontrast(img: Image.Image, severity: float) -> Image.Image:
    return ImageOps.autocontrast(img)


def equalize(img: Image.Image, severity: float) -> Image.Image:
    return ImageOps.equalize(img)


def rotate(img: Image.Image, severity: float) -> Image.Image:
    degrees = _float_parameter(severity, 30)
    if random.random() > 0.5:
        degrees = -degrees
    return img.rotate(degrees, resample=Image.BILINEAR)


def solarize(img: Image.Image, severity: float) -> Image.Image:
    threshold = 256 - _int_parameter(severity, 256)
    return ImageOps.solarize(img, threshold)


def posterize(img: Image.Image, severity: float) -> Image.Image:
    bits = max(1, 8 - _int_parameter(severity, 4))
    return ImageOps.posterize(img, bits)


def color(img: Image.Image, severity: float) -> Image.Image:
    return ImageEnhance.Color(img).enhance(1.0 + random.choice([-1, 1]) * _float_parameter(severity, 0.9))


def contrast(img: Image.Image, severity: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(1.0 + random.choice([-1, 1]) * _float_parameter(severity, 0.9))


def brightness(img: Image.Image, severity: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(1.0 + random.choice([-1, 1]) * _float_parameter(severity, 0.9))


def sharpness(img: Image.Image, severity: float) -> Image.Image:
    return ImageEnhance.Sharpness(img).enhance(1.0 + random.choice([-1, 1]) * _float_parameter(severity, 0.9))


def shear_x(img: Image.Image, severity: float) -> Image.Image:
    level = _float_parameter(severity, 0.3)
    if random.random() > 0.5:
        level = -level
    return img.transform(img.size, Image.AFFINE, (1, level, 0, 0, 1, 0), resample=Image.BILINEAR)


def shear_y(img: Image.Image, severity: float) -> Image.Image:
    level = _float_parameter(severity, 0.3)
    if random.random() > 0.5:
        level = -level
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, level, 1, 0), resample=Image.BILINEAR)


def translate_x(img: Image.Image, severity: float) -> Image.Image:
    pixels = _int_parameter(severity, img.size[0] // 3)
    if random.random() > 0.5:
        pixels = -pixels
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), resample=Image.BILINEAR)


def translate_y(img: Image.Image, severity: float) -> Image.Image:
    pixels = _int_parameter(severity, img.size[1] // 3)
    if random.random() > 0.5:
        pixels = -pixels
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), resample=Image.BILINEAR)


AUGMIX_OPS: list[Callable[[Image.Image, float], Image.Image]] = [
    autocontrast,
    equalize,
    rotate,
    solarize,
    posterize,
    color,
    contrast,
    brightness,
    sharpness,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
]


def augmix_image(
    img: Image.Image,
    severity: int = 3,
    width: int = 3,
    depth: int = -1,
    alpha: float = 1.0,
) -> Image.Image:
    """AugMix image augmentation from the ICLR 2020 paper."""
    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))
    mix = np.zeros_like(np.asarray(img)).astype("float32")

    for i in range(width):
        image_aug = img.copy()
        depth_i = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(depth_i):
            op = random.choice(AUGMIX_OPS)
            image_aug = op(image_aug, severity)
        mix += ws[i] * np.asarray(image_aug).astype("float32")

    mixed = (1 - m) * np.asarray(img).astype("float32") + m * mix
    mixed = np.clip(mixed, 0, 255).astype("uint8")
    return Image.fromarray(mixed)


def corrupt_pil_image(img: Image.Image, severity: int = 2) -> Image.Image:
    """Apply a lightweight random corruption for consistency training."""
    choice = random.choice(["gaussian_noise", "blur", "jpeg", "brightness", "contrast"])
    if choice == "gaussian_noise":
        arr = np.asarray(img).astype("float32") / 255.0
        sigma = 0.03 + 0.03 * severity
        arr = np.clip(arr + np.random.normal(0.0, sigma, arr.shape), 0.0, 1.0)
        return Image.fromarray((arr * 255).astype("uint8"))
    if choice == "blur":
        tensor = TF.to_tensor(img)
        kernel_size = 3 if severity <= 2 else 5
        blurred = TF.gaussian_blur(tensor, kernel_size=kernel_size, sigma=0.4 + 0.2 * severity)
        return TF.to_pil_image(blurred)
    if choice == "jpeg":
        buffer = io.BytesIO()
        quality = max(15, 95 - severity * 15)
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if choice == "brightness":
        return ImageEnhance.Brightness(img).enhance(1.0 + random.choice([-1, 1]) * 0.12 * severity)
    return ImageEnhance.Contrast(img).enhance(1.0 + random.choice([-1, 1]) * 0.12 * severity)
