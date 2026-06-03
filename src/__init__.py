from importlib import import_module


_EXPORTS = {
    "CLIPTextEncoder": ".clip",
    "UNet": ".unet",
    "VAE": ".vae",
    "StableDiffusionPipeline": ".pipeline",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
