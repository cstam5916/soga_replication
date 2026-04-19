import warnings

_SSFUG_VARIANTS = {"SOGA", "IMOnly", "SCOnly"}
_ALGO_TYPES = {"SSFUG", "MSFUG", "MSSG", "MSFU"}
_EMBEDMODE_UNUSED_VARIANTS = {"IMOnly"}


def parse_mode(mode: str):
    parts = mode.split("_", 2)
    algo_type = parts[0] if len(parts) >= 1 else None
    variant = parts[1] if len(parts) >= 2 else None
    embed_mode = parts[2] if len(parts) >= 3 else None

    if algo_type not in _ALGO_TYPES:
        raise ValueError(f"Unknown algo type {algo_type!r} in --mode {mode!r}")
    if algo_type != "SSFUG":
        raise NotImplementedError(f"Algorithm type {algo_type!r} is not yet implemented")
    if variant is not None and variant not in _SSFUG_VARIANTS:
        raise NotImplementedError(f"SSFUG variant {variant!r} is not yet implemented")

    if variant in _EMBEDMODE_UNUSED_VARIANTS:
        if embed_mode is not None:
            warnings.warn(
                f"EMBEDMODE {embed_mode!r} is ignored when VARIANT is {variant!r}",
                stacklevel=2,
            )
        embed_mode = None

    return algo_type, variant, embed_mode
