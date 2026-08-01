"""Where an axis is turned on.

`trainbench/applied.py` reads back what a model ended up running; this module is
the other half — what asks for it in the first place. They are separated because
they are checked against each other: an axis is only certified when the code that
applies it and the code that verifies it are both present, and `IMPLEMENTED` here
is required to equal the set of capture probes over there.

Everything an axis needs at model construction time goes through `load_kwargs`;
everything that happens to an already-built model goes through `apply`. Callers
never pass `attn_implementation=` or flip `requires_grad` themselves — a second
application site is how a run ends up measuring a setting nobody selected.

Axes not listed in `IMPLEMENTED` are not silently skipped: they have no capture
probe either, so `applied.assert_matches` blocks any run that would report a
number for them.
"""

from __future__ import annotations

from typing import Any

from trainbench.config_schema import BenchConfig

# gemma-4's per-layer embeddings. Every one of the 108 PLE tensors carries this
# in its name — `language_model.embed_tokens_per_layer.weight`,
# `layers.N.per_layer_input_gate`, `per_layer_model_projection`, and so on
# (docs/model-spec.md, read off model.safetensors.index.json). Substring rather
# than an enumerated list because the layer index is part of the name.
PLE_PARAM_MARKER = "per_layer"

# Axis knobs this module can actually put into effect. Kept honest by
# tests/test_axes.py, which requires this set to match applied._CAPTURES exactly:
# an axis that can be applied but not verified is the failure mode applied.py
# exists for, and one that can be verified but not applied certifies a default.
IMPLEMENTED = frozenset({"attn.name", "freeze.ple"})


def load_kwargs(config: BenchConfig) -> dict[str, Any]:
    """Keyword arguments for `from_pretrained`.

    Attention is set here rather than afterwards because transformers validates
    and may downgrade the request during construction; setting it later would mean
    the model was built once with the wrong one.
    """
    return {"attn_implementation": config.attn.impl}


def ple_parameters(model: Any) -> list[tuple[str, Any]]:
    """The per-layer embedding tensors, by name."""
    return [(n, p) for n, p in model.named_parameters() if PLE_PARAM_MARKER in n]


def apply(model: Any, config: BenchConfig) -> list[str]:
    """Put the post-construction axes into effect. Returns the axes applied.

    Does not report success — `applied.capture` does that by looking at the model
    afterwards. A function that both acts and certifies its own action cannot
    catch the case where the action did not take.
    """
    applied = []
    if config.freeze.ple:
        for _, param in ple_parameters(model):
            param.requires_grad_(False)
        applied.append("freeze.ple")
    return applied
