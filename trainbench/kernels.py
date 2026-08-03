"""Which attention kernel actually bound, and the two refusals that reads.

A resolved torch/transformers version does not identify the kernel that ran.
Requesting `flash_attention_2` without the `flash-attn` package rewrites the
request to a Hub repository id (`modeling_utils.py:1997-2003`,
`modeling_flash_attention_utils.py:63-71`) that is fetched while the run starts,
so identity is repo + revision. A run whose revision cannot be read is refused
rather than labelled with the version of the library that went looking.

Registration in the mask registry is a second, separate fact. An implementation
absent from `AttentionMaskInterface._global_mapping` makes `create_causal_mask`
return `None` (`masking_utils.py:821-827`, returned at `:936-940`), and packed
sequences then read each other with no exception and no warning. That flag is
what `assert_packing_is_isolated` refuses on.

Coverage is per sub-config: `attn_implementation` may be a dict and each
sub-config takes its own value (`configuration_utils.py:401-417`), so one string
cannot say which towers got the kernel.

`validate_fingerprint` is a second implementation of the rule that
`tests/contract/test_kernel_provenance.py` states, not an import of it. The
contract is deliberately import-free so a checkout without the `native` extra can
still judge a stored payload; that leaves the runtime free to allow what the
contract refuses. `tests/test_kernels.py` compares the two verdicts on every
mutation case the contract uses, and that comparison is the only thing keeping
them one rule.

This is not `applied._capture_attn`. That answers "did the `attn` axis apply",
counting implementations across every module in the tree. This answers "what is
the identity of the thing that applied, and will a mask be built for it", per
sub-config, and it is the payload the `kernel-provenance` boundary freezes.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import sys
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any

# --- the frozen payload ------------------------------------------------------
# Mirrors tests/contract/test_kernel_provenance.py. Nothing imports across that
# line in either direction; tests/test_kernels.py asserts the values are equal.

BUILD_FINGERPRINT_KEY = "attention"
RUN_RECORD_KEY = "build_fingerprint"
MASK_REGISTRY = "transformers.masking_utils.AttentionMaskInterface"

FINGERPRINT_KEYS = frozenset({"requested", "resolved", "backbones"})
REQUESTED_KEYS = frozenset({"axis", "value", "attn_implementation"})
RESOLVED_KEYS = frozenset({"attn_implementation", "identity", "mask_registered", "mask_registry"})
IDENTITY_KEYS = frozenset(
    {"source", "package", "package_version", "repo_id", "requested_ref", "revision"}
)
BACKBONE_KEYS = frozenset({"attn_implementation", "mask_registered", "requested"})
IDENTITY_SOURCES = frozenset({"hub", "package", "builtin"})

# The distribution that owns each non-Hub implementation name. Anything not here
# ships inside transformers itself.
PACKAGE_BACKED_IMPLEMENTATIONS = {
    "flash_attention_2": "flash-attn",
    "flash_attention_3": "flash-attn",
    "flash_attention_4": "flash-attn",
}

# --- closing the network during a measured run -------------------------------
# Pinning the image digest does not pin the kernel. These are the switches that
# decide whether one can arrive over the network mid-run.

RUNTIME_FETCH_ENV = {"USE_HUB_KERNELS": "NO", "HF_HUB_OFFLINE": "1"}

# Each entry read one of the above once, at import, into a module global. Setting
# the variable afterwards does not reach them, so they are closed by name too.
CACHED_FETCH_SWITCHES = (
    ("transformers.integrations.hub_kernels", "_TRANSFORMERS_USE_HUB_KERNELS", "NO"),
    ("transformers.integrations.hub_kernels", "_kernels_enabled", False),
    ("huggingface_hub.constants", "HF_HUB_OFFLINE", True),
    ("kernels.layer.globals", "_DISABLE_KERNEL_MAPPING", True),
)


class KernelProvenanceError(RuntimeError):
    """Base for every refusal in this module."""


class UnidentifiedKernel(KernelProvenanceError):
    """The kernel that ran cannot be named, so no number from this run is usable."""


class UnsafePacking(KernelProvenanceError):
    """Packing was requested where the mask that isolates the sequences is not built."""


class RuntimeKernelFetch(KernelProvenanceError):
    """A door a kernel could arrive through over the network is still open."""


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise UnidentifiedKernel(
            f"{where} must be a mapping, got {type(value).__name__}: {value!r}"
        )
    return value


def _exact_keys(value: dict[str, Any], expected: frozenset[str], where: str) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        raise UnidentifiedKernel(
            f"{where} keys are frozen by the kernel-provenance boundary; "
            f"missing={missing} unexpected={unexpected}"
        )


def _flag(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise UnidentifiedKernel(f"{where} must be a bool, got {type(value).__name__}: {value!r}")
    return value


def _name(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UnidentifiedKernel(f"{where} must be a non-empty string, got {value!r}")
    return value


def _validate_identity(identity: Any) -> dict[str, Any]:
    ident = _mapping(identity, "resolved.identity")
    _exact_keys(ident, IDENTITY_KEYS, "resolved.identity")
    source = ident["source"]
    if source not in IDENTITY_SOURCES:
        raise UnidentifiedKernel(
            f"resolved.identity.source must be one of {sorted(IDENTITY_SOURCES)}, got {source!r}"
        )
    if source == "hub":
        for field in ("repo_id", "revision"):
            if not isinstance(ident[field], str) or not ident[field].strip():
                raise UnidentifiedKernel(
                    "a Hub kernel is fetched at runtime and is identified by repo_id and "
                    f"revision; resolved.identity.{field}={ident[field]!r}. A package version "
                    "identifies the library that went looking, not the kernel that bound."
                )
        return ident
    for field in ("package", "package_version"):
        if not isinstance(ident[field], str) or not ident[field].strip():
            raise UnidentifiedKernel(
                f"resolved.identity.source={source!r} is identified by its distribution, so "
                f"{field} is required; got {ident[field]!r}"
            )
    for field in ("repo_id", "revision"):
        if ident[field] is not None:
            raise UnidentifiedKernel(
                f"resolved.identity.source={source!r} did not come from the Hub, so {field} "
                f"must be null; got {ident[field]!r}"
            )
    return ident


def validate_fingerprint(payload: Any) -> dict[str, Any]:
    """Raise `UnidentifiedKernel` unless `payload` is a kernel fingerprint.

    Returns the payload so a caller can validate and bind in one step.
    """
    fingerprint = _mapping(payload, "fingerprint")
    _exact_keys(fingerprint, FINGERPRINT_KEYS, "fingerprint")

    requested = _mapping(fingerprint["requested"], "requested")
    _exact_keys(requested, REQUESTED_KEYS, "requested")
    for field in sorted(REQUESTED_KEYS):
        _name(requested[field], f"requested.{field}")

    resolved = _mapping(fingerprint["resolved"], "resolved")
    _exact_keys(resolved, RESOLVED_KEYS, "resolved")
    resolved_impl = _name(resolved["attn_implementation"], "resolved.attn_implementation")
    identity = _validate_identity(resolved["identity"])
    resolved_registered = _flag(resolved["mask_registered"], "resolved.mask_registered")
    if resolved["mask_registry"] != MASK_REGISTRY:
        raise UnidentifiedKernel(
            "resolved.mask_registry must name the registry that decides mask creation, "
            f"{MASK_REGISTRY!r}; got {resolved['mask_registry']!r}"
        )
    if identity["source"] == "hub" and not resolved_impl.startswith(identity["repo_id"]):
        raise UnidentifiedKernel(
            f"resolved.attn_implementation={resolved_impl!r} does not name "
            f"resolved.identity.repo_id={identity['repo_id']!r}; the recorded provenance "
            "belongs to a different kernel than the one the model is running."
        )

    backbones = fingerprint["backbones"]
    if not isinstance(backbones, dict) or not backbones:
        raise UnidentifiedKernel(
            "backbones must be a non-empty mapping of sub-config name -> coverage. "
            "attn_implementation is set per sub-config and a backbone left out of the request "
            f"keeps SDPA, so a single value cannot say which towers got the kernel; got "
            f"{backbones!r}"
        )

    covered = []
    for name, entry in sorted(backbones.items()):
        where = f"backbones[{name!r}]"
        coverage = _mapping(entry, where)
        _exact_keys(coverage, BACKBONE_KEYS, where)
        impl = _name(coverage["attn_implementation"], f"{where}.attn_implementation")
        registered = _flag(coverage["mask_registered"], f"{where}.mask_registered")
        if _flag(coverage["requested"], f"{where}.requested"):
            covered.append(name)
            if impl != resolved_impl:
                raise UnidentifiedKernel(
                    f"{where} is marked as having received the request but runs {impl!r}, "
                    f"not resolved.attn_implementation={resolved_impl!r}"
                )
            if registered != resolved_registered:
                raise UnidentifiedKernel(
                    f"{where}.mask_registered={registered} contradicts "
                    f"resolved.mask_registered={resolved_registered} for the same implementation"
                )
        elif impl == resolved_impl:
            raise UnidentifiedKernel(
                f"{where} runs the requested implementation {impl!r} but is marked "
                "requested=false; coverage would read as narrower than it is"
            )
    if not covered:
        raise UnidentifiedKernel(
            "no backbone is marked as having received the requested implementation, so the "
            "request reached nothing and the fingerprint records a kernel that is not running"
        )
    return fingerprint


def packing_isolation_holds(fingerprint: Mapping[str, Any]) -> bool:
    """Whether packed sequences stay isolated under this fingerprint.

    One backbone whose implementation builds no mask is enough to lose isolation
    for the whole forward, which is why this is `all` and not a look at the text
    tower alone.
    """
    return all(entry["mask_registered"] for entry in fingerprint["backbones"].values())


def assert_packing_is_isolated(fingerprint: Mapping[str, Any]) -> None:
    """Refuse `dataloader.packing` on a build that would drop the isolation mask."""
    validate_fingerprint(fingerprint)
    unregistered = sorted(
        name for name, entry in fingerprint["backbones"].items() if not entry["mask_registered"]
    )
    if not unregistered:
        return
    running = {name: fingerprint["backbones"][name]["attn_implementation"] for name in unregistered}
    raise UnsafePacking(
        f"dataloader.packing=true is refused: {running} is absent from {MASK_REGISTRY}, so "
        "transformers skips mask creation and passes attention_mask=None to the attention "
        "layers. The block-diagonal mask that keeps packed sequences from reading each other "
        "silently disappears — no exception, no warning — and the run would report a "
        "throughput number for a batch whose sequences are each other's context. Request an "
        "implementation the mask registry knows, or turn packing off."
    )


def _sub_config_implementations(config: Any) -> tuple[dict[str, str], str | None]:
    """Per-sub-config attention implementation, and the name the `""` key reaches.

    The second element is the backbone the top-level `""` key addresses, which is
    the parent config and therefore exists only when the model has no sub-configs.
    A model that has them keeps its parent out of the coverage map, so `""` reaches
    nothing there — see `_requested_by_backbone`.
    """
    sub_configs = getattr(config, "sub_configs", None) or {}
    found: dict[str, str] = {}
    for key in sub_configs:
        sub = getattr(config, key, None)
        impl = getattr(sub, "_attn_implementation", None)
        if impl is None:
            # Dropping it would report coverage over fewer towers than the model
            # has, which is the reading the per-sub-config map exists to prevent.
            raise UnidentifiedKernel(
                f"sub-config {key!r} records no _attn_implementation, so the fingerprint would "
                "describe a model with one tower fewer than the one that ran"
            )
        found[key] = str(impl)
    if found:
        return found, None
    impl = getattr(config, "_attn_implementation", None)
    if impl is None:
        raise UnidentifiedKernel(
            "the built config records no _attn_implementation on itself or any sub-config, so "
            "nothing says which kernel the model is running"
        )
    parent = str(getattr(config, "model_type", None) or "model")
    return {parent: str(impl)}, parent


def _mask_registry() -> Mapping[str, Any]:
    from transformers.masking_utils import AttentionMaskInterface

    return AttentionMaskInterface._global_mapping


def _is_hub_implementation(implementation: str) -> bool:
    from transformers.integrations.hub_kernels import is_kernel

    return bool(is_kernel(implementation))


def _distribution_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _identify(
    implementation: str,
    requested_ref: str | None,
    revision_resolver: Callable[[str], str | None] | None,
) -> dict[str, Any]:
    if _is_hub_implementation(implementation):
        repo_id, _, inline_ref = implementation.partition("@")
        revision = inline_ref or (revision_resolver(repo_id) if revision_resolver else None)
        if not revision:
            raise UnidentifiedKernel(
                f"{implementation!r} was resolved from the Hub and no revision came back. A Hub "
                "kernel is identified by repo_id and revision; substituting a package version "
                "would name the library that went looking, not the kernel that bound. Pin the "
                "kernel (kernels.lock / an @revision suffix) or do not run this cell."
            )
        return {
            "source": "hub",
            "package": None,
            "package_version": None,
            "repo_id": repo_id,
            "requested_ref": requested_ref or (f"@{inline_ref}" if inline_ref else None),
            "revision": revision,
        }
    distribution = PACKAGE_BACKED_IMPLEMENTATIONS.get(implementation)
    if distribution is not None:
        version = _distribution_version(distribution)
        if version is None:
            raise UnidentifiedKernel(
                f"the model reports {implementation!r} but {distribution!r} is not installed, so "
                "the kernel behind that name cannot be identified"
            )
        return {
            "source": "package",
            "package": distribution,
            "package_version": version,
            "repo_id": None,
            "requested_ref": requested_ref,
            "revision": None,
        }
    version = _distribution_version("transformers")
    if version is None:
        raise UnidentifiedKernel(
            f"{implementation!r} ships inside transformers and transformers reports no version"
        )
    return {
        "source": "builtin",
        "package": "transformers",
        "package_version": version,
        "repo_id": None,
        "requested_ref": requested_ref,
        "revision": None,
    }


def _requested_by_backbone(
    requested: str | Mapping[str, str],
    backbones: Mapping[str, str],
    parent_key: str | None,
) -> dict[str, str]:
    """Which backbone each requested value actually landed on.

    A dict does not broadcast. transformers sets each sub-config from
    `value.get(<sub-config key>, <its current value>)` and the parent from
    `value.get("")` (`configuration_utils.py:401-417`), so `""` reaches the parent
    only, a key naming no sub-config reaches nothing, and an unnamed backbone keeps
    what it had. When the model has no sub-configs the parent is the sole backbone
    and `""` is the only key that lands on it.
    """
    if isinstance(requested, str):
        return {name: requested for name in backbones}
    if parent_key is not None:
        landed = {parent_key: str(requested[""])} if "" in requested else {}
    else:
        landed = {name: str(requested[name]) for name in backbones if name in requested}
    if not landed:
        raise UnidentifiedKernel(
            f"attn_implementation was requested per sub-config as {dict(requested)!r} and none of "
            f"those keys names a backbone of this model ({sorted(backbones)}). A dict does not "
            "broadcast: transformers reads each sub-config out of it by its own key and the "
            "parent out of the '' key, so this request bound no kernel anywhere and the run is "
            "measuring the implementation the checkpoint loaded with."
        )
    return landed


def _repository(implementation: str) -> str:
    """`repo@revision` without the revision, and anything else unchanged.

    The request names a repository; what binds may carry the `@ref` the resolution
    pinned, and comparing the two as raw strings reads one kernel as two.
    """
    return implementation.partition("@")[0]


def _names_the_request_can_bind_as(asked: str) -> set[str]:
    """Every implementation string `asked` may be reported as once the model is built.

    A flash-attention name is not a kernel name. With `flash-attn` absent
    transformers rewrites it to a Hub repository — `FLASH_ATTN_KERNEL_FALLBACK`
    (`modeling_flash_attention_utils.py:63-71`), applied inside
    `_check_and_adjust_attn_implementation` — so the string the config asked for and
    the string the backbone reports are two names for one request. Without this the
    rewrite reads as a backbone that did not receive the request.
    """
    from transformers.modeling_flash_attention_utils import FLASH_ATTN_KERNEL_FALLBACK

    fallback = FLASH_ATTN_KERNEL_FALLBACK.get(asked)
    return {asked} if fallback is None else {asked, fallback}


def _bound_by_backbone(landed: Mapping[str, str], backbones: Mapping[str, str]) -> dict[str, str]:
    """The backbones running what the request asked them for, and what they run.

    A backbone the request reached is not a backbone the request bound. transformers
    resolves each submodule separately — `submodule.get_correct_attn_implementation(
    sub_implementation)` at `modeling_utils.py:2238-2239`, whose sdpa branch falls
    back to `eager` at `:2088-2093` — so one string request routinely lands
    different implementations on different towers. That is the state the
    per-sub-config map exists to record: `fa2_hub_fallback_qwen3_vl` is exactly it,
    a string request with the vision tower left on SDPA. Reading every reached
    backbone as bound made that frozen sample unproducible and refused the whole
    `attn=fa2/fa3/fa4` x Qwen3-VL column as an unidentified kernel.
    """
    return {
        name: backbones[name]
        for name, asked in landed.items()
        if _repository(backbones[name]) in _names_the_request_can_bind_as(asked)
    }


def _one(values: list[str], what: str) -> str:
    unique = sorted(set(values))
    if len(unique) != 1:
        raise UnidentifiedKernel(
            f"the backbones the request reached do not agree on {what}: {unique}. One axis value "
            "bound more than one kernel and the frozen payload records a single identity, so "
            "this build cannot be described rather than being described wrongly."
        )
    return unique[0]


def read_fingerprint(
    model: Any,
    *,
    axis: str,
    value: str,
    requested: str | Mapping[str, str],
    requested_ref: str | None = None,
    revision_resolver: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Read the bound attention kernel back off `model` as a frozen fingerprint.

    `requested` is what the run asked for — the string or per-sub-config dict handed
    to `from_pretrained`. Everything else comes off the built config and the live
    mask registry, because the same requested name is safe or unsafe depending on
    what it resolved to.
    """
    config = getattr(model, "config", None)
    if config is None:
        raise UnidentifiedKernel("the built model carries no config to read the kernel back from")

    backbones, parent_key = _sub_config_implementations(config)
    registry = _mask_registry()
    landed = _requested_by_backbone(requested, backbones, parent_key)
    bound = _bound_by_backbone(landed, backbones)
    if not bound:
        raise UnidentifiedKernel(
            f"the request {requested!r} reached {sorted(landed)} and no backbone is running it: "
            f"{ {name: backbones[name] for name in sorted(landed)} }. Every tower resolved to "
            "something else, so this build is measuring an implementation nobody asked for."
        )
    resolved_impl = _one([bound[name] for name in sorted(bound)], "the implementation that bound")
    asked = (
        requested if isinstance(requested, str) else _one(list(landed.values()), "what was asked")
    )
    resolved_registered = resolved_impl in registry

    return validate_fingerprint(
        {
            "requested": {
                "axis": axis,
                "value": value,
                "attn_implementation": asked,
            },
            "resolved": {
                "attn_implementation": resolved_impl,
                "identity": _identify(resolved_impl, requested_ref, revision_resolver),
                "mask_registered": resolved_registered,
                "mask_registry": MASK_REGISTRY,
            },
            "backbones": {
                name: {
                    "attn_implementation": impl,
                    "mask_registered": impl in registry,
                    "requested": impl == resolved_impl,
                }
                for name, impl in sorted(backbones.items())
            },
        }
    )


def _env_door(environ: Mapping[str, str], name: str, want: str) -> str | None:
    got = environ.get(name)
    if got is not None and got.upper() == want.upper():
        return None
    return f"${name}={got!r}, want {want!r}"


def open_fetch_doors(
    environ: Mapping[str, str] | None = None,
    modules: Mapping[str, Any] | None = None,
) -> list[str]:
    """Every way a kernel could still arrive over the network during this run.

    An empty list is the only state a timing run may start in. Reading the modules
    as well as the environment is the point: both variables below are read once,
    at import, into a module global, so a process that already imported
    transformers keeps the value it read no matter what is set afterwards.

    An imported module that does not carry the name is a door, not a closed one.
    `CACHED_FETCH_SWITCHES` is read off one version of each library and the six pod
    images do not share versions; a renamed global read as absent would leave the
    live switch untouched and report the process as closed.
    """
    env = os.environ if environ is None else environ
    loaded = sys.modules if modules is None else modules
    doors = [
        door
        for name, want in sorted(RUNTIME_FETCH_ENV.items())
        if (door := _env_door(env, name, want)) is not None
    ]
    for module_name, attribute, want in CACHED_FETCH_SWITCHES:
        module = loaded.get(module_name)
        if module is None:
            continue
        if not hasattr(module, attribute):
            doors.append(
                f"{module_name}.{attribute} is absent from the imported module — this check "
                "reads the switch by name, so the installed version renamed or dropped it and "
                "whatever it uses instead is still whatever it was"
            )
            continue
        got = getattr(module, attribute)
        if got != want:
            doors.append(
                f"{module_name}.{attribute}={got!r}, want {want!r} — cached at import, so the "
                "environment variable was set too late to reach it"
            )
    return doors


def forbid_runtime_kernel_fetch(
    environ: MutableMapping[str, str] | None = None,
    modules: Mapping[str, Any] | None = None,
) -> list[str]:
    """Close every door in `open_fetch_doors` and return the ones that were open.

    Reading training data off a network volume and downloading a kernel mid-run are
    the same contamination: the measurement stops being of the pipeline. Call this
    before the model is built.

    A name the imported module does not already carry is left alone. Creating it
    would close nothing — the library reads its own global, not this one — while
    making the door this function just reported disappear from the next reading.
    """
    env = os.environ if environ is None else environ
    loaded = sys.modules if modules is None else modules
    was_open = open_fetch_doors(env, loaded)
    env.update(RUNTIME_FETCH_ENV)
    for module_name, attribute, want in CACHED_FETCH_SWITCHES:
        module = loaded.get(module_name)
        if module is not None and hasattr(module, attribute):
            setattr(module, attribute, want)
    return was_open


def assert_no_runtime_kernel_fetch(
    environ: Mapping[str, str] | None = None,
    modules: Mapping[str, Any] | None = None,
) -> None:
    """Refuse to start a measured run while a kernel could still be downloaded."""
    doors = open_fetch_doors(environ, modules)
    if doors:
        raise RuntimeKernelFetch(
            "a kernel can still be fetched over the network during this run, and a kernel that "
            "arrives mid-run is not the kernel the image pins: " + "; ".join(doors)
        )


def flash_attention_falls_back_to_the_hub(implementation: str) -> bool:
    """Whether requesting `implementation` would rewrite to a Hub repository id.

    `USE_HUB_KERNELS` does not close this path: the rewrite at
    `modeling_utils.py:1997-2003` is gated on `is_kernels_available()` and nothing
    else, so a flash-attention request with `flash-attn` absent and `kernels`
    installed becomes a repo id whatever that variable says. What closes it is
    `HF_HUB_OFFLINE`, which turns the download into a refusal instead of a fetch.
    """
    if implementation not in PACKAGE_BACKED_IMPLEMENTATIONS:
        return False
    if importlib.util.find_spec("flash_attn") is not None:
        return False
    from transformers.utils.import_utils import is_kernels_available

    return bool(is_kernels_available())
