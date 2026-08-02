# axis-libraries — 핀 원문 리서치 브리프

소비 레인: `kernels`, `axes`.
작성일 2026-08-02. 호스트: darwin/aarch64 (CUDA 없음).

이 브리프의 모든 주장은 **lock이 기록한 URL에서 받아 sha256을 lock과 대조한 아티팩트**의 원문에서만
나온다. 웹 검색·기억·업스트림 문서는 근거로 쓰지 않았다. 대조에 실패했거나 받지 못한 것은
마지막 절 "이 호스트에서 확정하지 못한 것"에 있다.

---

## 0. 핀 해석 — 무엇을 어떻게 열었나

### 0.1 이 호스트에 캐시된 것은 transformers 뿐이었다

프로토콜이 지시한 dist-info 해석을 7개 대상 전부에 대해 돌렸고, **전부 미스**였다.

```
ls -d ~/.cache/uv/archive-v0/*/{liger_kernel,kernels,fla,fla_core,flash_linear_attention,bitsandbytes,deepspeed,transformer_engine,nvidia_dali,causal_conv1d}-*.dist-info
  → (eval):2: no matches found
```

이유는 명확하다. 이 패키지들은 `envs/*/uv.lock`에서 전부 `marker = "sys_platform == 'linux'"`를
달고 있고 (예: `/Users/jwcho/Codes/train-comparison/envs/native/uv.lock:152-156`), darwin 호스트의
`uv sync`는 이들을 애초에 받지 않는다. 그래서 이 브리프의 소스는 전량 **lock의 URL에서 직접
받아 sha256을 대조한 것**이다.

한편 transformers는 캐시에 있었고, **디코이가 10개 함께 있었다.** 프로토콜의 경고가
이 저장소에서 실측으로 재현된다:

```
ls -d ~/.cache/uv/archive-v0/*/transformers-*.dist-info
/Users/jwcho/.cache/uv/archive-v0/70OhQvhQj042zLtn/transformers-5.13.0.dist-info/
/Users/jwcho/.cache/uv/archive-v0/Bh3N2cKaCNDMURw4/transformers-5.12.0.dist-info/
/Users/jwcho/.cache/uv/archive-v0/JV7sX-v4goOSOToR/transformers-5.13.1.dist-info/
/Users/jwcho/.cache/uv/archive-v0/Kur5R2PrM3RUwEti/transformers-5.14.1.dist-info/   ← native/axolotl/st/tevatron 핀
/Users/jwcho/.cache/uv/archive-v0/Oqy6RtNB3vvkLBO3/transformers-5.10.2.dist-info/
/Users/jwcho/.cache/uv/archive-v0/Rj8Th9Bs2T6mue_z/transformers-4.57.6.dist-info/
/Users/jwcho/.cache/uv/archive-v0/SMXDMezQuUQYpkT2/transformers-5.3.0.dist-info/
/Users/jwcho/.cache/uv/archive-v0/nB97V5Iacpret1f5/transformers-5.9.0.dist-info/
/Users/jwcho/.cache/uv/archive-v0/plcyRhzg-LE7LDvn/transformers-5.5.0.dist-info/   ← unsloth 핀
/Users/jwcho/.cache/uv/archive-v0/u8fTA70ydUgg54UN/transformers-5.11.0.dist-info/
/Users/jwcho/.cache/uv/archive-v0/uOhIKcY-1QKGNf7V/transformers-5.12.1.dist-info/  ← ms-swift 핀
```

경로에는 어느 버전인지 적혀 있지 않다. `transformers-*` 로 glob 하면 11개 중 하나를 임의로
집는다. 이 브리프의 transformers 인용은 전부 `Kur5R2PrM3RUwEti`(=5.14.1) 하위다.

### 0.2 `uv pip download` 는 이 호스트에 없다

프로토콜이 준 명령은 이 uv에서 동작하지 않는다.

```
$ uv --version
uv 0.11.16 (135a36367 2026-05-21 aarch64-apple-darwin)
$ uv pip download 'liger-kernel==0.8.1' --no-deps --only-binary=:all: ...
error: unrecognized subcommand 'download'
```

대신 **lock이 기록한 정확한 wheel URL을 curl로 받아 lock의 `hash = "sha256:..."` 와 직접 대조**했다.
이쪽이 오히려 엄밀하다 — 리졸버를 거치지 않으므로 "이름이 같은 다른 아티팩트"가 끼어들 여지가 없다.
9개 전부 OK:

```
OK   liger_kernel-0.8.1-py3-none-any.whl  90836c1e9de22a1c57a640775fb79151ea704748cf9c3f6658dddaa06fff1989
OK   liger_kernel-0.8.0-py3-none-any.whl  e1f03eeb4ba6a6a413d585dacf92c4c15d164bab5844fa4ead2fede6bcac469c
OK   kernels-0.16.0-py3-none-any.whl  794af6a10fd888bb4f46ad1b9b2f4f61b5b0b104475a6415c5322b58a7bf02ed
OK   flash_linear_attention-0.5.2-py3-none-any.whl  dcf405d81f5426393b59037097aa700d0f4a841465d5028d5aa543f4502f2400
OK   fla_core-0.5.2-py3-none-any.whl  5e830c85bad3d0d34677f98ac7074d08687a3756f0f0499d95ceb96eb6920761
OK   transformer_engine-2.17.0-py3-none-any.whl  c274fd74ea2e4caa7132921e1ecfa3847931abbed9f6b8cab61346cdb833bc76
OK   bitsandbytes-0.50.0-py3-none-manylinux_2_24_x86_64.whl  173d137610468bec9cddbaa2e049254e97792657ab984e3e737bec1772c1668c
OK   deepspeed-0.19.3.tar.gz  82b4b92d0cd58f8fc461b80428a5be2838f4adad48ae3aef509495c640c9ed2f
OK   causal_conv1d-1.6.2.post1.tar.gz  245e314ea21064ded7a5bf6b3b842b644aa6f92e45cecfe3e935629744c35ff4
OK   nvidia_dali_cuda130-2.2.0-py3-none-manylinux_2_28_x86_64.whl  a957b654bd851f36b65ab8c99705bba40cf699d10a11cccecdc995523f739524
```

### 0.3 lock별 핀 표 (lock에서 복사, 기억 아님)

| 패키지 | native | axolotl | ms-swift | unsloth | tevatron | st |
|---|---|---|---|---|---|---|
| `transformers` | 5.14.1 | 5.14.1 | 5.12.1 | 5.5.0 | 5.14.1 | 5.14.1 |
| `torch` | 2.13.0+cu130 | 2.12.1+cu130 | 2.13.0+cu130 | 2.11.0+cu130 | 2.13.0+cu130 | 2.13.0+cu130 |
| `triton` | 3.7.1 | 3.7.1 | 3.7.1 | 3.6.0 | 3.7.1 | 3.7.1 |
| `liger-kernel` | **0.8.1** | **0.8.0** | — | — | — | — |
| `kernels` | **0.16.0** | **0.15.2** | — | — | — | — |
| `flash-linear-attention` | 0.5.2 | 0.4.1 | 0.5.2 | 0.5.2 | 0.5.2 | 0.5.2 |
| `fla-core` | 0.5.2 | 0.4.1 | 0.5.2 | 0.5.2 | 0.5.2 | 0.5.2 |
| `causal-conv1d` | 1.6.2.post1 | 1.6.2.post1 | 1.6.2.post1 | 1.6.2.post1 | 1.6.2.post1 | 1.6.2.post1 |
| `bitsandbytes` | 0.50.0 | 0.49.1 | — | 0.50.0 | — | — |
| `deepspeed` | 0.19.3 | — | — | — | — | — |
| `transformer-engine` | 2.17.0 | — | — | — | — | — |
| `nvidia-dali-cuda130` | 2.2.0 | — | — | — | — | — |

`deepspeed` / `transformer-engine` / `nvidia-dali` / `kernels` / `liger-kernel` 은 **native lock에만**
있다. 즉 `optim`/`parallel`/`precision`/`dataloader`/`kernel` 축의 라이브러리 구현은 native 프레임워크
전용이고, 다른 다섯 프레임워크 lock에는 그 코드가 아예 없다.

---

## 1. liger-kernel — 철자 확정. 그리고 표가 두 군데 틀렸다

### 1.1 `apply_liger_kernel_to_qwen3_5` — 철자 확정 (OK)

`trainbench/axes.py:113-116`이 "철자 미검증"으로 남겨둔 이름은 **맞다.**

`/private/tmp/.../pins/liger-kernel-0.8.1/liger_kernel/transformers/monkey_patch.py:3057-3064`:

```python
def apply_liger_kernel_to_qwen3_5(
    rope: bool = False,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
```

`liger_kernel/transformers/__init__.py:243` 의 `__all__`에도 있다:

```python
            "apply_liger_kernel_to_qwen3_5",
            "apply_liger_kernel_to_qwen3_5_moe",
```

`monkey_patch.py:3569-3570` 의 model_type 매핑에도 있다:

```python
    "qwen3_5": apply_liger_kernel_to_qwen3_5,
    "qwen3_5_text": apply_liger_kernel_to_qwen3_5,
```

### 1.2 `LIGER_ENTRYPOINTS` 에 qwen3_vl 이 빠져 있다

`trainbench/axes.py:113-116`은 qwen3_5 하나만 기록한다. 그런데 이 스터디의 세 모델 중 하나가
Qwen3-VL-Embedding-2B 이고, liger 0.8.1은 그 엔트리포인트를 갖고 있다.

`monkey_patch.py:2044-2051`:

```python
def apply_liger_kernel_to_qwen3_vl(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = False,
    model: PreTrainedModel = None,
) -> None:
```

`monkey_patch.py:3573-3574`:

```python
    "qwen3_vl": apply_liger_kernel_to_qwen3_vl,
    "qwen3_vl_text": apply_liger_kernel_to_qwen3_vl,
```

현재 상태로는 `kernel=liger` × `model=qwen3_vl_emb_2b` 조합이 `UnappliedAxis`로 거부된다
(`axes.py:318-324`). 그건 "liger가 못 미친다"가 아니라 "표에 안 적혀 있다"이고, 이 두 상태를
구분하겠다는 것이 `axes.py:118-120` 주석의 취지였다.

### 1.3 `LIGER_UNSUPPORTED["gemma4"]` 는 native 핀에서 사실이 아니다

`trainbench/axes.py:121-123`:

```python
LIGER_UNSUPPORTED = {
    "gemma4": "Liger-Kernel#1186 is open (PLAN.md), so gemma-4 has no Liger path",
}
```

liger-kernel **0.8.1** (native 핀)에는 gemma4 경로가 두 개 있다.
`monkey_patch.py:1392-1400`:

```python
def apply_liger_kernel_to_gemma4(
    rope: bool = False,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    layer_norm: bool = False,
    rms_norm: bool = True,
    geglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Gemma4
    multimodal models (`Gemma4ForConditionalGeneration`).

    For text-only Gemma 4 (`Gemma4ForCausalLM`, `Gemma4TextForCausalLM`,
    `Gemma4TextModel`), use :func:`apply_liger_kernel_to_gemma4_text` instead.
```

`monkey_patch.py:3540-3541`:

```python
    "gemma4_text": apply_liger_kernel_to_gemma4_text,
    "gemma4": apply_liger_kernel_to_gemma4,
```

**버전 경계가 정확히 여기 있다.** axolotl 핀인 liger-kernel **0.8.0** 에는 `_gemma4_text` 만 있고
멀티모달 `apply_liger_kernel_to_gemma4` 는 없다:

```
$ grep -n "^def apply_liger_kernel_to_gemma4" liger-kernel-0.8.0/.../monkey_patch.py
1242:def apply_liger_kernel_to_gemma4_text(
$ grep -n '"gemma4' liger-kernel-0.8.0/.../monkey_patch.py
3364:    "gemma4_text": apply_liger_kernel_to_gemma4_text,
```

그래서 "gemma-4에 liger 경로가 없다"는 진술은 **0.8.0에서는 멀티모달에 한해 참, 0.8.1에서는 거짓**이다.
`kernel=liger` × `model=gemma4_e2b` 를 어느 lock에서 도느냐로 답이 갈린다.

### 1.4 실패 메시지가 빈 목록을 출력한다 — `dir()` 이 안 통한다

`trainbench/axes.py:332-339`는 엔트리포인트를 못 찾으면 `dir(module)`로 후보를 뽑아 보여준다:

```python
    apply = getattr(module, entrypoint, None)
    if not callable(apply):
        exported = sorted(n for n in dir(module) if n.startswith("apply_liger_kernel_to_"))
```

`liger_kernel/transformers/__init__.py` 는 **지연 모듈**이다. `apply_liger_kernel_to_*` 는
`if TYPE_CHECKING:` 블록(38행 이하)과 `__getattr__`(103행) 안에만 있고, 모듈 `__dict__` 에는 없다.

`__init__.py:103-113`:

```python
def __getattr__(name: str):
    """
    Handles lazy access to transformer-dependent attributes.
    If 'transformers' is not installed, raises a user-friendly ImportError.
    """
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError(
            f"The attribute '{name}' requires the 'transformers' library, which is not installed.\n"
            f"Please install it with `pip install transformers` to use this functionality."
        )
```

그리고 이 파일에는 `__dir__` 정의가 없다 (`grep -n "__dir__" → NOT DEFINED`). CPython에서
`dir(module)` 은 `module.__dict__` 의 키를 돌려주므로, 아직 접근하지 않은 지연 심볼은 나오지 않는다.
실측:

```
$ python3 -c "
import types
m = types.ModuleType('x')
exec(\"__all__=['a','b']\ndef __getattr__(n): return 1\", m.__dict__)
print('dir(m) =', [n for n in dir(m) if not n.startswith('__')])"
dir(m) = []
```

즉 `axes.py:336`의 메시지는 `it exports []` 를 찍는다. 고치려면 `dir()` 대신
**`getattr(module, "__all__", [])`** 를 읽어야 한다 — `__all__` 은 `_TRANSFORMERS_AVAILABLE` 일 때
실제 이름들로 채워진다 (`__init__.py:206-255`).

### 1.5 인자 없이 부르면 무엇이 켜지는가

`axes.py:340`은 `apply()` 를 인자 없이 부른다. 그러면 qwen3_5 기준 기본값은
`rope=False, cross_entropy=False, fused_linear_cross_entropy=True, rms_norm=True, swiglu=True, model=None`
이고, `model=None` 경로는 모듈 전역을 갈아끼운다 (`monkey_patch.py:3104-3128`):

```python
    if rms_norm:
        modeling_qwen3_5.Qwen3_5RMSNorm = LigerRMSNormForQwen3Next
...
        else:
            modeling_qwen3_5.Qwen3_5ForCausalLM.forward = qwen3_5_lce_forward
            if Qwen3_5ForConditionalGeneration is not None:
                modeling_qwen3_5.Qwen3_5ForConditionalGeneration.forward = qwen3_5_lce_forward_for_multimodal

    if swiglu:
        modeling_qwen3_5.Qwen3_5MLP = LigerQwen3MoeSwiGLUMLP
```

`kernels` 레인이 알아야 할 점: 기본값의 주력은 `fused_linear_cross_entropy=True`, 즉 **LM head의
CE를 융합하는 것**이다. 이 스터디의 손실은 InfoNCE(`loss=mnrl`/`cached_mnrl`)이고 LM head를 통과하지
않는다. 그러면 liger가 실제로 기여하는 것은 RMSNorm과 SwiGLU 뿐이고, FLCE 패치는 한 번도 불리지
않는 forward를 갈아끼운 것이 된다. 축 이름이 약속하는 것과 측정되는 것이 갈라지는 지점이므로,
`_capture_kernel` 이 "무엇이 실제로 갈렸는지"를 되읽어야 한다.

또한 `monkey_patch.py:3085-3087`은 함수 본문에서 `from transformers.models.qwen3_5 import modeling_qwen3_5`
를 한다. transformers가 그 모델을 모르면 `ImportError` 이고, `axes.py:326-331`의 try/except는
`liger_kernel.transformers` import만 감싸므로 여기서 새는 예외는 잡히지 않는다.

---

## 2. flash-linear-attention — 핀 하나로는 `fla.ops` 가 안 들어온다

### 2.1 `flash-linear-attention` wheel 에는 `fla/ops` 도 `fla/modules` 도 없다

이게 이번 리서치에서 가장 놀란 지점이다. sha256이 lock과 일치하는 그 wheel의 전체 매니페스트는
**154개**이고, 내용은 `fla/layers/` 40개 + `fla/models/` 109개 + dist-info 5개가 전부다.

```
$ python3 - <<'EOF'   # flash_linear_attention-0.5.2-py3-none-any.whl
total entries: 154
     40  fla/layers
    109  fla/models
      1  flash_linear_attention-0.5.2.dist-info/METADATA
      1  flash_linear_attention-0.5.2.dist-info/RECORD
      1  flash_linear_attention-0.5.2.dist-info/WHEEL
      1  flash_linear_attention-0.5.2.dist-info/licenses
      1  flash_linear_attention-0.5.2.dist-info/top_level.txt
--- any fla/__init__.py? []
EOF
```

`fla/__init__.py` 조차 없다. RECORD도 154행으로 일치하니 잘린 것이 아니다. axolotl 핀인 0.4.1
wheel도, 0.5.2 sdist도 같은 모양이다 (`direct children of fla/: ['', 'layers', 'models']`).

### 2.2 이유: `fla-core` 로 쪼개져 있고, lock은 그것도 핀한다

0.5.2 sdist의 `pyproject.toml`:

```toml
[project]
name = "flash-linear-attention"
version = "0.5.2"
description = "Fast linear attention models and layers"
readme = "README.md"
license = {file = "LICENSE"}
requires-python = ">=3.10"
dependencies = ["fla-core==0.5.2", "transformers>=4.45.0"]
...
[project.optional-dependencies]
cuda = ["fla-core[cuda]==0.5.2"]
...
conv1d = ["causal-conv1d>=1.4.0"]
...
[tool.setuptools.packages.find]
include = ["fla*"]
namespaces = true
```

`fla-core` 는 여섯 lock 전부에 핀되어 있다 (native `/Users/jwcho/Codes/train-comparison/envs/native/uv.lock:411-412`),
그리고 거기에 나머지가 들어 있다:

```
$ unzip -Z1 fla_core-0.5.2-py3-none-any.whl | cut -d/ -f1-2 | sort -u
fla/__init__.py
fla/modules
fla/ops
fla/utils
```

두 배포판은 `pkgutil` 네임스페이스로 합쳐진다. `fla-core-0.5.2/fla/__init__.py:8-12`:

```python
import importlib
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
__version__ = "0.5.2"
```

transformers가 import하는 심볼은 전부 `fla-core` 쪽에 있다 (확인 완료):
`fla/modules/__init__.py:17,36` 에 `FusedRMSNormGated`, `fla/ops/gated_delta_rule/__init__.py:8-9` 에

```python
from .chunk import chunk_gated_delta_rule, chunk_gdn
from .fused_recurrent import fused_recurrent_gated_delta_rule, fused_recurrent_gdn
```

**결론: 현재 lock들은 정상이다.** 다만 `axes.py:147`의

```python
FLA_DISTRIBUTIONS = ("flash-linear-attention", "fla")
```

는 `fla.ops`/`fla.modules`/`fla.__version__` 을 실제로 공급하는 배포판인 **`fla-core` 를 이름에서
빠뜨리고 있다**. 지금은 두 배포판이 같은 버전으로 핀되어 있어 결과가 갈리지 않지만,
`flash-linear-attention` 만 있고 `fla-core` 가 없는 환경은 버전 검사를 통과하면서 import에서 죽는다.
축이 잡아야 하는 상태가 정확히 그것이다.

### 2.3 `axes.py:143-145` 의 주장은 원문과 다르다

```python
# `fla` publishes no entry in transformers' `PACKAGE_DISTRIBUTION_MAPPING`, so
# that predicate resolves the version by distribution name and falls back to
# importing the package — the same two steps, in the same order, as below.
```

`PACKAGE_DISTRIBUTION_MAPPING` 은 transformers가 관리하는 표가 아니다. import 시점에 설치 환경에서
계산된다. `Kur5R2PrM3RUwEti/transformers/utils/import_utils.py:47`:

```python
PACKAGE_DISTRIBUTION_MAPPING = importlib.metadata.packages_distributions()
```

`fla-core` 와 `flash-linear-attention` 이 둘 다 top-level `fla` 를 선언하므로 (`top_level.txt` = `fla`),
`packages_distributions()["fla"]` 에는 **엔트리가 존재한다**. 따라서 `import_utils.py:59-69` 의
distribution-name 경로가 잡히고, 70-77행의 import 폴백은 돌지 않는다:

```python
            distributions = PACKAGE_DISTRIBUTION_MAPPING[pkg_name]
            # Per PEP 503, underscores and hyphens are equivalent in package names.
            # Prefer the distribution that matches the (normalized) package name.
            normalized_pkg_name = pkg_name.replace("_", "-")
            if normalized_pkg_name in distributions:
                distribution_name = normalized_pkg_name
            elif pkg_name in distributions:
                distribution_name = pkg_name
            else:
                distribution_name = distributions[0]
            package_version = importlib.metadata.version(distribution_name)
```

`"fla"` 는 정규화해도 두 배포판 이름 어느 쪽과도 같지 않으므로 **`distributions[0]`** 로 떨어진다 —
즉 읽히는 버전은 `fla-core` 와 `flash-linear-attention` 중 **리스트 순서가 정하는 쪽**이다.
현재 lock들은 둘을 같은 버전에 묶어 두었으므로 문제가 드러나지 않지만, 이건 우연이지 보장이 아니다.

### 2.4 `FLA_MIN_VERSION = (0, 2, 2)` 는 맞다. 그리고 CUDA 게이트가 있다

`import_utils.py:869-877`:

```python
@lru_cache
def is_flash_linear_attention_available():
    is_available, fla_version = _is_package_available("fla", return_version=True)
    return is_torch_cuda_available() and is_available and version.parse(fla_version) >= version.parse("0.2.2")


@lru_cache
def is_causal_conv1d_available() -> bool:
    return is_torch_cuda_available() and _is_package_available("causal_conv1d")[0]
```

(`axes.py:138` 은 이를 `import_utils.py:869` 로 인용한다. 869행은 `@lru_cache` 데코레이터이고 `def` 는
870행이다. 내용은 인용대로 맞다.)

`is_torch_cuda_available()` 이 앞에 있으므로 **CPU에서는 fla가 절대 바인딩되지 않는다.** 이 축은
이 호스트에서 구현 검증이 불가능하다. `@lru_cache` 이므로 한 프로세스에서 한 번만 계산된다.

### 2.5 causal-conv1d 가 없으면 무엇으로 떨어지는가

`Kur5R2PrM3RUwEti/transformers/models/qwen3_5/modeling_qwen3_5.py:68-78`:

```python
if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

if is_flash_linear_attention_available():
    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
else:
    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule = None, None
    FusedRMSNormGated = None
```

`modeling_qwen3_5.py:219-221` — 네 심볼이 **전부** 있어야 fast path다:

```python
is_fast_path_available = all(
    (causal_conv1d_fn, causal_conv1d_update, chunk_gated_delta_rule, fused_recurrent_gated_delta_rule)
)
```

`modeling_qwen3_5.py:421-431` — 폴백 결선과, 예외 없는 경고 한 줄:

```python
        self.causal_conv1d_fn = causal_conv1d_fn
        self.causal_conv1d_update = causal_conv1d_update or torch_causal_conv1d_update
        self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because one of the required library is not installed. Falling back to "
                "torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )
```

`causal_conv1d_fn` 만 `or` 폴백이 없다. 대신 forward에서 분기한다 (`modeling_qwen3_5.py:492-501`):

```python
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=kwargs.get("seq_idx"),
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])
```

`torch_causal_conv1d_update` 원문은 `modeling_qwen3_5.py:224-239`.

정규화도 갈린다 (`modeling_qwen3_5.py:409-417`):

```python
        self.norm = (
            Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
            if FusedRMSNormGated is None
            else FusedRMSNormGated(
                self.head_v_dim,
                eps=self.layer_norm_epsilon,
                activation=self.activation,
            )
        )
```

**정리 (kernels 레인용):** GDN 경로에는 독립적으로 켜지는 스위치가 4개 있고, 어느 것이 꺼져도
예외가 아니라 로그 한 줄이다. 그리고 `logger.warning_once` 는 프로세스당 한 번이므로 첫 레이어에서만
찍힌다. `kernel=fla` 를 측정하려면 **`self.causal_conv1d_fn`/`self.chunk_gated_delta_rule` 이
어느 함수 객체인지 인스턴스에서 되읽는 것**이 유일하게 믿을 만한 확인이다. 설정값이나 로그로는
안 된다.

liger와의 상호작용도 여기서 보인다. liger는 `Qwen3_5RMSNorm` 을 갈아끼우고 (§1.5), fla는
`Qwen3_5RMSNormGated` 자리에 `FusedRMSNormGated` 를 넣는다. **서로 다른 클래스**이므로 두 축은
충돌하지 않고 겹치지도 않는다.

---

## 3. kernels — native 핀 0.16.0 은 transformers 5.14.1 이 거부한다

### 3.1 버전 창이 상한 배타적이고, 핀이 정확히 상한이다

`Kur5R2PrM3RUwEti/transformers/utils/import_utils.py:144-145`:

```python
KERNELS_MIN_VERSION = "0.15.2"
KERNELS_MAX_VERSION = "0.16.0"
```

`import_utils.py:693-701`:

```python
@lru_cache
def is_kernels_available(MIN_VERSION: str = KERNELS_MIN_VERSION, MAX_VERSION: str = KERNELS_MAX_VERSION) -> bool:
    is_available, kernels_version = _is_package_available("kernels", return_version=True)
    viable_version = False
    if kernels_version != "N/A":
        viable_version = version.parse(kernels_version) >= version.parse(MIN_VERSION) and version.parse(
            kernels_version
        ) < version.parse(MAX_VERSION)
    return is_available and viable_version
```

`envs/native/uv.lock:634-635` 는 `kernels == 0.16.0` 을 핀한다. 상한은 **`<`** 이다:

```
$ python3 -c "from packaging import version; print(version.parse('0.16.0') < version.parse('0.16.0'))"
False
```

**그래서 native 환경에서 `is_kernels_available()` 은 False다.** axolotl 핀인 0.15.2 는 창 안에 있다
(`0.15.2 in range -> True`) — 하지만 axolotl lock에는 `kernel=kernels_hub` 를 적용할 native 경로가 없다.

### 3.2 False가 되면 조용히 no-op이 된다

`Kur5R2PrM3RUwEti/transformers/integrations/hub_kernels.py:61` 이 `if is_kernels_available():` 이고,
`387-409` 가 그 else다:

```python
else:
    _kernels_enabled = False

    # Stub to make decorators int transformers work when `kernels`
    # is not installed.
    def use_kernel_forward_from_hub(*args, **kwargs):
        def decorator(cls):
            return cls

        return decorator
...
    class LayerRepository:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("LayerRepository requires `kernels` to be installed. Run `pip install kernels`.")
```

데코레이터는 **조용히 항등함수**가 되고, `LayerRepository` 를 만들면 *"requires `kernels` to be
installed"* 라고 말한다 — 설치는 되어 있는데. AGENTS.md가 경고하는 실패 모양 그대로다:
증상이 원인을 가리키지 않는다.

이 사실은 `axes.py:522-542` 의 판단(=이 축은 이 호출 지점에서 적용할 수 없다)을 뒤집지 않는다.
오히려 그 위에 조건을 하나 더 얹는다: **native lock에서는 어느 지점으로 옮겨도 적용되지 않는다.**
kernels를 `>=0.15.2,<0.16.0` 으로 내리거나, transformers를 창이 넓어진 버전으로 올려야 한다.

### 3.3 환경변수는 import 시점에 한 번만 읽힌다 (질문 3의 핵심)

`hub_kernels.py:57-58` — **모듈 최상위**:

```python
_TRANSFORMERS_USE_HUB_KERNELS = os.environ.get("USE_HUB_KERNELS", "YES").upper()
_kernels_enabled = _TRANSFORMERS_USE_HUB_KERNELS in ENV_VARS_TRUE_VALUES
```

`kernels-0.16.0/kernels/layer/globals.py:6` — 마찬가지로 모듈 최상위:

```python
_DISABLE_KERNEL_MAPPING: bool = bool(int(os.environ.get("DISABLE_KERNEL_MAPPING", "0")))
```

**따라서 `axes.py:530` 의 서술("read when that module is first imported")은 원문대로 맞다.**
`USE_HUB_KERNELS=NO` 를 프로세스 시작 전에 걸면 런타임 fetch를 끌 수 있다. 단 조건이 있다:
`transformers.integrations.hub_kernels` 와 `kernels.layer.globals` 가 import되기 **전**이어야 한다.
`axes.py` 의 patch 시점은 이미 늦을 수 있으므로, 이건 프로세스 진입점(또는 컨테이너 env)에서
걸어야 하는 값이다.

반면 캐시/오버라이드 계열은 호출 시마다 읽는다 (`kernels/utils.py:171-181`):

```python
def _get_cache_dir() -> str | None:
    """Returns the kernels cache directory."""
    return os.environ.get("KERNELS_CACHE", None)


def _get_local_kernel_overrides() -> dict[str, Path]:
    """Returns list local overrides for kernels."""
    local_kerels = os.environ.get("LOCAL_KERNELS", None)
    if local_kerels is None:
        return dict()
    return _parse_local_kernel_overrides(local_kerels)
```

### 3.4 version / revision / lockfile 시그니처

`kernels-0.16.0/kernels/layer/layer.py:65-77` — `version`과 `revision`은 **배타적이고 둘 중 하나는 필수**:

```python
    def __init__(
        self,
        repo_id: str,
        *,
        layer_name: str,
        revision: str | None = None,
        version: int | None = None,
        trust_remote_code: bool | list[str] = False,
    ):
        if revision is not None and version is not None:
            raise ValueError("Either a revision or a version must be specified, not both.")
        if revision is None and version is None:
            raise ValueError("Either a revision or a version must be specified.")
```

`version` 은 **`int`** 다 (semver 문자열이 아니다). transformers의 기본 매핑도 그렇게 쓴다
(`hub_kernels.py:237-241`):

```python
                    Mode.INFERENCE: LayerRepository(
                        repo_id="kernels-community/mlx_rmsnorm",
                        layer_name="RMSNorm",
                        version=1,
                    )
```

`kernels/utils.py:706-731` — `get_locked_kernel()`:

```python
def get_locked_kernel(repo_id: str, local_files_only: bool = False) -> ModuleType:
    """
    Get a kernel using a lock file.

    Args:
        repo_id (`str`):
            The Hub repository containing the layer.
        local_files_only (`bool`, *optional*, defaults to `False`):
            Whether to only use local files and not download from the Hub.

    Returns:
        `ModuleType`: The imported kernel module.
    """
    locked_sha = _get_caller_locked_kernel(repo_id)

    if locked_sha is None:
        raise ValueError(f"Kernel `{repo_id}` is not locked")

    variant_path = install_kernel(
        repo_id,
        revision=locked_sha,
        local_files_only=local_files_only,
        validate_dependencies=True,
    )

    return _import_from_path(variant_path)
```

**주의: `get_locked_kernel` 은 기본값에서 네트워크를 탄다.** `install_kernel` 이 `snapshot_download`
를 부른다. 런타임 fetch 금지를 강제하려면 `local_files_only=True` 를 넘기거나, 아래
`load_kernel` 을 쓰는 쪽이 맞다 (`utils.py:646-652`) — 이건 **이미 받아둔 것만** 쓴다:

```python
def load_kernel(
    repo_id: str,
    *,
    lockfile: Path | None,
    backend: str | None = None,
    revision: str | None = None,
) -> ModuleType:
```

`utils.py:697-702` 가 미다운로드 시의 거동:

```python
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Locked kernel `{repo_id}` was not downloaded or does not have an "
            "applicable variant. Make sure it's downloaded locally via "
            "`kernels download <project>`."
        ) from e
```

### 3.5 `kernels.lock` 은 파일이 아니라 **설치된 배포판의 메타데이터**에서 읽힌다

`utils.py:734-750`:

```python
def _get_caller_locked_kernel(repo_id: str) -> str | None:
    for dist in _get_caller_distributions():
        lock_json = dist.read_text("kernels.lock")
        if lock_json is None:
            continue
        locked_sha = _get_locked_kernel(repo_id, lock_json)
        if locked_sha is not None:
            return locked_sha
    return None


def _get_locked_kernel(repo_id: str, lock_json: str) -> str | None:
    for kernel_lock_json in json.loads(lock_json):
        kernel_lock = KernelLock.from_json(kernel_lock_json)
        if kernel_lock.repo_id == repo_id:
            return kernel_lock.sha
    return None
```

즉 작업 디렉터리에 `kernels.lock` 을 두는 것으로는 안 되고, **호출자 모듈이 속한 배포판의
dist-info 안에** 들어가 있어야 한다 (`trainbench` 를 설치할 때 패키징되어야 한다는 뜻).
스키마는 `kernels/lockfile.py:15-30`:

```python
class VariantLock:
    hash: str
    hash_type: str = "git_lfs_concat"


@strict
@dataclass
class KernelLock:
    repo_id: str
    sha: str
    variants: dict[str, VariantLock]
```

---

## 4. bitsandbytes — 8bit 클래스명과 4bit kwargs

### 4.1 8bit optimizer 클래스

`bitsandbytes-0.50.0/bitsandbytes/optim/__init__.py` 전문 중 관련부:

```python
from .adagrad import Adagrad, Adagrad8bit, Adagrad32bit
from .adam import Adam, Adam8bit, Adam32bit, PagedAdam, PagedAdam8bit, PagedAdam32bit
from .adamw import (
    AdamW,
    AdamW8bit,
    AdamW32bit,
    PagedAdamW,
    PagedAdamW8bit,
    PagedAdamW32bit,
)
from .ademamix import AdEMAMix, AdEMAMix8bit, AdEMAMix32bit, PagedAdEMAMix, PagedAdEMAMix8bit, PagedAdEMAMix32bit
from .lamb import LAMB, LAMB8bit, LAMB32bit
from .lars import LARS, LARS8bit, LARS32bit, PytorchLARS
from .lion import Lion, Lion8bit, Lion32bit, PagedLion, PagedLion8bit, PagedLion32bit
from .optimizer import GlobalOptimManager
from .rmsprop import RMSprop, RMSprop8bit, RMSprop32bit
from .sgd import SGD, SGD8bit, SGD32bit
```

`configs/optim/adamw_8bit.yaml` 이 겨냥하는 것은 `bitsandbytes.optim.AdamW8bit` 이다.

### 4.2 생성자 시그니처 — `optim_bits` 함정

`bitsandbytes/optim/adamw.py:62-75`:

```python
class AdamW8bit(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        is_paged=False,
    ):
```

`optim_bits` 의 기본값이 **32**인데 이건 8bit를 끄지 않는다. `adamw.py:105-123`:

```python
            raise ValueError("AdamW8bit does not support amsgrad=True")

        if optim_bits != 32:
            # We allow the default value of 32 to maintain compatibility with the function signature,
            # but any other value is invalid since AdamW8bit always uses 8-bit optimization
            raise ValueError("AdamW8bit only supports optim_bits=32 (default value for compatibility)")

        super().__init__(
            "adam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            8,  # Hardcoded to 8 bits
            args,
            min_8bit_size,
            is_paged=is_paged,
        )
```

**`optim_bits=8` 을 "명시적으로 켜려고" 넘기면 `ValueError` 로 죽는다.** 넘기지 않는 것이 정답이다.
그리고 `amsgrad=True` 도 거부된다.

`axes` 레인에 중요한 값: **`min_8bit_size=4096`**. 원소 4096개 미만인 파라미터는 8bit로 안 간다.
0.8B 모델의 norm/bias 계열은 전부 32bit로 남으므로, "optimizer state가 8bit"라는 서술은 파라미터
전체에 대한 참이 아니다. 메모리 축 결과를 해석할 때 이 문턱이 설명 변수다.

### 4.3 4bit 적재 kwargs

`Kur5R2PrM3RUwEti/transformers/utils/quantization_config.py:441-453`:

```python
    def __init__(
        self,
        load_in_8bit=False,
        load_in_4bit=False,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=False,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=None,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_storage=None,
        **kwargs,
    ):
```

**`bnb_4bit_quant_type` 의 기본값은 `"fp4"` 이지 `"nf4"` 가 아니다.** 그리고
`bnb_4bit_compute_dtype=None` 은 `float32` 로 떨어진다 (`quantization_config.py:469-470`).
`trainbench/axes.py:562-567` 의 `QLORA_4BIT` 는 셋 다 명시하고 있어서 이 함정을 이미 피해 간다 —
그 명시성이 필수라는 것이 여기 근거다.

`load_in_4bit` 와 `load_in_8bit` 동시 지정은 거부된다 (`quantization_config.py:457-458`).

---

## 5. deepspeed — 엔진에서 되읽을 수 있는 속성 경로

### 5.1 `deepspeed.initialize` 시그니처

`deepspeed-0.19.3/deepspeed/__init__.py:93-107`:

```python
def initialize(
    args: Any = None,
    model: torch.nn.Module = None,
    optimizer: Optional[Union[Optimizer, DeepSpeedOptimizerCallable]] = None,
    model_parameters: Optional[torch.nn.Module] = None,
    training_data: Optional[torch.utils.data.Dataset] = None,
    lr_scheduler: Optional[Union[_LRScheduler, DeepSpeedSchedulerCallable]] = None,
    distributed_port: int = TORCH_DISTRIBUTED_DEFAULT_PORT,
    mpu: Any = None,
    dist_init_required: Optional[bool] = None,
    collate_fn: Optional[Callable] = None,
    config: Optional[Union[str, Dict[str, Any]]] = None,
    mesh_param: Any = None,
    config_params: Optional[Union[str, Dict[str, Any]]] = None
) -> Tuple[DeepSpeedEngine, Optional[Union[Optimizer, DeepSpeedOptimizer]], Optional[DeepSpeedDataLoader], Any]:
```

`axes.py:27` 이 인용하는 `deepspeed.initialize(model=..., model_parameters=..., training_data=...)`
는 세 인자 모두 실재한다 (95, 97, 98행).

### 5.2 capture 가 읽을 속성 — 전부 `DeepSpeedEngine` 의 메서드다

`deepspeed/runtime/engine.py`:

```python
1110:    def zero_optimization(self):
1111:        return self._config.zero_enabled

1125:    def zero_offload_optimizer(self):
1126:        return self._config.zero_config.offload_optimizer

1128:    def zero_offload_param(self):
1129:        return self._config.zero_config.offload_param

1136:    def zero_cpu_offload(self):
1137:        if self._config.zero_config.offload_optimizer is not None:
1138:            return self._config.zero_config.offload_optimizer.device == OffloadDeviceEnum.cpu
1139:        return False

1153:    def zero_optimization_stage(self):
1154:        return self._config.zero_optimization_stage

1176:    def zero_optimization_partition_gradients(self):
1177:        return self.zero_optimization_stage() >= ZeroStageEnum.gradients

1179:    def zero_optimization_partition_weights(self):
1180:        return self.zero_optimization_stage() >= ZeroStageEnum.weights
```

**capture 권고 (axes 레인):**
- zero stage → `engine.zero_optimization_stage()` (반환은 `ZeroStageEnum`, int 서브클래스)
- offload target → `engine.zero_offload_optimizer()` / `engine.zero_offload_param()`
  (반환은 config 객체 또는 `None`; `.device` 가 `OffloadDeviceEnum`)

값 도메인 (`deepspeed/runtime/zero/config.py:81-87`):

```python
class ZeroStageEnum(int, Enum):
    """ Enum class for possible zero stages """
    disabled = 0
    optimizer_states = 1
    gradients = 2
    weights = 3
    max_stage = 3
```

`deepspeed/runtime/zero/offload_config.py:14-18`:

```python
class OffloadDeviceEnum(str, Enum):
    """ Enum for valid offload devices """
    none = "none"
    cpu = "cpu"
    nvme = "nvme"
```

**단서 하나.** 이 메서드들은 전부 `self._config` 를 읽는다 — 즉 DeepSpeed가 **정규화한 config**이지
우리가 넘긴 dict이 아니다. 그래서 "config가 아니라 엔진에서 읽는다"는 요구는 만족한다(기본값 적용
후의 실제 값이 나온다). 다만 이것도 여전히 *설정*이지 *동작*은 아니다. 옵티마이저가 실제로 그
stage로 돌았는지까지 보려면 `type(engine.optimizer).__name__` 을 함께 기록해야 한다 —
이 호스트에서는 확인 못 했다 (§9).

### 5.3 deepspeed 는 sdist 뿐이다

`envs/native/uv.lock:343-359` 의 블록에 `wheels = [...]` 가 없다. `sdist` 한 줄이 전부다.
의존에 `ninja`, `torch` 가 있으므로 **파드에서 소스 빌드**된다. `causal-conv1d` 도 같다
(`envs/native/uv.lock:149-157`, sdist only, `ninja`+`torch` 의존).
이미지 빌드 시간과 실패 지점이 여기 있다.

---

## 6. transformer-engine — recipe 는 있고, 하드웨어가 A100/H100이면 못 쓴다

### 6.1 `transformer-engine` 은 shim 이고, lock 이 실제 구현을 따로 핀한다

`transformer_engine-2.17.0.dist-info/METADATA:9-18`:

```
Provides-Extra: core
Requires-Dist: transformer_engine_cu12==2.17.0; extra == "core"
Provides-Extra: core-cu12
Requires-Dist: transformer_engine_cu12==2.17.0; extra == "core-cu12"
Provides-Extra: core-cu13
Requires-Dist: transformer_engine_cu13==2.17.0; extra == "core-cu13"
Provides-Extra: pytorch
Requires-Dist: transformer_engine_torch==2.17.0; extra == "pytorch"
Provides-Extra: jax
Requires-Dist: transformer_engine_jax==2.17.0; extra == "jax"
```

native lock은 둘 다 잡아 두었다 — `envs/native/uv.lock:1899` (`transformer-engine-cu13`),
`:1912` (`transformer-engine-torch`), 그리고 `:1890-1894` 의 optional-dependencies 블록:

```
[package.optional-dependencies]
core-cu13 = [
    { name = "transformer-engine-cu13", marker = "sys_platform == 'linux'" },
]
pytorch = [
```

파이썬 소스(`transformer_engine/pytorch/*.py`)는 shim wheel 안에 들어 있어서 읽을 수 있다.
동작하려면 `transformer_engine_torch` 의 C 확장이 필요하므로 이 호스트에서는 import가 불가능하다.

`transformer-engine-torch` 는 **파이썬 API를 하나도 담고 있지 않다.** sdist를 받아(sha256 일치) 열어 보면
`csrc/`, `common_headers/`, `build_tools/`, `setup.py` 가 전부다 — 파이썬 파일은 빌드 스크립트뿐이다.
그리고 lock에 `wheels` 항목이 없어 **소스 빌드**다 (`envs/native/uv.lock:1925`, sdist only).
따라서 TE의 파이썬 표면(recipe, autocast, 지원 검사)은 §6.2-6.4에서 인용한 shim wheel이 전부이며,
이 브리프는 그 표면을 빠짐없이 확인했다. 남는 미지수는 컴파일과 하드웨어뿐이다.

### 6.2 recipe 객체

`transformer-engine-2.17.0/transformer_engine/common/recipe/__init__.py`:

```
336:@dataclass(repr=False)
337:class MXFP8BlockScaling(Recipe):
478:@dataclass(repr=False)
479:class NVFP4BlockScaling(Recipe):
```

`MXFP8BlockScaling` 파라미터 (`__init__.py:354-364`):

```
    Parameters
    ----------
    fp8_format : {Format.E4M3, Format.HYBRID}, default = Format.E4M3
                Controls the FP8 data format used during forward and backward
                pass.
    backward_override : {None, 'high_precision', 'dequantized'}, default = None
            Backward precision mode. None does not modify backward behavior,
            `high_precision` keeps original high-precision operands for backward,
            and `dequantized` dequantizes saved operands to the active high-precision
            compute dtype (e.g. BF16/FP16/FP32) for backward.
```

`NVFP4BlockScaling` (`__init__.py:513-517`):

```
    Parameters
    ----------
    fp4_format : {Format.E2M1}, default = Format.E2M1
             FP4 data type.
    disable_rht : bool, default = False
```

recipe 종류 판별은 클래스메서드다 (`__init__.py:136-144`):

```python
    @classmethod
    def nvfp4(cls):
        """Whether the given recipe is NVFP4 1D block scaling."""
        return issubclass(cls, NVFP4BlockScaling)

    @classmethod
    def mxfp8(cls):
        """Whether the given recipe is MXFP8 block scaling."""
        return issubclass(cls, MXFP8BlockScaling)
```

### 6.3 forward 를 감싸는 컨텍스트 — `fp8_autocast` 는 **deprecated**

`transformer_engine/pytorch/quantization.py:931-959`:

```python
def fp8_autocast(
    enabled: bool = True,
    calibrating: bool = False,
    fp8_recipe: Optional[Recipe] = None,
    fp8_group: Optional[dist_group_type] = None,
    _graph: bool = False,
) -> "autocast":
    """
    .. warning::

       ``fp8_autocast`` is deprecated and will be removed in a future release.
       Use ``autocast(enabled=..., calibrating=..., recipe=..., group=..., _graph=...)`` instead.

    """

    warnings.warn(
        "fp8_autocast is deprecated and will be removed in a future release. "
        "Use autocast(enabled=..., calibrating=..., recipe=..., group=..., _graph=...) instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )

    return autocast(
        enabled=enabled,
        calibrating=calibrating,
        recipe=fp8_recipe,
        amax_reduction_group=fp8_group,
        _graph=_graph,
    )
```

현행 API (`quantization.py:1012-1019`):

```python
    def __init__(
        self,
        enabled: bool = True,
        calibrating: bool = False,
        recipe: Optional["Recipe"] = None,
        amax_reduction_group: Optional["dist_group_type"] = None,
        _graph: bool = False,
    ) -> None:
```

인자 이름이 바뀐다: `fp8_recipe` → **`recipe`**, `fp8_group` → **`amax_reduction_group`**.
(deprecation 경고문 자체는 `group=...` 라고 쓰지만, 실제 파라미터명은 `amax_reduction_group` 이다.)
둘 다 `transformer_engine/pytorch/__init__.py:44,46` 에서 export된다.

같은 인스턴스를 중첩 진입하면 거부된다 (`quantization.py:1027-1032`).

### 6.4 mxfp8 / nvfp4 는 compute capability 10.0 이상을 요구한다

`transformer_engine/pytorch/quantization.py:162-175`:

```python
def _compute_mxfp8_support() -> Tuple[bool, str]:
    """Return if fp8 support is available"""
    if get_device_compute_capability() >= (12, 0):
        return False, "MXFP8 (for all gemm layouts) is not supported on 12.0+ architectures yet."
    if get_device_compute_capability() >= (10, 0):  # blackwell and above
        return True, ""
    return False, "Device compute capability 10.0 or higher required for MXFP8 execution."


def _compute_nvfp4_support() -> Tuple[bool, str]:
    """Return if nvfp4 support is available"""
    if get_device_compute_capability() >= (10, 0):  # blackwell and above
        return True, ""
    return False, "Device compute capability 10.0 or higher required for NVFP4 execution."
```

**A100 = (8, 0), H100 = (9, 0). 둘 다 거부된다.** `precision=mxfp8` / `precision=nvfp4` 축은
**Blackwell (B200/GB200 급, CC 10.0)** 에서만 값이 있다. AGENTS.md가 기록한 1차 캠페인은 A100 18대였다 —
그 하드웨어로는 이 축의 두 값이 원리적으로 측정되지 않는다. 그리고 mxfp8은 CC 12.0 이상(소비자
Blackwell)에서도 거부되므로 **정확히 CC 10.x** 가 필요하다.

**추가 함정:** `check_recipe_support` 는 NVFP4를 검사하지 않는다 (`quantization.py:224-239`):

```python
def check_recipe_support(recipe: Recipe) -> None:
    """Check if the given recipe is supported."""
    if torch.compiler.is_compiling() and isinstance(recipe, DelayedScaling):
        raise RuntimeError(
            "DelayedScaling is not supported under torch.compile. Please use other recipes instead."
        )
    recipe_supported = True
    unsupported_reason = ""
    if isinstance(recipe, (DelayedScaling, Float8CurrentScaling)):
        recipe_supported, unsupported_reason = check_fp8_support()
    elif isinstance(recipe, Float8BlockScaling):
        recipe_supported, unsupported_reason = check_fp8_block_scaling_support()
    elif isinstance(recipe, MXFP8BlockScaling):
        recipe_supported, unsupported_reason = check_mxfp8_support()
    if not recipe_supported:
        raise RuntimeError(unsupported_reason)
```

`elif` 사슬에 `NVFP4BlockScaling` 이 없다. MXFP8은 지원 안 되는 하드웨어에서 `RuntimeError` 로
막히지만, **NVFP4는 이 관문을 그냥 통과한다.** `precision=nvfp4` 축은 반드시
`is_nvfp4_available()` (`quantization.py:360`) 을 직접 불러 게이트해야 한다. 안 그러면
A100에서 "돌긴 돌았는데 nvfp4가 아닌" 숫자가 나온다 — 이 저장소가 가장 두려워하는 실패 모양이다.

---

## 7. nvidia-dali — iterator 생성 API

`nvidia_dali_cuda130-2.2.0` wheel 의 `nvidia/dali/plugin/pytorch/__init__.py` 에 세 개가 있다:

```
43:class DALIGenericIterator(_DaliBaseIterator):
289:class DALIClassificationIterator(DALIGenericIterator):
410:class DALIRaggedIterator(_DaliBaseIterator):
```

`DALIGenericIterator.__init__` (`plugin/pytorch/__init__.py:132-144`):

```python
    def __init__(
        self,
        pipelines: Union[List[Pipeline], Pipeline],
        output_map: List[str],
        size: int = -1,
        reader_name: Optional[str] = None,
        auto_reset: Union[str, bool, None] = False,
        fill_last_batch: Optional[bool] = None,
        dynamic_shape: Optional[bool] = False,
        last_batch_padded: bool = False,
        last_batch_policy: LastBatchPolicy = LastBatchPolicy.FILL,
        prepare_first_batch: bool = True,
    ) -> None:
```

docstring 원문 (`:50-60`):

```
    pipelines : list of nvidia.dali.Pipeline
                List of pipelines to use
    output_map : list of str
                List of strings which maps consecutive outputs
                of DALI pipelines to user specified name.
                Outputs will be returned from iterator as dictionary
                of those names.
                Each name should be distinct
    size : int, default = -1
                Number of samples in the shard for the wrapped pipeline (if there is more than
                one it is a sum)
```

`configs/dataloader/dali.yaml` / `dali_packed.yaml` 이 겨냥할 진입점은
`nvidia.dali.plugin.pytorch.DALIGenericIterator` 이고, `output_map` 이름은 서로 달라야 한다
(`__init__.py:146`: `assert len(set(output_map)) == len(output_map)`).
`prepare_first_batch=True` 가 기본이므로 **생성자에서 첫 배치를 미리 돌린다** — 타이밍 측정 시
iterator 생성 비용에 한 배치가 섞인다. 워밍업 경계를 여기에 맞춰야 한다.

`nvidia/dali/plugin/pytorch/experimental/proxy/` 와 `loader_evaluator/` 도 wheel에 있으나
이번에 열지 않았다.

---

## 8. 축 값 × 이 호스트에서 무엇이 가능한가

"구현 검증 가능"은 **핀된 소스를 열어 시그니처·상수·분기를 확정할 수 있는가**를 뜻한다.
darwin/CPU 호스트에서 import나 실행이 되는지가 아니다 (거의 전부 안 된다).

| 축 | 값 | 이 호스트에서 구현 검증 | 이미지 필요 | GPU 필요 | 비고 |
|---|---|---|---|---|---|
| `kernel` | `none` | 해당 없음 | 아니오 | 아니오 | 패치 부재 |
| `kernel` | `liger` | **가능 — 완료** | 예 (linux wheel) | 예 | 철자 확정. 표 2건 수정 필요 (§1.2, §1.3) |
| `kernel` | `fla` | **가능 — 완료** | 예 (`fla-core` 포함) | **예** — `is_torch_cuda_available()` 게이트 | 폴백 4스위치 확정 (§2.5) |
| `kernel` | `kernels_hub` | **가능 — 완료** | 예 | 예 | native 핀 조합이 이미 죽어 있다 (§3.1) |
| `optim` | `adamw_fused` | 가능 (torch 내장) | 예 | 예 | 이번 범위 밖 |
| `optim` | `adamw_8bit` | **가능 — 완료** | 예 | 예 | `min_8bit_size=4096` 문턱 (§4.2) |
| `optim` | `muon` | 확인 안 함 | 예 | 예 | `pytorch-optimizer` — 이번에 안 열었다 |
| `precision` | `bf16` | 해당 없음 | 아니오 | 예 | |
| `precision` | `mxfp8` | **가능 — 완료** | 예 (TE cu13 + torch ext) | **CC 10.x 전용** | A100/H100 불가 (§6.4) |
| `precision` | `nvfp4` | **가능 — 완료** | 예 | **CC ≥ 10.0 전용** | 지원 검사가 없다 (§6.4) |
| `parallel` | `single`/`ddp`/`fsdp2` | 확인 안 함 | 예 | 예 | torch 내장, 이번 범위 밖 |
| `parallel` | `zero2`/`zero3` | **가능 — 완료** | 예 (**소스 빌드**) | 예 | 엔진 되읽기 경로 확정 (§5.2) |
| `dataloader` | `torch*` | 해당 없음 | 아니오 | 아니오 | |
| `dataloader` | `dali`/`dali_packed` | **가능 — 완료** | 예 (185MB wheel) | 예 | `prepare_first_batch` 주의 (§7) |
| `peft` | `qlora` (4bit 적재) | **가능 — 완료** | 예 | 예 | `QLORA_4BIT` 는 이미 올바르다 (§4.3) |

---

## 9. 이 호스트에서 확정하지 못한 것 — 파드/이미지가 답해야 할 질문

추측은 적지 않는다. 아래는 전부 "확인 안 함"이다.

1. `kernel=liger` × `model=qwen3_vl_emb_2b` 에서 `apply_liger_kernel_to_qwen3_vl()` 을 인자 없이
   불렀을 때, 이 스터디의 임베딩 모델 클래스(`Qwen3VLForConditionalGeneration` 이 아닐 수 있다)에
   대해 `fused_linear_cross_entropy=True` 경로가 예외를 내는가 아니면 조용히 무의미한 패치가 되는가.
2. `kernel=liger` × `model=gemma4_e2b` 에서 native(0.8.1)의 `apply_liger_kernel_to_gemma4` 가
   `Gemma4ForConditionalGeneration` 이 아닌 인스턴스에 대해 `monkey_patch.py:1447-1448` 의
   `TypeError` 를 내는지. 그리고 base 체크포인트 `google/gemma-4-E2B` 가 어느 클래스로 로드되는지.
3. `kernel=fla` 를 켠 실제 파드에서 `Qwen3_5GatedDeltaNet` 인스턴스의
   `self.causal_conv1d_fn` / `self.chunk_gated_delta_rule` / `type(self.norm)` 이
   각각 무엇인가. 네 스위치가 전부 fla/causal-conv1d 쪽인지, 하나라도 torch 폴백인지.
4. `envs/native` 에서 `is_kernels_available()` 이 실제로 False를 돌려주는가 (§3.1은 소스 독해이고,
   실행 확인이 아니다). 그리고 kernels를 0.15.x로 내렸을 때 native lock이 리졸브되는가.
5. `USE_HUB_KERNELS=NO` 를 프로세스 진입 전에 걸었을 때, `bench.py` 의 import 순서에서 실제로
   `transformers.integrations.hub_kernels` 보다 먼저 잡히는가.
6. `AdamW8bit` 에서 `min_8bit_size=4096` 문턱에 걸려 32bit로 남는 파라미터가 세 모델 각각 몇 개이고
   전체 파라미터의 몇 %인가.
7. `deepspeed.initialize` 가 돌려준 엔진에서 `engine.zero_optimization_stage()` 와
   `engine.zero_offload_optimizer()` 가 zero2/zero3 config에 대해 각각 무엇을 반환하는가.
   그리고 `type(engine.optimizer).__name__` 이 stage별로 어떻게 다른가.
8. `deepspeed==0.19.3` 과 `causal-conv1d==1.6.2.post1` 이 파드 이미지에서 소스 빌드에 성공하는가.
   실패한다면 어느 단계(ninja/nvcc/torch 헤더)에서인가. 빌드 시간은 얼마인가.
9. RunPod에서 확보 가능한 GPU 중 compute capability가 **정확히 10.x** 인 것이 있는가.
   없으면 `precision=mxfp8`/`nvfp4` 두 축은 이 스터디에서 측정 불가로 확정해야 한다.
10. `transformer_engine.pytorch.autocast(recipe=NVFP4BlockScaling())` 를 CC < 10.0 에서 진입했을 때
    실제로 무엇이 일어나는가 — 조용히 고정밀로 도는지, 커널 레벨에서 죽는지 (§6.4의 검사 누락).
11. `optim=muon` 의 구현체(`pytorch-optimizer`)와 그 생성자 시그니처. 이번에 열지 않았다.
12. `attn` 축(`fa2`/`fa3`/`fa4`/`flex`/`sdpa`)과 `flash-attn` 의 GitHub 릴리스 wheel
    (`envs/native/uv.lock:423-432`). 이번 범위 밖이다.
