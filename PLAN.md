# 임베딩 모델 학습 속도 최적화 비교 연구

## 목적

Qwen3-VL-Embedding-2B / Qwen3.5-0.8B / gemma-4-E2B 세 모델에 대해 텍스트+이미지
임베딩(contrastive) 학습의 속도 최적화 기법을 실측 비교하고, 모델별 권장 레시피와
throughput x memory x 품질 Pareto frontier를 리포트로 산출한다.

## 핵심 가설

**임베딩 학습 내부에서 축별 효과가 갈리고, 그 갈림이 모델마다 다르다.**

측정 대상은 두 가지다.

1. **축별 효과** — attention backend, 커널, precision, compile, 옵티마이저,
   데이터 파이프라인, GradCache, freeze, PEFT, 병렬화 각각이 임베딩 학습의
   throughput / peak memory에 얼마를 기여하는가
2. **모델별 병목** — 같은 축이 세 아키텍처에서 다른 크기의 효과를 낸다

세 모델의 병목 추정:

- Qwen3-VL-Embedding-2B: attention (28/28 레이어 full attention)
- Qwen3.5-0.8B: linear attention 커널 (18/24가 Gated DeltaNet)
- gemma-4-E2B: 옵티마이저 메모리 (전체 5.104B 중 PLE 2.390B(46.8%),
  주 `embed_tokens`까지 합친 전체 embedding은 2.751B — `docs/model-spec.md` 실측)

따라서 "모델 무관 최적 레시피"는 존재하지 않는다는 것이 예상 결론이다.

### SFT 대비 주장은 철회했다 (2026-08-01)

당초 가설은 "임베딩 학습의 최적화 우선순위가 SFT와 다르다"였다. **철회한다.**
동일 조건 SFT arm이 이 연구에 없고, 공개된 제3자 SFT 수치는 하드웨어·데이터·시퀀스
길이가 달라 대조군이 되지 못한다. 반증 가능한 형태로 검증할 수단이 없는 주장이다.

관련해서 Liger에 대한 서술도 정정한다. 임베딩 학습에는 LM head가 없으므로
**Liger-Kernel의 FLCE(fused linear cross-entropy) 경로는 정의상 비활성**이다.
이것은 "Liger가 무력화된다"와 다르다 — Liger는 RMSNorm / SwiGLU / RoPE 커널도
제공하고 그 경로들은 임베딩 학습에서도 그대로 동작한다. Axolotl의 Cut Cross
Entropy도 같은 이유로 CE 경로만 해당한다. 따라서 `kernel` 축은 "효과 없음"을
전제하지 않고 **FLCE를 뺀 나머지 커널의 기여를 측정하는 축**으로 둔다.

## 대상 모델 (HF config 직접 확인, 2026-07-31)

| | Qwen3-VL-Embedding-2B | Qwen3.5-0.8B | gemma-4-E2B |
|---|---|---|---|
| arch | `qwen3_vl` | `qwen3_5` | `gemma4` |
| 파라미터 | 2,127.5M | 873.4M | 5,123.2M (effective 2.3B) |
| 토큰 믹싱 | full attention 28L | linear(GDN) 18L : full 6L | sliding(512) 28L : full 7L |
| 특이사항 | DeepStack (5/11/17), mRoPE interleaved, ViT 24L | `attn_output_gate`, MTP 1L, `mamba_ssm_dtype: float32` | PLE (256 dim x 262144 vocab), `num_kv_shared_layers: 20`, p-RoPE |
| vocab | 151,936 | 248,320 | 262,144 |
| config상 transformers | 4.57.1 | 4.57.0.dev0 | 5.5.0.dev0 |
| 출발점 | 이미 임베딩 모델 (sentence-transformers) | 생성형 VLM | 생성형 any-to-any |

현재 transformers 안정판은 5.14.1.

## 실행 환경

- 데이터: MMEB-train. **실제로 읽는 저장소는 `TIGER-Lab/MMEB-train`이 아니다** —
  그 저장소는 이미지 *경로*만 담고 이미지 자체가 없다. 이미지를 포함한 커뮤니티
  미러 `MrZilinXiao/MMEB_train_with_image`(20 config / 1,068,472행)를 upstream으로
  쓰고, 커뮤니티 유지 저장소이므로 브랜치가 아니라 커밋을 고정한다.
  런은 upstream을 직접 읽지 않고 여기서 만든 고정 서브셋
  `jinwoo-cho/mmeb-subset`(private)을 revision으로 고정해 읽는다
  (`configs/data/speed.yaml`). eval은 `TIGER-Lab/MMEB-eval` 서브셋 예정 —
  `configs/data/quality.yaml`은 아직 revision 미고정이다
- 모달리티: 텍스트 + 이미지 (오디오 제외)
- 산출물: 벤치마크 리포트 중심 + 측정 하네스

### GPU 3단 전략

개발 단계를 저가 GPU로 내려 비용과 재고 리스크를 동시에 줄인다.

| | A100 SXM 80GB | H200 SXM 141GB | B200 180GB |
|---|---|---|---|
| $/GPU-hr (secure) | 1.49 | 4.39 | 5.89 |
| 가용성 (2026-07-31) | HIGH | MEDIUM | LOW |
| FA2 / FA3 / FA4 | O / X / X | O / O / X | O / O / O |
| fp8 (E4M3/E5M2) | X | O | O |
| MXFP8 / NVFP4 | X | X | O |
| Liger, fla, torch.compile, GradCache, Muon, packing, PEFT | O | O | O |

- **Phase 0~1 (환경 검증 + 하네스 개발) -> A100.** "로드되는가 / Unsloth 패칭이
  깨지는가 / Axolotl이 Qwen3-VL을 받는가"는 GPU 아키텍처와 무관하다. 재고 LOW인
  B200을 개발 중에 점유하는 것이 더 큰 손해다.
  단 **A100 80GB로는 gemma-4-E2B full FT가 안 들어간다** (5.1B x 16 bytes 약 82GB,
  활성값 제외). full FT 적재 확인만 B200에서 별도 수행.
- **Phase 2~3 (본 측정) -> B200.** A100에서 돌던 것이 Blackwell 빌드에서 그대로
  돈다는 보장이 없으므로 스모크 테스트 1회 필수.
- **B200 확보 실패 시 -> H200 전면 전환.** FA4/MXFP8/NVFP4를 리포트에 "미측정"으로
  명시. **혼용 금지** — 비교가 무효화된다.

FA4는 Blackwell 전용이다. 원문이 TMEM(SM당 256KB), 5세대 비동기 텐서코어, 2-CTA MMA
등 Blackwell 전용 기능 의존을 명시한다. FA3는 Hopper 전용으로 Ampere에서 동작하지
않는다. ("FlexAttention이 Hopper에서 FA4 백엔드 지원"이라는 2차 요약이 있으나
1차 출처와 배치되어 채택하지 않음.)

### 데이터센터 제약 (2026-07-31 스냅샷)

재고가 있는 DC가 GPU 타입별로 **완전히 분리**되어 있다. 교집합 0.

| GPU | 재고 보유 DC (전부 LOW) |
|---|---|
| B200 | EU-RO-1, US-NC-2, US-NE-1 (31개 중 3개) |
| H200 | AP-JP-1, EU-FR-1, EUR-IS-4, US-CA-2, US-GA-2, US-NC-1 |
| A100 SXM 80GB | EUR-IS-1, US-KS-2, US-MD-1, US-MO-1, US-WA-1 |

**DC 종속은 해소되었다.** network volume이 pod을 특정 DC에 고정시키는 유일한
요인이었는데, 아래 "스토리지" 절의 결정으로 볼륨을 쓰지 않게 되었다. 따라서:

- B200 재고가 있는 3개 DC를 **모두** 후보로 쓸 수 있다. 어느 한 곳이 마르면 다른
  곳으로 그냥 옮기면 된다
- A100(Phase 0~1) -> B200(Phase 2~3) 전환에 사전 준비가 필요 없다
- H200 fallback이 실제로 즉시 전환 가능하다

남는 주의점: A100의 집계 가용성은 HIGH로 표기되나 개별 DC는 전부 LOW다.
**표기 불일치이므로 실행 시점에 재확인**한다. H200도 스냅샷 간 MEDIUM -> LOW로
변동했다. 위 표는 스냅샷이지 보장이 아니다.

### 다중 pod 병렬 실행

여러 pod을 동시에 띄워 wall-clock을 단축한다. GPU-hour 총합은 변하지 않으므로
비용 증가는 pod별 셋업 오버헤드뿐이다.

**분할 규칙 — 이걸 어기면 측정이 오염된다**

pod이 다르면 물리 호스트가 다르다. CPU 모델, 메모리 대역폭, 스토리지 I/O, 이웃
워크로드 노이즈가 전부 달라진다. 특히 **데이터로딩이 CPU 바운드일 때 호스트 vCPU
수 차이가 그대로 throughput 차이로 잡힌다.**

1. **같은 축 내 설정은 절대 pod을 가르지 않는다.** attention backend 4종 비교는
   반드시 한 pod에서 연속 실행한다.
   - 허용: pod A = Qwen3-VL 전 축 / pod B = Qwen3.5 전 축 / pod C = gemma-4 전 축
   - 허용: pod A = attention 축(3모델 전부) / pod B = precision 축(3모델 전부)
   - **금지**: pod A = FA2 / pod B = FA3
2. **분할 단위는 모델 또는 축 블록 전체**로 한다.
3. **canonical baseline run.** 모든 pod에서 동일한 baseline 설정 1개를 반드시
   실행한다. pod 간 편차를 측정해 임계값(3%) 초과 시 해당 pod 결과를 폐기하거나
   재실행한다. 이것이 없으면 병렬 실행 결과를 한 표에 넣을 수 없다.
4. **pod spec 기록 의무화.** GPU UUID, nvidia-smi 출력, vCPU 수/모델, RAM, 디스크
   타입, DC id, 드라이버/CUDA 버전. 리포트 부록에 첨부한다.

**스토리지 — network volume 미사용**

RunPod network volume은 통상 200~400 MB/s의 네트워크 연결형 스토리지다. 여기에
학습 데이터를 두면 **Phase 1의 데이터로딩 병목 판정이 파이프라인이 아니라 볼륨을
측정하게 되어 실험 축 하나가 무효화된다.** 편의 문제가 아니라 측정 유효성 문제이므로
쓰지 않는다.

원칙: 소스가 무엇이든 **측정 중에는 모든 것이 pod-local NVMe에 있어야 한다.**

- **의존성 = Docker 이미지.** 공통 베이스(CUDA + torch + transformers) 위에
  프레임워크별 얇은 레이어. 레이어 공유로 레지스트리 저장량과 호스트 캐시가
  재사용된다. 셋업 편차가 사라지는 것이 이미지를 쓰는 진짜 이유다.
  빌드는 amd64 네이티브(RunPod CPU pod)에서 한다
- **모델·데이터 = HF Hub -> pod-local NVMe.** 모델 3종은 공식 repo에서, 전처리한
  MMEB 고정 서브셋은 private dataset repo에서 받는다. repo revision이 곧 데이터
  버전이 되어 컨벤션 07의 데이터 버전 기록 요건을 자연히 충족한다
- `HF_XET_HIGH_PERFORMANCE=1`을 쓴다. `HF_HUB_ENABLE_HF_TRANSFER`는 현재
  huggingface_hub에서 무시되므로 쓰지 않는다
- 결과·프로파일 산출물은 같은 private repo의 pod별 경로로 push

**분할 입도 — 무한정 쪼개는 것이 최선은 아니다**

pod 개수 제한은 없으나, pod을 잘게 쪼갤수록 (a) pod마다 canonical baseline을 돌리는
오버헤드가 커지고 (b) pod 간 하드웨어 편차에 노출되는 표면적이 넓어진다.

| 입도 | pod 수 | baseline 오버헤드 | pod 간 편차 리스크 |
|---|---|---|---|
| 모델당 1개 | 3 | 최소 | 최소 |
| 축 그룹당 1개 (권장) | 12~18 | 중간 | 중간 |
| 축당 1개 | 36 | 최대 (~6 GPU-h 추가) | 최대 |

**축 그룹 단위 12~18개**를 기본으로 한다. wall-clock 이득의 대부분을 가져가면서
baseline 오버헤드가 감당 가능한 지점이다.

**예상 wall-clock**

| 구간 | 순차 | 병렬 | 분할 |
|---|---|---|---|
| Phase 0 환경 검증 | 8h | ~0.7h | 프레임워크 x 모델 = 18 pod |
| Phase 1 하네스 | 10h | 10h | 사람 작업이라 병렬 이득 없음 |
| Phase 2 ablation | 35h | ~3h | 모델 x 축그룹 = 12~18 pod |
| Phase 3 프레임워크 | 15h | ~1h | 프레임워크 x 모델 = 18 pod |
| 품질 검증 런 | 30h | ~3h | Pareto 후보별 |
| **합계** | ~98h | **~18h** | |

B200 재고 LOW 상황에서 12~18개 pod 동시 확보가 되는지는 실행 시점에 확인한다.
확보 실패 시 **확보된 수만큼 실행하고 나머지는 큐잉**하도록 오케스트레이션을 설계한다
(전량 순차 폴백이 아니라 부분 병렬).

---

## Phase 0 — 환경 정합성 검증

여기서 막히면 이후가 전부 무너지므로 최우선. 모든 항목은 가정이 아니라 실측 대상.

- [ ] 세 모델이 **동일 단일 transformers 환경**(5.14.x)에서 로드/학습되는가.
      실패 시 모델별 환경 분리 -> 비교 공정성 훼손 -> 설계 변경 필요
- [ ] `sentence-transformers` v5.5 x transformers v5 호환
- [ ] Qwen3.5 GDN 레이어가 `fla` 없이 학습되는가 / `fla`·FlashQLA 설치 시 커널 경로
- [ ] gemma-4-E2B PLE 파라미터의 freeze 가능 여부 및 LoRA target module 인식
- [ ] Unsloth 일반 VLM 경로 + 커스텀 InfoNCE loss에서 패칭이 깨지지 않는가 (모델 3종)
- [ ] Unsloth `FastSentenceTransformer`가 VLM 체크포인트를 거부하는가 (문서에 언급
      없음 != 동작 안 함)
- [ ] Axolotl의 Qwen3-VL 지원 여부 (Qwen2-VL까지만 문서 확인됨)
- [ ] Tevatron 2.0의 세 모델 지원 여부
- [ ] 모델별 동일 이미지의 실제 visual token 수 (patch 16 공통이나 merge/pooling 상이)

**산출물**: `docs/support-matrix.md` — 최적화 축 x 모델 3종, 셀마다 근거 URL +
검증 버전 + 실측 로그. 미확인 셀은 "미확인"으로 명시하고 추측으로 채우지 않는다.

## Phase 1 — 측정 하네스

### 측정 규율

- **고정 step 기준 비교 + 실제 토큰 수 기록.** vocab/토크나이저가 모델마다 다르므로
  같은 step 수가 같은 토큰 수를 뜻하지 않는다. 그렇다고 토큰 예산을 단위로 삼으면
  모델마다 step 수가 달라져 step 단위 속성(peak VRAM, step 시간 분포)을 비교할 수
  없다. 그래서 `train.steps`를 단위로 두고 **소비된 토큰 수를 측정해 결과에 남긴다** —
  정규화는 리포트 단계에서 하고 측정 단계에서 하지 않는다 (2026-08-01 확정)
- **이미지 토큰 예산 고정.** 미고정 시 나머지 측정이 전부 오염됨
- **타이밍 런과 프로파일링 런 분리.** 숫자는 프로파일러 off 상태에서만 측정하고,
  프로파일은 원인 분석 전용이다. 부풀림 폭은 **미측정**이며 출처도 확보하지
  못했다 — `docs/methodology.md` 참조. 규율 자체는 폭과 무관하게 유지한다
- warmup step 폐기, 명시적 CUDA sync, 동일 seed 및 동일 데이터 순서

### 소수 샘플 정책 (속도 런과 품질 런의 분리)

throughput은 steady-state 속성이고 peak VRAM은 step 단위 속성이다. 따라서
**속도·메모리 측정은 소수 샘플로 충분하며 그것이 정석**이다. 전체 데이터셋 불필요.

단 다음 조건을 지킨다.

1. **분포는 실제 MMEB를 따른다.** 시퀀스 길이·이미지 해상도 분포가 달라지면 속도
   수치가 왜곡된다. 무작위 서브샘플은 OK, "짧은 것만 골라 빠르게"는 금지
2. **품질 가드레일은 소수 샘플로 불가.** loss curve와 Recall@k는 학습이 어느 정도
   진행돼야 의미가 생긴다. 속도 런(소수 샘플)과 품질 런(장기)을 완전히 분리하고,
   품질 런은 Pareto 후보로 좁혀진 소수 설정에만 돌린다
3. **GradCache / 배치 스케일링 축은 예외.** effective batch 16k를 재려면 샘플 수가
   배치보다 훨씬 커야 한다
4. **torch.compile max-autotune은 warmup이 길다.** 소수 샘플이면 warmup이 전체를
   차지해 측정이 무의미해진다. 이 축만 step 수를 늘린다
5. **워밍업 폐기 구간을 축마다 다르게** 잡는다 (컴파일 축은 길게, 나머지는 짧게)

### 지표

samples/s, tokens/s, MFU(Megatron-LM/NeMo 구현 참고), peak VRAM,
step time p50/p95, torch.profiler 커널 breakdown(별도 런), memory snapshot,
HTA 트레이스 분석

### 품질 가드레일

loss curve + MMEB-eval 서브셋 Recall@k. 속도만 재면 "빠른데 망가진" 설정을 못 거른다.

**주의**: Unsloth 문서상 멀티모달 gemma-4 E2B/E4B는 학습 loss가 13~15로 나오는 것이
정상이다(텍스트 전용은 1~3). loss 절대값으로 품질을 판정하면 안 된다.

### 데이터로딩 병목 판정 (선행 필수)

MMEB는 이미지 헤비다. CPU 디코딩이 GPU를 굶기면 커널/attention 최적화가 측정에
잡히지 않는다. Phase 2 시작 전 dataloader가 병목인지 판정하고, 병목이면
DALI(GPU 디코딩) 또는 사전 디코딩/캐싱으로 해소한 뒤 진행한다.

## Phase 2 — 축별 ablation

1xB200 단일 GPU. 베이스라인에서 one-factor-at-a-time 스크리닝 후 상위 인자만
상호작용 검증. 전수 조합은 비용상 불가.

| 축 | 모델별 예상 차별점 |
|---|---|
| attention backend (sdpa/FA2/FA3/FA4/Flex) | Qwen3-VL 28/28, Qwen3.5 6/24, gemma-4 sliding 512 |
| 커널 (Liger / kernels hub / fla·FlashQLA) | Qwen3.5 Liger 지원, gemma-4 미지원(#1186 open) |
| precision (bf16 / MXFP8, 여유 시 NVFP4) | Blackwell 전용 이득 |
| torch.compile (off/default/max-autotune/regional) | GDN·PLE 컴파일 호환성 |
| gradient checkpointing (full/selective) + offload | |
| 옵티마이저 (AdamW fused / 8-bit / Muon) | gemma-4-E2B 가설 검증에 한정 (아래) |
| **데이터 파이프라인** (packing, 사전 토크나이즈, DALI) | ms-swift "100%+" 주장 검증 |
| **GradCache on/off + batch size 스케일링** | 오버헤드 주장이 20% vs 2~2.4배로 상충 |
| **vision tower freeze / unfreeze** | 속도-품질 trade-off 곡선 |
| **gemma-4-E2B PLE freeze / train** | 이 모델 결과를 지배하는 축 |
| PEFT (full / LoRA rank sweep / QLoRA) | |
| 병렬화 (DDP / FSDP2 / ZeRO / cross-device negative) | 별도 멀티 GPU 섹션 |

### Muon 가설 (좁게 검증)

NVIDIA GB300 실측상 Muon의 throughput은 AdamW와 거의 동등하고(Kimi K2: 1,080 vs
1,051 TFLOPs/s/GPU) 이득은 수렴 속도 + optimizer state 메모리(2D weight ~45% 절감)다.

Muon은 통상 embedding을 제외하고 AdamW에 맡긴다. gemma-4-E2B는 파라미터의 절반
이상이 PLE embedding 테이블이므로 **Muon 이득이 이 모델에서만 유독 작을 가능성**이
있다. 이는 검증되지 않은 추론이며, 전 모델 전 설정 조합이 아니라 이 가설 검증
목적으로만 좁게 측정한다.

## Phase 3 — 프레임워크 베이스라인

자체 하네스가 축을 분리하고, 프레임워크는 "실무에서 실제로 나오는 수치"를 제시하는
이원 구조. 프레임워크는 각자 오버헤드가 섞여 기여도 분리가 안 되므로 역할을 나눈다.

| 프레임워크 | 포함 근거 | 리스크 |
|---|---|---|
| 자체 얇은 하네스 | 축 분리 기준선 | - |
| Unsloth | Gemma4 E2B 멀티모달 FFT/LoRA, B200 지원, "FA2 대비 1.5x / VRAM 60% 절감" 주장 | 임베딩 경로는 encoder-only. VLM 임베딩 미지원 가능성 |
| ms-swift v4.0 | infonce 임베딩 학습, Qwen3-VL/Gemma4 지원, packing | - |
| sentence-transformers v5.5 | VLM 임베딩 학습 정식 지원, GradCache 내장 | transformers v5 호환 미확인 |
| Tevatron 2.0 | 검색/임베딩 전용, 멀티모달, LoRA + DeepSpeed + FlashAttention | 세 모델 지원 미확인 |
| Axolotl v0.8+ | 멀티모달 first-class, Cut Cross Entropy, Sequence Parallelism, FSDP2, Gemma4 | Qwen3-VL 지원 미확인 |

Phase 0에서 동작하지 않는 조합은 "미지원"으로 기록한다. 그 자체가 리포트의 결과다.

**제외**: Megatron-LM(VLM은 torch checkpoint 포맷만 지원, 변환 비용 과다),
TorchTune(멀티모달 임베딩 불확실 + 자체 하네스와 중복), NeMo(과함),
LlamaFactory(SFT 중심), 커스텀 커널 DSL(Helion/ThunderKittens/CuTe, 스코프 밖)

## Phase 4 — 리포트

- throughput x peak memory x 품질 3축 Pareto frontier
- 모델별 권장 레시피
- full FT vs LoRA 손익분기점
- 공개 주장 수치(Unsloth 1.5~3.3x, ms-swift packing 100%+, FlashQLA 2~3x,
  NVFP4 1.73x)는 **참고 수치로만 병기**한다. 하드웨어·데이터·시퀀스 길이가 달라
  대조군이 아니므로 "얼마나 남는가"라는 비교로 쓰지 않는다

---

## 비용 개략

| 구간 | GPU | 시간 | 비용 |
|---|---|---|---|
| Phase 0 환경 검증 (6 프레임워크 x 3 모델) | A100 $1.49 | 8h | ~$12 |
| Phase 1 하네스 + 데이터로딩 판정 | A100 | 10h | ~$15 |
| gemma-4-E2B full FT 적재 확인 | B200 | 1h | ~$6 |
| Phase 2 속도·메모리 ablation (소수 샘플) | B200 | ~35h | ~$206 |
| Phase 3 프레임워크 6종 (소수 샘플) | B200 | ~15h | ~$88 |
| 품질 검증 런 (Pareto 후보 한정) | B200 | ~30h | ~$177 |
| 멀티 GPU 섹션 | 8xB200 | 8 GPU-h | ~$47 |
| 다중 pod 셋업 오버헤드 (~10%) | | | ~$55 |
| 이미지 빌드용 CPU pod | | | 소액 |
| 버퍼 15% | | | ~$91 |
| **합계** | | | **~$700** |

다중 pod 병렬화는 GPU-hour 총합을 바꾸지 않는다. wall-clock만 약 98h -> 33h로
줄고, 비용 증가분은 pod별 셋업 오버헤드뿐이다.

품질 런은 Pareto 후보로 좁힌 소수 설정에만 돌리는 전제. 전 설정을 수렴까지 학습하면
수 배가 된다. H200 전면 전환 시 GPU 단가는 25% 낮으나 FA4/MXFP8/NVFP4 축을 잃는다.

## 리스크

1. **B200 재고 LOW + DC 3곳 한정.** 캠페인 중간 pod 손실 시 비교가 깨진다. 완화책:
   (a) Phase 0~1을 A100으로 내려 B200 점유 시간 최소화,
   (b) 소수 샘플 정책으로 축별 런을 짧게 쪼개 재개 가능하게 설계,
   (c) 다중 pod 확보 실패 시 순차 실행으로 자동 폴백,
   (d) 최후에는 H200 전면 전환 + 해당 축 "미측정" 명시.
   GPU 혼용은 어떤 경우에도 금지
2. **다중 pod 간 하드웨어 편차.** pod마다 호스트 CPU/메모리대역폭/스토리지가 달라
   throughput이 흔들린다. 완화: 같은 축은 한 pod에 묶고, 전 pod에서 canonical
   baseline을 돌려 편차 3% 초과 시 폐기·재실행. pod spec 전량 기록
3. **transformers 버전 단일화 실패 가능성** (Phase 0에서 판명). 모델별 환경 분리가
   필요하면 "동일 조건 3자 비교" 전제 자체를 조정해야 한다
4. **데이터로딩이 병목이면** Phase 2 전체가 무의미해진다. Phase 1에서 반드시 선판정

---

# 실행 계획

## 개발 컨벤션 적용 범위

`~/Codes/develop-convention` 기준. 이 프로젝트는 **본질이 ablation study이고 산출물이
리포트**이므로, 그 성격에 값어치가 있는 것만 적용하고 나머지는 명시적으로 뺀다.

| 컨벤션 | 적용 | 근거 |
|---|---|---|
| 02 Config | **전면** | Hydra config group + Pydantic 검증, **코드 변경 없이 config 조합만으로 ablation**, run별 resolved config + git hash 스냅샷. 이 프로젝트의 뼈대 그 자체 |
| 03 Environment | **전면** | uv + uv.lock + ruff. torch platform marker로 macOS(CPU/MPS) <-> RunPod(CUDA) 무수정 이식. device는 `torch.accelerator` 기반 단일 헬퍼로만 선택, 인라인 `.cuda()` 금지 |
| 04 Pipeline | **전면** | 모든 stage `--limit N` 소수 샘플 지원, atomic save(temp -> `os.replace`), resume. 소수 샘플 정책 및 pod 손실 대비와 정확히 일치 |
| 05 Performance | **전면** | GPU util/VRAM/RAM/CPU + throughput 구조화(JSON) 로그. 이 프로젝트의 산출물 자체 |
| 07 ML | **전면** | seed 단일 헬퍼, run별 config+commit 기록, **중단을 가정한 설계**(재개 가능). B200 재고 LOW에서 필수 |
| 08 LLM | **적용** | FSDP2 + bf16 기본, 평가 harness/task 버전 기록. 컨벤션이 **torchtune을 deprecated로 금지** — 앞서의 제외 판단과 독립적으로 일치 |
| 16 Research | **이미 적용 중** | 근거 없는 사실 금지. support-matrix의 "미확인" 규칙이 이것 |
| 17 Commit | **전면** | Conventional Commits 영문 헤더 + 한국어 Why/What/How/Result 본문. 측정 안 한 수치는 "측정 안 함" |
| 01 Structure | **적용** | flat layout(src/ 아님), 시맨틱 네이밍, `_v2` 금지, 주석 최소, 코드 주석 이모지 금지 |
| 06 Testing | **경량** | config 스키마 + MFU 계산 유닛 테스트 + **CPU 소수 샘플 E2E 스모크 1개**. 그 이상은 안 만든다 |
| 13 Secret | **경량** | RunPod API key / HF token만 대상. `.env.example` + gitignore + 런타임 env 주입. Infisical은 이미 쓰고 있을 때만 |
| 09 Agentic | **경량** | 하네스 코드에 작성자와 분리된 리뷰 1레인 |
| 15 Doc tracking | **미적용** | 4계층 + docsync 마커 + ADR은 단기 리포트 프로젝트에 과함. `PLAN.md`(설계) + `docs/`(계약·규격·방법론·결과) 평면 구성으로 간다 |
| 10 / 11 LLM API | **해당 없음** | 학습 프로젝트. API 추론 없음 |

## 저장소 구조

flat layout. config group이 곧 실험 축이다.

**아래 블록은 계획이 아니라 현재 존재하는 것만 적는다.** `scripts/audit_plan.py`의
`plan-files` 체크가 이 블록을 트리로 파싱해 실제 파일과 대조하며, 없는 파일이 적혀
있으면 게이트가 막는다. 계획 중인 파일은 블록 아래 "미작성" 목록에 둔다 — 아직 없는
것을 구조도에 그려두면 어느 것이 이미 있는지 읽는 사람이 구분할 수 없다.

```
train-comparison/
├── AGENTS.md                  # 공유 지침 (컨벤션 경로 참조)
├── CLAUDE.md                  # @AGENTS.md
├── PLAN.md                    # 이 문서
├── README.md
├── pyproject.toml             # uv + ruff + torch platform marker + extras
├── uv.lock
├── .env.example               # RUNPOD_API_KEY, HF_TOKEN (키 이름만)
├── configs/                   # Hydra config groups = 실험 축
│   ├── config.yaml
│   ├── model/                 # qwen3_vl_emb_2b, qwen3_5_0_8b, gemma4_e2b
│   ├── data/                  # speed(소수 샘플), quality(장기)
│   ├── run/                   # probe, timing, profile, quality
│   ├── train/                 # 단일 플래그 knob (batch, seed, checkpointing 등)
│   ├── experiment/            # orchestrate.py가 읽는 매니페스트. defaults에 없어 합성되지 않음
│   ├── attn/                  # sdpa, fa2, fa3, fa4, flex
│   ├── kernel/                # none, liger, fla
│   ├── precision/             # bf16 (mxfp8/nvfp4 제거— A100은 CC 8.0, 둘 다 CC 10.x 전용)
│   ├── compile/               # none, default, max_autotune, regional
│   ├── optim/                 # adamw_fused, adamw_8bit, muon
│   ├── loss/                  # mnrl, cached_mnrl
│   ├── peft/                  # full, lora, qlora
│   ├── freeze/                # none, vision_tower, ple, vision_and_ple
│   ├── dataloader/            # torch, torch_packed, dali, dali_packed
│   ├── parallel/              # single, ddp, fsdp2, zero2, zero3
│   └── framework/             # native, unsloth, ms_swift, st, tevatron, axolotl
├── trainbench/
│   ├── config_schema.py       # Pydantic 검증 (fail-fast) + 축 마커
│   ├── config.py              # 해석된 config JSON 입출력
│   ├── compose.py             # Hydra 조합 -> 검증된 BenchConfig
│   ├── device.py              # torch.accelerator 단일 헬퍼
│   ├── seed.py                # 시드 단일 헬퍼
│   ├── record.py              # run 기록 + 원자적 쓰기
│   ├── collate.py             # 행 -> MicroBatch. collate-metrics 경계가 이 이름을 찾는다
│   ├── embedding.py           # 풀링 + InfoNCE
│   ├── prompt.py              # 모델의 prompt_format을 읽는 유일한 지점
│   ├── axes.py                # 축을 켜는 유일한 지점
│   ├── applied.py             # 켜졌는지 읽는 유일한 지점
│   ├── kernels.py             # 바인딩된 어텐션 커널의 신원 + 런타임 fetch 차단
│   ├── loader.py              # 프레임워크 어댑터 레지스트리 + 빌드 지문
│   ├── pods.py                # RunPod pod 수명주기
│   ├── metrics/               # 타이밍 런이 보고하는 지표와 그 측정 방법
│   └── probe/                 # 프레임워크별 적재·1step 검증 어댑터
├── scripts/
│   ├── bench.py               # 단일 런 측정 진입점. assert_matches를 호출하는 유일한 하네스
│   ├── verify_env.py          # Phase 0 프레임워크 x 모델 probe
│   ├── env_report.py          # 하네스 관통 경로 점검 (모델 미적재)
│   ├── compose_config.py      # 로컬에서 config JSON 해석 -> pod 전달
│   ├── prepare_data.py        # MMEB 고정 서브셋 생성 + push
│   ├── orchestrate.py         # RunPod 다중 pod 기동/큐잉/수집
│   ├── publish_result.py      # 결과를 HF repo로 push
│   ├── report.py              # pod별 결과 병합
│   └── audit_plan.py          # 계획-문서-코드 정합 회귀 추적기
├── tests/
│   ├── contract/              # 레인 경계 계약 — 어느 레인도 소유하지 않고 팬아웃 전에 쓰인다
│   ├── fixtures/              # 경계별 대표 페이로드 — 계약의 진실은 산문이 아니라 이 파일이다
│   ├── conftest.py
│   ├── test_collate.py
│   ├── test_config.py
│   ├── test_applied.py
│   ├── test_audit.py
│   ├── test_axes.py
│   ├── test_data.py
│   ├── test_device_seed.py
│   ├── test_embedding.py
│   ├── test_kernels.py
│   ├── test_loader.py
│   ├── test_metrics.py
│   ├── test_pods.py
│   ├── test_probe.py
│   ├── test_prompt.py
│   ├── test_report.py
│   └── test_smoke_cpu.py
├── docker/                    # Dockerfile.base + Dockerfile.framework + entrypoint
├── envs/                      # 프레임워크별 독립 프로젝트 + 독립 lock
└── docs/
    ├── CONTRACTS.md           # 레인 간 공유 계약 (Wave 0 확정)
    ├── audit-baseline.json    # audit_plan.py가 KNOWN으로 통과시키는 실패 + 그 결과
    ├── evidence/              # 커밋된 런 기록 (evidence-committed가 대조)
    ├── methodology.md         # 측정 규율과 그 근거
    ├── model-spec.md          # 모델별 공식 규격 검증 (산문)
    ├── model-spec.yaml        # 같은 내용의 기계 판독본 (audit이 대조)
    ├── open-verdicts.json     # 재검증 판정서의 미착지 항목 (verdicts-closed가 대조)
    ├── prebuilt-wheels.yaml   # URL로 고정한 자체 빌드 휠의 출처와 ABI (prebuilt-wheels가 대조)
    ├── support-matrix.md      # Phase 0 산출물
    └── review-findings.md     # 리뷰 레인 결과와 수정 순서
```

**미작성**

| 파일 | 담당 | 역할 |
|---|---|---|
| `docs/report.md` | Phase 4 | 최종 산출물 |

Wave 2~3에서 이 목록에 있던 `tests/test_axes.py`, `configs/experiment/`,
`scripts/bench.py`, `tests/test_metrics.py`, `tests/test_smoke_cpu.py`가 전부
작성되어 위 구조 블록으로 옮겨졌다. 이 표가 뒤처졌던 이유는 `plan-files`가 한쪽
방향만 봤기 때문이다 — 지금은 저장소에 있는데 블록에 없는 파일도 막으므로 구조
블록은 뒤처질 수 없다. 이 표는 여전히 손으로 관리한다.

**설계 결정**

- ablation은 `bench.py model=gemma4_e2b attn=fa4 freeze=ple` 조합으로만 표현한다.
  축을 추가할 때 코드를 고치면 컨벤션 위반이다
- 상호배타 변형이 3개 이상인 축만 config group으로 만든다. 단일 플래그
  (gradient checkpointing 등)는 `train.yaml`의 필드로 둔다
- 프레임워크 어댑터는 공통 인터페이스(`prepare -> step -> metrics`) 뒤에 숨겨
  `framework=` 한 줄로 교체되게 한다. 이게 Phase 3의 전제

## 작업 분해

### Task 1 — 저장소 부트스트랩 (GPU 불필요)

- [ ] 컨벤션 템플릿에서 `AGENTS.md` / `CLAUDE.md` / `pyproject.toml` 복사 후 채우기
- [ ] torch platform marker 설정 (macOS CPU / Linux CUDA), `uv.lock` 커밋
- [ ] ruff + pre-commit + gitleaks
- [ ] `.env.example` (`RUNPOD_API_KEY`, `HF_TOKEN` 키 이름만)
- [ ] `device.py` / `seed.py` 단일 헬퍼
- [ ] config group 골격 + Pydantic 스키마 + fail-fast 검증
- [ ] `test_config.py` (잘못된 조합이 실행 전에 죽는지)

### Task 2 — Phase 0 검증 하네스 (A100, 18 pod 병렬)

- [ ] `verify_env.py`: 프레임워크 x 모델 적재/1-step 학습 검증
- [ ] `orchestrate.py`: RunPod pod 기동(이미지 지정), 결과를 HF repo의 pod별 경로에
      push, 확보 실패분은 큐잉
- [ ] MMEB 고정 서브셋 생성 -> private HF dataset repo push
- [ ] 베이스 이미지 + 프레임워크 이미지 빌드·푸시 (GHCR)
- [ ] `docs/support-matrix.md` 자동 생성 (미확인 셀은 "미확인"으로)

### Task 3 — Phase 1 측정 하네스 (A100)

- [ ] MMEB 로더 + `--limit N` + **이미지 토큰 예산 고정** (모델별 실측 후 보정)
- [ ] 3종 모델 어댑터 (풀링 + 임베딩 헤드), InfoNCE / GradCache
- [ ] 지표 수집: throughput, MFU, peak VRAM, step p50/p95, 구조화 JSON 로그
- [ ] **타이밍 런 / 프로파일 런 분리** + 축별 warmup 폐기 구간 설정
- [ ] **canonical baseline run** + pod 간 편차 3% 게이트
- [ ] pod spec 자동 기록 (GPU UUID, vCPU, RAM, 드라이버/CUDA, DC)
- [ ] atomic save + resume
- [ ] `test_smoke_cpu.py` (CPU + 소수 샘플 E2E)
- [ ] **데이터로딩 병목 선판정** — 병목이면 DALI/사전 디코딩으로 해소 후 진행
- [ ] 리뷰 게이트: 하네스 코드를 작성자와 분리된 레인에서 리뷰

### Task 3.5 — 축 구현 (Phase 2의 선행 조건, 대부분 GPU 불필요)

Task 4는 축을 **조합해서 실행**하는 것만 적고 있었고, 그 축들을 **구현하는** 작업은
어느 Task에도 없었다. 원인은 위 "설계 결정"의 첫 줄이다 — "축을 추가할 때 코드를
고치면 컨벤션 위반"은 축을 *변화시키는 방법*으로는 맞지만, 조용히 *축이 이미
존재한다*는 전제를 깔았다. ms-swift나 axolotl을 쓴다면 맞는 전제다. 우리는 `native`
하네스를 자체로 만들고 있고 거기서는 축 하나가 config 한 줄이 아니라 구현 하나다.
**측정 인프라를 짓는 일과 측정 대상 기법을 구현하는 일은 다른 작업이다.**

미구현 축은 config가 값을 제공하므로 스윕이 돌아가고, `axes.py`가 그 값을 거부하므로
런은 기본값과 동일해진다. 그 상태로 ablation을 돌리면 라벨만 다른 동일 결과가 나오고
**"이 기법들은 효과 없음"으로 읽힌다.**

거부는 전부 `trainbench/axes.py`의 **무조건적** `UnappliedAxis`이지 import 실패가
아니다. 따라서 **이미지를 빌드해도 이 목록은 줄지 않는다** — 패키지 설치(`axis-packages`)는
선행 조건이지 이 작업이 아니다.

**`axis-values`를 완료 근거로 쓰지 않는다.** 앞서 이 자리에 "현재 상태의 유일한
기준은 `axis-values`"라고 적었는데 **그 문장이 틀렸다.** 이 체크가 재는 것은 축 값
하나를 4개 호출 지점에 통과시켰을 때 `UnappliedAxis`가 나지 않는다는 것뿐이다 —
**적용됐다가 아니라 거부되지 않았다**이다. 적대 검증에서 `dataloader` 축을 완전히
무력화해도 이 숫자가 글자 단위로 같다는 것이 확인됐다. 그리고 이 체크는 `_Tiny`
모듈(`torch.nn.Linear` 하나)로 돌므로 실제 모델에서만 드러나는 것은 원리적으로 볼 수
없다.

축이 실제로 무언가를 바꿨다는 근거는 **`applied.py`의 capture가 요청과 다른 값을
읽어낼 수 있는가**와 **`tests/test_axes.py`가 그 차이를 고정하는가** 두 가지다.
`axis-values`는 그 앞단의 하한선이고, 오르면 진전이지만 올랐다고 축이 도는 것은
아니다. 아래 목록은 2026-08-02 01:37 스냅샷이며 축 레인이 동시에 랜딩 중이다.

CPU에서 구현·검증이 끝나는 것과 GPU가 있어야 판정되는 것을 나눈다. 앞의 것을 GPU
파드에서 디버깅하면 시간당 요금을 내며 `NotImplementedError`를 읽게 된다.

| 축 값 | CPU/GPU | 비고 |
|---|---|---|
| `kernel=liger` / `fla` | GPU 판정 | 패칭 자체는 CPU에서 확인 가능하나 커널 경로는 CUDA 전용 |
| `parallel=ddp` / `fsdp2` | GPU | 프로세스 그룹 필요 |
| `parallel=zero2` / `zero3` + `train.offload` | **GPU** | `deepspeed.initialize`가 모델·옵티마이저·로더를 한 번에 만든다(`docs/CONTRACTS.md` §2) |
| `dataloader=dali` / `dali_packed` | **GPU** | 하드웨어 JPEG 디코드가 이 축의 측정 대상 |
| `optim=adamw_8bit` | **GPU** | bitsandbytes |
| `peft=qlora` | **GPU** | 4-bit 양자화가 CUDA 전용 |
| `compile=regional` | CPU | |

- [ ] 위 표의 미구현 값에 `axes.py` 적용 지점 추가
- [ ] **capture probe를 같은 커밋에서 확장** — 아래 3건은 구현만으로 측정이 열리지 않는다
- [ ] `axis-values`가 그룹당 2개 이상을 보고하는지 확인 후 Task 4 착수

**capture probe 확장이 구현과 한 세트인 2건.** `applied=None`(미확인)은 불일치와
동일하게 timing 런을 차단하므로(`docs/CONTRACTS.md` §2 불변식), 구현만 하고 probe를
두면 그 축은 **영구히 측정 불가**다.

| 축 | 지금 상태 | 구현과 함께 해야 하는 것 |
|---|---|---|
| `optim=adamw_8bit` | `_capture_optim`이 클래스명을 `kind.lower()`로 돌려준다 → `AdamW8bit` → `"adamw8bit"` | config 값 `adamw_8bit`와 철자가 다르다. `_REQUESTED_OVERRIDES`나 capture 쪽에서 맞춰야 영구 불일치가 안 된다 |
| `train.offload` | deepspeed 아래에서는 **의도적으로** undetermined (`_capture_offload` docstring) | deepspeed config를 읽어내는 경로가 없으면 offload는 켜자마자 차단된다 |

`train.gradient_checkpointing=selective`도 같은 형태였으나 **양쪽 다 랜딩됐다**
(`axes.py`의 `create_selective_checkpoint_contexts` + `_capture_gradient_checkpointing`이
`context_fn`이 실어나르는 **정책의 동일성**으로 `full`과 구분한다 — `context_fn`의 존재
자체는 근거가 아니다. 같은 팩토리로 만든 남의 정책은 `none`이나 `full`의 backward를
`selective` 라벨 아래 넣으므로 undetermined로 거부된다). 남은 것은 위 2건이다.
`precision=mxfp8`/`nvfp4`도 같은 형태였으나 Transformer Engine 의존성째 제거됐다 — A100
(CC 8.0)에서는 둘 다(CC 10.x 전용) 원리적으로 열릴 수 없었다.

### Task 4 — Phase 2 ablation (B200, 12~18 pod 병렬)

- [ ] B200 스모크 테스트 1회 (A100에서 돌던 것이 Blackwell 빌드에서 도는지)
- [ ] gemma-4-E2B full FT 적재 확인
- [ ] `configs/experiment/`에 축 그룹별 조합 정의 후 실행
- [ ] **`attn` 축 매니페스트.** 지금 21개 매니페스트 중 `attn`을 스윕하는 것이 0개이고
      전부 `configs/config.yaml`의 기본값 `sdpa`로 돈다. 5개 값이 전부 적용 가능한
      유일한 축인데(2026-08-02 `axis-values` 5/5) 한 번도 비교된 적이 없다.
      Qwen3-VL의 병목 가설이 attention이므로 이 축이 빠지면 핵심 가설 하나가 미검증으로
      남는다. FA3는 Hopper·FA4는 Blackwell 전용이므로 GPU 선택과 함께 정한다
- [ ] baseline 편차 게이트 통과 여부 확인, 실패 pod 재실행

### Task 5 — Phase 3 프레임워크 (B200, 18 pod 병렬)

- [ ] 프레임워크 어댑터 6종
- [ ] 동작하지 않는 조합은 "미지원"으로 기록 (그 자체가 결과)

### Task 6 — 품질 검증 + Phase 4 리포트

- [ ] Pareto 후보 선별 -> 장기 품질 런 (loss curve + MMEB-eval Recall@k)
- [ ] `report.py`로 집계, `docs/report.md` 작성
- [ ] 공개 주장 수치 대비 실측 비교표

## 착수 순서

Task 1은 GPU가 필요 없으므로 즉시 시작 가능하다. Task 2의 이미지 빌드부터 RunPod
접근이 필요하고, 실행 시점에 A100/B200 재고를 재확인한다. DC 종속이 없으므로
재고가 있는 곳 아무 데나 배치하면 된다.

## 근거 출처

모델 config: HF Hub 직접 조회 (2026-07-31)
- https://hf.co/Qwen/Qwen3-VL-Embedding-2B
- https://hf.co/Qwen/Qwen3.5-0.8B
- https://hf.co/google/gemma-4-E2B
- 논문: https://arxiv.org/abs/2601.04720

생태계
- Liger-Kernel Gemma4 (open): https://github.com/linkedin/Liger-Kernel/issues/1186
- Liger-Kernel Qwen3.5: https://github.com/linkedin/Liger-Kernel/issues/1119
- FlashAttention-4: https://tridao.me/blog/2026/flash4/
- FlashQLA: https://qwen.ai/blog?id=flashqla
- flash-linear-attention: https://github.com/fla-org/flash-linear-attention
- ms-swift Embedding: https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Embedding.md
- ST 멀티모달: https://huggingface.co/blog/train-multimodal-sentence-transformers
- ST x Unsloth: https://sbert.net/examples/sentence_transformer/training/unsloth/README.html
- Unsloth Gemma4: https://unsloth.ai/docs/models/gemma-4/train
- Unsloth 임베딩: https://unsloth.ai/docs/basics/embedding-finetuning
- Axolotl Sequence Parallelism: https://axolotlai.substack.com/p/enabling-long-context-training-with
- Tevatron 2.0: https://arxiv.org/html/2505.02466v1
- GradCache: https://github.com/luyug/GradCache
- Transformer Engine: https://github.com/NVIDIA/TransformerEngine
- HTA: https://hta.readthedocs.io/
- 데이터(원본, 이미지 경로만): https://hf.co/datasets/TIGER-Lab/MMEB-train
- 데이터(실제 upstream, 이미지 포함 커뮤니티 미러):
  https://hf.co/datasets/MrZilinXiao/MMEB_train_with_image
