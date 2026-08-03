# lane-logs 레인 노트 — 머지 단계로 넘기는 것

## 1. 소유 파일 변경

- `trainbench/pods.py`: `_build_http_request`/`_http_error_message` 추출(순수 리팩터,
  `send()` 동작 불변), `logs_request`/`stream_log`/`parse_log_sse`/`fetch_log`/`LogFetch` 신설.
- `scripts/orchestrate.py`: `huggingface_hub.HfApi` import 추가, `POD_LOG_NAME` 상수,
  `pod_log_destination`/`capture_pod_log` 신설, 종료 루프 인라인 블록을
  `handle_pod_outcome()` 로 추출(동작 불변 — 로그 캡처를 그 안에 추가). 원장 항목에
  `"log": None` 필드 추가.
- `tests/test_pod_logs.py` 신규 — `tests/test_pods.py` 를 잡고 있는 다른 레인 것이라
  건드리지 않았다. **머지 시 이 파일 전체(29개 테스트)를 `tests/test_pods.py` 로
  옮겨달라.**

## 2. API 표면 — 근거

RunPod REST v1(`rest.runpod.io/v1/openapi.json`)에는 로그 엔드포인트가 없다(직접
curl 로 `paths` 확인, 2026-08-03). 실제 엔드포인트는 **다른 호스트**의 v2 REST —
`GET https://v2-rest.runpod.io/v2/pods/{id}/logs`, SSE, `tail`/`source`/`since`
쿼리 — 로, 그 호스트의 `openapi.json` 을 직접 curl 로 받아 `getPodLogs` 오퍼레이션과
파라미터 정의(`LogTail`/`LogSince`/`LogSourceParam`)를 확인했고, 이 세션에 설치된
`mcp__runpod__stream-pod-logs` MCP 툴 스키마 및 `runpod/runpod-mcp` 저장소의
`src/tools/logs.ts`(`readSseSnapshot`, `parseLogSse`)로 파싱 모양·헤더까지
교차 확인했다. `RUNPOD_API_KEY`로 실제 파드에 대고 쳐본 적은 없다 — 아래 열린 질문의
1번.

## 3. 실 파드로만 답할 수 있는 것 (한 줄씩)

1. SSE `data:` 프레임이 openapi 문서/`runpod-mcp` 참조 구현과 실제로 바이트 단위까지
   일치하는지 — 이 레인은 실 API를 호출한 적이 없다(확인 안 함).
2. 백필(`tail`)이 끝난 뒤 이미 종료된 컨테이너의 스트림이 정말로 idle 로 넘어가는지,
   아니면 RunPod 가 주기적 SSE 코멘트(`: ping` 류)를 계속 보내 `LOG_IDLE_SECONDS`
   타이머를 계속 리셋시키는지 — 리셋된다면 매 파드 종료가 `LOG_TOTAL_SECONDS`(15초)
   만큼 늘어난다(확인 안 함, `stream_log`의 두 번째 방어선인 벽시계 데드라인이 이
   경우를 여전히 막아준다).
3. 오케스트레이터 쪽 `dev` Infisical 환경에 `HF_TOKEN` 이 실제로 있는지 — 있다고
   가정하고 `capture_pod_log` 를 짰다(`.env.example`/AGENTS.md 교차 근거는 있으나 직접
   `os.environ` 을 찍어 확인하지는 않았다). 없다면 업로드가 매번 실패로 잡히고(파드
   종료는 막지 않음) 로그가 한 건도 안 남는다.
4. 배치로 동시에 끝나는 파드가 많을 때 `capture_pod_log`(최악 25초 대기 + 업로드
   1회)가 순차 실행되며 다음 파드 종료를 늦추는 정도 — 실측 없음.
5. 프레임워크가 verbose 할 때(axolotl 등) 원인이 되는 줄이 `tail=5000` 백필 범위
   밖으로 밀려나는 실제 빈도 — 이번 캠페인 초반 진짜 실패 로그로 검증한 적 없다.
