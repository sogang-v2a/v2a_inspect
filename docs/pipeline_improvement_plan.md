# V2A Inspect 파이프라인 고도화 계획 요약

> 작성일: 2026-03-13

---

## 1. 현재 파이프라인

```
영상 → [Gemini ① 씬 분석] → VideoSceneAnalysis
     → [Gemini ② 텍스트 그룹핑] → TrackGroup[]
     → [Gemini ③ VLM 그룹 검증] → TrackGroup[] (분할/유지)
     → [Gemini ④ VLM 모델 판정] → ModelSelection (TTA/VTA)
     → GroupedAnalysis → Streamlit UI
```

### 주요 문제

| 문제 | 심각도 | 원인 |
|------|:------:|------|
| Model Selection 93%가 "TTA 50%"로 수렴 | 매우 높음 | LLM central tendency bias + 1.5 임계값 자의성 |
| 고FPS → 과잉 분할 → 그룹핑 실패 | 높음 | 트랙 폭증으로 텍스트 그룹핑 정확도 저하 |
| VLM이 전체 영상에서 시간 지시를 무시 가능 | 높음 | 전체 URI 전송, hard-clip 미적용 |
| object ≤2 → background 오염 | 높음 | 3번째 이상 음원이 비구조적으로 혼입 |

---

## 2. 개선 계획

### 제안 파이프라인

```
영상 → [Gemini ① 씬 분석] (object ≤3, TTA-aware 제약, 길이 제약)
     → [extract_raw_tracks]
     → [Gemini ② 텍스트 그룹핑] (음향 맥락 분리 규칙 추가)
     → [Gemini ③ 프레임 기반 VLM 검증] (대표 프레임 2-3장)
     → [Gemini ④ CoT 기반 모델 선택] (숫자 점수 폐기 → 자연어 추론)
     → GroupedAnalysis
```

---

## 3. 단기 수정 사항 (즉시 적용)

### 3-1. analyze_video_scenes.py

| 변경 | 위치 | 내용 |
|------|------|------|
| object 수 확장 | `Scene.objects` `max_length` | 2 → 3 |
| TTA-aware 제약 | `PROMPT` / `EXTENDED_PROMPT` | "각 description 20~50 단어, 음향 속성 중심" 추가 |
| Scene 길이 제약 | `PROMPT` / `EXTENDED_PROMPT` | "각 scene은 최소 1.5초, 최대 12.0초" 추가 |

### 3-2. track_grouper.py — 그루핑 프롬프트

| 변경 | 위치 | 내용 |
|------|------|------|
| 음향 맥락 분리 규칙 | `_GROUPING_PROMPT_TEMPLATE` | "같은 entity라도 행동/acoustic context가 다르면 별도 그룹" 규칙 추가 |

### 3-3. track_grouper.py — Model Selection (핵심)

| 변경 | 위치 | 내용 |
|------|------|------|
| 프롬프트 교체 | `_MODEL_SELECT_PROMPT_TEMPLATE` | 숫자 3점수 → CoT 자연어 추론 프롬프트 |
| 판정 함수 교체 | `_select_model_from_scores()` | 스코어 공식 폐기 → CoT 응답 파싱 |
| 그룹 레벨 집계 | `assign_model_selections()` | 평균 → majority voting + confidence geometric mean |
| Background 규칙 | 유지 | background → TTA 고정 (rule_based=True) |

**현재 공식 (폐기 대상)**:
```
vta_raw = (motion + coupling) / 2.0
tta_raw = source_diversity + n_obj 보정
diff ≥ 1.5 → VTA, diff ≤ -1.5 → TTA, else → TTA 50%
→ 93%가 "TTA 50%"로 수렴 (사실상 무력화)
```

**CoT 대체 방식**:
```
LLM에게 자연어 추론 요청 → 최종 판단 {model: VTA|TTA, confidence: 0~1, reasoning: str}
그룹 레벨: 멤버 majority voting, confidence = geometric mean
```

---

## 4. 중기 수정 사항 (1-2주)

### 4-1. track_grouper.py — VLM 검증 방식 전환

| 변경 | 내용 |
|------|------|
| 현재 | 전체 영상 URI + 시간 텍스트 지시 (집중 보장 없음) |
| 제안 | 대표 프레임 2-3장을 moviepy로 추출 → 이미지로 Gemini 전송 |
| 효과 | API 비용 60%+ 절감, 해당 시점 집중 보장 |

### 4-2. track_grouper.py + pipeline_client.py — 2-Pass entity 매핑

| 변경 | 내용 |
|------|------|
| 현재 | 텍스트 그룹핑(1회) + VLM 검증(그룹당 N회) = 1+N회 |
| 제안 | 1차: 씬 분석, 2차: 결과 JSON + 비디오로 entity_id 매핑 = 2회 |
| 효과 | 시각+텍스트 동시 활용, 호출 횟수 감소 |

### 4-3. app.py — UI 분리

| 변경 | 내용 |
|------|------|
| 버튼 분리 | `Analyze & Group` → `🔍 Analyze` + `🔗 Group` |
| 단계별 status | 그루핑 검증 / 모델 선정 각각 별도 표시 |

### 4-4. analyze_video_scenes.py — Event 하이브리드

| 변경 | 내용 |
|------|------|
| 스키마 추가 | `AudioEvent(timestamp, duration, description, intensity)` |
| Scene 확장 | `events: list[AudioEvent]` 필드 추가 |
| 프롬프트 | "순간적 충격음/접촉음을 events로 별도 기술" |

---

## 5. 장기 연구 확장 (1개월+)

| 작업 | 설명 |
|------|------|
| Oracle baseline | TTA/VTA 둘 다 실행 → 인간 평가로 ground truth |
| Audio Source Timeline | Scene 폐기, 음원 단위 추적 → grouping 불필요화 |
| 계층적 그루핑 | Entity-level (모델 선택) + Instance-level (canonical) |
| Embedding 클러스터링 | 고FPS 대규모 트랙 대응 |

---

## 6. 수정 파일 매핑

```
v2a_inspect/
├── pipeline/
│   ├── analyze_video_scenes.py    ← [단기] object ≤3, 프롬프트 제약, 길이 제약
│   │                                 [중기] Event 스키마 추가
│   │
│   └── track_grouper.py           ← [단기] 그루핑 프롬프트 규칙, CoT 모델 선택
│                                     [중기] 프레임 기반 VLM 검증, 2-Pass entity 매핑
│
├── pipeline_client.py             ← [중기] 2-Pass 호출 지원
│
└── app.py                         ← [중기] 버튼 분리, 단계별 status
```

---

## 7. 검증 계획

### Baseline 비교

| # | 방법 | 설명 |
|---|------|------|
| 1 | TTA-only | 모든 트랙에 TangoFlux |
| 2 | VTA-only | 모든 트랙에 VTA 모델 |
| 3 | Random | 50% 확률 TTA/VTA |
| 4 | Score-based (현재) | 숫자 스코어 공식 |
| 5 | CoT-routing (제안) | CoT 기반 라우팅 |
| 6 | Oracle | 인간 최적 선택 (상한) |

### 핵심 가설

1. VLM 장면 이해 기반 모델 라우팅이 단일 모델 대비 V2A 품질을 개선하는가?
2. CoT 추론이 숫자 스코어보다 안정적이고 oracle에 가까운가?
3. 음향 맥락 기반 그루핑이 CLAP score를 향상시키는가?

### 평가 지표

| 메트릭 | 측정 대상 | 자동/수동 |
|--------|----------|:--------:|
| FAD | 생성 품질 | 자동 |
| KLD | 분포 유사도 | 자동 |
| AV-Align | 오디오-비디오 동기화 | 자동 |
| CLAP score | 텍스트-오디오 정합도 | 자동 |
| MOS | 전체 인지 품질 | 수동 |

### 예상 실패 모드

| 실패 | 대응 |
|------|------|
| LLM 판단 불안정 | temperature=0 + 3회 majority voting |
| TTA/VTA 품질 격차 예측 불가 | 모델-specific 프로파일링 |
| 그루핑이 오히려 품질 저하 | description 통일만, 모델은 per-segment |
| 전체 routing 무의미 | negative result로 연구 기여 전환 |

---

## 8. 연구 프레이밍

**제목 후보**: "Chain-of-Thought Guided Model Routing for Scene-Aware Video-to-Audio Generation"

**핵심 기여**:
1. CoT 기반 TTA/VTA 모델 라우팅 — 숫자 스코어 대비 안정성/정확도 개선
2. 음향 맥락 기반 그루핑 — entity 동일성이 아닌 acoustic context 기준 분리
3. 프레임 기반 VLM 검증 — 비용 60%+ 절감, 정확도 동등 이상
