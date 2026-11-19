# Data-Mining-26-2-Project
아주대학교 2026 2학기 데이터마이닝 팀프로젝트 

팀원 : 202422077 최지예 / 202422054 오수언 / 202020751 송건

# 프로젝트 주제

Steam 게임 이용 이력과 메타데이터를 활용한 Hybrid 추천 시스템 및 Cold Start 문제 해결

* 사용자의 게임 상호작용 이력과 게임의 장르, 태그 데이터를 결합해 개인화된 게임을 추천하고, 사용자 또는 게임의 상호작용 데이터가 부족한 Cold Start 상황에서도 추천 성능을 유지할 수 있는지를 확인한다.

# 본 프로젝트의 핵심 과제

1. Hybrid 추천의 효과 : 사용자 상호작용 데이터와 게임 메타데이터를 결합하면 단일 추천 알고리즘보다 추천 성능이 개선되는가?

2. 사용자 Cold Start : 사용자의 상호작용 이력이 적을 때 Content-Based Filtering과 사용자 군집 정보를 활용하면 추천 성능 저하를 완화할 수 있는가?

3. 게임 Cold Start : 사용자 평가 이력이 부족한 신규 게임에 대해 메타데이터 기반 추천이 유효한 추천 결과를 제공할 수 있는가?

# Steam 게임 추천 머신 (SCE331 Data Mining, Team #)

K-means · Apriori · Item-based CF · Content-based filtering을 결합한 하이브리드 Steam 게임 추천 시스템.

## Overview
- **문제**: 사용자의 Steam 리뷰/플레이 이력으로 아직 리뷰하지 않은 게임 Top-K를 추천한다.
- **데이터**: Kaggle *Game Recommendations on Steam* (Anton Kozyriev).
- **수업 알고리즘 4개**
  1. **K-means clustering**: 태그별 플레이 비중(표준화)으로 사용자 취향 군집 생성 → 군집 내 인기 게임 점수
  2. **Apriori**: 사용자별 긍정 리뷰 게임 basket → 빈발 itemset·연관규칙 → 규칙 confidence 점수
  3. **Item-based collaborative filtering**: 사용자 공동 선호 기반 cosine 유사도(top-N 이웃)
  4. **Content-based filtering**: 태그·설명 TF-IDF 프로필 cosine 유사도, 이력이 적은(cold) 게임에서 가중치 증가
- **하이브리드 (v2, 2단계)**: ① Item-based CF가 사용자별 후보 100개 생성 → ② K-means·Apriori·Content 점수로 후보 안에서만 순위 재조정(re-rank)

## v1 → v2 변경 사항
v1 실험에서 하이브리드 가중치가 모두 0으로 선택되어(= CF 단독) 다음을 수정했다. v1 결과는 `results/`에 보존해 비교한다.

| 구성요소 | v1 | v2 | 이유 |
|---|---|---|---|
| K-means 특징 | 상위 60개 태그의 플레이 비중 | 게임 30% 이상에 붙은 범용 태그(Singleplayer 등) 제외, 태그별 z-score 표준화 | v1 군집이 모두 Action/Singleplayer 위주(silhouette 0.08) |
| K-means k 후보 | 4–14 | 2–12 | v1에서 최솟값 4가 선택됨 |
| Apriori | min_support 0.01–0.02, min_conf 0.2 | min_support 0.003–0.01, min_conf 0.1 | v1 규칙 15개, 테스트 사용자 54%만 적용 |
| CF 이웃 수 | 50–200 | 50–400 | v1에서 최댓값 200이 선택됨 |
| 결합 방식 | 전체 게임 점수의 가중합 | CF 후보 100개 안에서만 re-rank | 약한 신호가 CF 상위권을 흐트러뜨림 |
| 가중치 후보 | 0, 0.1–1.0 | 0, 0.01–0.5 (작은 값 포함) | 0.1에서도 성능이 떨어짐 |
| 선택 기준 | NDCG@20 | NDCG@20 (정확도형) + NDCG 99% 이상 중 tail recall 최대 (tail형) | Content의 비인기 게임 보완 효과 측정 |
| 진단 | – | `missing_metadata.csv`, 후보 recall 상한 | v1에서 게임 18%에 태그 없음 |
| 전처리 | pandas, k-core 후 샘플링 | 순수 파이썬, 샘플링 후 k-core, 단계별 현황표 | 기준을 코드에서 직접 확인하기 위해 |

## v2 → v3: 메타데이터 보완
v2의 `missing_metadata.csv` 분석 결과, Kaggle `games_metadata.json`에서 **행은 있지만 tags·description이 비어 있는 게임이 505개**였다.
이들은 Team Fortress 2, The Witcher 3, Cyberpunk 2077 등 가장 인기 있는 게임이어서 **긍정 리뷰의 약 45%**가 태그 없는 게임에 몰려 있었고,
K-means 사용자 특징과 Content-based 프로필이 데이터의 절반 가까이를 반영하지 못했다.
v3는 이 게임들만 공개 API에서 보완한다 (Kaggle에 값이 있는 게임은 덮어쓰지 않음).

| 항목 | 출처 | 비고 |
|---|---|---|
| 태그 | SteamSpy `api.php?request=appdetails` | 사용자 투표 수 상위 20개, 초당 1회 제한 |
| 설명 | Steam 스토어 `api/appdetails` 의 `short_description` | 약 5분에 200회 제한 |
| 태그 대체 | Steam 스토어 `genres` | SteamSpy에서 태그를 못 받은 경우만, `tag_source`에 기록 |

```bash
# 1) v2(또는 --no-supplement) 결과의 missing_metadata.csv 로 대상 게임을 정하고 받는다 (약 23분, 이어받기 가능)
python scripts/fetch_missing_tags.py --missing results_v2/missing_metadata.csv --out data/supplement/supplement_metadata.jsonl
# 2) 보완 파일을 넣어 실행
python run_pipeline.py --data-dir data/raw --out-dir results_v3 --supplement data/supplement/supplement_metadata.jsonl
```
`data/supplement/supplement_metadata.jsonl`은 저장소에 함께 커밋한다(수집 시각 `fetched_utc` 포함). API 값은 시간이 지나면 바뀔 수 있으므로,
재현할 때는 새로 받지 말고 커밋된 파일을 사용한다.
- **Baseline**: Popularity (학습 구간 긍정 리뷰 수)

## Team
| 이름 | 학번 | 기여 |
|---|---|---|
| | | |

## Repository layout
```
run_pipeline.py              전체 실험 실행 (데이터 → 튜닝 → 테스트 → 표/그림)
colab_run.ipynb              Colab 실행용 노트북 (설치·데이터 다운로드·실행·결과 확인)
src/config.py                모든 하이퍼파라미터와 seed
src/preprocess.py            전처리: 로딩, 사용자 샘플링, 중복 정리, k-core, 시간순 split (pandas 미사용)
src/data.py                  전처리 결과 → 모델 입력용 희소행렬, 기술 통계
src/kmeans.py                K-means (k-means++ 초기화 + Lloyd), 사용자 태그 특징, k 선택
src/apriori.py               A-Priori (bitset 지지도 계산, 후보 생성·가지치기), 연관규칙, 규칙 점수
src/models.py                Popularity / ClusterPop / RuleRec / ItemCF / ContentBased / RerankHybrid
src/metrics.py               Precision/Recall/NDCG/Hit@K, coverage, 인기도 그룹별 recall
src/plots.py                 보고서용 그림
scripts/fetch_missing_tags.py   빈 태그·설명을 SteamSpy·Steam 스토어에서 받아 보완 파일 생성 (v3)
scripts/make_synthetic_data.py  동일 스키마의 가짜 데이터 (코드 동작 확인용, 결과 보고 금지)
data/supplement/             보완 메타데이터 (저장소에 커밋)
```

## Environment / setup
Python ≥ 3.10.
```bash
pip install -r requirements.txt
```
전체 `recommendations.csv`(약 4,100만 행)를 읽으므로 **RAM 8GB 이상**을 권장한다 (Google Colab 가능).
메모리가 부족하면 `--max-rows 10000000`처럼 앞부분만 읽을 수 있다(보고서에 명시할 것).

## Data sources and acquisition
1. https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam 에서 다운로드
   (또는 `kaggle datasets download -d antonkozyriev/game-recommendations-on-steam -p data/raw --unzip`).
2. `data/raw/`에 `recommendations.csv`, `games.csv`, `users.csv`, `games_metadata.json`을 둔다.
3. 보고서에 **dataset version과 다운로드 날짜, license**를 기록한다. 데이터는 repository에 올리지 않는다(`.gitignore`).

## Preprocessing
`src/preprocess.py`에 있다. 기준을 눈으로 확인할 수 있도록 pandas·numpy 없이 파이썬 기본 문법(for/if/dict)과
표준 모듈(csv, json, random)만 사용했고, 파일 맨 위 주석에 모든 기준이 순서대로 정리되어 있다.
단계별 리뷰·사용자·게임 수와 제거된 리뷰 수는 `results_v2/preprocessing_report.csv`에 저장된다.

| 단계 | 기준 |
|---|---|
| [1] 긍정 리뷰 | `is_recommended = true` |
| [2] 후보 사용자 | 긍정 리뷰 ≥ 10개 |
| [3] 샘플링 | 후보 중 30,000명 무작위 (seed 42) |
| [4] 게임 확인 | `games.csv`에 없는 app_id 제거 |
| [5] 중복 정리 | 같은 (사용자, 게임) 리뷰는 날짜가 가장 늦은 1개만 |
| [6] k-core 반복 | 게임: 긍정 사용자 ≥ 30명 / 사용자: 긍정 리뷰 ≥ 10개, 변화가 없을 때까지 반복 |
| [7] 시간순 분할 | 사용자별 날짜순 정렬 → 최근 20% test, 직전 (나머지의 20%) validation, 나머지 train (반올림, 최소 1개) |
| [8] 메타데이터 보완 (v3) | Kaggle tags/description이 빈 게임만 보완 파일 값으로 채움 |

- 상호작용 가중치(confidence) = 1 + log(1 + hours)
- 부정 리뷰는 학습 신호로 쓰지 않고, 이미 리뷰한 게임을 추천 후보에서 빼는 데만 사용
- 원본 4,100만 행을 한 줄씩 두 번 읽는다(전체 약 3분). v1과 달리 **샘플링을 k-core보다 먼저** 하므로, 최종 사용자 수는 30,000명보다 조금 적다.

## Experiment protocol
1. **튜닝 (train으로 학습, validation으로 평가, NDCG@20 기준)**
   - K-means k: silhouette(+ elbow) → `kmeans_k_selection.csv`
   - Apriori min_support ∈ {0.003, 0.005, 0.01}, CF 이웃 수 ∈ {50, 100, 200, 400}
   - Hybrid re-rank 가중치: w_CF = 1 고정, w_KM, w_AR, w_CB ∈ {0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5} (512개 조합)
   - 두 하이브리드 선택: **정확도형**(NDCG@20 최대), **tail형**(NDCG@20이 최대값의 99% 이상인 조합 중 tail 게임 recall@20 최대)
2. **최종 평가**: 선택된 설정으로 train+validation 재학습 → test를 **한 번만** 평가
3. 모든 모델이 같은 사용자, 같은 후보(이미 리뷰한 게임 제외), 같은 타깃을 사용
4. Ablation(구성요소 하나씩 제거), 인기도 그룹별 recall(head/mid/tail), Wilcoxon 대응표본 검정

## Reproducing results
```bash
python run_pipeline.py --data-dir data/raw --out-dir results_v3 --supplement data/supplement/supplement_metadata.jsonl  # 본 실험 (v3)
python run_pipeline.py --data-dir data/raw --out-dir results_v2 --no-supplement     # v2 재현 (Kaggle 메타데이터만)
python run_pipeline.py --data-dir data/raw --out-dir results_random --split random  # 민감도: 무작위 split
python run_pipeline.py --data-dir data/raw --out-dir results_share --feature-scaling share  # K-means 특징 v1 방식
```
코드 동작만 확인할 때:
```bash
python scripts/make_synthetic_data.py --out data/synthetic
python run_pipeline.py --data-dir data/synthetic --out-dir results_synthetic --min-item-pos 10 --quick
```

Colab에서는 `colab_run.ipynb`를 위에서부터 실행한다.

## Outputs (`results_v3/`)
| 파일 | 내용 |
|---|---|
| `preprocessing_report.csv` | 전처리 단계별 리뷰·사용자·게임 수, 제거된 리뷰 수 |
| `metadata_fill.json` | 보완 전 빈 태그·설명 수, 채운 수, 태그 출처별 개수 |
| `data_stats.json` | subset 통계 (사용자·게임·리뷰 수, density, split 크기 등) |
| `missing_metadata.csv` | 태그·설명이 없는 게임 목록 (메타데이터 행 자체가 없는지 / 비어 있는지) |
| `kmeans_k_selection.csv`, `clusters.csv` | k별 SSE/silhouette, 군집별 크기·평균보다 많이/적게 하는 태그·인기 게임·특징적 게임 |
| `apriori_top_rules.csv` | 상위 100개 연관규칙 (support, confidence, lift) |
| `tuning_val.csv`, `hybrid_weight_grid_val.csv`, `chosen_params.json` | 검증 구간 튜닝 기록 (규칙 수·규칙 적용 사용자 비율, 512개 가중치 조합, CF 후보 recall 상한) |
| `metrics_test.csv` | 최종 test 지표 (모델별 Precision/Recall/NDCG/Hit@10·20, 95% CI, coverage, 그룹별 recall) |
| `significance_test.csv` | 두 하이브리드 vs 각 모델 per-user NDCG Wilcoxon 검정 |
| `component_analysis.json` | 규칙 수, 규칙 적용 사용자 비율, 군집 크기 등 |
| `figures/*.png` | k 선택, 군집 태그 heatmap, 모델 비교, ablation, 인기도 그룹 recall, 가중치 민감도, 정확도–tail trade-off |

## References and external code
- 알고리즘(K-means, A-Priori, item-based CF, content-based, hybrid, 평가 지표)은 본 팀이 직접 구현했다.
- 외부 라이브러리: NumPy, pandas, SciPy(희소행렬, Wilcoxon), scikit-learn(`TfidfVectorizer`, `normalize`, `silhouette_score`), Matplotlib.
- 데이터: A. Kozyriev, "Game Recommendations on Steam," Kaggle, [version], accessed [date].
- 보완 메타데이터: SteamSpy, https://steamspy.com/api.php, accessed [date]; Valve, Steam Store appdetails API, accessed [date].

## AI-use statement
본 팀은 [기간] 동안 코드 초안 작성을 위해 Anthropic Claude (Claude Opus 5.5)를 사용했다.
도움받은 부분은 `src/*.py`, `run_pipeline.py`, `scripts/make_synthetic_data.py`이다.
[검증 방법: 코드 리뷰, 가짜 데이터 테스트, 실제 데이터 결과 확인 등]을 통해 검증하고 [수정한 부분]을 수정했다.
