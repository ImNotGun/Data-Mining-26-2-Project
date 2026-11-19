"""
데이터 전처리 (pandas / numpy 를 쓰지 않는 버전)
================================================

이 파일은 원본 Kaggle 파일을 읽어 실험에 쓸 데이터를 만든다.
기준이 한눈에 보이도록 pandas·numpy 대신 파이썬 기본 문법(for, if, dict, list)과
표준 모듈(csv, json, random, math)만 사용했다.

모든 기준은 아래 "전처리 기준" 블록에 모여 있고, 기본값은 src/config.py 에 있다.
각 단계가 끝날 때마다 (리뷰 수, 사용자 수, 게임 수, 제거된 수)를 기록해서
results/preprocessing_report.csv 로 저장한다.

──────────────────────────────────────────────────────────────────────────────
전처리 기준 (순서대로 적용)
──────────────────────────────────────────────────────────────────────────────
 [1] 긍정 리뷰 정의      is_recommended 값이 true 인 리뷰 = 사용자가 그 게임을 "좋아함"
 [2] 후보 사용자          긍정 리뷰가 MIN_USER_POS(10)개 이상인 사용자
 [3] 사용자 샘플링        후보 사용자 중 MAX_USERS(30,000)명을 무작위 추출 (seed 고정)
 [4] 게임 목록 확인       games.csv 에 없는 app_id 의 리뷰는 제거
 [5] 중복 리뷰 정리       같은 (사용자, 게임) 리뷰가 여러 개면 날짜가 가장 늦은 1개만 유지
                          (날짜가 같으면 파일에서 더 뒤에 나온 것)
 [6] 반복 필터링 (k-core) 아래 두 조건을 둘 다 만족할 때까지 반복
       - 게임: 긍정 리뷰를 남긴 사용자가 MIN_ITEM_POS(30)명 이상
       - 사용자: 남은 게임 중 긍정 리뷰가 MIN_USER_POS(10)개 이상
     (긍정 기준으로 남길 사용자/게임을 고르고, 그들의 부정 리뷰도 함께 남긴다.
      부정 리뷰는 학습 신호로는 안 쓰고 '이미 리뷰한 게임'을 추천 후보에서 빼는 데만 쓴다.)
 [7] 시간순 분할          사용자마다 리뷰를 날짜순(같으면 파일 순서)으로 정렬한 뒤
       - 가장 최근 TEST_RATIO(20%) → test
       - 그 직전, 나머지의 VAL_RATIO(20%) → validation
       - 나머지(오래된 리뷰) → train
     개수는 반올림하고 최소 1개 이상이다.
 [8] 메타데이터 보완      Kaggle 메타데이터에서 tags(또는 description)가 비어 있는 게임만,
                          보완 파일(scripts/fetch_missing_tags.py 로 SteamSpy·Steam 스토어에서 받은 것)의
                          값으로 채운다. Kaggle 에 값이 있는 게임은 절대 덮어쓰지 않는다.
──────────────────────────────────────────────────────────────────────────────
"""
import csv
import json
import math
import os
import random
import logging

log = logging.getLogger(__name__)

TRUE_VALUES = {"true", "True", "TRUE", "1"}


# =============================================================================
# 0. 기록용: 단계별 현황표
# =============================================================================
class StepReport:
    """각 단계 뒤의 현황(리뷰·사용자·게임 수)을 표로 모은다."""

    def __init__(self):
        self.rows = []

    def add(self, step, rule, reviews, users, games, note=""):
        prev = self.rows[-1] if self.rows else None
        row = {
            "step": step,
            "rule": rule,
            "reviews": len(reviews) if isinstance(reviews, list) else reviews,
            "users": users,
            "games": games,
            "reviews_removed": "" if prev is None else prev["reviews"] - (len(reviews) if isinstance(reviews, list) else reviews),
            "note": note,
        }
        self.rows.append(row)
        log.info("[%s] %s | reviews=%s users=%s games=%s %s",
                 step, rule, f"{row['reviews']:,}", f"{users:,}", f"{games:,}", note)

    def save(self, path):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: 엑셀에서 한글 깨짐 방지
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)


def count_users_games(reviews):
    users, games = set(), set()
    for r in reviews:
        users.add(r["user_id"])
        games.add(r["app_id"])
    return len(users), len(games)


# =============================================================================
# 1. 파일 읽기 (csv, json 표준 모듈)
# =============================================================================
def read_games(path):
    """games.csv -> {app_id: title}"""
    titles = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            titles[int(row["app_id"])] = row.get("title", "")
    return titles


def read_metadata(path):
    """games_metadata.json (한 줄에 JSON 하나) -> {app_id: {"tags": [...], "description": "..."}}"""
    meta = {}
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read().strip()
    if text.startswith("["):                       # 혹시 JSON 배열 형식이면
        records = json.loads(text)
    else:                                          # 기본: JSON Lines 형식 (한 줄 = 게임 하나)
        # 주의: str.splitlines()는 설명 안의 특수 줄바꿈 문자( , \x85 등)에서도
        # 줄을 끊어 JSON이 깨진다. 그래서 진짜 줄바꿈 "\n" 에서만 나눈다.
        records, bad = [], 0
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
        if bad:
            log.warning("games_metadata.json: %d lines could not be parsed and were skipped", bad)
    for rec in records:
        app_id = int(rec["app_id"])
        tags = rec.get("tags") or []
        desc = rec.get("description") or ""
        if app_id not in meta:                     # 같은 app_id가 또 나오면 첫 번째 것 사용
            meta[app_id] = {"tags": [str(t) for t in tags], "description": str(desc)}
    return meta


def fill_missing_metadata(meta, supplement_path, used_games):
    """기준 [8]: Kaggle 메타데이터가 비어 있는 게임만 보완 파일 값으로 채운다.
    used_games: 실험에 남은 게임(app_id) 집합 -> 이 게임들 기준으로 현황을 센다."""
    stats = {"supplement_file": supplement_path, "empty_tags_before": 0, "tags_filled": 0,
             "empty_desc_before": 0, "desc_filled": 0, "tag_sources": {}}
    for app_id in used_games:
        m = meta.get(app_id, {"tags": [], "description": ""})
        stats["empty_tags_before"] += not m["tags"]
        stats["empty_desc_before"] += not m["description"].strip()
    if not supplement_path or not os.path.exists(supplement_path):
        if supplement_path:
            log.warning("metadata supplement file not found: %s -> using Kaggle metadata only "
                        "(run scripts/fetch_missing_tags.py first)", supplement_path)
        stats["supplement_file"] = "(none)"
        return meta, stats
    with open(supplement_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sup = json.loads(line)
            app_id = int(sup["app_id"])
            if app_id not in used_games:
                continue
            m = meta.setdefault(app_id, {"tags": [], "description": ""})
            if not m["tags"] and sup.get("tags"):                  # 비어 있을 때만 채운다
                m["tags"] = [str(t) for t in sup["tags"]]
                stats["tags_filled"] += 1
                src = sup.get("tag_source", "")
                stats["tag_sources"][src] = stats["tag_sources"].get(src, 0) + 1
            if not m["description"].strip() and sup.get("description"):
                m["description"] = str(sup["description"])
                stats["desc_filled"] += 1
    return meta, stats


def iter_reviews(path, max_rows=None):
    """recommendations.csv 를 한 줄씩 읽어 필요한 5개 열만 dict 로 돌려준다.
    (4,100만 줄 전체를 한 번에 메모리에 올리지 않기 위해 한 줄씩 처리)"""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        i_user, i_app = col["user_id"], col["app_id"]
        i_rec, i_hours = col["is_recommended"], col["hours"]
        i_date = col.get("date")
        for line_no, row in enumerate(reader):
            if max_rows is not None and line_no >= max_rows:
                break
            yield {
                "line": line_no,                                   # 파일 안 순서 (동점 처리용)
                "user_id": int(row[i_user]),
                "app_id": int(row[i_app]),
                "is_recommended": row[i_rec] in TRUE_VALUES,       # 기준 [1]
                "hours": float(row[i_hours]) if row[i_hours] else 0.0,
                "date": row[i_date] if i_date is not None else "",  # "YYYY-MM-DD" 문자열은 그대로 정렬 가능
            }


# =============================================================================
# 2. 전처리 단계
# =============================================================================
def step_count_positive_per_user(rec_path, max_rows):
    """1차 통과: 사용자별 긍정 리뷰 수를 센다 (원본 전체 현황도 함께)."""
    positive_count = {}            # user_id -> 긍정 리뷰 수
    all_users, all_games = set(), set()
    n_reviews = n_positive = 0
    for r in iter_reviews(rec_path, max_rows):
        n_reviews += 1
        all_games.add(r["app_id"])
        all_users.add(r["user_id"])
        if r["is_recommended"]:
            n_positive += 1
            positive_count[r["user_id"]] = positive_count.get(r["user_id"], 0) + 1
    return positive_count, n_reviews, n_positive, len(all_users), len(all_games)


def step_candidate_users(positive_count, min_user_pos):
    """기준 [2]: 긍정 리뷰가 min_user_pos 개 이상인 사용자."""
    candidates = []
    for user_id, n_pos in positive_count.items():
        if n_pos >= min_user_pos:
            candidates.append(user_id)
    candidates.sort()              # 정렬해 두어야 같은 seed에서 항상 같은 샘플이 나온다
    return candidates


def step_sample_users(candidates, max_users, seed):
    """기준 [3]: 후보 중 max_users 명 무작위 추출."""
    if max_users is None or len(candidates) <= max_users:
        return set(candidates)
    rng = random.Random(seed)
    return set(rng.sample(candidates, max_users))


def step_collect_reviews(rec_path, max_rows, sampled_users):
    """2차 통과: 뽑힌 사용자의 리뷰만 모은다."""
    reviews = []
    for r in iter_reviews(rec_path, max_rows):
        if r["user_id"] in sampled_users:
            reviews.append(r)
    return reviews


def step_keep_known_games(reviews, known_games):
    """기준 [4]: games.csv 에 있는 게임의 리뷰만 남긴다."""
    kept = []
    for r in reviews:
        if r["app_id"] in known_games:
            kept.append(r)
    return kept


def step_remove_duplicates(reviews):
    """기준 [5]: 같은 (사용자, 게임) 리뷰는 가장 늦은 것 하나만."""
    latest = {}                    # (user_id, app_id) -> 리뷰
    for r in reviews:
        key = (r["user_id"], r["app_id"])
        if key not in latest:
            latest[key] = r
        else:
            old = latest[key]
            if (r["date"], r["line"]) > (old["date"], old["line"]):
                latest[key] = r
    kept = list(latest.values())
    kept.sort(key=lambda r: r["line"])
    return kept


def step_kcore(reviews, min_user_pos, min_item_pos, report):
    """기준 [6]: 게임/사용자 조건을 번갈아 적용, 더 이상 제거되는 것이 없을 때까지 반복."""
    round_no = 0
    while True:
        round_no += 1
        before = len(reviews)

        # (a) 게임별로 긍정 리뷰를 남긴 사용자 수
        game_pos_users = {}
        for r in reviews:
            if r["is_recommended"]:
                game_pos_users[r["app_id"]] = game_pos_users.get(r["app_id"], 0) + 1
        good_games = set()
        for app_id, n in game_pos_users.items():
            if n >= min_item_pos:
                good_games.add(app_id)
        reviews = [r for r in reviews if r["app_id"] in good_games]

        # (b) 사용자별 (남은 게임 중) 긍정 리뷰 수
        user_pos = {}
        for r in reviews:
            if r["is_recommended"]:
                user_pos[r["user_id"]] = user_pos.get(r["user_id"], 0) + 1
        good_users = set()
        for user_id, n in user_pos.items():
            if n >= min_user_pos:
                good_users.add(user_id)
        reviews = [r for r in reviews if r["user_id"] in good_users]

        u, g = count_users_games(reviews)
        report.add(f"6-{round_no}", f"k-core round {round_no}: game >= {min_item_pos} positive users, "
                   f"user >= {min_user_pos} positive reviews", reviews, u, g)
        if len(reviews) == before:
            return reviews


def step_temporal_split(reviews, test_ratio, val_ratio):
    """기준 [7]: 사용자별 날짜순 정렬 → 최근 test, 그 직전 validation, 나머지 train."""
    by_user = {}
    for r in reviews:
        by_user.setdefault(r["user_id"], []).append(r)

    for user_id, items in by_user.items():
        items.sort(key=lambda r: (r["date"], r["line"]))        # 오래된 것 → 최근 것
        n = len(items)
        n_test = max(1, int(n * test_ratio + 0.5))               # 반올림, 최소 1개
        n_val = max(1, int((n - n_test) * val_ratio + 0.5))
        n_train = n - n_test - n_val
        for idx, r in enumerate(items):
            if idx < n_train:
                r["part"] = "train"
            elif idx < n_train + n_val:
                r["part"] = "val"
            else:
                r["part"] = "test"
    return reviews


def step_random_split(reviews, test_ratio, val_ratio, seed):
    """(민감도 실험용) 날짜 대신 무작위 순서로 같은 비율 분할."""
    rng = random.Random(seed)
    for r in reviews:
        r["date_backup"] = r["date"]
        r["date"] = f"{rng.random():.12f}"
    reviews = step_temporal_split(reviews, test_ratio, val_ratio)
    for r in reviews:
        r["date"] = r.pop("date_backup")
    return reviews


# =============================================================================
# 3. 전체 실행
# =============================================================================
def run(cfg, out_dir):
    """전처리 전체를 실행하고 (리뷰 목록, 게임 제목, 메타데이터, 현황표)를 돌려준다."""
    rec_path = os.path.join(cfg.data_dir, "recommendations.csv")
    report = StepReport()

    # --- 1차 통과: 원본 현황 + 사용자별 긍정 리뷰 수
    positive_count, n_reviews, n_positive, n_users, n_games = \
        step_count_positive_per_user(rec_path, cfg.max_rows)
    report.add("0", "raw recommendations.csv" + (f" (first {cfg.max_rows:,} rows)" if cfg.max_rows else ""),
               n_reviews, n_users, n_games, f"positive reviews = {n_positive:,}")

    # --- [2] 후보 사용자
    candidates = step_candidate_users(positive_count, cfg.min_user_pos)
    report.add("2", f"users with >= {cfg.min_user_pos} positive reviews (candidates)",
               n_reviews, len(candidates), n_games, "only counts candidate users; no review removed yet")
    del positive_count

    # --- [3] 샘플링 + 2차 통과
    sampled = step_sample_users(candidates, cfg.max_users, cfg.seed)
    reviews = step_collect_reviews(rec_path, cfg.max_rows, sampled)
    u, g = count_users_games(reviews)
    if cfg.max_users and len(candidates) > cfg.max_users:
        rule, note = f"random sample of {cfg.max_users:,} candidate users (seed {cfg.seed})", ""
    else:
        rule, note = "keep all candidate users", "no sampling needed (candidates <= max_users)"
    report.add("3", rule, reviews, u, g, note)

    # --- [4] games.csv 확인
    titles = read_games(os.path.join(cfg.data_dir, "games.csv"))
    reviews = step_keep_known_games(reviews, set(titles))
    u, g = count_users_games(reviews)
    report.add("4", "drop games missing from games.csv", reviews, u, g)

    # --- [5] 중복 정리
    reviews = step_remove_duplicates(reviews)
    u, g = count_users_games(reviews)
    report.add("5", "one review per (user, game): keep the latest", reviews, u, g)

    # --- [6] k-core 반복
    reviews = step_kcore(reviews, cfg.min_user_pos, cfg.min_item_pos, report)

    # --- [7] 분할
    if cfg.split == "temporal":
        reviews = step_temporal_split(reviews, cfg.test_ratio, cfg.val_ratio)
    else:
        reviews = step_random_split(reviews, cfg.test_ratio, cfg.val_ratio, cfg.seed)
    for part in ("train", "val", "test"):
        part_rows = [r for r in reviews if r["part"] == part]
        n_pos = sum(1 for r in part_rows if r["is_recommended"])
        u, g = count_users_games(part_rows)
        report.add(f"7-{part}", f"{cfg.split} split: {part}", part_rows, u, g,
                   f"positive = {n_pos:,}")
        report.rows[-1]["reviews_removed"] = ""

    # --- [8] 메타데이터 보완
    meta = read_metadata(os.path.join(cfg.data_dir, "games_metadata.json"))
    used_games = set(r["app_id"] for r in reviews)
    meta, fill = fill_missing_metadata(meta, cfg.supplement_path, used_games)
    u, g = count_users_games(reviews)
    report.add("8", "fill empty tags/description from supplement file (only where Kaggle is empty)",
               reviews, u, g,
               f"tags filled {fill['tags_filled']}/{fill['empty_tags_before']}, "
               f"descriptions filled {fill['desc_filled']}/{fill['empty_desc_before']} "
               f"(file: {fill['supplement_file']})")
    report.rows[-1]["reviews_removed"] = ""
    report.save(os.path.join(out_dir, "preprocessing_report.csv"))
    with open(os.path.join(out_dir, "metadata_fill.json"), "w", encoding="utf-8") as f:
        json.dump(fill, f, indent=2, ensure_ascii=False)
    return reviews, titles, meta, report
