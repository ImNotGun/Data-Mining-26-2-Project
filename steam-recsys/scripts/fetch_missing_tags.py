"""
태그·설명이 비어 있는 게임의 메타데이터를 공개 API에서 받아 보완 파일을 만든다.

배경
----
Kaggle games_metadata.json 에는 행은 있지만 tags 와 description 이 비어 있는 게임이 있다.
실험 subset 에서는 505개였고, 하필 가장 인기 있는 게임들(Team Fortress 2, The Witcher 3 ...)이라
긍정 리뷰의 약 45%가 태그 없는 게임에 몰려 있었다. 이 스크립트는 그 게임들만 골라 채운다.

출처 (둘 다 로그인·키 불필요)
-----------------------------
  태그   SteamSpy   https://steamspy.com/api.php?request=appdetails&appid=<id>
                    "tags": {"태그": 투표수, ...} -> 투표수 순으로 상위 MAX_TAGS 개 사용
                    제한: 상세 조회 초당 1회 -> 요청 사이 STEAMSPY_SLEEP 초 대기
  설명   Steam 스토어 https://store.steampowered.com/api/appdetails?appids=<id>&l=english
                    "short_description" 사용, "genres" 는 SteamSpy 가 실패했을 때 태그 대용
                    제한: 약 5분에 200회 -> 요청 사이 STORE_SLEEP 초 대기

결과
----
JSON Lines 파일 (한 줄 = 게임 하나):
  {"app_id", "tags", "description", "tag_source", "description_source", "fetched_utc"}
게임 하나를 받을 때마다 바로 파일에 추가하므로, 중간에 끊겨도 다시 실행하면
이미 받은 게임은 건너뛰고 이어서 받는다.

사용법
------
  python scripts/fetch_missing_tags.py \
      --missing results_v2/missing_metadata.csv \
      --out data/supplement/supplement_metadata.jsonl
"""
import argparse
import csv
import datetime
import json
import os
import time
import urllib.error
import urllib.request

MAX_TAGS = 20             # Kaggle 메타데이터의 태그 수와 비슷하게 상위 20개
STEAMSPY_SLEEP = 1.1      # SteamSpy appdetails: 초당 1회
STORE_SLEEP = 1.6         # Steam 스토어: 5분에 약 200회 (1.5초에 1회)
MAX_RETRIES = 4
HEADERS = {"User-Agent": "SCE331-course-project/1.0 (academic use)"}


def get_json(url, sleep_after):
    """URL 에서 JSON 을 받는다. 429/5xx 는 기다렸다가 재시도. 실패하면 (None, 이유)."""
    reason = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            time.sleep(sleep_after)
            try:
                return json.loads(body), ""
            except json.JSONDecodeError:
                return None, "response was not JSON (blocked or error page)"
        except urllib.error.HTTPError as e:
            reason = f"HTTP {e.code}"
            if e.code == 429 or e.code >= 500:
                wait = 10 * attempt
                print(f"    {reason} -> {wait}s 기다린 뒤 재시도 ({attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            time.sleep(sleep_after)
            return None, reason
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            reason = f"network error: {e}"
            time.sleep(5 * attempt)
    return None, reason


def fetch_steamspy_tags(app_id, base):
    data, reason = get_json(f"{base}?request=appdetails&appid={app_id}", STEAMSPY_SLEEP)
    if not data:
        return [], reason
    tags = data.get("tags")
    if not isinstance(tags, dict) or not tags:        # 태그가 없으면 SteamSpy 는 [] 를 준다
        return [], "no tags on SteamSpy"
    ranked = sorted(tags.items(), key=lambda kv: (-int(kv[1]), kv[0]))   # 투표수 많은 순
    return [name for name, _ in ranked[:MAX_TAGS]], ""


def fetch_store_details(app_id, base):
    data, reason = get_json(f"{base}?appids={app_id}&l=english", STORE_SLEEP)
    if not data:
        return "", [], reason
    entry = data.get(str(app_id), {})
    if not entry.get("success"):
        return "", [], "store says success=false (delisted or region-locked)"
    info = entry.get("data", {})
    desc = (info.get("short_description") or "").strip()
    genres = [g.get("description", "") for g in info.get("genres", []) if g.get("description")]
    return desc, genres, ""


def read_missing(path):
    """missing_metadata.csv 에서 태그 또는 설명이 비어 있는 게임 id (리뷰 많은 순)."""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["no_tags"] == "True" or row["no_description"] == "True":
                rows.append((int(row["positive_reviews"]), int(row["app_id"]), row["title"]))
    rows.sort(reverse=True)
    return [(app_id, title) for _, app_id, title in rows]


def already_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["app_id"])
    return done


def main(a):
    todo = read_missing(a.missing)
    done = already_done(a.out)
    todo = [(i, t) for i, t in todo if i not in done]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    per_game = STEAMSPY_SLEEP + (0 if a.no_description else STORE_SLEEP)
    print(f"보완 대상 {len(todo) + len(done)}개 중 이미 받은 것 {len(done)}개, 남은 것 {len(todo)}개")
    print(f"예상 소요 시간: 약 {len(todo) * per_game / 60:.0f}분")

    n_tag_ok = n_desc_ok = 0
    failures = []
    with open(a.out, "a", encoding="utf-8") as out:
        for k, (app_id, title) in enumerate(todo, 1):
            tags, tag_reason = fetch_steamspy_tags(app_id, a.steamspy_url)
            tag_source = "steamspy" if tags else ""
            desc, genres, desc_reason = ("", [], "skipped")
            if not a.no_description:
                desc, genres, desc_reason = fetch_store_details(app_id, a.store_url)
            if not tags and genres:                    # SteamSpy 실패 -> 스토어 장르로 대신
                tags, tag_source = genres, "steam_store_genres"
            record = {"app_id": app_id, "tags": tags, "description": desc,
                      "tag_source": tag_source,
                      "description_source": "steam_store" if desc else "",
                      "fetched_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            n_tag_ok += bool(tags)
            n_desc_ok += bool(desc)
            if not tags:
                failures.append((app_id, title, tag_reason))
            if k % 25 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)}  태그 성공 {n_tag_ok}, 설명 성공 {n_desc_ok}  (마지막: {title})")

    print(f"\n완료: 태그 {n_tag_ok}/{len(todo)}, 설명 {n_desc_ok}/{len(todo)} (이번 실행 기준)")
    if failures:
        print(f"태그를 못 받은 게임 {len(failures)}개 (앞 10개):")
        for app_id, title, reason in failures[:10]:
            print(f"  {app_id} {title}: {reason}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--missing", default="results_v2/missing_metadata.csv")
    p.add_argument("--out", default="data/supplement/supplement_metadata.jsonl")
    p.add_argument("--no-description", action="store_true", help="태그만 받기 (스토어 API 호출 생략, 약 2.5배 빠름)")
    p.add_argument("--steamspy-url", default="https://steamspy.com/api.php")
    p.add_argument("--store-url", default="https://store.steampowered.com/api/appdetails")
    main(p.parse_args())
