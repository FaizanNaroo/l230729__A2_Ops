"""Task 8 evidence: send several test images through the Flask app (which forwards
them to the model API on Vercel), then read the database trail back.

    python task8_demo.py                   # 6 images via Flask -> Vercel
    python task8_demo.py -n 10             # more images
    python task8_demo.py --direct          # call the Vercel API directly (Task 7 check)

Flask must be running (python app.py or docker compose up) with INFERENCE_MODE=remote.
Results are written to reports/task8_results.csv.
"""
import argparse
import csv
import os
import random
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
LABELS = {"0": "Real", "1": "AI-generated"}


def pick_images(n, seed=42):
    """Half real, half AI images from the held-out test split."""
    with open("data/splits/test.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if Path(r["path"]).is_file()]
    rng = random.Random(seed)
    by_class = {k: [r for r in rows if r["label"] == k] for k in LABELS}
    picked = []
    for k, items in by_class.items():
        picked += rng.sample(items, min(len(items), (n + 1) // 2 if k == "1" else n // 2))
    return picked


def via_flask(base, path):
    with open(path, "rb") as f:
        r = requests.post(f"{base}/predict", files={"image": (Path(path).name, f)},
                          allow_redirects=False, timeout=90)
    if r.status_code != 302:
        return None, f"HTTP {r.status_code}"
    pred_id = int(r.headers["Location"].rstrip("/").rsplit("/", 1)[-1])
    rec = next((h for h in requests.get(f"{base}/api/history", params={"limit": 100},
                                        timeout=30).json() if h["id"] == pred_id), None)
    return rec, None


def via_api(api_url, api_key, path):
    with open(path, "rb") as f:
        r = requests.post(f"{api_url.rstrip('/')}/predict", headers={"X-API-Key": api_key},
                          files={"image": (Path(path).name, f)}, timeout=90)
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}: {r.text[:120]}"
    return r.json(), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=6, help="number of test images")
    ap.add_argument("--flask", default=os.getenv("FLASK_URL", "http://127.0.0.1:5000"))
    ap.add_argument("--direct", action="store_true", help="call the Vercel API directly")
    args = ap.parse_args()

    target = os.getenv("API_URL") if args.direct else args.flask
    print(f"Target: {target}  ({'Vercel API directly' if args.direct else 'Flask -> Vercel API'})")
    if not args.direct:
        print("Flask health:", requests.get(f"{args.flask}/health", timeout=60).json())

    out, correct = [], 0
    print(f"\n{'#':>2}  {'true':<13}{'predicted':<14}{'P(AI)':>7}{'conf':>7}{'ms':>7}  file")
    for i, row in enumerate(pick_images(args.n), 1):
        t0 = time.perf_counter()
        if args.direct:
            rec, err = via_api(os.environ["API_URL"], os.environ["API_KEY"], row["path"])
        else:
            rec, err = via_flask(args.flask, row["path"])
        ms = int((time.perf_counter() - t0) * 1000)
        true = LABELS[row["label"]]
        name = Path(row["path"]).name[:45]
        if err:
            print(f"{i:>2}  {true:<13}ERROR {err}  {name}")
            continue
        correct += rec["label"] == true
        print(f"{i:>2}  {true:<13}{rec['label']:<14}{rec['prob_ai']:>7.3f}"
              f"{rec['confidence']:>7.2f}{ms:>7}  {name}")
        out.append({"file": row["path"], "true_label": true, "predicted": rec["label"],
                    "prob_ai": rec["prob_ai"], "confidence": rec["confidence"],
                    "uncertain": rec["uncertain"], "threshold": rec["threshold"],
                    "model": rec["model"], "round_trip_ms": ms,
                    "db_id": rec.get("id", "")})

    if out:
        print(f"\n{correct}/{len(out)} correct")
        Path("reports").mkdir(exist_ok=True)
        with open("reports/task8_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        print("Saved reports/task8_results.csv")

    if not args.direct:
        print("\nLatest database records (GET /api/history):")
        for h in requests.get(f"{args.flask}/api/history", params={"limit": 5}, timeout=30).json():
            print(f"  #{h['id']:<4} {h['created_at'][:19]}  {h['label']:<13} "
                  f"conf={h['confidence']:.2f}  {h['model']}  {h['filename'][:30]}")


if __name__ == "__main__":
    main()
