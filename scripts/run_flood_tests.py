"""Run the Locust flood test against 1, 2 and 3 API containers and write results/flood_results.md.

Needs: Docker Desktop running, `pip install locust`, and models/heart_cnn.keras in place.
Usage (from the repo root):
    python scripts/run_flood_tests.py                      # 1, 2, 3 containers, 40 users, 60 s each
    python scripts/run_flood_tests.py --counts 1 3 --users 80 --time 90s
"""
import argparse
import csv
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = ["docker", "compose", "-f", "docker-compose.scale.yml"]
URL = "http://localhost:8080"


def run(cmd, **kw):
    print("$", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT, check=kw.pop("check", True), **kw)


def wait_until_ready(timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{URL}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def read_row(csv_path, name):
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["Name"] == name:
                return row
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--users", type=int, default=40)
    ap.add_argument("--rate", type=int, default=10)
    ap.add_argument("--time", default="60s")
    ap.add_argument("--table-only", action="store_true",
                    help="do not run anything; build the table from the existing results/flood_N_containers_stats.csv files")
    args = ap.parse_args()

    (ROOT / "results").mkdir(exist_ok=True)
    table = []
    for n in args.counts:
        if args.table_only:
            prefix = f"results/flood_{n}_containers"
            agg = read_row(ROOT / f"{prefix}_stats.csv", "Aggregated")
            pred = read_row(ROOT / f"{prefix}_stats.csv", "/predict")
            if agg:
                table.append((n, agg, pred))
            continue
        print(f"\n===== {n} API container(s) =====")
        run(COMPOSE + ["up", "-d", "--build", "--scale", f"api={n}"])
        if not wait_until_ready():
            print("The API did not become ready in time; skipping this run.")
            run(COMPOSE + ["down"], check=False)
            continue
        time.sleep(10)                                        # let every container finish loading the model
        prefix = f"results/flood_{n}_containers"
        run([sys.executable, "-m", "locust", "-f", "locustfile.py", "--host", URL, "--headless",
             "-u", str(args.users), "-r", str(args.rate), "-t", args.time, "--csv", prefix, "--only-summary"],
            check=False)                                      # Locust exits with code 1 when some requests failed; that is a result, not a crash
        agg = read_row(ROOT / f"{prefix}_stats.csv", "Aggregated")
        pred = read_row(ROOT / f"{prefix}_stats.csv", "/predict")
        if agg:
            table.append((n, agg, pred))
        run(COMPOSE + ["down"], check=False)

    lines = [f"Flood test: {args.users} simulated users, spawn rate {args.rate}/s, {args.time} per run, "
             "requests go through nginx to the API containers.\n",
             "| Containers | Requests | Failures | Requests/s | Median (ms) | Average (ms) | 95th percentile (ms) | /predict average (ms) |",
             "|---|---|---|---|---|---|---|---|"]
    for n, a, p in table:
        lines.append(f"| {n} | {a['Request Count']} | {a['Failure Count']} | {float(a['Requests/s']):.1f} | "
                     f"{a['Median Response Time']} | {float(a['Average Response Time']):.0f} | {a['95%']} | "
                     f"{float(p['Average Response Time']):.0f} |" if p else
                     f"| {n} | {a['Request Count']} | {a['Failure Count']} | {float(a['Requests/s']):.1f} | "
                     f"{a['Median Response Time']} | {float(a['Average Response Time']):.0f} | {a['95%']} | n/a |")
    out = ROOT / "results" / "flood_results.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()