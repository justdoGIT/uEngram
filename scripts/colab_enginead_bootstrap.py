"""Runs ON the Colab VM: download EngineAD v2.2 (md5/size-verified against /content/manifest.json), then run the
committed driver /content/colab_run_study.py. Progress goes to /content/bootstrap.log; the driver logs to /content/colab_study.log."""
import hashlib, json, os, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path("/content/enginead/2.2"); DATA.mkdir(parents=True, exist_ok=True)
files = json.loads(Path("/content/manifest.json").read_text())["files"]
t0 = time.time()

def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

def fetch(rec):
    dest = DATA / rec["name"]
    for attempt in range(1, 6):
        try:
            req = urllib.request.Request(f"https://borealisdata.ca/api/access/datafile/{rec['id']}", headers={"User-Agent": "Mozilla/5.0"})
            md5 = hashlib.md5(); n = 0
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest.with_suffix(".part"), "wb") as out:
                while chunk := resp.read(8 * 1024 * 1024):
                    out.write(chunk); md5.update(chunk); n += len(chunk)
            if n != rec["size_bytes"] or md5.hexdigest() != rec["md5_expected"]:
                raise ValueError(f"size/md5 mismatch: got {n} bytes md5 {md5.hexdigest()}")
            os.replace(dest.with_suffix(".part"), dest)
            log(f"ok {rec['name']} {n/1e6:.0f} MB"); return rec["name"]
        except Exception as exc:
            log(f"retry {attempt} {rec['name']}: {exc}"); time.sleep(10 * attempt)
    raise SystemExit(f"download failed: {rec['name']}")

with ThreadPoolExecutor(4) as pool:
    done = list(pool.map(fetch, files))
log(f"downloaded and verified {len(done)} files, {sum(r['size_bytes'] for r in files)/1e9:.2f} GB")
with open("/content/colab_study.log", "w") as out:
    rc = subprocess.run([sys.executable, "-u", "/content/colab_run_study.py"], cwd="/content", stdout=out, stderr=subprocess.STDOUT).returncode
log(f"driver exited with code {rc}")
