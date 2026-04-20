#!/usr/bin/env python3
"""
Download all INFORM Severity monthly snapshots from the EU JRC DRMKC site.

Source page: https://drmkc.jrc.ec.europa.eu/inform-index/INFORM-Severity/Results-and-data
Writes:      Data/Third-Party/DRMKC-INFORM/snapshots/*.xlsx  (~130 MB, 67 files)

Use `consolidate.py` afterwards to collapse them into a tidy long-format CSV.

Usage:
    python3 Data/Third-Party/DRMKC-INFORM/download.py           # download missing
    python3 Data/Third-Party/DRMKC-INFORM/download.py --force   # re-download all
"""
from __future__ import annotations

import argparse
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "snapshots"
BASE = "https://drmkc.jrc.ec.europa.eu"
INDEX = f"{BASE}/inform-index/INFORM-Severity/Results-and-data"

UA = "Mozilla/5.0 (GEO-Insight-Datathon2026)"
POLITE_DELAY_S = 1.2
MIN_XLSX_BYTES = 10_000


def fetch(url: str, retries: int = 4) -> bytes | None:
    """GET with polite retries; None on 404, raises on repeated failure."""
    enc = urllib.parse.quote(url, safe="/?:&=#%")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(enc, headers={"User-Agent": UA, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, socket.error, ConnectionResetError, TimeoutError):
            if attempt == retries - 1:
                raise
        time.sleep(2 * (attempt + 1))
    return None


def scrape_xlsx_urls() -> list[str]:
    """Scrape the DRMKC page for all .xlsx download URLs."""
    html = fetch(INDEX)
    if not html:
        raise RuntimeError(f"could not fetch index page {INDEX}")
    urls = re.findall(r'href="([^"]*\.xlsx)"', html.decode("utf-8", errors="ignore"), re.IGNORECASE)
    return list(dict.fromkeys(urls))  # de-dup preserving order


def clean_name(url: str) -> str:
    """Produce a stable local filename YYYYMM_INFORM_Severity[_variant].xlsx."""
    months = {n: i for i, n in enumerate(
        "january february march april may june july august september october november december".split(), 1
    )}
    m = re.search(r"_(" + "|".join(months) + r")[\s_\-]+(\d{4})", url, re.IGNORECASE)
    if not m:
        return url.rsplit("/", 1)[-1]
    month = months[m.group(1).lower()]
    year = int(m.group(2))
    suffix = ""
    lower = url.lower()
    if "mid_month" in lower or "mid_december" in lower or "_mid_" in lower:
        suffix = "_mid"
    elif "late" in lower:
        suffix = "_late"
    return f"{year}{month:02d}_INFORM_Severity{suffix}.xlsx"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--force", action="store_true", help="Re-download even if file exists.")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    urls = scrape_xlsx_urls()
    print(f"Index page lists {len(urls)} xlsx files.")

    ok = skipped = failed = 0
    for u in urls:
        name = clean_name(u)
        dest = OUT / name
        if not args.force and dest.exists() and dest.stat().st_size > MIN_XLSX_BYTES:
            skipped += 1
            continue
        try:
            data = fetch(BASE + u)
        except Exception as e:
            print(f"  FAIL  {name}  ({e})")
            failed += 1
            time.sleep(POLITE_DELAY_S)
            continue
        if data and len(data) > MIN_XLSX_BYTES:
            with open(dest, "wb") as f:
                f.write(data)
            print(f"  ok    {name}  ({len(data)/1024:.0f} KB)")
            ok += 1
        else:
            print(f"  FAIL  {name}  (empty/too small)")
            failed += 1
        time.sleep(POLITE_DELAY_S)

    print(f"\nDownloaded {ok}, skipped {skipped}, failed {failed}.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
