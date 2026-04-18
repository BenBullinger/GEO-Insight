#!/usr/bin/env python3
"""
Facet — reproducible download of the 5 official source datasets.

Populates Data/{hno,hrp,cod-ps,fts,cbpf}/ with files matching the structure
described in Data/README.md. The five official sources listed in sources.txt
are all resolved to their underlying HDX resource files (see the DATASETS
table below for the full mapping).

Usage:
    python3 Data/download.py             # download missing files
    python3 Data/download.py --force     # re-download everything
    python3 Data/download.py --check     # list files and their current size, don't download

No third-party dependencies — standard library only.

Total size on disk after a full run: ~270 MB (CoD-PS admin2 is the largest
single file at ~144 MB).
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).resolve().parent
USER_AGENT = (
    "GEO-Insight-Datathon2026/1.0 "
    "(+https://github.com/BenBullinger/GEO-Insight)"
)
TIMEOUT_S = 120
CHUNK = 65536
MAX_WORKERS = 4

# (category, filename, url)
#
# URLs verified against HDX CKAN API on 2026-04-18. If a download 404s, the
# resource ID may have been re-issued; regenerate via:
#   curl -sL "https://data.humdata.org/api/3/action/package_show?id=<slug>"
# for slugs: global-hpc-hno, humanitarian-response-plans, cod-ps-global,
# global-requirements-and-funding-data, cbpf-allocations-and-contributions.
#
# URLs listed in NON_CRITICAL_URLS below will warn-and-continue on network
# failure rather than aborting the whole run; everything else is required.
DATASETS: list[tuple[str, str, str]] = [
    # --- HNO (source #1: https://data.humdata.org/dataset/global-hpc-hno) ---
    ("hno", "hpc_hno_2026.csv",
     "https://data.humdata.org/dataset/8326ed53-8f3a-47f9-a2aa-83ab4ecee476/resource/edb91329-0e6b-4ebc-b6cb-051b2a11e536/download/hpc_hno_2026.csv"),
    ("hno", "hpc_hno_2025.csv",
     "https://data.humdata.org/dataset/8326ed53-8f3a-47f9-a2aa-83ab4ecee476/resource/22093993-e23b-45c8-b84f-61e4a414ebbb/download/hpc_hno_2025.csv"),
    ("hno", "hpc_hno_2024.csv",
     "https://data.humdata.org/dataset/8326ed53-8f3a-47f9-a2aa-83ab4ecee476/resource/8e3931a5-452b-4583-9d02-2247a34e397b/download/hpc_hno_2024.csv"),

    # --- HRP (source #2: https://data.humdata.org/dataset/humanitarian-response-plans) ---
    ("hrp", "humanitarian-response-plans.csv",
     "https://data.humdata.org/dataset/f0e95437-b6d9-4897-9e3f-84853bcbfaee/resource/d4e67ae7-a910-47c9-b283-304877c7e5eb/download/humanitarian-response-plans.csv"),
    ("hrp", "hpc_tools_plans.json",
     "https://api.hpc.tools/v2/public/plan"),

    # --- CoD-PS (source #3: https://data.humdata.org/dataset/cod-ps-global) ---
    ("cod-ps", "cod_population_admin0.csv",
     "https://data.humdata.org/dataset/27e3d1c6-c57a-4159-85a4-adb6b7aca6b9/resource/d4ea8fba-3d98-4d6e-85c8-84a5b0b4ebd9/download/cod_population_admin0.csv"),
    ("cod-ps", "cod_population_admin1.csv",
     "https://data.humdata.org/dataset/27e3d1c6-c57a-4159-85a4-adb6b7aca6b9/resource/3f7e17af-4ffa-455a-874c-6bf75e031730/download/cod_population_admin1.csv"),
    ("cod-ps", "cod_population_admin2.csv",
     "https://data.humdata.org/dataset/27e3d1c6-c57a-4159-85a4-adb6b7aca6b9/resource/87186331-46fd-4cc2-9be8-bceb715abb91/download/cod_population_admin2.csv"),
    ("cod-ps", "cod_population_admin3.csv",
     "https://data.humdata.org/dataset/27e3d1c6-c57a-4159-85a4-adb6b7aca6b9/resource/3a58aef6-7884-40fc-8a0f-5bca8886e358/download/cod_population_admin3.csv"),
    ("cod-ps", "cod_population_admin4.csv",
     "https://data.humdata.org/dataset/27e3d1c6-c57a-4159-85a4-adb6b7aca6b9/resource/11c76189-42fa-48d2-b794-b89517f0c811/download/cod_population_admin4.csv"),

    # --- FTS (source #4: https://data.humdata.org/dataset/global-requirements-and-funding-data) ---
    ("fts", "fts_requirements_funding_global.csv",
     "https://data.humdata.org/dataset/b2bbb33c-2cfb-4809-8dd3-6bbdc080cbb9/resource/b3232da8-f1e4-41ab-9642-b22dae10a1d7/download/fts_requirements_funding_global.csv"),
    ("fts", "fts_requirements_funding_covid_global.csv",
     "https://data.humdata.org/dataset/b2bbb33c-2cfb-4809-8dd3-6bbdc080cbb9/resource/11219233-1663-4d06-9d44-4b5706ee890b/download/fts_requirements_funding_covid_global.csv"),
    ("fts", "fts_requirements_funding_cluster_global.csv",
     "https://data.humdata.org/dataset/b2bbb33c-2cfb-4809-8dd3-6bbdc080cbb9/resource/80975d5b-508b-47b2-a10c-b967104d3179/download/fts_requirements_funding_cluster_global.csv"),
    ("fts", "fts_requirements_funding_globalcluster_global.csv",
     "https://data.humdata.org/dataset/b2bbb33c-2cfb-4809-8dd3-6bbdc080cbb9/resource/eac18f6a-c0b6-4a84-a23d-6df4ebe5ce03/download/fts_requirements_funding_globalcluster_global.csv"),
    ("fts", "fts_incoming_funding_global.csv",
     "https://data.humdata.org/dataset/b2bbb33c-2cfb-4809-8dd3-6bbdc080cbb9/resource/557eb027-8c66-4741-9ff4-43dab4458b1d/download/fts_incoming_funding_global.csv"),
    ("fts", "fts_internal_funding_global.csv",
     "https://data.humdata.org/dataset/b2bbb33c-2cfb-4809-8dd3-6bbdc080cbb9/resource/4b793b71-02a1-420a-9910-ce86d7b79b6c/download/fts_internal_funding_global.csv"),
    ("fts", "fts_outgoing_funding_global.csv",
     "https://data.humdata.org/dataset/b2bbb33c-2cfb-4809-8dd3-6bbdc080cbb9/resource/f5ea63a4-f299-4e8c-80a3-21cf359300da/download/fts_outgoing_funding_global.csv"),

    # --- CBPF (source #5: https://cbpf.data.unocha.org) ---
    # The portal page is a static dashboard; its underlying datasets live on HDX
    # as cbpf-allocations-and-contributions. CSV mirrors are hosted by the
    # OCHA CBPF team on Google Sheets; JSON APIs are at cbpfapi.unocha.org.
    ("cbpf", "cbpf_project_summary.csv",
     "https://docs.google.com/spreadsheets/d/e/2PACX-1vRyEbNqi7QufuCwGCgbcdWCC3O7dFzwoZPm6tjUJ4RAI0ah12nTZLr5Gdaz-l44bTTOcIg9l2LP3GK_/pub?gid=0&single=true&output=csv"),
    ("cbpf", "cbpf_contributions.csv",
     "https://docs.google.com/spreadsheets/d/e/2PACX-1vRyEbNqi7QufuCwGCgbcdWCC3O7dFzwoZPm6tjUJ4RAI0ah12nTZLr5Gdaz-l44bTTOcIg9l2LP3GK_/pub?gid=1866794021&single=true&output=csv"),
]

# URLs allowed to fail without aborting the run. These are supplementary
# resources whose failure does not break the methodology (the core HRP data
# is already in humanitarian-response-plans.csv; the live API snapshot is a
# nice-to-have that some networks block with 406 Not Acceptable).
NON_CRITICAL_URLS: set[str] = {
    "http://api.hpc.tools/v2/public/plan",
}

# Per-host Accept overrides. The HPC Tools API returns 406 when no explicit
# Accept is sent; default urllib doesn't set one. Using `application/json`
# unblocks teammates whose network strips unusual header combinations.
EXTRA_HEADERS: dict[str, dict[str, str]] = {
    "api.hpc.tools": {"Accept": "application/json"},
}


def format_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:6.1f} {unit}"
        n /= 1024
    return f"{n:6.1f} TB"


def download_one(category: str, filename: str, url: str, force: bool) -> tuple[str, str, int, float]:
    dest = DATA_DIR / category / filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not force and dest.exists() and dest.stat().st_size > 0:
        return (f"{category}/{filename}", "skipped", dest.stat().st_size, 0.0)

    tmp = dest.with_suffix(dest.suffix + ".partial")
    t0 = time.monotonic()

    # Build headers — always send UA; layer on host-specific overrides.
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    try:
        host = url.split("//", 1)[1].split("/", 1)[0].lower()
        if host in EXTRA_HEADERS:
            headers.update(EXTRA_HEADERS[host])
    except Exception:
        pass

    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=TIMEOUT_S) as resp, open(tmp, "wb") as f:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
        tmp.replace(dest)
        return (f"{category}/{filename}", "ok", dest.stat().st_size, time.monotonic() - t0)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        status = "warn" if url in NON_CRITICAL_URLS else "error"
        return (f"{category}/{filename}", f"{status}: {exc}", 0, time.monotonic() - t0)


def cmd_check() -> int:
    print(f"Checking {DATA_DIR}")
    total = 0
    missing = 0
    for category, filename, _ in DATASETS:
        p = DATA_DIR / category / filename
        if p.exists():
            size = p.stat().st_size
            total += size
            print(f"  {'present':<8} {format_size(size)}  {category}/{filename}")
        else:
            missing += 1
            print(f"  {'MISSING':<8} {'       ':<11}  {category}/{filename}")
    print(f"\nTotal on disk: {format_size(total)}"
          + (f"    Missing files: {missing}" if missing else ""))
    return 1 if missing else 0


def cmd_download(force: bool) -> int:
    print(f"Downloading to {DATA_DIR}  (workers={MAX_WORKERS}, force={force})")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(download_one, cat, fn, url, force) for cat, fn, url in DATASETS]
        ok_bytes = 0
        errors: list[str] = []
        warnings: list[str] = []
        for fut in as_completed(futures):
            name, status, size, dt = fut.result()
            rate = f"{format_size(size/dt)}/s" if dt > 0.1 else ""
            label = status.split(":", 1)[0] if ":" in status else status
            print(f"  {label:<8} {format_size(size)}  {name:<60} {rate}")
            if status == "ok":
                ok_bytes += size
            elif status.startswith("error"):
                errors.append(f"{name} ({status})")
            elif status.startswith("warn"):
                warnings.append(f"{name} ({status})")
    print(f"\nDownloaded {format_size(ok_bytes)} this run.")
    if warnings:
        print(f"Skipped {len(warnings)} non-critical file(s): {', '.join(warnings)}")
        print("These are supplementary and don't block the methodology.")
    if errors:
        print(f"\nErrors in {len(errors)} file(s): {', '.join(errors)}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if file already exists.")
    parser.add_argument("--check", action="store_true",
                        help="Report which files are present; do not download.")
    args = parser.parse_args()
    if args.check:
        return cmd_check()
    return cmd_download(args.force)


if __name__ == "__main__":
    sys.exit(main())
