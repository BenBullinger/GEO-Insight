"""Generate or refresh the per-country LLM explanations embedded in globe.js.

Usage:
    python analysis/llm_explain.py           # prints JSON to stdout
    python analysis/llm_explain.py --write   # patches globe.js in-place

Requires ANTHROPIC_API_KEY in the environment.  The script reads the enriched
frame to ground each explanation in actual computed values rather than generic
knowledge, then calls Claude to produce a 2-sentence paragraph per country.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import anthropic
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))           # exposes `analysis` as a package
sys.path.insert(0, str(ROOT / "analysis"))  # exposes `features`, `ontology`, etc. directly
sys.path.insert(0, str(ROOT / "dashboard"))

import features  # noqa: E402


def _load_countries_from_globe() -> list[dict]:
    """Parse the OVERLOOKED array from globe.js — single source of truth."""
    globe_js = (ROOT / "landing" / "globe.js").read_text()
    pattern = re.compile(
        r'\{\s*iso:\s*"(?P<iso>[A-Z]{3})"'
        r'.*?name:\s*"(?P<name>[^"]+)"'
        r'.*?rank:\s*(?P<rank>[\d.]+)'
        r'(?:.*?ci_width:\s*(?P<ci_width>[\d.]+))?'
        r'.*?type:\s*"(?P<type>[^"]+)"'
        r'.*?cerf:\s*(?P<cerf>true|false)',
        re.DOTALL,
    )
    countries = []
    for m in pattern.finditer(globe_js):
        countries.append({
            "iso":      m.group("iso"),
            "name":     m.group("name"),
            "rank":     float(m.group("rank")),
            "ci_width": float(m.group("ci_width")) if m.group("ci_width") else None,
            "type":     m.group("type"),
            "cerf":     m.group("cerf") == "true",
        })
    if not countries:
        raise ValueError("Could not parse OVERLOOKED array from globe.js")
    return countries


COUNTRIES = _load_countries_from_globe()

# CI width range across the full HRP-eligible pool for context
CI_WIDTH_MIN = 0.30
CI_WIDTH_MAX = 0.85

SYSTEM_PROMPT = """\
You are a humanitarian-data analyst writing concise, factual copy for a
research tool called Geo-Insight. Geo-Insight ranks humanitarian crises by
how overlooked they are relative to their documented need, using a Bayesian
hierarchical model across HRP-eligible countries (those with an active
humanitarian response plan).

Key concepts used in the tool:
- Typology: each crisis is classified along two axes:
    * consensus (narrow credible interval) vs. contested (wide credible interval)
    * overlooked (even underfunding across sectors) vs. sector-starved (uneven sector funding)
- CI width is the 90% credible interval width on the latent overlooked score,
  ranging from 0.30 (very certain) to 0.85 (very uncertain) across the full pool.
  Values above 0.52 indicate meaningful model uncertainty about the country's rank.
- Rank is the posterior-median position among all HRP-eligible countries (1 = most overlooked).
- CERF UFE = OCHA's Underfunded Emergencies list, an independent human-curated benchmark.

Rules:
- If the country is NOT on CERF UFE, do NOT speculate about why CERF excluded it. Instead,
  flag the discrepancy: note that the model ranks this crisis as severely overlooked yet it
  is absent from CERF UFE, framing this as an open question worth investigating.
- Write 2 sentences, max 45 words total. Choose at most one or two numbers — pick the most
  telling signal, not all of them. Lead with the human meaning, not the metric.
- Do NOT start with the country name. Do NOT use bullet points. Plain prose only.\
"""


def _row_context(iso: str, enriched) -> str:
    if enriched is None or iso not in enriched.index:
        return ""
    row = enriched.loc[iso]
    parts = []
    # Column names valid for the Bayesian enriched frame
    for col in ["coverage_shortfall", "per_pin_gap", "need_intensity",
                "severity_category", "donor_hhi", "cluster_gini",
                "posterior_median", "ci_lower", "ci_upper"]:
        if col in row and pd.notna(row[col]):
            parts.append(f"{col}={row[col]:.3f}")
    return ", ".join(parts)


def generate_explanation(client: anthropic.Anthropic, country: dict, enriched,
                         max_retries: int = 3) -> str:
    ctx = _row_context(country["iso"], enriched)
    cerf_status = "on CERF UFE" if country["cerf"] else "NOT on CERF UFE"

    ci_line = ""
    if country.get("ci_width") is not None:
        ci_relative = (country["ci_width"] - CI_WIDTH_MIN) / (CI_WIDTH_MAX - CI_WIDTH_MIN)
        ci_label = "high" if ci_relative > 0.5 else "moderate" if ci_relative > 0.25 else "low"
        ci_line = (
            f"Model uncertainty (90% CI width): {country['ci_width']} — {ci_label} uncertainty "
            f"relative to the full pool range of {CI_WIDTH_MIN}–{CI_WIDTH_MAX}\n"
        )

    user_msg = (
        f"Country: {country['name']} ({country['iso']})\n"
        f"Posterior-median rank: {country['rank']} (1 = most overlooked among HRP-eligible countries)\n"
        + ci_line
        + f"Typology: {country['type']}\n"
        f"CERF UFE status: {cerf_status}\n"
        + (f"Computed values: {ctx}\n" if ctx else "")
        + "\nWrite the explanation paragraph."
    )

    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_msg}],
            )
            text = msg.content[0].text.strip()
            return text.replace("\\$", "$")
        except anthropic.RateLimitError:
            if attempt < max_retries - 1:
                wait = 2 ** attempt * 5
                print(f" rate-limited, retrying in {wait}s…", end=" ", flush=True, file=sys.stderr)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="Patch the explain fields directly into globe.js")
    parser.add_argument("--iso", nargs="+", metavar="ISO3",
                        help="Regenerate only these countries (e.g. --iso GTM NER)")
    args = parser.parse_args()

    client = anthropic.Anthropic()

    print("Loading enriched frame…", file=sys.stderr)
    try:
        enriched = features.load_cached_enriched_frame()
        if enriched is None:
            enriched = features.build_enriched_frame()
    except Exception as exc:
        print(f"Warning: could not load enriched frame ({exc}); proceeding without it.",
              file=sys.stderr)
        enriched = None

    targets = [c for c in COUNTRIES if not args.iso or c["iso"] in args.iso]

    results: dict[str, str] = {}
    for country in targets:
        print(f"  {country['iso']}…", end=" ", flush=True, file=sys.stderr)
        results[country["iso"]] = generate_explanation(client, country, enriched)
        print("done", file=sys.stderr)

    if not args.write:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    globe_path = ROOT / "landing" / "globe.js"
    source = globe_path.read_text()

    lines = source.splitlines()
    for iso, text in results.items():
        escaped = json.dumps(text)
        n = 0
        for i, line in enumerate(lines):
            if f'iso: "{iso}"' not in line:
                continue
            if "explain:" in line:
                lines[i] = re.sub(r'explain:\s*"[^"]*"', lambda _, s=escaped: f"explain: {s}", line)
            else:
                lines[i] = line.rstrip()
                if lines[i].endswith("},"):
                    lines[i] = lines[i][:-2] + f", explain: {escaped}" + " },"
                elif lines[i].endswith("}"):
                    lines[i] = lines[i][:-1] + f", explain: {escaped}" + " }"
            n += 1
        if n == 0:
            print(f"Warning: could not find entry for {iso}", file=sys.stderr)
    source = "\n".join(lines) + "\n"

    globe_path.write_text(source)
    print(f"Patched {len(results)} explanations into {globe_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
