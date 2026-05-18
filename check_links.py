#!/usr/bin/env python3
"""
check_links.py — Dead-link checker for the Langmead Lab Quarto site.

Scans .qmd files for every hyperlink, then checks each URL concurrently
using per-domain rate limiting so nothing gets throttled or blacklisted.

Strategy
--------
* HEAD first (no body transfer, fast).  Falls back to GET if the server
  rejects HEAD with 405.
* Global semaphore caps total open connections (default: 20).
* Per-domain semaphores cap per-site concurrency (default: 2; 1 for
  known rate-sensitive domains such as YouTube and Google Scholar).
* Known bot-blocking domains (LinkedIn, Twitter/X) are skipped rather
  than falsely reported as dead.
* SSL verification is disabled so a dodgy cert doesn't mask a live link.

Requirements
------------
    pip install aiohttp
    # or: pip install -r requirements-linkcheck.txt

Usage
-----
    python check_links.py                   # all *.qmd in current dir
    python check_links.py teaching.qmd      # one file
    python check_links.py --verbose         # show every result
    python check_links.py --concurrency 10  # lower global limit
    python check_links.py --timeout 20      # longer per-request timeout

Exit code: 0 = all clear, 1 = one or more dead links found.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

try:
    import aiohttp
except ImportError:
    sys.exit("aiohttp is required.  Install with: pip install aiohttp")

# ── Configuration ──────────────────────────────────────────────────────────────

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Domains that block automated requests regardless of headers.
# Links to these are reported as "skipped" rather than dead.
BOT_BLOCKED: frozenset[str] = frozenset({
    "twitter.com",
    "x.com",
    "linkedin.com",
    "www.linkedin.com",
})

# Rate-sensitive domains: limit to 1 concurrent request each.
CAUTIOUS: frozenset[str] = frozenset({
    "youtube.com",
    "www.youtube.com",
    "youtu.be",
    "scholar.google.com",
    "coursera.org",
    "www.coursera.org",
    "bsky.app",
    "genomic.social",
    "speakerdeck.com",
    "nbviewer.jupyter.org",
    "colab.research.google.com",
})

DEFAULT_PER_DOMAIN = 2   # simultaneous requests to most hosts
CAUTIOUS_PER_DOMAIN = 1  # simultaneous requests to cautious hosts


# ── URL extraction ─────────────────────────────────────────────────────────────

# [text](url) and [text](url "title")
_RE_INLINE = re.compile(r'\[[^\[\]]*\]\((\S+?)(?:\s+"[^"]*")?\)')
# ![alt](url)
_RE_IMAGE  = re.compile(r'!\[[^\[\]]*\]\((\S+?)(?:\s+"[^"]*")?\)')
# [id]: url  (reference-style definitions, e.g. _links.qmd)
_RE_REFDEF = re.compile(r'^\s*\[[^\[\]]+\]:\s*(\S+)', re.MULTILINE)


def extract_links(path: Path) -> list[tuple[str, int]]:
    """Return [(url, line_number), ...] for every http(s) link in *path*."""
    text = path.read_text(encoding="utf-8")
    found: list[tuple[str, int]] = []

    # Inline and image links — line-by-line for accurate line numbers
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pat in (_RE_INLINE, _RE_IMAGE):
            for m in pat.finditer(line):
                url = m.group(1)
                if url.startswith(("http://", "https://")):
                    found.append((url, lineno))

    # Reference-style definitions span the whole file
    for m in _RE_REFDEF.finditer(text):
        url = m.group(1)
        if url.startswith(("http://", "https://")):
            lineno = text[: m.start()].count("\n") + 1
            found.append((url, lineno))

    return found


# ── Per-URL HTTP check ─────────────────────────────────────────────────────────

def _host(url: str) -> str:
    return urlparse(url).netloc.lower()


def _matches(host: str, domain_set: frozenset[str]) -> bool:
    return host in domain_set or any(
        host == d or host.endswith("." + d) for d in domain_set
    )


async def _check(
    session: aiohttp.ClientSession,
    url: str,
    global_sem: asyncio.Semaphore,
    domain_sems: dict[str, asyncio.Semaphore],
    timeout: float,
) -> tuple[str, bool, str]:
    """Return (status_label, is_alive, note)."""
    host = _host(url)

    if _matches(host, BOT_BLOCKED):
        return "skipped", True, "bot-blocking domain, not checked"

    if host not in domain_sems:
        limit = CAUTIOUS_PER_DOMAIN if _matches(host, CAUTIOUS) else DEFAULT_PER_DOMAIN
        domain_sems[host] = asyncio.Semaphore(limit)

    headers = {"User-Agent": USER_AGENT}
    to = aiohttp.ClientTimeout(total=timeout)

    async with global_sem, domain_sems[host]:

        # ── Attempt 1: HEAD (no body transfer) ──────────────────────────────
        try:
            async with session.head(
                url, headers=headers, allow_redirects=True, timeout=to, ssl=False
            ) as r:
                if r.status != 405:
                    return str(r.status), r.status < 400, ""
                # 405 Method Not Allowed → fall through to GET
        except asyncio.TimeoutError:
            return "timeout", False, f"no response in {timeout:.0f}s"
        except aiohttp.ClientConnectorError as e:
            return "conn error", False, _trim(e)
        except aiohttp.ClientError:
            pass  # unexpected HEAD error → try GET anyway

        # ── Attempt 2: GET fallback ──────────────────────────────────────────
        # (context manager closes connection before reading body)
        try:
            async with session.get(
                url, headers=headers, allow_redirects=True, timeout=to, ssl=False
            ) as r:
                return str(r.status), r.status < 400, "HEAD→GET"
        except asyncio.TimeoutError:
            return "timeout", False, f"no response in {timeout:.0f}s"
        except aiohttp.ClientConnectorError as e:
            return "conn error", False, _trim(e)
        except aiohttp.ClientError as e:
            return "error", False, _trim(e)


def _trim(exc: Exception, n: int = 80) -> str:
    return str(exc)[:n]


# ── Concurrent runner ─────────────────────────────────────────────────────────

_GREEN = "\033[32m"
_RED   = "\033[31m"
_DIM   = "\033[2m"
_RESET = "\033[0m"


async def _run_all(
    urls: list[str],
    global_limit: int,
    timeout: float,
    verbose: bool,
) -> dict[str, tuple[str, bool, str]]:
    global_sem = asyncio.Semaphore(global_limit)
    domain_sems: dict[str, asyncio.Semaphore] = {}
    results: dict[str, tuple[str, bool, str]] = {}
    completed = 0
    width = len(str(len(urls)))
    print_lock = asyncio.Lock()

    connector = aiohttp.TCPConnector(limit=global_limit)

    async def run_one(url: str) -> None:
        nonlocal completed
        result = await _check(session, url, global_sem, domain_sems, timeout)
        async with print_lock:
            completed += 1
            results[url] = result
            status, ok, note = result
            if verbose or not ok:
                if status == "skipped":
                    color = _DIM
                elif ok:
                    color = _GREEN
                else:
                    color = _RED
                note_str = f"  ({note})" if note else ""
                print(
                    f"  [{completed:{width}d}/{len(urls)}]"
                    f" {color}{status:>9}{_RESET}  {url}{note_str}"
                )

    async with aiohttp.ClientSession(connector=connector) as session:
        await asyncio.gather(*[run_one(url) for url in urls])

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Check every hyperlink in the site's .qmd files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exit 0 = all clear.  Exit 1 = dead links found.",
    )
    ap.add_argument(
        "files", nargs="*", metavar="FILE",
        help="QMD files to scan (default: all *.qmd in current directory)",
    )
    ap.add_argument(
        "--concurrency", type=int, default=20, metavar="N",
        help="Max simultaneous connections overall (default: 20)",
    )
    ap.add_argument(
        "--timeout", type=float, default=15.0, metavar="T",
        help="Per-request timeout in seconds (default: 15)",
    )
    ap.add_argument(
        "--verbose", action="store_true",
        help="Print every URL checked, not just failures",
    )
    args = ap.parse_args()

    paths = (
        [Path(f) for f in args.files]
        if args.files
        else sorted(Path(".").glob("*.qmd"))
    )
    if not paths:
        print("No .qmd files found.", file=sys.stderr)
        return 1

    # ── Collect links ────────────────────────────────────────────────────────
    occurrences: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for p in paths:
        for url, lineno in extract_links(p):
            occurrences[url].append((p, lineno))

    unique = list(occurrences)
    total_refs = sum(len(v) for v in occurrences.values())
    print(
        f"Scanning {len(paths)} file(s) — "
        f"{len(unique)} unique URLs ({total_refs} total references)\n"
    )

    # ── Check ────────────────────────────────────────────────────────────────
    t0 = time.monotonic()
    results = asyncio.run(_run_all(unique, args.concurrency, args.timeout, args.verbose))
    elapsed = time.monotonic() - t0

    # ── Report ───────────────────────────────────────────────────────────────
    dead    = {u: r for u, r in results.items() if not r[1]}
    skipped = [u for u, r in results.items() if r[0] == "skipped"]
    alive   = len(results) - len(dead) - len(skipped)

    if dead:
        print(f"\n{_RED}{'─' * 60}{_RESET}")
        print(f"{_RED}DEAD LINKS  ({len(dead)}){_RESET}\n")
        for url in sorted(dead):
            status, _, note = dead[url]
            note_str = f"  ({note})" if note else ""
            print(f"  [{status}]  {url}{note_str}")
            for src_path, src_line in occurrences[url]:
                print(f"        {_DIM}{src_path}:{src_line}{_RESET}")
        print()

    print("─" * 60)
    check = "✓" if not dead else "✗"
    print(
        f"{check}  {_GREEN}{alive} alive{_RESET}  "
        f"{_RED}{len(dead)} dead{_RESET}  "
        f"{_DIM}{len(skipped)} skipped{_RESET}  "
        f"— {elapsed:.1f}s"
    )

    return 1 if dead else 0


if __name__ == "__main__":
    sys.exit(main())
