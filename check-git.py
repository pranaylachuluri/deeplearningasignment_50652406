"""
check_git.py  –  CSE 676 Assignment 1
======================================
Verifies that the submitted directory contains a valid Git repository with
at least 5 commits spread across at least 2 different calendar days.

Usage
-----
    python check_git.py <path-to-submission-directory>

Examples
--------
    python check_git.py LASTNAME_FIRSTNAME_A3/
    python check_git.py .

Exit codes
----------
    0  – PASS (all requirements satisfied)
    1  – FAIL (requirements not met; 15-point deduction applies)
    2  – ERROR (not a git repo, or git not installed)
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ── Requirements (must match the handout) ──────────────────────────────────
MIN_COMMITS       = 5
MIN_CALENDAR_DAYS = 2


def run_git(args: list[str], cwd: Path) -> tuple[int, str, str]:
    """Run a git command in *cwd* and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        ["git"] + args,
        cwd    = cwd,
        capture_output = True,
        text   = True,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def check(submission_dir: str) -> int:
    """
    Perform all checks.  Prints a human-readable report and returns
    0 (PASS) or 1 (FAIL) or 2 (ERROR).
    """
    path = Path(submission_dir).resolve()

    print(f"\n{'='*55}")
    print(f"  Git History Check  –  CS 676 Assignment 3")
    print(f"{'='*55}")
    print(f"  Directory : {path}")

    # ── 1. Check that git is installed ────────────────────────────────────
    rc, _, err = run_git(["--version"], Path("."))
    if rc != 0:
        print("\n  ERROR: 'git' command not found.  Is Git installed?")
        return 2

    # ── 2. Check that the directory is a git repository ───────────────────
    rc, _, err = run_git(["rev-parse", "--git-dir"], path)
    if rc != 0:
        print(f"\n  ERROR: '{path}' is not a Git repository.")
        print("  Make sure you ran 'git init' inside your project folder and")
        print("  that the .git directory was included when you zipped.")
        return 2

    # ── 3. Fetch all commit timestamps (author date, ISO strict format) ───
    rc, log_out, err = run_git(
        ["log", "--format=%aI", "--no-walk=unsorted"],
        path,
    )
    # --format=%aI prints one ISO-8601 timestamp per commit.
    # Fall back to a simpler log if the above returns nothing.
    if rc != 0 or not log_out:
        rc, log_out, err = run_git(["log", "--format=%aI"], path)

    if rc != 0 or not log_out:
        print(f"\n  ERROR: Could not read commit history.\n  {err}")
        return 2

    timestamps = [line.strip() for line in log_out.splitlines() if line.strip()]
    num_commits = len(timestamps)

    # Parse dates (just the YYYY-MM-DD portion is sufficient).
    calendar_days: set[str] = set()
    for ts in timestamps:
        try:
            # ISO-8601 with offset, e.g. "2026-03-15T21:34:02+00:00"
            dt = datetime.fromisoformat(ts)
            calendar_days.add(dt.date().isoformat())
        except ValueError:
            # Fallback: take the first 10 characters.
            if len(ts) >= 10:
                calendar_days.add(ts[:10])

    num_days = len(calendar_days)

    # ── 4. Print summary ──────────────────────────────────────────────────
    print(f"\n  Commits found        : {num_commits}")
    print(f"  Unique calendar days : {num_days}")

    if num_commits >= 1:
        print(f"\n  Commit dates:")
        for day in sorted(calendar_days):
            day_count = sum(1 for ts in timestamps if ts[:10] == day)
            print(f"    {day}  ({day_count} commit{'s' if day_count != 1 else ''})")

    # ── 5. Evaluate ───────────────────────────────────────────────────────
    passed  = True
    reasons = []

    if num_commits < MIN_COMMITS:
        passed = False
        reasons.append(
            f"Need at least {MIN_COMMITS} commits; found {num_commits}."
        )
    if num_days < MIN_CALENDAR_DAYS:
        passed = False
        reasons.append(
            f"Need commits on at least {MIN_CALENDAR_DAYS} different calendar "
            f"days; found {num_days}."
        )

    print()
    if passed:
        print("  ✓  PASS: git history requirement satisfied.")
        print(f"     ({num_commits} commits across {num_days} day(s))")
        print()
        return 0
    else:
        print("  ✗  FAIL: git history requirement NOT satisfied.")
        print("     The following issues were found:")
        for r in reasons:
            print(f"       • {r}")
        print()
        print("     This will result in a 15-point deduction.")
        print("     Make additional commits on separate days before submitting.")
        print()
        return 1


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage: python check_git.py <submission-directory>")
        sys.exit(2)

    submission_dir = sys.argv[1]
    exit_code = check(submission_dir)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()