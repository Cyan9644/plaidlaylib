#!/usr/bin/env python3
"""Recover an example's <name>_scale.csv/.png from CSV, lines already printed
to a terminal or log file, without re-running the binary.

io_trace.py (as of this fix) and run_benches.py both capture each `CSV,` line
as the sweep runs. This script is for the case where a run already happened
and printed everything needed, but nothing captured it at the time (e.g. an
expensive out-of-core sweep run before that capture step existed) — as long
as you still have the console output (scrollback, a saved log, a paste),
the comparison graph can be rebuilt with no drive access at all.

  usage:
    python3 benchmarks/csv_from_log.py <example> run.log
    python3 benchmarks/csv_from_log.py <example> < run.log
    pbpaste | python3 benchmarks/csv_from_log.py <example>

`n` (element count) is already in every CSV, line, so input_bytes is
recomputed from it (n * elem_bytes * input_seqs) — the original --size
strings don't need to be known or match anything.

Reuses run_benches.py's EXAMPLES registry, write_csv, and plot_example, so the
output matches every other *_scale.csv/.png in the repo exactly.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benches as rb  # noqa: E402  (sibling module; import-safe)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: csv_from_log.py <example> [logfile]  (reads stdin if no logfile)\n"
                 "  choices: " + ", ".join(e["name"] for e in rb.EXAMPLES))
    example = sys.argv[1]
    entry = next((e for e in rb.EXAMPLES if e["name"] == example), None)
    if entry is None:
        sys.exit(f"unknown example {example!r}; choices: "
                 + ", ".join(e["name"] for e in rb.EXAMPLES))

    if len(sys.argv) > 2:
        with open(sys.argv[2]) as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    per_n = entry["elem_bytes"] * entry["input_seqs"]
    rows = []
    seen_n = set()
    skipped = 0
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("CSV,"):
            continue
        fields = line[len("CSV,"):].split(",")
        if len(fields) != len(entry["cols"]):
            print(f"  !!! skipping malformed CSV, line (got {len(fields)} fields, "
                 f"want {len(entry['cols'])} for {example!r}): {line}", file=sys.stderr)
            skipped += 1
            continue
        row = dict(zip(entry["cols"], fields))
        n = int(row["n"])
        if n in seen_n:
            print(f"  !!! duplicate n={n}; keeping the first occurrence", file=sys.stderr)
            continue
        seen_n.add(n)
        row["input_bytes"] = str(n * per_n)
        rows.append(row)

    if not rows:
        sys.exit(f"no CSV, lines found matching {example!r}'s column count "
                 f"({len(entry['cols'])}) — wrong example name, or the log "
                 f"doesn't contain them" + (f" ({skipped} malformed line(s) seen)"
                 if skipped else ""))
    rows.sort(key=lambda r: int(r["n"]))
    print(f"found {len(rows)} point(s) for {example!r}: n="
         + ",".join(r["n"] for r in rows), file=sys.stderr)

    csv_path = f"{entry['name']}_scale.csv"
    rb.write_csv(csv_path, ["input_bytes"] + entry["cols"], rows)
    try:
        rb.plot_example(rows, entry, f"{entry['name']}_scale.png")
    except Exception as exc:
        print(f"  !!! plotting failed ({exc}); CSV was written", file=sys.stderr)


if __name__ == "__main__":
    main()
