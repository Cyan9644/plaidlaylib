import argparse
from itertools import chain
from pathlib import Path
import re
import sys
from dataclasses import dataclass


def convert_group(output: str, measure_cpu: bool = False) -> dict[str, str]:
    result = {}
    for line in output.split("\n"):
        r = re.search(r"(?<!,) bw=([0-9.gmkib]+)/s \(([0-9.gmkib]+)/s\)", line, re.IGNORECASE)
        if r is not None:
            result['throughput'] = r.group(2)
        r = re.search(r" lat \(([^)]+)\).*avg=([0-9.]+),", line)
        if r is not None:
            result['latency'] = r.group(2) + r.group(1)
        r = re.search(r"cpu.*usr=([\d.]+)%.*sys=([\d.]+)%", line)
        if r is not None and measure_cpu:
            result['cpu'] = "{:.2f}".format(float(r.group(1)) + float(r.group(2)))
    return result


def main():
    assert len(sys.argv) > 1
    files = sys.argv[1:]
    result: list[list[dict[str, str]]] = []

    def add_data_point(i: int, d: dict[str, str]) -> None:
        while i >= len(result):
            result.append([])
        result[i].append(d)

    for file_name in files:
        index = 0
        f = Path(file_name)
        if not f.exists():
            f = Path("results") / file_name
        assert f.exists()
        with open(f, "r", encoding="utf-8") as f:
            lines: list[str] = f.readlines()
        current_group = []
        for line in lines:
            current_group.append(line)
            if "run=" in line:
                add_data_point(index, convert_group("\n".join(current_group)))
                current_group = []
                index += 1
        
    
    for row, result_dict in enumerate(result, 1):
        print(",".join([str(row), *chain.from_iterable(d.values() for d in result_dict)]))


if __name__ == "__main__":
    main()
