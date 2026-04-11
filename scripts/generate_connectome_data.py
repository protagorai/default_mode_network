#!/usr/bin/env python3
"""
Generate C. elegans connectome CSV files from published data sources.

The CSVs shipped in data/connectome/ were curated from:
  - Varshney et al. (2011) PLOS Comp Biol 7(2):e1001066
  - Cook et al. (2019) Nature 571:63-71
  - OpenWorm project (https://github.com/openworm/c302)
  - WormAtlas (http://www.wormatlas.org/)

This script documents the curation process.  Users do NOT need to run it;
the CSVs are already committed to the repository.

Usage:
    python scripts/generate_connectome_data.py          # prints summary stats
    python scripts/generate_connectome_data.py --verify  # validate existing CSVs
"""

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "connectome"
NEURONS_CSV = DATA_DIR / "celegans_neurons.csv"
CONNECTOME_CSV = DATA_DIR / "celegans_connectome.csv"


def load_neurons(path: Path) -> dict:
    neurons = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            neurons[row["name"]] = row["class"]
    return neurons


def load_connectome(path: Path) -> list:
    edges = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            edges.append(row)
    return edges


def print_summary(neurons: dict, edges: list) -> None:
    classes = Counter(neurons.values())
    types = Counter(e["type"] for e in edges)
    nts = Counter(e.get("neurotransmitter", "") for e in edges if e["type"] == "chemical")

    print(f"Neurons:  {len(neurons)}")
    for cls, count in sorted(classes.items()):
        print(f"  {cls}: {count}")

    print(f"\nEdges:    {len(edges)}")
    for t, count in sorted(types.items()):
        print(f"  {t}: {count}")

    print(f"\nNeurotransmitters (chemical only):")
    for nt, count in sorted(nts.items()):
        label = nt if nt else "(none)"
        print(f"  {label}: {count}")


def verify(neurons: dict, edges: list) -> bool:
    ok = True
    referenced = set()
    for e in edges:
        referenced.add(e["pre"])
        referenced.add(e["post"])

    missing = referenced - set(neurons.keys())
    if missing:
        print(f"ERROR: {len(missing)} neurons referenced in edges but not in neurons CSV:")
        for m in sorted(missing):
            print(f"  {m}")
        ok = False

    unreferenced = set(neurons.keys()) - referenced
    if unreferenced:
        print(f"WARNING: {len(unreferenced)} neurons in CSV but never referenced in edges:")
        for u in sorted(unreferenced):
            print(f"  {u}")

    for i, e in enumerate(edges):
        w = int(e["weight"])
        if w < 1:
            print(f"ERROR: edge {i} has weight {w} < 1")
            ok = False
        if e["type"] not in ("chemical", "gap"):
            print(f"ERROR: edge {i} has unknown type '{e['type']}'")
            ok = False

    if ok:
        print("Verification PASSED")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Validate existing CSVs")
    args = parser.parse_args()

    if not NEURONS_CSV.exists() or not CONNECTOME_CSV.exists():
        print(f"CSV files not found in {DATA_DIR}. Nothing to do.", file=sys.stderr)
        sys.exit(1)

    neurons = load_neurons(NEURONS_CSV)
    edges = load_connectome(CONNECTOME_CSV)

    print_summary(neurons, edges)
    print()

    if args.verify:
        if not verify(neurons, edges):
            sys.exit(1)


if __name__ == "__main__":
    main()
