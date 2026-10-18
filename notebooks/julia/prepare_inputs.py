"""Download public sources and recreate local inputs for the Julia notebooks."""

from __future__ import annotations

import argparse
import importlib
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATOR_DIRECTORY = Path(__file__).resolve().parent / "input_generators"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GENERATOR_DIRECTORY))

from book_support import DATASETS, ensure_dataset
from input_common import save_exact


GENERATORS = {
    "12": "prepare_mif_chapter12_inputs",
    "14": "prepare_mif_chapter14_inputs",
    "15": "prepare_mif_chapter15_inputs",
    "16": "prepare_mif_chapter16_inputs",
    "17": "prepare_mif_chapter17_inputs",
    "18": "prepare_mif_chapter18_inputs",
    "19": "prepare_mif_chapter19_inputs",
    "20": "prepare_mif_chapter20_inputs",
    "21": "prepare_mif_chapter21_inputs",
    "22": "prepare_mif_chapter22_inputs",
    "23": "prepare_mif_chapter23_inputs",
    "25": "prepare_mif_chapter25_inputs",
    "26": "prepare_mif_chapter26_inputs",
    "27": "prepare_mif_chapter27_inputs",
    "28": "prepare_mif_chapter28_inputs",
    "31": "prepare_mif_chapter31_inputs",
    "appendix": "prepare_mif_appendix_inputs",
}


def prepare_public_data() -> None:
    paths = {name: ensure_dataset(ROOT, name) for name in DATASETS}
    native_directory = ROOT / "notebooks" / "julia" / "data"
    for name, destination_name in (
        ("binance_btcusdt", "binance_btcusdt.csv"),
        ("french_49_industries", "french_49_industries.csv"),
    ):
        with zipfile.ZipFile(paths[name]) as archive:
            content = archive.read(archive.namelist()[0])
        save_exact(native_directory / destination_name, content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter", action="append", choices=[*GENERATORS, "all"], default=[])
    parser.add_argument("--public-data-only", action="store_true")
    arguments = parser.parse_args()
    prepare_public_data()
    if arguments.public_data_only:
        return
    selected = list(GENERATORS) if not arguments.chapter or "all" in arguments.chapter else arguments.chapter
    for chapter in selected:
        importlib.import_module(GENERATORS[chapter]).main()


if __name__ == "__main__":
    main()
