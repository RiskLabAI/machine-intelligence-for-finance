"""Retrieve and verify the public datasets used by the notebooks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen


DATASETS = {
    "binance_btcusdt": {
        "path": "data/public/binance/BTCUSDT-1m-2021-01.zip",
        "url": "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2021-01.zip",
        "sha256": "9364A063EFA0604D5B9D2536FB68F922DA5AFA3D8F94A26A527471A1E2F87801",
        "name": "Binance Data Vision BTCUSDT one-minute spot archive"
    },
    "fred_nasdaq": {
        "path": "data/public/fred/NASDAQCOM_1994-01-01_2025-12-31.csv",
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=NASDAQCOM&cosd=1994-01-01&coed=2025-12-31",
        "sha256": "2E988948FFCB108DA4902DFEDAA5DF756E7E102DC523A99A8916A0533291F2D7",
        "name": "Federal Reserve Bank of St. Louis NASDAQ Composite"
    },
    "fred_sp500": {
        "path": "data/public/fred/SP500_2014-01-02_2025-12-31.csv",
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500&cosd=2014-01-02&coed=2025-12-31",
        "sha256": "2098B209EE3AFDB3B2C1E783AC90A2BB7DD3865FEB416D0FABEDBB4A7A542754",
        "name": "Federal Reserve Bank of St. Louis S&P 500"
    },
    "french_49_industries": {
        "path": "data/public/ken_french/49_Industry_Portfolios_daily_CSV.zip",
        "url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_daily_CSV.zip",
        "sha256": "8F394FE34BEA54D41B9AAFED410425EE8F8E252EDE3C71A7C1CD20BAB83040DE",
        "name": "Kenneth R. French Data Library 49 Industry Portfolios"
    },
    "mpdta": {
        "path": "data/public/callaway_santanna/mpdta.csv",
        "url": "https://raw.githubusercontent.com/pedrohcgs/csdid-stata/31a577382d7c09757ed3baa095659f45cc63c107/examples/data/mpdta.csv",
        "sha256": "1B789C34E12FF490B2F432217A1F70AF334117523EB44D20EB842ED92A574661",
        "name": "Callaway and Sant'Anna county panel"
    }
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def ensure_dataset(repository_root: Path, key: str) -> Path:
    record = DATASETS[key]
    path = repository_root / record["path"]
    if path.is_file():
        actual = file_sha256(path)
        if actual != record["sha256"]:
            raise ValueError(f"Unexpected SHA-256 for {path}: {actual}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    temporary.write_bytes(urlopen(record["url"], timeout=120).read())
    actual = file_sha256(temporary)
    if actual != record["sha256"]:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Unexpected SHA-256 for downloaded {path.name}: {actual}")
    temporary.replace(path)
    return path
