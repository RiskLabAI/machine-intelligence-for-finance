# Public data sources

The notebooks retrieve only the public files used in their examples. Downloaded data, extracted copies, and generated simulation inputs remain outside version control. Users remain responsible for the terms stated by each source.

Python notebooks retrieve files when needed. For the Julia notebooks, `python notebooks/julia/prepare_inputs.py` creates the same local cache from these public sources and from deterministic simulation code.

| Dataset | Source | Repository path |
| --- | --- | --- |
| Binance Data Vision BTCUSDT one-minute spot archive | [https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2021-01.zip](https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2021-01.zip) | `data/public/binance/BTCUSDT-1m-2021-01.zip` |
| Federal Reserve Bank of St. Louis NASDAQ Composite | [https://fred.stlouisfed.org/graph/fredgraph.csv?id=NASDAQCOM&cosd=1994-01-01&coed=2025-12-31](https://fred.stlouisfed.org/graph/fredgraph.csv?id=NASDAQCOM&cosd=1994-01-01&coed=2025-12-31) | `data/public/fred/NASDAQCOM_1994-01-01_2025-12-31.csv` |
| Federal Reserve Bank of St. Louis S&P 500 | [https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500&cosd=2014-01-02&coed=2025-12-31](https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500&cosd=2014-01-02&coed=2025-12-31) | `data/public/fred/SP500_2014-01-02_2025-12-31.csv` |
| Kenneth R. French Data Library 49 Industry Portfolios | [https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_daily_CSV.zip](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_daily_CSV.zip) | `data/public/ken_french/49_Industry_Portfolios_daily_CSV.zip` |
| Callaway and Sant'Anna county panel | [https://raw.githubusercontent.com/pedrohcgs/csdid-stata/31a577382d7c09757ed3baa095659f45cc63c107/examples/data/mpdta.csv](https://raw.githubusercontent.com/pedrohcgs/csdid-stata/31a577382d7c09757ed3baa095659f45cc63c107/examples/data/mpdta.csv) | `data/public/callaway_santanna/mpdta.csv` |
