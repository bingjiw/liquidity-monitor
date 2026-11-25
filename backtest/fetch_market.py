"""抓取美股历史价格(含分红的复权价),存到 output/market.csv。

优先用 Yahoo 的复权收盘价(SPY、QQQ),失败时退回 FRED 的 SP500 / NASDAQCOM 价格指数
(不含分红)。云端会话连不上这些网站,所以在 GitHub Actions 里运行。
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

START = "2018-01-01"
EXTRA = {"GLD": "GLD", "IWM": "IWM", "SOXX": "SOXX", "HYG": "HYG", "^VIX": "VIX_Y"}
OUT = Path(__file__).resolve().parent.parent / "output" / "market.csv"


def yahoo(sym: str) -> pd.Series:
    t0 = int(pd.Timestamp(START).timestamp())
    t1 = int(time.time()) + 86400
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
    r = requests.get(url, params={"period1": t0, "period2": t1, "interval": "1d", "events": "div,splits"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s").normalize()
    ind = res["indicators"]
    adj = ind["adjclose"][0]["adjclose"] if ind.get("adjclose") else ind["quote"][0]["close"]
    s = pd.Series(adj, index=idx, name=sym).dropna()
    return s[~s.index.duplicated(keep="last")]


def fred(sid: str) -> pd.Series:
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": sid, "cosd": START}, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s.dropna().rename(sid)


def main() -> int:
    cols = {}
    for sym, fb in [("SPY", "SP500"), ("QQQ", "NASDAQCOM")]:
        for name, fn in [(sym, lambda s=sym: yahoo(s)), (fb, lambda s=fb: fred(s))]:
            try:
                cols[name] = fn()
                print(f"{name}: {len(cols[name])} 条, 最新 {cols[name].index.max().date()}")
            except Exception as e:  # noqa: BLE001
                print(f"[警告] {name} 失败: {e}", file=sys.stderr)
    if not cols:
        return 1
    # 观察指标用的附加资产(只用 Yahoo,失败就跳过):黄金、小盘、半导体、高收益债 ETF、当天的 VIX
    for sym, name in EXTRA.items():
        try:
            cols[name] = yahoo(sym)
            print(f"{name}: {len(cols[name])} 条, 最新 {cols[name].index.max().date()}")
        except Exception as e:  # noqa: BLE001
            print(f"[警告] {name} 失败: {e}", file=sys.stderr)
    df = pd.DataFrame(cols).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, float_format="%.4f")
    print(f"写入 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
