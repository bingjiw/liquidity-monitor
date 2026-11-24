"""把每个序列按“数据实际公布的时间”往后挪,重算综合指数,避免回测里偷看未来数据。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from liquidity_monitor import score  # noqa: E402

# 每个衍生序列从观测日到可用日的大致滞后(工作日),取其组成部分里最慢的那个
PUB_LAG = {
    "reserves": 2, "fed_assets": 2, "reserves_chg": 2, "fed_bills_chg": 2, "net_liq_chg": 2,
    "tga": 1, "tga_chg": 1, "rrp": 0,
    "sofr_iorb": 1, "sofr99_iorb": 1, "tgcr_rrp": 1, "srf": 0,
    "ccc_oas": 1, "ccc_yield": 1, "hy_oas": 1, "ig_oas": 1, "ccc_bb_gap": 1,
    "real_policy_rate": 45,   # 核心 PCE 按月初记日期,实际约两个月后才公布
    "tips5": 1, "ust2_chg": 1, "term_premium": 5,
    "vix": 0, "usd_chg": 5, "usdjpy_chg": 5, "nfci": 5, "stlfsi": 6,
}


def pit_scores(raw_csv: Path = ROOT / "output" / "raw_data.csv") -> pd.DataFrame:
    f = pd.read_csv(raw_csv, index_col=0, parse_dates=True)
    g = f.copy()
    for c, lag in PUB_LAG.items():
        if c in g:
            g[c] = g[c].shift(lag)
    return score(g)
