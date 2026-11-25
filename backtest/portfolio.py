"""综合组合:比特币 + 美股 + 美债 + 现金,用一个流动性信号统一调风险仓位。

事先定好、不按结果挑:
- 目标比例(可改):BTC 40%、美股 VTI 25%、海外股 VXUS 10%、美债 IEF 15%、现金 SGOV 10%。
- 信号用规则 ①(综合指数 20 日均值 <35 连续 20 天 → 减半,>50 连续 20 天 → 恢复)。
  选它的理由是事前的:它是唯一在 2004–2018、2019–2022、2023 起三段美股上都不明显亏的规则,
  而不是因为它在 2024 年后的比特币上最好。③、⑤ 只作对照。
- 减半时:比特币和股票的目标比例各乘 0.5,腾出的钱放现金;美债不动。
- 每天按目标比例计算(相当于持续再平衡),信号次日执行,仓位变动部分扣 0.05%。
- 未来推演:把 2024-01 以来(和 2004 年以来,不含 BTC)的每日收益按 63 天一段打乱重抽,
  模拟 5000 个“未来一年”。它假设未来和过去这段相似,只是区间,不是预测。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHTS = {"BTC": 0.40, "VTI": 0.25, "VXUS": 0.10, "IEF": 0.15}  # 余下 10% 现金
RISKY = ("BTC", "VTI", "VXUS")
COST = 0.0005
BTC_START = "2024-01-11"
LONG_START = "2004-01-02"
NAMES = {"BTC": "比特币", "VTI": "美股 VTI", "VXUS": "海外股 VXUS", "IEF": "美债 IEF", "CASH": "现金 SGOV"}


def _returns(m: pd.DataFrame, idx) -> dict:
    px = {}
    px["BTC"] = m["BTC"] if "BTC" in m else None
    px["VTI"] = m["VTI"] if "VTI" in m and m["VTI"].notna().sum() > 1000 else m["SPY"]
    v = m["VXUS"] if "VXUS" in m else None
    e = m["EFA"] if "EFA" in m else None
    r = {}
    for k in ("BTC", "VTI", "IEF"):
        s = m.get(k) if k != "VTI" else px["VTI"]
        r[k] = None if s is None else s.dropna().pct_change().reindex(idx)
    # 海外股:VXUS 2011 年才有,之前用 EFA 接上
    if v is not None or e is not None:
        rv = v.dropna().pct_change().reindex(idx) if v is not None else pd.Series(np.nan, idx)
        re_ = e.dropna().pct_change().reindex(idx) if e is not None else pd.Series(np.nan, idx)
        r["VXUS"] = rv.combine_first(re_)
    else:
        r["VXUS"] = None
    return r


def _weights(sig: pd.Series, wts: dict) -> pd.DataFrame:
    w = pd.DataFrame({k: v * (sig if k in RISKY else 1.0) for k, v in wts.items()}, index=sig.index)
    w["CASH"] = 1 - w.sum(axis=1)
    return w


def _daily(w: pd.DataFrame, R: pd.DataFrame, cash: pd.Series) -> pd.Series:
    ww = w.shift(1).bfill()
    ret = (ww.drop(columns="CASH") * R[ww.columns.drop("CASH")]).sum(axis=1) + ww["CASH"] * cash
    turn = ww.diff().abs().sum(axis=1).fillna(0) / 2
    return ret - turn * COST


def _stats(ret: pd.Series, cash: pd.Series) -> dict:
    eq = (1 + ret).cumprod()
    yrs = len(ret) / 252
    vol = ret.std() * np.sqrt(252)
    return {"total": float(eq.iloc[-1] - 1), "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1),
            "maxdd": float((eq / eq.cummax() - 1).min()), "vol": float(vol),
            "sharpe": float((ret - cash).mean() * 252 / vol) if vol else None}


def _boot(rets: dict, n=5000, horizon=252, block=63, seed=7) -> dict:
    """成对抽样:同一段日子,所有策略一起抽,保证可比。"""
    df = pd.DataFrame(rets).dropna()
    a = df.values
    T = len(a)
    rng = np.random.default_rng(seed)
    k = int(np.ceil(horizon / block))
    tot = np.empty((n, a.shape[1]))
    dd = np.empty((n, a.shape[1]))
    for i in range(n):
        st = rng.integers(0, T - block, k)
        path = np.concatenate([a[s:s + block] for s in st])[:horizon]
        eq = np.cumprod(1 + path, axis=0)
        tot[i] = eq[-1] - 1
        dd[i] = (eq / np.maximum.accumulate(eq, axis=0) - 1).min(axis=0)
    cols = list(df.columns)
    out = {}
    for j, c in enumerate(cols):
        out[c] = {"p10": float(np.percentile(tot[:, j], 10)), "p50": float(np.percentile(tot[:, j], 50)),
                  "p90": float(np.percentile(tot[:, j], 90)), "dd50": float(np.percentile(dd[:, j], 50)),
                  "dd10": float(np.percentile(dd[:, j], 10)), "loss20": float((tot[:, j] < -0.2).mean())}
    if len(cols) > 1:
        out["_beat"] = {c: float((tot[:, j] > tot[:, 0]).mean()) for j, c in enumerate(cols) if j > 0}
    out["_n"] = T
    return out


def run(idx, cash, R_rules: dict) -> dict | None:
    from pathlib import Path
    m = pd.read_csv(Path(__file__).resolve().parent.parent / "output" / "market.csv", index_col=0, parse_dates=True)
    m = m[~m.index.duplicated(keep="last")]
    r = _returns(m, idx)
    if r["IEF"] is None or r["VXUS"] is None:
        return None
    ks = list(R_rules)
    sigs = {"一直持有(不择时)": pd.Series(1.0, idx), "① 综合指数(主方案)": R_rules[ks[1]],
            "③ 净流动性(对照)": R_rules[ks[3]], "⑤ ①或③(对照)": R_rules[ks[-1]]}
    res = {"weights": {NAMES[k]: v for k, v in WEIGHTS.items()} | {NAMES["CASH"]: round(1 - sum(WEIGHTS.values()), 4)}}

    # A) 2024-01 起,含比特币
    R = pd.DataFrame({k: r[k] for k in WEIGHTS}).fillna(0)
    on = idx >= pd.Timestamp(BTC_START)
    daily = {n: _daily(_weights(s, WEIGHTS), R, cash)[on] for n, s in sigs.items()}
    rows, eqs = [], {}
    for n, d in daily.items():
        st = _stats(d, cash[on])
        sw = int((sigs[n][on].diff().abs() > 0).sum())
        rows.append([n, st["total"], st["cagr"], st["maxdd"], st["vol"], st["sharpe"], sw])
        eqs[n] = [round(float(v), 4) for v in (1 + d).cumprod()]
    # 单一资产一直持有,供对照
    single = []
    for k in list(WEIGHTS) + ["CASH"]:
        d = (cash if k == "CASH" else R[k])[on]
        st = _stats(d, cash[on])
        single.append([NAMES[k], st["total"], st["cagr"], st["maxdd"], st["sharpe"]])
    res["btc_era"] = {"start": BTC_START, "dates": [d.strftime("%Y-%m-%d") for d in idx[on]], "equity": eqs,
                      "rows": rows, "single": single, "boot": _boot({k: daily[k] for k in list(daily)[:2]})}

    # B) 2004 起,不含比特币(把 BTC 的比例按比例分给其他),检验股债部分是否稳健
    w2 = {k: v for k, v in WEIGHTS.items() if k != "BTC"}
    sc = (1 - (1 - sum(WEIGHTS.values()))) / sum(w2.values())
    w2 = {k: round(v * sc, 4) for k, v in w2.items()}
    R2 = pd.DataFrame({k: r[k] for k in w2}).fillna(0)
    periods = [("2004–2018", "2004-01-02", "2018-12-31"), ("2019–2022", "2019-01-01", "2022-12-31"),
               ("2023 起", "2023-01-01", "2100-01-01"), ("全期 2004 起", LONG_START, "2100-01-01")]
    lrows = []
    dl = {n: _daily(_weights(s, w2), R2, cash) for n, s in sigs.items()}
    for n, d in dl.items():
        row = [n]
        for _, a, b in periods:
            st = _stats(d[a:b], cash[a:b])
            row += [st["cagr"], st["maxdd"], st["sharpe"]]
        lrows.append(row)
    res["long"] = {"weights": {NAMES[k]: v for k, v in w2.items()} | {NAMES["CASH"]: round(1 - sum(w2.values()), 4)},
                   "periods": [p[0] for p in periods], "rows": lrows,
                   "boot": _boot({k: dl[k][LONG_START:] for k in list(dl)[:2]})}

    # 今天的建议比例(主方案)
    wt = _weights(sigs["① 综合指数(主方案)"], WEIGHTS).iloc[-1]
    res["today"] = {"signal": float(sigs["① 综合指数(主方案)"].iloc[-1]), "date": idx[-1].strftime("%Y-%m-%d"),
                    "alloc": {NAMES[k]: round(float(v), 4) for k, v in wt.items()},
                    "nl_signal": float(sigs["③ 净流动性(对照)"].iloc[-1])}
    return res


def md(p: dict) -> list[str]:
    f = lambda v, s=True: "—" if v is None else (f"{v:+.1%}" if s else f"{v:.1%}")  # noqa: E731
    L = ["## 7. 综合组合(比特币 + 美股 + 美债 + 现金,一个流动性信号统一调仓)", "",
         "目标比例:" + "、".join(f"{k} {v:.0%}" for k, v in p["weights"].items())
         + "。主方案信号用规则 ①,减半时比特币和股票各打五折,腾出的钱放现金,美债不动。", "",
         f"### 2024-01-11 起(含比特币)", "", "| 方案 | 总收益 | 年化 | 最大回撤 | 波动 | 夏普 | 调仓 |", "|---|---|---|---|---|---|---|"]
    for r in p["btc_era"]["rows"]:
        L.append(f"| {r[0]} | {f(r[1])} | {f(r[2], False)} | {f(r[3])} | {f(r[4], False)} | {r[5]:.2f} | {r[6]} |")
    L += ["", "| 单一资产一直持有 | 总收益 | 年化 | 最大回撤 | 夏普 |", "|---|---|---|---|---|"]
    for r in p["btc_era"]["single"]:
        L.append(f"| {r[0]} | {f(r[1])} | {f(r[2], False)} | {f(r[3])} | {r[4]:.2f} |")
    lo = p["long"]
    L += ["", "### 2004 年起(不含比特币,检验股债部分):" + "、".join(f"{k} {v:.0%}" for k, v in lo["weights"].items()), "",
          "| 方案 | " + " | ".join(f"{x} 年化 | {x} 回撤 | {x} 夏普" for x in lo["periods"]) + " |",
          "|---|" + "---|" * (3 * len(lo["periods"]))]
    for r in lo["rows"]:
        cells = []
        for i in range(len(lo["periods"])):
            c, d, s = r[1 + 3 * i: 4 + 3 * i]
            cells += [f(c, False), f(d), f"{s:.2f}"]
        L.append(f"| {r[0]} | " + " | ".join(cells) + " |")
    L += ["", "### 未来一年推演(重抽历史收益,5000 次;不是预测)", "",
          "| 样本 | 方案 | 差的情况(10%) | 中间(50%) | 好的情况(90%) | 一年内回撤中位数 | 亏超 20% 的概率 |", "|---|---|---|---|---|---|---|"]
    for lab, b in (("2024 起(含 BTC)", p["btc_era"]["boot"]), ("2004 起(股债)", lo["boot"])):
        for k, v in b.items():
            if k.startswith("_"):
                continue
            L.append(f"| {lab} | {k} | {f(v['p10'])} | {f(v['p50'])} | {f(v['p90'])} | {f(v['dd50'])} | {v['loss20']:.0%} |")
        if "_beat" in b:
            L.append(f"| {lab} | 择时胜过不择时的概率 | " + "、".join(f"{v:.0%}" for v in b["_beat"].values()) + " | | | | |")
    t = p["today"]
    L += ["", f"今天({t['date']})主方案信号 {'满仓' if t['signal'] >= 1 else '减半'},对应比例:"
          + "、".join(f"{k} {v:.0%}" for k, v in t["alloc"].items()) + "。", ""]
    return L
