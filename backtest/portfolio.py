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
# 宽松加码档(2026-11-25 用户要求加,在看结果前定好):综合指数 20 日均值 >70(页面上本来就标为“宽松”)连续 20 天 → 加码;
# 跌回 60 以下连续 20 天 → 回到正常。减半信号优先。
LOOSE = {"BTC": 0.70, "VTI": 0.18, "VXUS": 0.07, "IEF": 0.05}  # 现金 0
LOOSE_LONG = {"VTI": 0.70, "VXUS": 0.25, "IEF": 0.05}          # 不含比特币时
LOOSE_IN, LOOSE_OUT, LOOSE_N = 70, 60, 20
# D 方案(2026-11-25 加):美联储宣布 QE / 扩表的次日加码,持有 250 个交易日(约一年);减半信号优先。
# 名单要人工维护:以后联储宣布新的大规模购债,在这里加一行。
QE_EVENTS = [("2008-11-25", "QE1:宣布购买 MBS 和机构债"), ("2009-03-18", "QE1 扩大并开始买国债"),
             ("2010-11-03", "QE2:6000 亿美元国债"), ("2012-09-13", "QE3:每月 400 亿 MBS,不设上限"),
             ("2019-10-11", "宣布每月 600 亿美元购买短债"), ("2020-03-23", "无限量 QE"),
             ("2025-12-10", "宣布开始补充准备金购买短债(RMP)")]
QE_HOLD = 250
GUIDE_MODE = "D"  # 首页指引用哪种加码:"A" 指数 >70 连续 20 天;"D" 宣布 QE 次日
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


def loose_flag(sm: pd.Series) -> pd.Series:
    hi = (sm > LOOSE_IN).astype(int).rolling(LOOSE_N).sum() == LOOSE_N
    lo = (sm < LOOSE_OUT).astype(int).rolling(LOOSE_N).sum() == LOOSE_N
    f = pd.Series(np.nan, index=sm.index)
    f[hi] = 1.0
    f[lo] = 0.0
    return f.ffill().fillna(0.0)


def qe_flag(fa: pd.Series) -> pd.Series:
    """B 方案:联储总资产 13 周增长 >5%(只有 QE 才会这么快)连续 10 天 → 加码;13 周变化转负连续 20 天 → 退出。"""
    g = fa / fa.shift(65) - 1
    hi = (g > 0.05).astype(int).rolling(10).sum() == 10
    lo = (g < 0).astype(int).rolling(20).sum() == 20
    f = pd.Series(np.nan, index=fa.index)
    f[hi] = 1.0
    f[lo] = 0.0
    return f.ffill().fillna(0.0)


def easing_events() -> list:
    """D 方案用的“大规模宽松事件”:research.py 里的 17 个联储宽松事件(QE、紧急贷款工具、停止/放慢缩表、买短债等),
    这份名单在检验加码之前就已经定好。以后出现新的宽松手段(SLR 放松、降准备金率、财政部大规模回购等),在 research.EVENTS 加一行。"""
    try:
        from research import EVENTS
        return EVENTS
    except Exception:  # noqa: BLE001
        return QE_EVENTS


def nl_surge_flag(nl_chg: pd.Series) -> pd.Series:
    """E 方案(只作对照/提示):净流动性 13 周增量高于自己过去 3 年的 90% 分位,连续 10 天 → 加码;13 周变化转负连续 20 天 → 退出。
    TGA 下降、逆回购释放这类“不是 QE 的放水”会体现在这里。"""
    p90 = nl_chg.rolling(756, min_periods=250).quantile(0.9)
    hi = (nl_chg > p90).astype(int).rolling(10).sum() == 10
    lo = (nl_chg < 0).astype(int).rolling(20).sum() == 20
    f = pd.Series(np.nan, index=nl_chg.index)
    f[hi] = 1.0
    f[lo] = 0.0
    return f.ffill().fillna(0.0)


def qe_event_flag(idx, events=None) -> pd.Series:
    f = np.zeros(len(idx))
    for d, _ in (events or QE_EVENTS):
        j = idx.searchsorted(pd.Timestamp(d)) + 1
        if j < len(idx):
            f[j:j + QE_HOLD] = 1.0
    return pd.Series(f, index=idx)


def tier(sig: pd.Series, loose: pd.Series) -> pd.Series:
    """0.5 = 减半,1 = 正常,2 = 宽松加码(减半优先)。"""
    return pd.Series(np.where(sig < 1, 0.5, np.where(loose >= 1, 2.0, 1.0)), index=sig.index)


def _tw(t: pd.Series, wts: dict, loose: dict) -> pd.DataFrame:
    w = _weights(t.clip(upper=1.0), wts)
    lw = pd.Series({**loose, "CASH": 1 - sum(loose.values())}).reindex(w.columns).fillna(0)
    w.loc[t >= 2, :] = lw.values
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


def run(idx, cash, R_rules: dict, sm: pd.Series | None = None, fed_assets: pd.Series | None = None,
        net_liq: pd.Series | None = None) -> dict | None:
    from pathlib import Path
    m = pd.read_csv(Path(__file__).resolve().parent.parent / "output" / "market.csv", index_col=0, parse_dates=True)
    m = m[~m.index.duplicated(keep="last")]
    r = _returns(m, idx)
    if r["IEF"] is None or r["VXUS"] is None:
        return None
    ks = list(R_rules)
    sigs = {"一直持有(不择时)": pd.Series(1.0, idx), "① 综合指数(主方案)": R_rules[ks[1]],
            "③ 净流动性(对照)": R_rules[ks[3]], "⑤ ①或③(对照)": R_rules[ks[-1]]}
    res = {"weights": {NAMES[k]: v for k, v in WEIGHTS.items()} | {NAMES["CASH"]: round(1 - sum(WEIGHTS.values()), 4)},
           "loose_weights": {NAMES[k]: v for k, v in LOOSE.items()} | {NAMES["CASH"]: round(1 - sum(LOOSE.values()), 4)},
           "loose_rule": [LOOSE_IN, LOOSE_OUT, LOOSE_N]}
    MAIN = "①+宽松加码(三档 A)"
    lf = loose_flag(sm) if sm is not None else pd.Series(0.0, idx)
    T3 = tier(sigs["① 综合指数(主方案)"], lf)
    T3b = tier(sigs["① 综合指数(主方案)"], qe_flag(fed_assets)) if fed_assets is not None else T3
    EV = easing_events()
    T3d = tier(sigs["① 综合指数(主方案)"], qe_event_flag(idx, EV))
    T3q = tier(sigs["① 综合指数(主方案)"], qe_event_flag(idx, QE_EVENTS))
    T3e = tier(sigs["① 综合指数(主方案)"], nl_surge_flag(net_liq.diff(65))) if net_liq is not None else T3
    MD = "①+宽松事件加码(三档 D,主方案)"
    sigs = {"一直持有(不择时)": sigs["一直持有(不择时)"], MD: T3d, "①+只认 QE 加码(对照)": T3q, "①+净流动性激增加码 E(对照)": T3e,
            MAIN: T3, "①+扩表加码(三档 B)": T3b, "① 两档(减半/正常)": sigs["① 综合指数(主方案)"],
            "③ 净流动性(对照)": sigs["③ 净流动性(对照)"], "⑤ ①或③(对照)": sigs["⑤ ①或③(对照)"]}
    W = lambda t, wts, lo: _tw(t, wts, lo)  # noqa: E731

    # A) 2024-01 起,含比特币
    R = pd.DataFrame({k: r[k] for k in WEIGHTS}).fillna(0)
    on = idx >= pd.Timestamp(BTC_START)
    daily = {n: _daily(W(s, WEIGHTS, LOOSE), R, cash)[on] for n, s in sigs.items()}
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
                      "rows": rows, "single": single, "boot": _boot({k: daily[k] for k in [list(daily)[0], MD, "① 两档(减半/正常)"]})}

    # B) 2004 起,不含比特币(把 BTC 的比例按比例分给其他),检验股债部分是否稳健
    w2 = {k: v for k, v in WEIGHTS.items() if k != "BTC"}
    sc = (1 - (1 - sum(WEIGHTS.values()))) / sum(w2.values())
    w2 = {k: round(v * sc, 4) for k, v in w2.items()}
    R2 = pd.DataFrame({k: r[k] for k in w2}).fillna(0)
    periods = [("2004–2018", "2004-01-02", "2018-12-31"), ("2019–2022", "2019-01-01", "2022-12-31"),
               ("2023 起", "2023-01-01", "2100-01-01"), ("全期 2004 起", LONG_START, "2100-01-01")]
    lrows = []
    dl = {n: _daily(W(s, w2, LOOSE_LONG), R2, cash) for n, s in sigs.items()}
    for n, d in dl.items():
        row = [n]
        for _, a, b in periods:
            st = _stats(d[a:b], cash[a:b])
            row += [st["cagr"], st["maxdd"], st["sharpe"]]
        lrows.append(row)
    res["long"] = {"weights": {NAMES[k]: v for k, v in w2.items()} | {NAMES["CASH"]: round(1 - sum(w2.values()), 4)},
                   "periods": [p[0] for p in periods], "rows": lrows,
                   "boot": _boot({k: dl[k][LONG_START:] for k in [list(dl)[0], MD, "① 两档(减半/正常)"]})}
    res["long"]["loose_weights"] = {NAMES[k]: v for k, v in LOOSE_LONG.items()}
    # 宽松档在历史上出现的时段(按年统计天数占比),以及每段宽松期的股票表现
    res["loose_years_b"] = {str(y): round(float(v), 2) for y, v in (T3b >= 2).groupby(idx.year).mean().items() if v > 0}
    on_l = ((T3d if GUIDE_MODE == "D" else T3) >= 2)
    res["loose_years"] = {str(y): round(float(v), 2) for y, v in on_l.groupby(on_l.index.year).mean().items() if v > 0}
    segs, st0 = [], None
    for i, (d, v) in enumerate(on_l.items()):
        if v and st0 is None:
            st0 = d
        if st0 is not None and (not v or i == len(on_l) - 1):
            a, b = st0, d
            vt = (1 + R2["VTI"][a:b]).prod() - 1
            segs.append([a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d"), int(on_l[a:b].sum()), float(vt),
                         float(((1 + R2["VTI"][a:b]).cumprod() / (1 + R2["VTI"][a:b]).cumprod().cummax() - 1).min())])
            st0 = None
    res["loose_segments"] = segs
    # 参考:2020–2023 用 ETF 前的比特币跑同一套(只作参考,那时比特币还不受美元流动性驱动)
    ref = idx >= pd.Timestamp("2020-01-01")
    Rb = pd.DataFrame({k: r[k] for k in WEIGHTS}).fillna(0)
    rr = []
    for n in ("一直持有(不择时)", MD, "①+只认 QE 加码(对照)", "①+净流动性激增加码 E(对照)", MAIN, "① 两档(减半/正常)"):
        d = _daily(W(sigs[n], WEIGHTS, LOOSE), Rb, cash)[ref]
        d = d[: "2023-12-31"]
        st = _stats(d, cash[d.index])
        rr.append([n, st["total"], st["cagr"], st["maxdd"], st["sharpe"]])
    res["ref_2020"] = rr

    # 首页操作指引图:每天的目标比例、信号、综合指数 20 日均值、到当天为止的资金
    # 从 2004 年起给出每天规则会建议的比例(比特币那份在 2024 年前只是“规则会怎么说”,不计入收益);资金只从 2024-01-11 算
    gon = idx >= pd.Timestamp(LONG_START)
    GK = MD if GUIDE_MODE == "D" else MAIN
    sg = sigs[GK][gon]
    wd = W(sigs[GK], WEIGHTS, LOOSE)[gon]
    npre = int((idx[gon] < pd.Timestamp(BTC_START)).sum())
    res["guide"] = {"dates": [d.strftime("%Y-%m-%d") for d in idx[gon]], "sig": [float(v) for v in sg], "btc_start": BTC_START,
                    "sm": [None if sm is None or pd.isna(v) else round(float(v), 1) for v in (sm[gon] if sm is not None else sg * np.nan)],
                    "alloc": {NAMES[k]: [round(float(v), 3) for v in wd[k]] for k in wd.columns},
                    "mode": GUIDE_MODE, "name": GK, "qe_events": EV, "nl_surge_now": bool(T3e.iloc[-1] >= 2 or nl_surge_flag(net_liq.diff(65)).iloc[-1] >= 1) if net_liq is not None else False, "qe_hold": QE_HOLD,
                    "eq_timed": [None] * npre + eqs[GK], "eq_hold": [None] * npre + eqs["一直持有(不择时)"]}
    # 今天的建议比例(主方案)
    wt = W(sigs[GK], WEIGHTS, LOOSE).iloc[-1]
    res["today"] = {"signal": float(sigs[GK].iloc[-1]), "name": GK, "date": idx[-1].strftime("%Y-%m-%d"),
                    "alloc": {NAMES[k]: round(float(v), 4) for k, v in wt.items()},
                    "nl_signal": float(sigs["③ 净流动性(对照)"].iloc[-1])}
    return res


def md(p: dict) -> list[str]:
    f = lambda v, s=True: "—" if v is None else (f"{v:+.1%}" if s else f"{v:.1%}")  # noqa: E731
    L = ["## 7. 综合组合(比特币 + 美股 + 美债 + 现金,一个流动性信号统一调仓)", "",
         "目标比例:" + "、".join(f"{k} {v:.0%}" for k, v in p["weights"].items())
         + "。信号用规则 ①,减半时比特币和股票各打五折,腾出的钱放现金,美债不动。", "",
         "宽松加码档(三档 A):综合指数 20 日均值 >70 连续 20 天 → 改为" + "、".join(f"{k} {v:.0%}" for k, v in p["loose_weights"].items())
         + ";跌回 60 以下连续 20 天 → 恢复正常;减半信号优先。三档 B:联储总资产 13 周增长 >5% 连续 10 天 → 加码,13 周变化转负连续 20 天 → 退出。"
         "三档 D(主方案):美联储任一大规模宽松事件(QE、紧急贷款工具、停止/放慢缩表、买短债等,共 17 次)的次日加码,持有约一年;"
         "对照:只认 QE 的 7 次;E 净流动性 13 周增量高于过去 3 年 90% 分位连续 10 天。"
         "两个加码条件都是在看结果之前定的。", "",
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
    L += ["", f"今天({t['date']})主方案信号 { {0.5: '减半', 1.0: '正常', 2.0: '宽松加码'}.get(t['signal'], t['signal'])},对应比例:"
          + "、".join(f"{k} {v:.0%}" for k, v in t["alloc"].items()) + "。", ""]
    L += [f"### 首页所用加码档({p['guide']['mode']})在历史上出现的时段", "", "| 开始 | 结束 | 天数 | 期间美股 VTI | 期间 VTI 最大回撤 |", "|---|---|---|---|---|"]
    for a, b, n, v, d in p.get("loose_segments", []):
        L.append(f"| {a} | {b} | {n} | {f(v)} | {f(d)} |")
    L += ["", "### 参考:2020–2023 用 ETF 前的比特币跑同一套(那时比特币还不受美元流动性驱动,只作参考)", "",
          "| 方案 | 总收益 | 年化 | 最大回撤 | 夏普 |", "|---|---|---|---|---|"]
    for r in p.get("ref_2020", []):
        L.append(f"| {r[0]} | {f(r[1])} | {f(r[2], False)} | {f(r[3])} | {r[4]:.2f} |")
    L.append("")
    return L
