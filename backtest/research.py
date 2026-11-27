"""第二批:检验视频里提出的几类新指标,看它们历史上有没有用。只做研究,不进仓位规则。

1. 回购管道分档(SOFR − IORB):各档之后美股的涨跌和回撤。
2. 反抽体检:纳指单日大涨之后,"管道 / 恐慌 / 信用 不再恶化"满足几条,与之后是反转还是反抽的关系。
3. 跨资产砍仓:黄金与纳指同跌、相关性抬升之后,美股和黄金的表现("砍仓后黄金必然修复"是否成立)。
4. 美联储流动性干预事件:之后黄金、高收益债、半导体、纳指谁先涨、谁涨最多。

防偷看:FRED 序列按公布滞后后移一天(VIX 和 ETF 价格当天收盘可得)。
输出:output/research_summary.md、output/research.json(仪表盘读取)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

START = "2004-01-02"
UP_DAY = 0.03          # 纳指单日大涨阈值
DD = -0.01             # 同日双跌:各跌超 1%
CORR_HI = 0.5

# 美联储流动性相关的政策动作(公告日)。周末公告以下一个交易日为第一天,基准用公告前最后一个收盘。
EVENTS = [
    ("2008-03-16", "贝尔斯登救助,推出一级交易商融资工具 PDCF"),
    ("2008-10-08", "全球央行联合降息"),
    ("2008-11-25", "宣布购买 MBS 和机构债(QE1)"),
    ("2009-03-18", "QE1 扩大并开始买国债"),
    ("2010-11-03", "QE2:6000 亿美元国债"),
    ("2012-09-13", "QE3:每月 400 亿 MBS,不设上限"),
    ("2019-01-30", "FOMC 转向“耐心”,表示缩表可灵活调整"),
    ("2019-03-20", "FOMC 宣布 9 月停止缩表"),
    ("2019-07-31", "首次降息并提前结束缩表"),
    ("2019-09-17", "回购利率飙升,纽约联储重启隔夜回购操作"),
    ("2019-10-11", "宣布每月 600 亿美元购买短债(补充准备金)"),
    ("2020-03-15", "紧急降息到 0 并重启 7000 亿 QE"),
    ("2020-03-23", "无限量 QE 并设立公司债工具"),
    ("2023-03-12", "SVB 后推出 BTFP 银行定期融资工具"),
    ("2024-05-01", "宣布放慢缩表(国债每月上限 600 亿降到 250 亿)"),
    ("2025-10-29", "宣布 12 月 1 日停止缩表"),
    ("2025-12-10", "宣布开始补充准备金购买短债(RMP)"),
]
EVENT_ASSETS = [("GLD", "黄金"), ("HYG", "高收益债"), ("SOXX", "半导体"), ("QQQ", "纳指 100"), ("SPY", "标普 500")]


def _fwd(px: pd.Series, h: int) -> pd.Series:
    return px.shift(-h) / px - 1


def _fwd_mdd(px: pd.Series, h: int) -> pd.Series:
    """之后 h 个交易日内相对今天收盘的最大跌幅。"""
    fut_min = px[::-1].rolling(h, min_periods=1).min()[::-1].shift(-1)
    return fut_min / px - 1


def _episodes(mask: pd.Series) -> int:
    m = mask.fillna(False).astype(int)
    return int(((m.diff().fillna(m.iloc[0] if len(m) else 0)) == 1).sum())


def _pct(v, nd=1, sign=True):
    return "—" if v is None or pd.isna(v) else (f"{v:+.{nd}%}" if sign else f"{v:.{nd}%}")


def load(f: pd.DataFrame | None = None):
    if f is None:
        f = pd.read_csv(ROOT / "output" / "raw_data.csv", index_col=0, parse_dates=True)
    m = pd.read_csv(ROOT / "output" / "market.csv", index_col=0, parse_dates=True)
    m = m[~m.index.duplicated(keep="last")].sort_index()
    idx = m["SPY"].dropna().index
    idx = idx[idx >= START]
    F = f.reindex(f.index.union(idx)).sort_index().ffill().reindex(idx)
    # FRED 日度序列次日早上才公布:当天收盘决策只能用前一天的值
    for c in ("sofr_iorb", "sofr99_iorb", "ccc_oas", "hy_oas", "tips10", "tips5", "baa_spread"):
        if c in F:
            F[c] = F[c].shift(1)
    vix = F["vix"]
    if "VIX_Y" in m:  # FRED VIX 与 CBOE 收盘相同,Yahoo 补缺
        vix = vix.combine_first(m["VIX_Y"].reindex(idx))
    F["vix_close"] = vix
    px = {k: m[k].reindex(idx).ffill() for k in ["SPY", "QQQ", "GLD", "HYG", "SOXX", "IWM"] if k in m and m[k].notna().sum() > 250}
    return F, px


def group_table(labels: pd.Series, px: dict, order: list[str], assets=("SPY", "QQQ")) -> dict:
    headers = ["状态", "天数", "独立段数"]
    for a in assets:
        headers += [f"{a} 后 5 日", f"{a} 后 20 日", f"{a} 后 60 日", f"{a} 60 日上涨概率", f"{a} 20 日内最大跌幅均值"]
    rows = []
    for lab in order + ["全部"]:
        mask = labels.notna() if lab == "全部" else labels == lab
        if mask.sum() == 0:
            continue
        row = [lab, int(mask.sum()), "—" if lab == "全部" else _episodes(mask)]
        for a in assets:
            p = px[a]
            f5, f20, f60, mdd = (_fwd(p, 5)[mask], _fwd(p, 20)[mask], _fwd(p, 60)[mask], _fwd_mdd(p, 20)[mask])
            row += [_pct(f5.mean()), _pct(f20.mean()), _pct(f60.mean()),
                    _pct((f60.dropna() > 0).mean(), 0, False) if f60.notna().any() else "—", _pct(mdd.mean())]
        rows.append(row)
    return {"headers": headers, "rows": rows}


def study_pipe(F, px) -> dict:
    bp = F["sofr_iorb"]
    lab = pd.Series(np.select([bp <= 0, bp < 3, bp < 5, bp < 10, bp >= 10],
                              ["≤0 宽松", "0–3 正常", "3–5 偏紧", "5–10 紧张", "≥10 瘫痪"], default=""), index=bp.index)
    lab = lab.where(bp.notna() & (lab != ""))
    t = group_table(lab, px, ["≤0 宽松", "0–3 正常", "3–5 偏紧", "5–10 紧张", "≥10 瘫痪"])
    start = bp.dropna().index.min()
    return {"title": "1. 回购管道分档:SOFR − IORB 各档之后的美股表现",
            "note": f"数据从 {start.date() if start is not None else '—'} 起(SOFR 2018-04 才有;2021-07 前用 IOER 代替 IORB)。用前一天公布的利差。"
                    "“独立段数”= 连续处于该档算一段,段数少说明样本其实很小。", **t}


def study_rebound(F, px) -> dict:
    q = px["QQQ"]
    r = q.pct_change()
    big = r >= UP_DAY
    vix = F["vix_close"]
    c_pipe = F["sofr_iorb"] < 3
    c_vix = vix <= vix.shift(1).rolling(20).max()
    # 信用:CCC OAS 只有约 3 年,用高收益债 ETF(HYG)不创 20 日新低作为全期代理
    hyg = px.get("HYG")
    c_cred = hyg >= hyg.shift(1).rolling(20).min() if hyg is not None else pd.Series(np.nan, index=q.index)
    n_ok = c_vix.astype(int) + c_cred.astype(int) + c_pipe.where(F["sofr_iorb"].notna(), True).astype(int)
    prior_low = q.shift(1).rolling(20).min()
    fut_min60 = q[::-1].rolling(60, min_periods=1).min()[::-1].shift(-1)
    fail = fut_min60 < prior_low  # 60 天内跌破大涨前 20 日低点 = 反抽失败
    days = q.index[big.fillna(False)]
    headers = ["条件满足数", "大涨天数", "QQQ 后 20 日", "QQQ 后 60 日", "60 日上涨概率", "60 日内跌破前低(反抽失败)比例"]
    rows = []
    for lab, sel in [("3 条都满足", n_ok[days] == 3), ("2 条", n_ok[days] == 2), ("0–1 条", n_ok[days] <= 1), ("全部大涨日", n_ok[days] >= 0)]:
        d = days[sel.values]
        if len(d) == 0:
            continue
        f20, f60 = _fwd(q, 20)[d], _fwd(q, 60)[d]
        rows.append([lab, len(d), _pct(f20.mean()), _pct(f60.mean()),
                     _pct((f60.dropna() > 0).mean(), 0, False) if f60.notna().any() else "—",
                     _pct(fail[d][f60.notna()].mean(), 0, False) if f60.notna().any() else "—"])
    detail = [[d.strftime("%Y-%m-%d"), _pct(r[d]), int(n_ok[d]),
               "—" if pd.isna(F["sofr_iorb"][d]) else f"{F['sofr_iorb'][d]:.0f}",
               "✓" if c_vix[d] else "✗", "✓" if c_cred[d] else "✗",
               _pct(_fwd(q, 60)[d]), "—" if pd.isna(_fwd(q, 60)[d]) else ("✗ 失败" if fail[d] else "✓ 守住")]
              for d in days[-15:]]
    return {"title": f"2. 反抽体检:QQQ 单日涨超 {UP_DAY:.0%} 之后",
            "note": "三条:SOFR − IORB < 3BP(2018-04 前无数据,按满足计)、VIX 不高于前 20 日最高、HYG 不低于前 20 日最低(代替 CCC 利差,"
                    "因为 CCC 在 FRED 只有约 3 年)。“反抽失败”= 之后 60 天内跌破大涨前 20 日的最低收盘。",
            "headers": headers, "rows": rows,
            "detail": {"headers": ["日期", "QQQ 当日", "满足", "管道 BP", "VIX", "信用", "后 60 日", "结果"], "rows": detail}}


def study_cross(F, px) -> dict:
    g, q = px["GLD"], px["QQQ"]
    rg, rq = g.pct_change(), q.pct_change()
    corr = rg.rolling(20).corr(rq)
    both = (rg < DD) & (rq < DD)
    dd20 = both.astype(int).rolling(20).sum()
    lab = pd.Series(np.where(dd20 >= 2, "近 20 日同跌 ≥2 天", np.where(corr > CORR_HI, "相关 >0.5(未同跌)", "其他")), index=q.index)
    lab = lab.where(corr.notna())
    order = ["近 20 日同跌 ≥2 天", "相关 >0.5(未同跌)", "其他"]
    t = group_table(lab, px, order, assets=("QQQ", "GLD"))
    # 黄金在实际利率下降时下跌
    tips = F.get("tips10")
    div = None
    if tips is not None and tips.notna().sum() > 250:
        dv = (tips.diff(5) < 0) & (g / g.shift(5) - 1 < -0.02)
        d = dv[dv.fillna(False)].index
        div = {"days": int(len(d)), "episodes": _episodes(dv),
               "gld20": _pct(_fwd(g, 20)[d].mean()), "gld60": _pct(_fwd(g, 60)[d].mean()),
               "base20": _pct(_fwd(g, 20).mean()), "base60": _pct(_fwd(g, 60).mean())}
    return {"title": "3. 跨资产砍仓:黄金与纳指同跌之后", "note":
            "同跌 = 当天 GLD、QQQ 各跌超 1%。看两件事:同跌之后美股是否更危险;“砍仓结束后黄金会修复”是否成立(看 GLD 之后的表现)。",
            **t, "diverge": div}


def study_diverge(F, px) -> dict:
    """信用走阔而美股仍在一年高点附近("2007 式背离")之后的表现。用 Baa 利差,因为它 2004 年起都有。"""
    sys.path.insert(0, str(ROOT))
    from watch import DIV_BAA_BP, DIV_NEAR_HIGH
    spy = px["SPY"]
    near = spy >= DIV_NEAR_HIGH * spy.rolling(252, min_periods=200).max()
    wide = F["baa_spread"].diff(63) * 100 >= DIV_BAA_BP
    lab = pd.Series(np.where(wide & near, "背离:信用走阔,SPY 仍在高点附近",
                             np.where(wide, "信用走阔,SPY 已明显回落", "其他")), index=spy.index)
    lab = lab.where(F["baa_spread"].diff(63).notna() & spy.rolling(252, min_periods=200).max().notna())
    order = ["背离:信用走阔,SPY 仍在高点附近", "信用走阔,SPY 已明显回落", "其他"]
    t = group_table(lab, px, order)
    d = (wide & near).fillna(False)
    starts, last = [], None
    for x in d[d].index:  # 相隔 90 天以上算新的一段
        if last is None or (x - last).days > 90:
            starts.append(x)
        last = x
    mdd = _fwd_mdd(spy, 126)
    detail = [[x.strftime("%Y-%m-%d"), f"{F['baa_spread'].diff(63)[x] * 100:+.0f}", _pct(_fwd(spy, 60)[x]),
               _pct(_fwd(spy, 126)[x]), _pct(mdd[x])] for x in starts]
    return {"title": "7. 信用与股市背离(2007 式):Baa 利差 3 个月走阔、SPY 仍在一年高点附近",
            "note": f"背离 = Baa 公司债 − 10 年美债利差 63 个交易日内走阔 ≥ {DIV_BAA_BP}BP,同时 SPY 距 252 日最高收盘不到 "
                    f"{round((1 - DIV_NEAR_HIGH) * 100)}%。Baa 利差用前一天公布的值。ICE 高收益利差在 FRED 只有约 3 年,所以历史检验用 Baa。",
            **t, "detail_title": "每段背离第一次出现的日子",
            "detail": {"headers": ["日期", "Baa 3 个月变化 BP", "SPY 后 60 日", "SPY 后 126 日", "126 日内最大跌幅"], "rows": detail}}


def study_events(px) -> dict:
    headers = ["日期", "事件"] + [f"{n} 5 日 / 60 日" for _, n in EVENT_ASSETS]
    rows, acc = [], {a: {"r5": [], "r60": []} for a, _ in EVENT_ASSETS}
    idx = px["SPY"].index
    for ds, name in EVENTS:
        d = pd.Timestamp(ds)
        pos = idx.searchsorted(d)  # 公告日(或其后第一个交易日)
        if pos == 0 or pos >= len(idx):
            continue
        base = pos - 1
        row = [ds, name]
        for a, _ in EVENT_ASSETS:
            p = px.get(a)
            if p is None:
                row.append("—"); continue
            r5 = p.iloc[base + 5] / p.iloc[base] - 1 if base + 5 < len(p) else np.nan
            r60 = p.iloc[base + 60] / p.iloc[base] - 1 if base + 60 < len(p) else np.nan
            acc[a]["r5"].append(r5); acc[a]["r60"].append(r60)
            row.append(f"{_pct(r5)} / {_pct(r60)}")
        rows.append(row)
    summ = []
    for a, n in EVENT_ASSETS:
        r5, r60 = pd.Series(acc[a]["r5"]).dropna(), pd.Series(acc[a]["r60"]).dropna()
        if len(r5) == 0:
            continue
        summ.append([n, len(r5), _pct(r5.mean()), f"{int((r5 > 0).sum())}/{len(r5)}",
                     len(r60), _pct(r60.mean()), f"{int((r60 > 0).sum())}/{len(r60)}", _pct(r60.median())])
    return {"title": "4. 美联储流动性干预之后,谁先涨、谁涨最多",
            "note": f"基准 = 公告前最后一个收盘。共 {len(EVENTS)} 次(2008–2012 的 QE 是新补的,视频只统计了 2019–2023)。2020-03 两次是史上最大放水,会拉高平均数,看中位数和上涨次数更稳。高收益债 ETF 2007-04、半导体 ETF 2001 年起有数据。",
            "headers": headers, "rows": rows,
            "summary": {"headers": ["资产", "样本", "5 日平均", "5 日上涨次数", "样本", "60 日平均", "60 日上涨次数", "60 日中位数"],
                        "rows": summ}}


# 截至 2026-11-25 的检验结论(人工撰写;数字每天会小幅变化,结论变了需要更新这里)
VERDICTS = {
    "pipe": "补上 2018-04~2021-07(用 IOER 代替 IORB)以后,各档之间差别很小:≥10BP 之后纳指 60 天 +4.6%、上涨概率 65%,"
            "只比全部日子(+4.8%、72%)略差。之前只用 2021 年以后数据时看到的“≥10BP 之后明显偏弱”没有经受住更长样本。管道分档只能当状态灯,不能用来择时。",
    "rebound": "约 100 次大涨里 96 次三条都满足,三条全满足时 60 天内跌破前低的比例 46%,和全部大涨日一样。"
               "这三条只能确认“没有继续恶化”,分不出反抽和反转。历史检验用 HYG 代替 CCC。",
    "cross": "2004 年以来,金纳同跌密集之后纳指 60 天平均 +8.2%、上涨概率 84%,明显好于平时的 +3.9%、73%:无差别砍仓多数接近短期低点。"
             "黄金在同跌之后 60 天 +2.8%,和平时 +2.7% 一样,“砍仓后黄金必然修复”不成立。黄金逆着实际利率下跌之后,20 天 +1.9%(平时 +0.9%),短期略好。",
    "events": "加入 2008–2012 年 6 次 QE 后共 17 次:高收益债 60 天 16/17 次上涨,最稳定;半导体 60 天平均 +13% 涨幅最大;"
              "黄金 5 天内只有 11/17 次上涨、平均 +0.4%(2008 年几次救市后黄金先跌 5–9%)。"
              "“半导体最后涨最多”成立,“黄金先涨”在更长样本里不成立。",
    "btc": "ETF 上市后,比特币开始对流动性“状态”有反应:综合指数 ≥45 时之后 60 天平均 +11.1%(上涨概率 60%),<30 时 −3.5%(36%);"
           "管道 SOFR−IORB ≥5BP 之后 60 天 −15.8%(14%)。ETF 前这些关系都不存在,指数 <30 时比特币反而上涨。这支持“机构化以后才受流动性影响”的判断。"
           "但月度涨跌和净流动性变化几乎不相关(+0.02),和综合指数变化只有 +0.19,只有 37 个样本,统计上不显著。"
           "按规则操作在 ETF 后把比特币最大回撤从 −53% 降到 −31%~−36%,年化也更高,但只调了 3–5 次仓,运气成分很大。",
    "diverge": "2004 年以来背离共 83 天、8 段。背离日之后 SPY 60 天平均 −1.1%、上涨概率 41%,明显差于平时的 +2.8%、75%;"
               "但 8 段里只有 2007-09 和 2015-05 两次之后半年内跌超 10%,其余 6 次之后半年都在涨。样本只有 8 段,只能当提醒,不能单独用来减仓。",
    "overlay": "补数据后最重要的发现:净流动性规则 ③ 在 2004–2018 年失效(择时增益 −2.2%/年,2008 年满仓跌了 54%),"
               "因为 2008 年联储扩表是对危机的反应,规则在暴跌中满仓、在 2009-03 底部减仓。叠加任何新预警都救不回来。"
               "新预警在 2019 年以后也没提高择时增益,只有 HYG 新低和 VIX ≥30 能降低回撤。",
}


def study_overlay() -> dict:
    """把新预警当作过滤条件叠加到净流动性规则 ③,看能否提升择时(各扣 25%,参数事先定,不调)。"""
    import backtest as B
    idx, F, S, cash, rets, spy = B.load()
    m = pd.read_csv(ROOT / "output" / "market.csv", index_col=0, parse_dates=True)
    m = m[~m.index.duplicated(keep="last")]
    q, g, h = (m[k].reindex(idx).ffill() for k in ("QQQ", "GLD", "HYG"))
    hold = lambda t, n=20: t.fillna(False).astype(int).rolling(n, min_periods=1).max().astype(bool)  # noqa: E731
    nl = F["net_liq"].diff(B.NL_WIN)
    base = B.persist(nl < 0, nl > 0, B.CONFIRM, B.LOW_POS)
    vix = F["vix"].combine_first(m["VIX_Y"].reindex(idx)) if "VIX_Y" in m else F["vix"]
    dd = ((g.pct_change() < DD) & (q.pct_change() < DD)).astype(int).rolling(20).sum() >= 2
    pipe = hold(F["sofr_iorb"] >= 10)
    cred = hold(h < h.shift(1).rolling(20).min())
    rules = [("③ 净流动性(现规则)", base),
             ("③ + 管道 ≥10BP 后 20 天扣 25%", base - 0.25 * pipe),
             ("③ + HYG 创 20 日新低后 20 天扣 25%", base - 0.25 * cred),
             ("③ + VIX ≥30 时扣 25%", base - 0.25 * (vix >= 30)),
             ("③ + 金纳同跌 ≥2 天时扣 25%", base - 0.25 * dd),
             ("③ + 金纳同跌 ≥2 天时加回满仓(逆向)", base.where(~dd, 1.0)),
             ("买入持有", pd.Series(1.0, idx))]
    rows = []
    for name, pos in rules:
        pos = pos.clip(0, 1)
        row = [name]
        for a, b in (("2004-01-02", "2019-03-14"), ("2019-03-15", "2022-12-31"), ("2023-01-01", "2100-01-01")):
            p = B.perf(pos, rets["SPY"], cash, a, b)
            c = B.const_benchmark(p["exposure"], rets["SPY"], cash, a, b)
            row += [_pct(p["cagr"], 1, False), _pct(p["cagr"] - c), _pct(p["maxdd"])]
        rows.append(row)
    hd = ["规则"] + [f"{t}{x}" for t in ("2004–18 ", "样本内 ", "样本外 ") for x in ("年化", "择时增益", "最大回撤")]
    return {"title": "5. 把新预警叠加到仓位规则,能否更好(SPY)",
            "note": "每个预警事先定为“扣 25% 仓位”,不调参数。择时增益 = 年化 − 同等平均仓位的年化。管道数据(SOFR)2018-04 才有,HYG 2007 年才有,之前这两个预警不起作用。",
            "headers": hd, "rows": rows}


BTC_START = "2024-01-11"  # 美国现货比特币 ETF 上市


def study_btc() -> dict | None:
    """比特币与流动性:只认现货 ETF 上市之后;ETF 之前(2018–2023)只作对照,检验“机构化后才受流动性影响”这个假设。"""
    import backtest as B
    idx, F, S, cash, rets, spy = B.load()
    m = pd.read_csv(ROOT / "output" / "market.csv", index_col=0, parse_dates=True)
    m = m[~m.index.duplicated(keep="last")]
    if "BTC" not in m or m["BTC"].notna().sum() < 250:
        return None
    Rr, X = B.rules(idx, F, S, spy)
    ks = list(Rr)
    pos = {"③ 净流动性": Rr[ks[3]], "① 综合指数水平": Rr[ks[1]], "⑤ ①或③任一减仓": Rr[ks[-1]]}
    btc = m["BTC"].reindex(idx).ffill()
    q = m["QQQ"].reindex(idx).ffill()
    g = m["GLD"].reindex(idx).ffill() if "GLD" in m else None
    eras = [("ETF 前 2018–2023(对照)", "2018-01-01", "2023-12-31"), ("ETF 后 2024-01~至今", BTC_START, "2100-01-01")]
    # 1) 相关性:每 20 个交易日取一次,不重叠
    nl20 = F["net_liq"].diff(20)
    li20 = S["liquidity_index"].diff(20)
    ret20 = lambda p: p / p.shift(20) - 1  # noqa: E731
    corr_rows = []
    for lab, a, b in eras:
        sl = slice(a, b)
        d = pd.DataFrame({"btc": ret20(btc), "nl": nl20, "li": li20, "qqq": ret20(q),
                          "gld": ret20(g) if g is not None else np.nan})[sl].iloc[::20].dropna(subset=["btc"])
        c = lambda x: f"{d['btc'].corr(d[x]):+.2f}" if d[x].notna().sum() > 5 else "—"  # noqa: E731
        corr_rows.append([lab, len(d), c("nl"), c("li"), c("qqq"), c("gld")])
    # 2) 各状态下之后的表现
    f20, f60 = _fwd(btc, 20), _fwd(btc, 60)
    state_rows = []
    li = S["liquidity_index"]
    states = [("③ 满仓(净流动性扩张)", pos["③ 净流动性"] >= 1), ("③ 半仓(净流动性收缩)", pos["③ 净流动性"] < 1),
              ("综合指数 <30(紧张)", li < 30), ("综合指数 30–45", (li >= 30) & (li < 45)), ("综合指数 ≥45", li >= 45),
              ("管道 SOFR−IORB ≥5BP", F["sofr_iorb"] >= 5)]
    for lab, a, b in eras:
        for name, mask in states:
            mk = mask[a:b].fillna(False)
            x20, x60 = f20[a:b][mk].dropna(), f60[a:b][mk].dropna()
            state_rows.append([lab, name, int(mk.sum()), _episodes(mk), _pct(x20.mean()) if len(x20) else "—",
                               _pct(x60.mean()) if len(x60) else "—",
                               _pct((x60 > 0).mean(), 0, False) if len(x60) else "—"])
    # 3) 按规则操作(只看 ETF 后)
    tm_rows = []
    for asset in ("BTC", "MSTR", "QQQ"):
        if asset not in m:
            continue
        r = m[asset].reindex(idx).ffill().pct_change().fillna(0)
        for name, p_ in [("一直满仓", pd.Series(1.0, idx))] + list(pos.items()):
            pf = B.perf(p_, r, cash, BTC_START, "2100-01-01")
            c = B.const_benchmark(pf["exposure"], r, cash, BTC_START, "2100-01-01")
            yrs = len(r[BTC_START:]) / 252
            tot = (1 + pf["cagr"]) ** yrs - 1
            tm_rows.append([asset, name, _pct(tot, 0), _pct(pf["cagr"], 1, False), _pct(pf["cagr"] - c) if name != "一直满仓" else "—",
                            _pct(pf["maxdd"]), f"{pf['sharpe']:.2f}", _pct(pf["exposure"], 0, False), pf["switches"]])
    rows60 = pd.DataFrame({"b": btc.pct_change(), "q": q.pct_change()}).rolling(60).corr().unstack()["b"]["q"]
    return {"title": "6. 比特币与流动性(只认现货 ETF 上市后的数据)",
            "note": f"现货比特币 ETF {BTC_START} 上市。ETF 前 2018–2023 只用来对照,检验“机构化以后比特币才跟着美元流动性走”这个假设。"
                    "ETF 后只有不到 3 年、约 35 个不重叠的月度样本,任何结论都只能当参考。MSTR 是靠发债囤币的公司,波动约为比特币的 2 倍。",
            "headers": ["时期", "月度样本", "与净流动性 4 周变化", "与综合指数 4 周变化", "与纳指 100", "与黄金"],
            "rows": corr_rows,
            "summary": {"headers": ["时期", "状态", "天数", "段数", "BTC 后 20 日", "BTC 后 60 日", "60 日上涨概率"], "rows": state_rows},
            "detail": {"headers": ["资产", "规则(ETF 后)", "总收益", "年化", "择时增益", "最大回撤", "夏普", "平均仓位", "调仓"], "rows": tm_rows},
            "detail_title": "按规则操作(只算 ETF 上市后)",
            "btc_qqq_corr60": {"dates": [d.strftime("%Y-%m-%d") for d in rows60.index[rows60.index >= "2020-01-01"]],
                               "v": [None if pd.isna(v) else round(float(v), 3) for v in rows60[rows60.index >= "2020-01-01"]]}}


def _md_table(h, rows):
    L = ["| " + " | ".join(map(str, h)) + " |", "|" + "---|" * len(h)]
    L += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return L


def run(f: pd.DataFrame | None = None, out: Path = ROOT / "output") -> dict:
    F, px = load(f)
    sections = [dict(study_pipe(F, px), verdict=VERDICTS["pipe"])]
    if "HYG" in px:
        sections.append(dict(study_rebound(F, px), verdict=VERDICTS["rebound"]))
    if "GLD" in px:
        sections.append(dict(study_cross(F, px), verdict=VERDICTS["cross"]))
    sections.append(dict(study_events(px), verdict=VERDICTS["events"]))
    try:
        sections.append(dict(study_overlay(), verdict=VERDICTS["overlay"]))
    except Exception as e:  # noqa: BLE001
        print(f"[警告] 叠加回测失败: {e}", file=sys.stderr)
    try:
        b = study_btc()
        if b:
            sections.append(dict(b, verdict=VERDICTS.get("btc", "")))
    except Exception as e:  # noqa: BLE001
        print(f"[警告] 比特币检验失败: {e}", file=sys.stderr)
    try:
        sections.append(dict(study_diverge(F, px), verdict=VERDICTS["diverge"]))
    except Exception as e:  # noqa: BLE001
        print(f"[警告] 信用背离检验失败: {e}", file=sys.stderr)
    L = ["# 新指标历史检验(视频对比讨论的第二批)", "",
         f"数据 {START} ~ {px['SPY'].index[-1].date()}。只做研究,不进仓位规则。样本小的结论只能当参考。", ""]
    for s in sections:
        L += [f"## {s['title']}", "", f"**结论:**{s['verdict']}", "", s["note"], ""] + _md_table(s["headers"], s["rows"]) + [""]
        if s.get("summary"):
            L += ["汇总:", ""] + _md_table(s["summary"]["headers"], s["summary"]["rows"]) + [""]
        if s.get("detail"):
            L += [s.get("detail_title", "最近 15 次大涨:"), ""] + _md_table(s["detail"]["headers"], s["detail"]["rows"]) + [""]
        if s.get("diverge"):
            d = s["diverge"]
            L += [f"黄金在 10 年 TIPS 实际利率 5 日下降时 5 日跌超 2%:共 {d['days']} 天({d['episodes']} 段),"
                  f"之后 GLD 20 日 {d['gld20']}、60 日 {d['gld60']}(所有日子平均 20 日 {d['base20']}、60 日 {d['base60']})。", ""]
    (out / "research_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    res = {"asof": px["SPY"].index[-1].strftime("%Y-%m-%d"), "sections": sections}
    (out / "research.json").write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return res


if __name__ == "__main__":
    run()
    print((ROOT / "output" / "research_summary.md").read_text())
