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

START = "2019-01-01"
UP_DAY = 0.03          # 纳指单日大涨阈值
DD = -0.01             # 同日双跌:各跌超 1%
CORR_HI = 0.5

# 美联储流动性相关的政策动作(公告日)。周末公告以下一个交易日为第一天,基准用公告前最后一个收盘。
EVENTS = [
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
    for c in ("sofr_iorb", "sofr99_iorb", "ccc_oas", "hy_oas", "tips10", "tips5"):
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
            "note": f"数据从 {start.date() if start is not None else '—'} 起(IORB 2021-07 才有)。用前一天公布的利差。"
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
            "note": "三条:SOFR − IORB < 3BP(2021-07 前无数据,按满足计)、VIX 不高于前 20 日最高、HYG 不低于前 20 日最低(代替 CCC 利差,"
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
            "note": "基准 = 公告前最后一个收盘。事件只有 11 次,其中 2020-03 两次是史上最大放水,会拉高平均数,看中位数和上涨次数更稳。",
            "headers": headers, "rows": rows,
            "summary": {"headers": ["资产", "样本", "5 日平均", "5 日上涨次数", "样本", "60 日平均", "60 日上涨次数", "60 日中位数"],
                        "rows": summ}}


def _md_table(h, rows):
    L = ["| " + " | ".join(map(str, h)) + " |", "|" + "---|" * len(h)]
    L += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return L


def run(f: pd.DataFrame | None = None, out: Path = ROOT / "output") -> dict:
    F, px = load(f)
    sections = [study_pipe(F, px)]
    if "HYG" in px:
        sections.append(study_rebound(F, px))
    if "GLD" in px:
        sections.append(study_cross(F, px))
    sections.append(study_events(px))
    L = ["# 新指标历史检验(视频对比讨论的第二批)", "",
         f"数据 {START} ~ {px['SPY'].index[-1].date()}。只做研究,不进仓位规则。样本小的结论只能当参考。", ""]
    for s in sections:
        L += [f"## {s['title']}", "", s["note"], ""] + _md_table(s["headers"], s["rows"]) + [""]
        if s.get("summary"):
            L += ["汇总:", ""] + _md_table(s["summary"]["headers"], s["summary"]["rows"]) + [""]
        if s.get("detail"):
            L += ["最近 15 次大涨:", ""] + _md_table(s["detail"]["headers"], s["detail"]["rows"]) + [""]
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
