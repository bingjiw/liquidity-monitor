"""观察指标:回购管道分档、反抽体检、跨资产砍仓检测、事件日历、券商融资余额。

这些指标来自对外部分析(视频)的对比讨论,只做展示和预警,**不进仓位规则**。
它们的历史检验在 backtest/research.py,结果见 output/research_summary.md。
"""
from __future__ import annotations

import io
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# 回购管道分档(SOFR − IORB,BP)
# ---------------------------------------------------------------------------
PIPE_TIERS = [  # (上限, 名称, 颜色变量)
    (0, "宽松(低于 IORB)", "--loose"),
    (3, "正常", "--neutral"),
    (5, "偏紧", "--semitight"),
    (10, "紧张", "--tight"),
    (float("inf"), "瘫痪", "--tight"),
]


def pipe_tier(bp: float) -> tuple[str, str]:
    if pd.isna(bp):
        return "无数据", "--neutral"
    for hi, name, color in PIPE_TIERS:
        if (bp <= 0) if hi == 0 else (bp < hi):
            return name, color
    return PIPE_TIERS[-1][1], PIPE_TIERS[-1][2]


# 原始序列 -> 衍生列,用来标注每个指标的最新数据日期
ASOF_SRC = {
    "reserves": "WRESBAL", "reserves_chg": "WRESBAL", "fed_assets": "WALCL", "net_liq": "WALCL",
    "tga": "TGA_DAILY", "tga_chg": "TGA_DAILY", "rrp": "RRPONTSYD", "fed_bills_chg": "WSHOBL",
    "sofr_iorb": "SOFR", "sofr99_iorb": "SOFR99", "tgcr_rrp": "TGCR", "srf": "RPONTSYD",
    "ccc_oas": "BAMLH0A3HYC", "ccc_yield": "BAMLH0A3HYCEY", "hy_oas": "BAMLH0A0HYM2", "ig_oas": "BAMLC0A0CM",
    "ccc_bb_gap": "BAMLH0A1HYBB", "ccc_cost_approx": "BAMLH0A3HYC", "ust5": "DGS5",
    "core_pce_yoy": "PCEPILFE", "real_policy_rate": "PCEPILFE", "tips5": "DFII5", "tips10": "DFII10",
    "ust2_chg": "DGS2", "term_premium": "THREEFYTP10", "vix": "VIXCLS", "usd_chg": "DTWEXBGS",
    "usdjpy": "DEXJPUS", "usdjpy_chg": "DEXJPUS", "nfci": "NFCI", "stlfsi": "STLFSI4",
}


def asof_dates(raw: pd.DataFrame) -> dict[str, str]:
    out = {}
    for k, src in ASOF_SRC.items():
        if src in raw and raw[src].notna().any():
            out[k] = raw[src].dropna().index.max().strftime("%Y-%m-%d")
    return out


def _last(s: pd.Series):
    s = s.dropna()
    return (s.index[-1], float(s.iloc[-1])) if len(s) else (None, float("nan"))


def _not_new_high(s: pd.Series, n: int = 20) -> tuple[bool | None, float, float, str | None]:
    """最新值是否不高于此前 n 个观测的最高值。返回 (是否通过, 最新值, 前高, 日期)。"""
    s = s.dropna()
    if len(s) < n + 1:
        return None, float("nan"), float("nan"), None
    prev = float(s.iloc[-n - 1:-1].max())
    v = float(s.iloc[-1])
    return v <= prev, v, prev, s.index[-1].strftime("%Y-%m-%d")


def load_market(out: Path) -> pd.DataFrame:
    p = out / "market.csv"
    return pd.read_csv(p, index_col=0, parse_dates=True) if p.exists() else pd.DataFrame()


def best_vix(raw: pd.DataFrame, m: pd.DataFrame) -> pd.Series:
    """FRED 的 VIX 通常滞后 1–2 天;Yahoo 有当天收盘就用它补上最新几天。"""
    fred = raw["VIXCLS"].dropna() if "VIXCLS" in raw else pd.Series(dtype=float)
    yah = m["VIX_Y"].dropna() if "VIX_Y" in m else pd.Series(dtype=float)
    return fred.combine_first(yah).sort_index()


def streak_above(raw_bp: pd.Series) -> int:
    n = 0
    for v in raw_bp.dropna()[::-1]:
        if v > 0:
            n += 1
        else:
            break
    return n


# ---------------------------------------------------------------------------
# 跨资产砍仓检测(黄金 vs 纳指)
# ---------------------------------------------------------------------------
DD_THRESH = -0.01     # 同日双跌:黄金、纳指各跌超 1%
CORR_HI = 0.5         # 20 日收益相关性高于 0.5 视为“同涨同跌”
QUADRANTS = {
    (True, True): ("同涨", "砍仓压力暂停;只能说明是反抽还是回暖,要看管道和信用"),
    (False, True): ("股跌金稳", "周期或基本面问题,不是流动性挤兑"),
    (True, False): ("股涨金跌", "风险偏好回升,资金从避险资产流出"),
    (False, False): ("同跌", "无差别砍仓的特征:为补保证金先卖最好卖的"),
}


def cross_asset(m: pd.DataFrame, tips10: pd.Series) -> dict | None:
    if not {"GLD", "QQQ"} <= set(m.columns):
        return None
    px = m[["GLD", "QQQ"]].dropna()
    if len(px) < 30:
        return None
    r = px.pct_change().dropna()
    corr = r["GLD"].rolling(20).corr(r["QQQ"])
    both = (r["GLD"] < DD_THRESH) & (r["QQQ"] < DD_THRESH)
    dd20 = int(both.iloc[-20:].sum())
    r5 = px.iloc[-1] / px.iloc[-6] - 1
    quad = QUADRANTS[(bool(r5["QQQ"] >= 0), bool(r5["GLD"] >= 0))]
    t = tips10.dropna()
    tips_chg = float((t.iloc[-1] - t.iloc[-6]) * 100) if len(t) > 6 else float("nan")
    diverge = pd.notna(tips_chg) and tips_chg < 0 and r5["GLD"] < -0.02
    hist = corr.iloc[-520:]
    return {
        "date": px.index[-1].strftime("%Y-%m-%d"),
        "corr20": round(float(corr.iloc[-1]), 2),
        "dd20": dd20,
        "dd_days": [d.strftime("%Y-%m-%d") for d in both.iloc[-20:][both.iloc[-20:]].index],
        "qqq5": round(float(r5["QQQ"]), 4), "gld5": round(float(r5["GLD"]), 4),
        "quadrant": quad[0], "quadrant_note": quad[1],
        "tips10_5d_bp": None if pd.isna(tips_chg) else round(tips_chg, 0),
        "gold_rate_diverge": bool(diverge),
        "alarm": bool((corr.iloc[-1] > CORR_HI and dd20 >= 2) or (quad[0] == "同跌" and dd20 >= 2)),
        "hist": {"dates": [d.strftime("%Y-%m-%d") for d in hist.index],
                 "corr20": [None if pd.isna(v) else round(float(v), 3) for v in hist]},
    }


# ---------------------------------------------------------------------------
# 事件日历
# ---------------------------------------------------------------------------
# FOMC 议息日(决议当天)。2027 年按 2026-11-12 修订版日程,10 月为不做决议的战略会议。
FOMC = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28",
        "2026-12-09", "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09", "2027-07-28", "2027-09-15",
        "2027-11-17", "2028-01-26"]
# 其他已知事件(需要人工维护)
KNOWN = {
    "2026-11-26": "感恩节休市",
    "2026-11-27": "感恩节次日,美股半天",
    "2026-12-11": "纽约联储公布下一期购债计划(11/16–12/11 期 RMP 为零,看是否恢复)",
    "2026-12-24": "平安夜,美股半天",
    "2026-12-25": "圣诞节休市",
    "2027-01-01": "元旦休市",
}


def third_friday(y: int, m: int) -> date:
    d = date(y, m, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def events(days: int = 30, today: date | None = None) -> dict[date, list[str]]:
    today = today or date.today()
    end = today + timedelta(days=days)
    ev: dict[date, list[str]] = {}
    add = lambda d, t: ev.setdefault(d, []).append(t) if today <= d <= end else None  # noqa: E731
    for s in FOMC:
        add(date.fromisoformat(s), "FOMC 议息决议")
    for s, t in KNOWN.items():
        add(date.fromisoformat(s), t)
    for y in (today.year, today.year + 1):
        for mth in range(1, 13):
            tf = third_friday(y, mth)
            add(tf, "月度期权到期" + ("(季度到期,量大)" if mth in (3, 6, 9, 12) else ""))
            me = (pd.Timestamp(y, mth, 1) + pd.offsets.BMonthEnd(0)).date()
            add(me, "年末" if mth == 12 else "季末" if mth in (3, 6, 9) else "月末")
        for mth in (1, 3, 4, 6, 9, 12):
            add(date(y, mth, 15), "缴税日")
    return ev


def fetch_maturities(days: int = 30) -> pd.Series:
    """未来 N 天每天到期的国债金额(十亿美元,按拍卖记录加总,约数)。"""
    today = date.today()
    end = today + timedelta(days=days)
    r = requests.get("https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query",
                     params={"filter": f"maturity_date:gte:{today.isoformat()},maturity_date:lte:{end.isoformat()}",
                             "fields": "maturity_date,cusip,offering_amt", "page[size]": 10000}, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json().get("data", []))
    if df.empty:
        return pd.Series(dtype=float)
    df["maturity_date"] = pd.to_datetime(df["maturity_date"], errors="coerce")
    df["offering_amt"] = pd.to_numeric(df["offering_amt"].replace({"null": None}), errors="coerce")
    df = df.drop_duplicates()
    return df.groupby("maturity_date")["offering_amt"].sum() / 1e9


# ---------------------------------------------------------------------------
# FINRA 券商融资余额(月度,约滞后 3 周)
# ---------------------------------------------------------------------------
FINRA_PAGE = "https://www.finra.org/finra-data/browse-catalog/industry-snapshot/margin-statistics"
# 页面链接常由脚本生成,抓不到时按历史上用过的固定地址依次尝试
FINRA_XLSX = ["https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx",
              "https://www.finra.org/sites/default/files/margin-statistics.xlsx"]


def fetch_margin(cache: Path) -> pd.Series:
    path = cache / "FINRA_MARGIN.csv"
    try:
        h = {"User-Agent": "Mozilla/5.0"}
        page = requests.get(FINRA_PAGE, headers=h, timeout=30).text
        links = re.findall(r'["\'(]([^"\'()]*margin[^"\'()]*\.xlsx)', page, flags=re.I)
        cands = [u if u.startswith("http") else "https://www.finra.org" + u for u in links] + FINRA_XLSX
        x, errs = None, []
        for url in dict.fromkeys(cands):
            try:
                rr = requests.get(url, headers=h, timeout=60)
                rr.raise_for_status()
                for hdr in range(5):  # 表头前可能有标题行
                    x = pd.read_excel(io.BytesIO(rr.content), sheet_name=0, header=hdr)
                    if any("debit" in str(c).lower() for c in x.columns):
                        break
                break
            except Exception as e:  # noqa: BLE001
                errs.append(f"{url}: {e}")
        if x is None:
            raise RuntimeError("; ".join(errs)[:500])
        dcol = next(c for c in x.columns if "month" in str(c).lower() or "date" in str(c).lower())
        vcol = next(c for c in x.columns if "debit" in str(c).lower() and "margin" in str(c).lower())
        s = pd.Series(pd.to_numeric(x[vcol], errors="coerce").values,
                      index=pd.to_datetime(x[dcol].astype(str), errors="coerce")).dropna()
        s = s[s.index.notna()].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        if s.empty:
            raise RuntimeError("解析结果为空")
        cache.mkdir(parents=True, exist_ok=True)
        s.rename("margin_debit_mn").to_csv(path, header=True)
        return s
    except Exception as e:  # noqa: BLE001
        print(f"[警告] FINRA 融资余额获取失败,尝试用缓存: {e}", file=sys.stderr)
        if path.exists():
            return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        return pd.Series(dtype=float)


def margin_summary(s: pd.Series) -> dict | None:
    s = s.dropna()
    if len(s) < 13:
        return None
    v, peak = float(s.iloc[-1]), float(s.max())
    return {"month": s.index[-1].strftime("%Y-%m"), "value_bn": round(v / 1e3, 0),
            "yoy": round(v / float(s.iloc[-13]) - 1, 4), "from_peak": round(v / peak - 1, 4),
            "peak_month": s.idxmax().strftime("%Y-%m")}


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def compute(f: pd.DataFrame, raw: pd.DataFrame, out: Path, offline: bool = False) -> dict:
    m = load_market(out)
    sofr_bp = ((raw.get("SOFR") - raw.get("IORB")) * 100).dropna() if {"SOFR", "IORB"} <= set(raw) else pd.Series(dtype=float)
    sofr_bp = sofr_bp.round(0)
    d_pipe, bp = _last(sofr_bp)
    tier, color = pipe_tier(bp)
    vix = best_vix(raw, m)
    ccc = raw["BAMLH0A3HYC"].dropna() if "BAMLH0A3HYC" in raw else pd.Series(dtype=float)
    v_ok, v_now, v_prev, v_date = _not_new_high(vix)
    c_ok, c_now, c_prev, c_date = _not_new_high(ccc)
    conds = [
        {"name": "回购管道没恶化:SOFR − IORB < 3BP", "ok": None if pd.isna(bp) else bool(bp < 3),
         "value": f"{bp:.0f}BP" if pd.notna(bp) else "—", "date": d_pipe.strftime("%Y-%m-%d") if d_pipe is not None else None},
        {"name": "恐慌没创新高:VIX 不高于前 20 日最高收盘", "ok": v_ok,
         "value": f"{v_now:.2f}(前高 {v_prev:.2f})" if v_ok is not None else "—", "date": v_date},
        {"name": "垃圾债没创新高:CCC OAS 不高于前 20 日最高", "ok": c_ok,
         "value": f"{c_now:.2f}%(前高 {c_prev:.2f}%)" if c_ok is not None else "—", "date": c_date},
    ]
    passed = sum(1 for c in conds if c["ok"])
    srf5 = float(f["srf"].iloc[-5:].max()) if "srf" in f else float("nan")
    hist = sofr_bp.iloc[-260:]
    margin = None if offline else margin_summary(fetch_margin(out / "cache"))
    if offline and (out / "cache" / "FINRA_MARGIN.csv").exists():
        margin = margin_summary(pd.read_csv(out / "cache" / "FINRA_MARGIN.csv", index_col=0, parse_dates=True).iloc[:, 0])
    try:
        mats = pd.Series(dtype=float) if offline else fetch_maturities(30)
    except Exception as e:  # noqa: BLE001
        print(f"[警告] 到期数据获取失败: {e}", file=sys.stderr)
        mats = pd.Series(dtype=float)
    return {
        "pipe": {"bp": None if pd.isna(bp) else bp, "date": d_pipe.strftime("%Y-%m-%d") if d_pipe is not None else None,
                 "tier": tier, "color": color, "streak": streak_above(sofr_bp),
                 "sofr99": None if pd.isna(f["sofr99_iorb"].dropna().iloc[-1]) else round(float(f["sofr99_iorb"].dropna().iloc[-1]), 0),
                 "srf5": round(srf5, 2),
                 "hist": {"dates": [d.strftime("%Y-%m-%d") for d in hist.index], "bp": [float(v) for v in hist]}},
        "rebound": {"conds": conds, "passed": passed, "total": len(conds)},
        "cross": cross_asset(m, raw["DFII10"] if "DFII10" in raw else pd.Series(dtype=float)),
        "vix_latest": {"value": None if pd.isna(vix.iloc[-1]) else round(float(vix.iloc[-1]), 2),
                       "date": vix.index[-1].strftime("%Y-%m-%d")} if len(vix) else None,
        "margin": margin,
        "maturities": {d.strftime("%Y-%m-%d"): round(float(v), 1) for d, v in mats.items()},
    }


def report_lines(w: dict, research_note: str | None = None) -> list[str]:
    L = ["## 观察指标(未经回测验证,不进仓位规则)", ""]
    p = w["pipe"]
    L.append(f"- **回购管道**:SOFR − IORB = {p['bp'] if p['bp'] is not None else '—'}BP({p['date']}),"
             f"分档「{p['tier']}」;已连续 {p['streak']} 天高于 IORB;SOFR99 − IORB {p['sofr99']}BP;SRF 近 5 日最高 {p['srf5']} 十亿美元。")
    L.append("  分档:≤0 宽松 / 0–3 正常 / 3–5 偏紧 / 5–10 紧张 / ≥10 瘫痪。")
    rb = w["rebound"]
    L.append(f"- **反抽体检**:满足 {rb['passed']}/{rb['total']} 条")
    for c in rb["conds"]:
        mark = "—" if c["ok"] is None else ("✓" if c["ok"] else "✗")
        L.append(f"  - {mark} {c['name']}:{c['value']}({c['date'] or '无数据'})")
    x = w.get("cross")
    if x:
        L.append(f"- **跨资产砍仓检测**({x['date']}):GLD 与 QQQ 20 日收益相关 {x['corr20']:+.2f};"
                 f"近 20 日同日双跌(各跌超 1%){x['dd20']} 天;近 5 日 QQQ {x['qqq5']:+.1%}、GLD {x['gld5']:+.1%} →「{x['quadrant']}」:{x['quadrant_note']}。"
                 + (f" 10 年 TIPS 5 日变化 {x['tips10_5d_bp']:+.0f}BP。" if x['tips10_5d_bp'] is not None else "")
                 + (" **黄金在实际利率下降时下跌,疑似被砍仓。**" if x["gold_rate_diverge"] else "")
                 + (" **砍仓警报。**" if x["alarm"] else ""))
    else:
        L.append("- **跨资产砍仓检测**:缺少 GLD 价格数据")
    if w.get("vix_latest"):
        L.append(f"- 最新 VIX(含 Yahoo 当日收盘):{w['vix_latest']['value']}({w['vix_latest']['date']})")
    mg = w.get("margin")
    if mg:
        L.append(f"- **券商融资余额(FINRA,月度)**:{mg['month']} 为 {mg['value_bn']:,.0f} 十亿美元,"
                 f"同比 {mg['yoy']:+.1%},距 {mg['peak_month']} 峰值 {mg['from_peak']:+.1%}。")
    if research_note:
        L.append(f"- {research_note}")
    L.append("")
    return L


def calendar_rows(cal: pd.DataFrame, w: dict | None, days: int = 30) -> list[dict]:
    """把国债结算、到期、FOMC、期权到期、缴税日、月末等合并成一张日历。"""
    ev = events(days)
    mats = (w or {}).get("maturities", {}) or {}
    cmap = {r.issue_date.date(): r for r in cal.itertuples()} if not cal.empty else {}
    out = []
    for d in sorted(set(ev) | set(cmap) | {date.fromisoformat(k) for k in mats}):
        r = cmap.get(d)
        gross = float(r.gross_bn) if r is not None and pd.notna(r.gross_bn) else None
        mat = mats.get(d.isoformat())
        out.append({"date": d.isoformat(), "gross_bn": None if gross is None else round(gross, 1),
                    "pending": r is not None and gross is None,
                    "mature_bn": mat, "net_bn": None if gross is None or mat is None else round(gross - mat, 1),
                    "events": ev.get(d, []), "detail": r.detail if r is not None else ""})
    return out
