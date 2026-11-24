#!/usr/bin/env python3
"""
美元流动性综合指标(Liquidity Monitor)

从公开免费数据源(FRED、纽约联储、美国财政部)抓取数据,
计算 0-100 的每日流动性指数(50 = 中性,越低越紧),并给出硬预警与结算日历。
不调用任何大模型。

用法:
    pip install pandas requests
    python3 liquidity_monitor.py                # 抓取数据、计算、输出报告
    python3 liquidity_monitor.py --out ./output # 指定输出目录
    python3 liquidity_monitor.py --offline      # 只用本地缓存(调试用)

输出(默认在脚本同目录的 output/ 下):
    liquidity_history.csv   每日综合指数、五个支柱分数、各序列 z 分数
    raw_data.csv            对齐后的原始数据
    report_YYYY-MM-DD.md    当日报告(数值、预警、未来 30 天结算日历)

仅用于研究,不构成投资建议。
"""
from __future__ import annotations

import argparse
import io
import math
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# 配置:权重、序列、阈值都在这里调
# ---------------------------------------------------------------------------

HISTORY_START = "2018-01-01"   # 抓取起点
ZSCORE_WINDOW = 756            # 约 3 年工作日,用于滚动标准化
ZSCORE_MIN_PERIODS = 250       # 至少约 1 年数据才开始出分数
CHANGE_WINDOW = 20             # "四周变化" = 20 个工作日
Z_CLIP = 3.0
MIN_PILLARS = 4                # 至少 4 个支柱有值才出综合指数

PILLAR_WEIGHTS = {
    "A_quantity": 0.25,  # 钱的数量
    "B_repo":     0.20,  # 回购管道的价格
    "C_credit":   0.25,  # 信用的价格
    "D_rates":    0.15,  # 利率与政策
    "E_stress":   0.15,  # 市场压力与美元
}

PILLAR_NAMES = {
    "A_quantity": "A 钱的数量",
    "B_repo": "B 回购管道价格",
    "C_credit": "C 信用价格",
    "D_rates": "D 利率与政策",
    "E_stress": "E 市场压力与美元",
}

# 需要从 FRED 下载的序列: id -> 中文说明
FRED_SERIES = {
    # 数量
    "WRESBAL": "银行准备金(百万美元,周)",
    "WALCL": "联储总资产(百万美元,周)",
    "WTREGEN": "财政部 TGA(百万美元,周)",
    "RRPONTSYD": "隔夜逆回购余额(十亿美元,日)",
    "WSHOBL": "联储持有短债(百万美元,周)",
    # 回购
    "SOFR": "SOFR(%)",
    "SOFR99": "SOFR 99 分位(%)",
    "IORB": "准备金利率 IORB(%)",
    "RRPONTSYAWARD": "逆回购利率(%)",
    "RPONTSYD": "常备回购 SRF 等回购操作(十亿美元,日)",
    # 信用
    "BAMLH0A3HYC": "CCC 及以下 OAS(%)",
    "BAMLH0A3HYCEY": "CCC 及以下有效收益率(%)",
    "BAMLH0A0HYM2": "高收益债 OAS(%)",
    "BAMLC0A0CM": "投资级 OAS(%)",
    "BAMLH0A1HYBB": "BB 级 OAS(%)",
    # 利率
    "DFF": "联邦基金有效利率(%)",
    "PCEPILFE": "核心 PCE 指数(月)",
    "DFII5": "5 年 TIPS 实际收益率(%)",
    "DGS2": "2 年美债收益率(%)",
    "DGS5": "5 年美债收益率(%)",
    "THREEFYTP10": "10 年期限溢价(%)",
    # 压力与美元
    "VIXCLS": "VIX",
    "DTWEXBGS": "广义美元指数",
    "DEXJPUS": "日元/美元(每美元兑日元)",
    "NFCI": "芝加哥联储金融条件指数(周)",
    "STLFSI4": "圣路易斯联储金融压力指数(周)",
}

# 计分用的衍生序列: (列名, 支柱, 方向, 中文说明)
# 方向 +1 表示数值上升 = 流动性变松;-1 表示上升 = 变紧
SCORED = [
    ("reserves",            "A_quantity", +1, "准备金水平"),
    ("reserves_chg",        "A_quantity", +1, "准备金四周变化"),
    ("net_liq_chg",         "A_quantity", +1, "净流动性四周变化"),
    ("tga_chg",             "A_quantity", -1, "TGA 四周变化"),
    ("rrp",                 "A_quantity", +1, "逆回购缓冲余额"),
    ("fed_bills_chg",       "A_quantity", +1, "联储持有短债四周变化"),

    ("sofr_iorb",           "B_repo", -1, "SOFR − IORB"),
    ("sofr99_iorb",         "B_repo", -1, "SOFR99 − IORB"),
    ("tgcr_rrp",            "B_repo", -1, "TGCR − 逆回购利率"),
    ("srf",                 "B_repo", -1, "SRF 用量"),

    ("ccc_oas",             "C_credit", -1, "CCC OAS"),
    ("ccc_yield",           "C_credit", -1, "CCC 有效收益率"),
    ("hy_oas",              "C_credit", -1, "高收益债 OAS"),
    ("ig_oas",              "C_credit", -1, "投资级 OAS"),
    ("ccc_bb_gap",          "C_credit", -1, "CCC 与 BB 利差之差"),

    ("real_policy_rate",    "D_rates", -1, "实际政策利率(EFFR − 核心 PCE 同比)"),
    ("tips5",               "D_rates", -1, "5 年 TIPS 实际收益率"),
    ("ust2_chg",            "D_rates", -1, "2 年美债四周变化"),
    ("term_premium",        "D_rates", -1, "10 年期限溢价"),

    ("vix",                 "E_stress", -1, "VIX"),
    ("usd_chg",             "E_stress", -1, "广义美元四周变化"),
    ("usdjpy_chg",          "E_stress", +1, "美元兑日元四周变化(急跌 = 套息平仓)"),
    ("nfci",                "E_stress", -1, "NFCI"),
    ("stlfsi",              "E_stress", -1, "金融压力指数"),
]

# 硬预警
ALERTS = {
    "sofr_iorb_bp": 10.0,        # SOFR − IORB 超过 10BP
    "sofr99_iorb_bp": 25.0,      # SOFR99 − IORB 超过 25BP
    "srf_bn": 10.0,              # SRF 单日超过 100 亿美元
    "rrp_floor_bn": 50.0,        # 逆回购低于 500 亿美元
    "tga_rise_bn": 50.0,         # 同时 TGA 四周上升超过 500 亿
    "reserves_floor_tn": 2.9,    # 准备金低于 2.9 万亿
    "ccc_oas_pct": 12.15,        # CCC OAS 高于 12.15%(视频中的企稳线)
    "usdjpy_drop_pct": 5.0,      # 美元兑日元 20 日跌幅超过 5%
}

TIMEOUT = 30
HEADERS: dict = {}  # FRED stalls requests with custom User-Agents; use the requests default

# ---------------------------------------------------------------------------
# 数据抓取
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None, retries: int = 3) -> requests.Response:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT, headers=HEADERS)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"请求失败 {url}: {last}")


def fetch_fred(series_id: str, start: str) -> pd.Series:
    """fredgraph.csv 不需要 API key。"""
    r = _get("https://fred.stlouisfed.org/graph/fredgraph.csv",
             {"id": series_id, "cosd": start})
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    s = pd.to_numeric(df[series_id], errors="coerce")
    s.index = df["date"]
    return s.dropna()


def fetch_nyfed_tgcr(start: str) -> pd.Series:
    r = _get("https://markets.newyorkfed.org/api/rates/secured/tgcr/search.json",
             {"startDate": start, "endDate": date.today().isoformat()})
    rows = r.json().get("refRates", [])
    s = pd.Series({pd.Timestamp(x["effectiveDate"]): float(x["percentRate"]) for x in rows})
    return s.sort_index()


FISCAL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"


def fetch_treasury_tga(start: str) -> pd.Series:
    """每日国库报告中的 TGA 收盘余额(百万美元)。"""
    out = {}
    page = 1
    while True:
        r = _get(f"{FISCAL}/v1/accounting/dts/operating_cash_balance", {
            "filter": f"record_date:gte:{start}",
            "fields": "record_date,account_type,open_today_bal,close_today_bal",
            "page[size]": 10000, "page[number]": page, "sort": "record_date",
        })
        js = r.json()
        for x in js.get("data", []):
            acct = (x.get("account_type") or "").lower()
            # 2021 年后的账户名为 "Treasury General Account (TGA) Closing Balance"
            if "closing balance" in acct and "treasury general account" in acct:
                val = x.get("open_today_bal") if x.get("close_today_bal") in (None, "null") else x.get("close_today_bal")
                try:
                    out[pd.Timestamp(x["record_date"])] = float(val)
                except (TypeError, ValueError):
                    pass
        meta = js.get("meta", {})
        if page >= int(meta.get("total-pages", 1)):
            break
        page += 1
    return pd.Series(out).sort_index()


def fetch_settlement_calendar(days: int = 30) -> pd.DataFrame:
    """未来 N 天的国债结算(发行)日与金额。来自财政部拍卖数据。"""
    today = date.today()
    end = today + timedelta(days=days)
    rows = []
    try:
        r = _get(f"{FISCAL}/v1/accounting/od/auctions_query", {
            "filter": f"issue_date:gte:{today.isoformat()},issue_date:lte:{end.isoformat()}",
            "fields": "issue_date,security_type,security_term,offering_amt",
            "page[size]": 1000,
        })
        got = r.json().get("data", [])
        print(f"  拍卖数据 auctions_query: {len(got)} 条")
        rows += got
    except Exception as e:  # noqa: BLE001
        print(f"[警告] 拍卖数据获取失败: {e}", file=sys.stderr)
    try:
        # 默认按最旧记录排序且分页,必须按发行日过滤
        r = _get(f"{FISCAL}/v1/accounting/od/upcoming_auctions", {
            "filter": f"issue_date:gte:{today.isoformat()},issue_date:lte:{end.isoformat()}",
            "sort": "issue_date", "page[size]": 1000,
        })
        got = r.json().get("data", [])
        print(f"  即将拍卖 upcoming_auctions: {len(got)} 条")
        for x in got:
            rows.append({k: x.get(k) for k in ("issue_date", "security_type", "security_term", "offering_amt")})
    except Exception as e:  # noqa: BLE001
        print(f"[警告] 即将拍卖数据获取失败: {e}", file=sys.stderr)
    # TreasuryDirect 的已公告待发行列表(财政部数据集常常没有未来记录)
    for kind in ("upcoming", "announced"):
        try:
            params = {"format": "json"} if kind == "upcoming" else {"format": "json", "days": 14}
            r = _get(f"https://www.treasurydirect.gov/TA_WS/securities/{kind}", params)
            got = r.json()
            print(f"  TreasuryDirect {kind}: {len(got)} 条")
            for x in got:
                rows.append({"issue_date": (x.get("issueDate") or "")[:10],
                             "security_type": x.get("securityType"),
                             "security_term": x.get("securityTerm"),
                             "offering_amt": x.get("offeringAmount")})
        except Exception as e:  # noqa: BLE001
            print(f"[警告] TreasuryDirect {kind} 获取失败: {e}", file=sys.stderr)
    if not rows:
        return pd.DataFrame(columns=["issue_date", "gross_bn", "detail"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["issue_date", "security_term", "offering_amt"])
    df["offering_amt"] = df["offering_amt"].replace({"null": None, "": None})
    df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce")
    df["offering_amt"] = pd.to_numeric(df["offering_amt"], errors="coerce")
    df = df[(df["issue_date"].dt.date >= today) & (df["issue_date"].dt.date <= end)]
    g = df.groupby("issue_date").agg(
        gross_bn=("offering_amt", lambda v: v.sum(min_count=1) / 1e9),
        detail=("security_term", lambda v: ", ".join(sorted(set(map(str, v))))),
    ).reset_index()
    return g.sort_values("issue_date")


def tax_dates(days: int = 30) -> list[date]:
    today = date.today()
    out = []
    for y in (today.year, today.year + 1):
        for m in (1, 3, 4, 6, 9, 12):  # 1/15 个人预缴, 其余为企业季度缴税
            d = date(y, m, 15)
            if today <= d <= today + timedelta(days=days):
                out.append(d)
    return out


def load_all(cache_dir: Path, offline: bool) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    fetchers = {sid: (lambda sid=sid: fetch_fred(sid, HISTORY_START)) for sid in FRED_SERIES}
    fetchers["TGCR"] = lambda: fetch_nyfed_tgcr(HISTORY_START)
    fetchers["TGA_DAILY"] = lambda: fetch_treasury_tga(HISTORY_START)

    for name, fn in fetchers.items():
        path = cache_dir / f"{name}.csv"
        s = None
        if not offline:
            try:
                s = fn()
                s.rename(name).to_csv(path, header=True)
            except Exception as e:  # noqa: BLE001
                print(f"[警告] {name} 下载失败,尝试用缓存: {e}", file=sys.stderr)
        if s is None and path.exists():
            s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        if s is None or s.empty:
            print(f"[警告] {name} 无数据,跳过", file=sys.stderr)
            continue
        frames[name] = s
        print(f"  {name:15s} {len(s):6d} 条, 最新 {s.index.max().date()}")
    raw = pd.DataFrame(frames).sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    return raw

# ---------------------------------------------------------------------------
# 计算
# ---------------------------------------------------------------------------

def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """对齐到工作日并生成衍生序列。单位统一为十亿美元 / 百分点。"""
    idx = pd.bdate_range(raw.index.min(), raw.index.max())
    d = raw.reindex(raw.index.union(idx)).sort_index().ffill().reindex(idx)
    f = pd.DataFrame(index=idx)
    col = lambda c: d[c] if c in d else pd.Series(float("nan"), index=idx)  # noqa: E731

    # A 数量(十亿美元)
    tga = col("TGA_DAILY") / 1e3
    tga = tga.combine_first(col("WTREGEN") / 1e3)
    f["reserves"] = col("WRESBAL") / 1e3
    f["tga"] = tga
    f["rrp"] = col("RRPONTSYD")
    f["fed_assets"] = col("WALCL") / 1e3
    f["net_liq"] = f["fed_assets"] - f["tga"] - f["rrp"].fillna(0)
    f["reserves_chg"] = f["reserves"].diff(CHANGE_WINDOW)
    f["net_liq_chg"] = f["net_liq"].diff(CHANGE_WINDOW)
    f["tga_chg"] = f["tga"].diff(CHANGE_WINDOW)
    f["fed_bills_chg"] = (col("WSHOBL") / 1e3).diff(CHANGE_WINDOW)

    # B 回购(BP / 十亿美元)
    f["sofr_iorb"] = (col("SOFR") - col("IORB")) * 100
    f["sofr99_iorb"] = (col("SOFR99") - col("IORB")) * 100
    f["tgcr_rrp"] = (col("TGCR") - col("RRPONTSYAWARD")) * 100
    srf = raw["RPONTSYD"].reindex(idx).fillna(0) if "RPONTSYD" in raw else pd.Series(0.0, index=idx)
    f["srf"] = srf  # 不前向填充:没操作就是 0

    # C 信用(%)
    f["ccc_oas"] = col("BAMLH0A3HYC")
    f["ccc_yield"] = col("BAMLH0A3HYCEY")
    f["hy_oas"] = col("BAMLH0A0HYM2")
    f["ig_oas"] = col("BAMLC0A0CM")
    f["ccc_bb_gap"] = col("BAMLH0A3HYC") - col("BAMLH0A1HYBB")
    f["ust5"] = col("DGS5")
    f["ccc_cost_approx"] = f["ust5"] + f["ccc_oas"]  # 视频里的算法:5 年美债 + CCC 利差

    # D 利率
    pce = raw["PCEPILFE"].dropna() if "PCEPILFE" in raw else pd.Series(dtype=float)
    pce_yoy = (pce.pct_change(12) * 100).reindex(idx, method="ffill") if not pce.empty else float("nan")
    f["core_pce_yoy"] = pce_yoy
    f["real_policy_rate"] = col("DFF") - f["core_pce_yoy"]
    f["tips5"] = col("DFII5")
    f["ust2_chg"] = col("DGS2").diff(CHANGE_WINDOW)
    f["term_premium"] = col("THREEFYTP10")

    # E 压力
    f["vix"] = col("VIXCLS")
    f["usd_chg"] = col("DTWEXBGS").pct_change(CHANGE_WINDOW, fill_method=None) * 100
    f["usdjpy"] = col("DEXJPUS")
    f["usdjpy_chg"] = f["usdjpy"].pct_change(CHANGE_WINDOW, fill_method=None) * 100
    f["nfci"] = col("NFCI")
    f["stlfsi"] = col("STLFSI4")
    return f


def robust_z(s: pd.Series) -> pd.Series:
    roll = s.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS)
    med = roll.median()
    mad = (s - med).abs().rolling(ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS).median()
    scale = (1.4826 * mad).replace(0, float("nan"))
    # MAD 为 0(例如 SRF 长期为 0)时退回标准差
    std = roll.std().replace(0, float("nan"))
    scale = scale.fillna(std)
    return ((s - med) / scale).clip(-Z_CLIP, Z_CLIP)


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def score(f: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=f.index)
    for colname, pillar, sign, _ in SCORED:
        if colname in f and f[colname].notna().sum() > ZSCORE_MIN_PERIODS:
            out[f"z_{colname}"] = sign * robust_z(f[colname])
    for p in PILLAR_WEIGHTS:
        cols = [f"z_{c}" for c, pl, _, _ in SCORED if pl == p and f"z_{c}" in out]
        # 支柱内至少要有一半序列有值,否则该支柱当天不计
        need = max(1, (len(cols) + 1) // 2)
        out[p] = out[cols].mean(axis=1).where(out[cols].notna().sum(axis=1) >= need) if cols else float("nan")
    # 缺失支柱时按剩余权重重新归一;至少 MIN_PILLARS 个支柱有值才出综合指数
    w = pd.Series(PILLAR_WEIGHTS)
    P = out[list(PILLAR_WEIGHTS)]
    wsum = P.notna().mul(w, axis=1).sum(axis=1).replace(0, float("nan"))
    out["composite_z"] = (P.mul(w, axis=1).sum(axis=1, min_count=1) / wsum).where(P.notna().sum(axis=1) >= MIN_PILLARS)
    out["pillars_used"] = P.notna().sum(axis=1)
    out["liquidity_index"] = out["composite_z"].apply(lambda z: round(100 * norm_cdf(z), 1) if pd.notna(z) else z)
    return out


def regime(v: float) -> str:
    if pd.isna(v):
        return "数据不足"
    if v >= 70: return "宽松"
    if v >= 55: return "偏松"
    if v >= 45: return "中性"
    if v >= 30: return "偏紧"
    return "紧张"


def check_alerts(f: pd.DataFrame) -> list[str]:
    a, last = [], f.ffill().iloc[-1]
    def ok(x): return pd.notna(x)
    if ok(last.sofr_iorb) and last.sofr_iorb > ALERTS["sofr_iorb_bp"]:
        a.append(f"回购管道压力:SOFR − IORB = {last.sofr_iorb:.1f}BP(> {ALERTS['sofr_iorb_bp']:.0f}BP)")
    if ok(last.sofr99_iorb) and last.sofr99_iorb > ALERTS["sofr99_iorb_bp"]:
        a.append(f"回购尾部压力:SOFR99 − IORB = {last.sofr99_iorb:.1f}BP(> {ALERTS['sofr99_iorb_bp']:.0f}BP)")
    srf_recent = f["srf"].iloc[-5:].max()
    if ok(srf_recent) and srf_recent > ALERTS["srf_bn"]:
        a.append(f"SRF 大量使用:近 5 日单日最高 {srf_recent:.1f} 十亿美元")
    if ok(last.rrp) and ok(last.tga_chg) and last.rrp < ALERTS["rrp_floor_bn"] and last.tga_chg > ALERTS["tga_rise_bn"]:
        a.append(f"缓冲池干涸:逆回购仅 {last.rrp:.1f} 十亿美元,TGA 四周 +{last.tga_chg:.0f} 十亿,将直接抽走准备金")
    if ok(last.reserves) and last.reserves / 1e3 < ALERTS["reserves_floor_tn"]:
        a.append(f"准备金偏低:{last.reserves/1e3:.2f} 万亿美元(< {ALERTS['reserves_floor_tn']} 万亿)")
    if ok(last.ccc_oas) and last.ccc_oas > ALERTS["ccc_oas_pct"]:
        a.append(f"垃圾债未企稳:CCC OAS = {last.ccc_oas:.2f}%(> {ALERTS['ccc_oas_pct']}%)")
    if ok(last.usdjpy_chg) and last.usdjpy_chg < -ALERTS["usdjpy_drop_pct"]:
        a.append(f"日元急升/套息平仓风险:美元兑日元 20 日 {last.usdjpy_chg:.1f}%")
    return a

# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def fmt(x, nd=2, suffix=""):
    return "—" if pd.isna(x) else f"{x:,.{nd}f}{suffix}"


def write_report(f: pd.DataFrame, sc: pd.DataFrame, alerts: list[str], cal: pd.DataFrame, out: Path) -> Path:
    sc_valid = sc.dropna(subset=["liquidity_index"])
    last_day = sc_valid.index[-1] if not sc_valid.empty else sc.index[-1]
    li = sc.loc[last_day, "liquidity_index"]
    def ago(n):
        s = sc_valid["liquidity_index"]
        return s.iloc[-1 - n] if len(s) > n else float("nan")
    lf = f.ffill().loc[last_day]

    L = [f"# 流动性日报 {last_day.date()}", ""]
    L.append(f"**综合流动性指数:{fmt(li,1)}({regime(li)})**  "
             f"一周前 {fmt(ago(5),1)} · 四周前 {fmt(ago(20),1)} · 三个月前 {fmt(ago(63),1)}")
    L.append("")
    L.append("50 = 三年中性水平;越低越紧。仅用于研究,不构成投资建议。")
    L.append("")
    L.append("## 硬预警")
    L += [f"- {x}" for x in alerts] if alerts else ["- 无"]
    L.append("")
    L.append("## 五大支柱(z 分数,正 = 宽松)")
    L.append("| 支柱 | 权重 | 今日 | 四周前 |")
    L.append("| --- | --- | --- | --- |")
    for p, w in PILLAR_WEIGHTS.items():
        s = sc_valid[p] if p in sc_valid else pd.Series(dtype=float)
        now = s.iloc[-1] if len(s) else float("nan")
        prev = s.iloc[-21] if len(s) > 20 else float("nan")
        L.append(f"| {PILLAR_NAMES[p]} | {int(w*100)}% | {fmt(now)} | {fmt(prev)} |")
    L.append("")
    L.append("## 关键读数")
    L.append("| 指标 | 数值 |")
    L.append("| --- | --- |")
    rows = [
        ("银行准备金", fmt(lf.reserves / 1e3, 3, " 万亿美元")),
        ("准备金四周变化", fmt(lf.reserves_chg, 0, " 十亿美元")),
        ("TGA", fmt(lf.tga, 0, " 十亿美元")),
        ("TGA 四周变化", fmt(lf.tga_chg, 0, " 十亿美元")),
        ("隔夜逆回购", fmt(lf.rrp, 1, " 十亿美元")),
        ("净流动性(总资产 − TGA − 逆回购)", fmt(lf.net_liq / 1e3, 3, " 万亿美元")),
        ("联储持有短债四周变化(RMP 是否在买)", fmt(lf.fed_bills_chg, 1, " 十亿美元")),
        ("SOFR − IORB", fmt(lf.sofr_iorb, 1, " BP")),
        ("SOFR99 − IORB", fmt(lf.sofr99_iorb, 1, " BP")),
        ("SRF 近 5 日最高", fmt(f['srf'].loc[:last_day].iloc[-5:].max(), 2, " 十亿美元")),
        ("CCC OAS", fmt(lf.ccc_oas, 2, "%")),
        ("CCC 借钱成本(5 年美债 + CCC OAS)", fmt(lf.ccc_cost_approx, 2, "%")),
        ("CCC 有效收益率", fmt(lf.ccc_yield, 2, "%")),
        ("高收益债 OAS", fmt(lf.hy_oas, 2, "%")),
        ("投资级 OAS", fmt(lf.ig_oas, 2, "%")),
        ("核心 PCE 同比", fmt(lf.core_pce_yoy, 2, "%")),
        ("实际政策利率", fmt(lf.real_policy_rate, 2, "%")),
        ("5 年 TIPS 实际收益率", fmt(lf.tips5, 2, "%")),
        ("VIX", fmt(lf.vix, 1)),
        ("美元兑日元", fmt(lf.usdjpy, 2)),
        ("NFCI", fmt(lf.nfci, 2)),
    ]
    L += [f"| {a} | {b} |" for a, b in rows]
    L.append("")
    L.append("## 各序列贡献(z 分数,正 = 宽松)")
    L.append("| 支柱 | 序列 | z |")
    L.append("| --- | --- | --- |")
    for c, p, _, name in SCORED:
        zc = f"z_{c}"
        if zc in sc:
            L.append(f"| {PILLAR_NAMES[p]} | {name} | {fmt(sc[zc].ffill().loc[last_day])} |")
    L.append("")
    L.append("## 未来 30 天结算与缴税日历")
    taxes = tax_dates(30)
    if cal.empty and not taxes:
        L.append("- 无数据")
    else:
        L.append("| 日期 | 国债结算(十亿美元,毛额) | 说明 |")
        L.append("| --- | --- | --- |")
        cal_map = {r.issue_date.date(): r for r in cal.itertuples()}
        month_ends = {(pd.Timestamp(d) + pd.offsets.BMonthEnd(0)).date() for d in pd.date_range(date.today(), periods=31)}
        for d in sorted(set(cal_map) | set(taxes)):
            r = cal_map.get(d)
            note = []
            if d in taxes: note.append("缴税日")
            if d in month_ends: note.append("月末")
            if d.month == 12 and d.day >= 29: note.append("年末")
            L.append(f"| {d} | {(fmt(r.gross_bn, 1) if pd.notna(r.gross_bn) else '待公布') if r else '—'} | {' · '.join(note + ([r.detail] if r else []))} |")
    L.append("")
    path = out / f"report_{last_day.date()}.md"
    path.write_text("\n".join(L), encoding="utf-8")
    (out / "report_latest.md").write_text("\n".join(L), encoding="utf-8")
    return path


def append_daily_log(sc: pd.DataFrame, f: pd.DataFrame, alerts: list[str], path: Path) -> None:
    """每次运行把当时算出的结果追加一行,作为不改写的历史快照(同一天重复运行只保留最后一次)。"""
    v = sc.dropna(subset=["liquidity_index"])
    if v.empty:
        return
    last = v.iloc[-1]
    row = {
        "run_date": date.today().isoformat(),
        "data_date": v.index[-1].date().isoformat(),
        "liquidity_index": last["liquidity_index"],
        "regime": regime(last["liquidity_index"]),
        **{p: round(last[p], 3) if pd.notna(last[p]) else None for p in PILLAR_WEIGHTS},
        "alerts": " | ".join(alerts),
    }
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not old.empty:
        old = old[old["run_date"] != row["run_date"]]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="美元流动性综合指标")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "output"))
    ap.add_argument("--offline", action="store_true", help="只用本地缓存")
    ap.add_argument("--dashboard", default=str(Path(__file__).resolve().parent / "docs" / "index.html"),
                    help="仪表盘 HTML 输出路径")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("抓取数据...")
    raw = load_all(out / "cache", args.offline)
    if raw.empty:
        print("没有任何数据,退出。", file=sys.stderr)
        return 1
    f = build_features(raw)
    sc = score(f)
    alerts = check_alerts(f)
    cal = pd.DataFrame(columns=["issue_date", "gross_bn", "detail"]) if args.offline else fetch_settlement_calendar(30)

    f.to_csv(out / "raw_data.csv", float_format="%.4f")
    hist = sc[["liquidity_index", "composite_z", *PILLAR_WEIGHTS]].join(sc.filter(like="z_"))
    hist.dropna(subset=["liquidity_index"]).to_csv(out / "liquidity_history.csv", float_format="%.3f")
    rp = write_report(f, sc, alerts, cal, out)
    append_daily_log(sc, f, alerts, out / "daily_log.csv")
    try:
        from dashboard import write_dashboard
        dash = Path(args.dashboard)
        write_dashboard(f, sc, alerts, cal, dash, out / "daily_log.csv")
        print(f"仪表盘: {dash}")
    except Exception as e:  # noqa: BLE001
        print(f"[警告] 仪表盘生成失败: {e}", file=sys.stderr)

    li = sc["liquidity_index"].dropna()
    print(f"\n综合流动性指数 {li.index[-1].date() if len(li) else '-'}: "
          f"{li.iloc[-1] if len(li) else '—'}({regime(li.iloc[-1]) if len(li) else '数据不足'})")
    for x in alerts:
        print("  预警:", x)
    print(f"报告: {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
