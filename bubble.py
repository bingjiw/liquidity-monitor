"""泡沫体检:按月给美股的泡沫程度打 0–100 分,可回看到 1950 年代(估值单项回到 1881 年)。

依据四个问题(技术是真的,破的往往是价格):
  1 利润追不追得上价格   —— Shiller CAPE、股票市值/GDP
  2 钱是借来的还是赚来的 —— 私人部门信贷缺口(BIS 方法)、券商融资余额同比、BAA 信用利差
  3 利率周期             —— 联邦基金利率 24 个月变化、10 年美债 − 联邦基金利率
  4 钱是不是在产业链里转 —— 企业设备/厂房投资占 GDP 偏离 10 年均值(只是代理指标,
                            供应商融资、客户拿投资款回头买货这类循环没有公开的宏观数据)

每个指标按“0 分线 → 100 分线”线性打分,支柱分 = 指标平均,总分 = 支柱加权平均(缺的支柱按权重归一)。
阈值和权重是经验设定,没有经过严格回测;只做观察,不进仓位规则。

按月计算,并按数据公布时间往后错开(季度数据错开 5 个月、月度数据错开 1 个月),
尽量只用当时已经能看到的数据。数据修订没有处理(用的是今天的修订后数值)。

输出:output/bubble_raw.csv(原始月度数据,抓取失败时用作备份)、output/bubble_history.csv、
output/bubble.json(页面用)。页面 HTML 和脚本也在本文件里,由 dashboard.py 插入。
"""
from __future__ import annotations

import io
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
PILLARS = {
    "P1": {"name": "利润追不追得上价格", "short": "估值", "weight": 0.35},
    "P2": {"name": "钱是借来的还是赚来的", "short": "杠杆", "weight": 0.25},
    "P3": {"name": "利率周期", "short": "利率", "weight": 0.20},
    "P4": {"name": "钱是不是在产业链里转", "short": "投资热", "weight": 0.20},
}

# key: (支柱, 名称, 单位, 0 分线, 100 分线, 说明)
INDICATORS = {
    "cape": ("P1", "Shiller CAPE(周期调整市盈率)", "倍", 15.0, 35.0,
             "股价 ÷ 过去 10 年平均实际盈利。越高说明价格跑得越远于利润。"),
    "mcap_gdp": ("P1", "股票市值 / GDP", "%", 60.0, 180.0,
                 "非金融企业股票市值(美联储 Z.1)÷ 名义 GDP,俗称巴菲特指标。"),
    "credit_gap": ("P2", "私人部门信贷缺口", "个百分点", 0.0, 10.0,
                   "私人非金融部门债务/GDP 高出其长期趋势的部分(BIS 方法,单边 HP 滤波)。BIS 以 +2 为警戒、+10 为高危。"),
    "margin_yoy": ("P2", "券商融资余额同比", "%", 0.0, 40.0,
                   "FINRA 券商客户融资余额一年增速,反映借钱买股的速度。1997 年起。"),
    "baa_spread": ("P2", "BAA 企业债 − 10 年美债", "个百分点", 3.0, 1.5,
                   "利差越窄,借钱越容易、越便宜(分越高)。"),
    "ff_chg24": ("P3", "联邦基金利率 24 个月变化", "个百分点", 0.0, 3.0,
                 "两年内加息越多,压力越大。历史上大泡沫多在加息周期里破。"),
    "curve": ("P3", "10 年美债 − 联邦基金利率", "个百分点", 2.0, -0.5,
              "收益率曲线越平、越倒挂,分越高。"),
    "capex_dev": ("P4", "企业投资占 GDP 偏离 10 年均值", "个百分点", 0.0, 1.5,
                  "私人非住宅固定投资 / GDP 比过去 10 年均值高多少。投资热过头时,收入常靠产业链内部互相采购撑起来(如 2000 年电信)。"),
}

LAG_Q = 5   # 季度数据:季度开始后约 5 个月可得(Q3 → 12 月)
LAG_M = 1   # 月度数据:次月可得
MIN_PILLARS = 3
EXTRAP_MAX = 60   # Shiller 数据之后,CAPE 最多延伸 60 个月
FIRST_MONTH = "1881-01-01"

LEVELS = [(0, "正常", "--loose"), (30, "偏热", "--semiloose"), (45, "过热", "--semitight"), (60, "泡沫区", "--tight")]

# 页面上标出的历史顶部(美股大跌前的高点月份)
EPISODES = [
    ("1929-09", "1929 大萧条前"), ("1968-11", "1968 电子股热"), ("1972-12", "1973 漂亮 50"),
    ("1987-08", "1987 股灾前"), ("2000-03", "2000 互联网泡沫"), ("2007-10", "2007 次贷"),
    ("2021-12", "2021 疫情后"),
]

FRED_IDS = {
    "FEDFUNDS": "联邦基金利率(月)", "GS10": "10 年美债(月)", "BAA10YM": "BAA − 10 年美债(月)",
    "GDP": "名义 GDP(季)", "PNFI": "私人非住宅固定投资(季)", "SP500": "标普 500(日)",
    "NCBEILQ027S": "非金融企业股票市值(季)", "QUSPAM770A": "私人非金融部门信贷/GDP(BIS,季)",
    "CMDEBT": "居民债务(季)", "BCNSDODNS": "非金融企业债务(季)",
    "CP": "企业税后利润(季,用于延伸 CAPE)", "CPIAUCSL": "CPI(月,用于延伸 CAPE)",
}
QUARTERLY = {"GDP", "PNFI", "NCBEILQ027S", "QUSPAM770A", "CMDEBT", "BCNSDODNS", "CP"}

SHILLER_PAGE = "https://shillerdata.com/"
SHILLER_XLS = ["http://www.econ.yale.edu/~shiller/data/ie_data.xls"]
SHILLER_CSV = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 取数
# ---------------------------------------------------------------------------
def _get(url: str, params: dict | None = None, ua: bool = False, retries: int = 3) -> requests.Response:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60,
                             headers={"User-Agent": "Mozilla/5.0"} if ua else None)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"{url}: {last}")


def _month(s: pd.Series, how: str = "last") -> pd.Series:
    s = s.dropna()
    s.index = pd.DatetimeIndex(s.index).to_period("M").to_timestamp()
    return s.groupby(level=0).mean() if how == "mean" else s.groupby(level=0).last()


def fred(sid: str) -> pd.Series:
    """fredgraph.csv 不需要 API key;不带自定义 User-Agent(FRED 会卡住)。"""
    r = _get("https://fred.stlouisfed.org/graph/fredgraph.csv", {"id": sid, "cosd": "1800-01-01"})
    df = pd.read_csv(io.StringIO(r.text))
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    s = s.dropna()
    if s.empty:
        raise RuntimeError("空序列")
    return _month(s, "mean" if sid == "SP500" else "last")


def _shiller_from_xls(content: bytes) -> pd.DataFrame:
    x = pd.read_excel(io.BytesIO(content), sheet_name="Data", header=None)
    hdr = next(i for i in range(40) if str(x.iat[i, 0]).strip().lower() == "date")
    heads = [str(v).strip() for v in x.iloc[hdr]]
    ccape = next(j for j, h in enumerate(heads) if h.upper() in ("CAPE", "P/E10"))
    body = x.iloc[hdr + 1:]
    d = pd.to_numeric(body.iloc[:, 0], errors="coerce")
    ok = d.notna()
    yr = d[ok].astype(float).apply(math.floor)
    mo = ((d[ok].astype(float) - yr) * 100).round().astype(int)   # 1871.1 = 10 月
    idx = pd.to_datetime(dict(year=yr.astype(int), month=mo, day=1))
    out = pd.DataFrame({"sp_price": pd.to_numeric(body.loc[ok].iloc[:, 1], errors="coerce").values,
                        "cape": pd.to_numeric(body.loc[ok].iloc[:, ccape], errors="coerce").values},
                       index=pd.DatetimeIndex(idx.values))
    return out


def shiller() -> tuple[pd.DataFrame, str]:
    errs = []
    try:
        page = _get(SHILLER_PAGE, ua=True).text
        links = re.findall(r'(https?://[^"\'\s<>]*ie_data\.xls[^"\'\s<>]*)', page)
        for url in list(dict.fromkeys(l.replace("&amp;", "&") for l in links)) + SHILLER_XLS:
            try:
                df = _shiller_from_xls(_get(url, ua=True).content)
                if df["cape"].notna().sum() > 1000:
                    print(f"[提示] Shiller 文件: {url}(页面上找到 {len(links)} 个链接)", file=sys.stderr)
                    return df, "Shiller ie_data.xls" + ("(官网)" if url not in SHILLER_XLS else "(耶鲁旧地址)")
            except Exception as e:  # noqa: BLE001
                errs.append(f"{url[:80]}: {e}")
    except Exception as e:  # noqa: BLE001
        errs.append(str(e))
    try:
        c = pd.read_csv(io.StringIO(_get(SHILLER_CSV).text), parse_dates=["Date"], index_col="Date")
        df = pd.DataFrame({"sp_price": c["SP500"].where(c["SP500"] > 0),
                           "cape": c["PE10"].where(c["PE10"] > 0)})
        print("[提示] Shiller 官网文件没取到,改用 GitHub datasets 的副本: " + "; ".join(errs)[:300], file=sys.stderr)
        return df, "Shiller 数据(GitHub datasets 副本)"
    except Exception as e:  # noqa: BLE001
        errs.append(str(e))
    raise RuntimeError("; ".join(errs)[:500])


def margin(out: Path) -> pd.Series:
    path = out / "cache" / "FINRA_MARGIN.csv"
    if path.exists():
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        from watch import fetch_margin
        s = fetch_margin(out / "cache")
    return _month(s) if len(s) else s


def load(out: Path, offline: bool = False) -> tuple[pd.DataFrame, dict]:
    """原始月度数据(按观测月份,未错开)。抓不到的列沿用上次提交的 bubble_raw.csv。"""
    prev_path = out / "bubble_raw.csv"
    prev = pd.read_csv(prev_path, index_col=0, parse_dates=True) if prev_path.exists() else pd.DataFrame()
    cols, status = {}, {}

    def take(name, fn, label):
        if offline:
            raise RuntimeError("离线")
        try:
            cols[name] = fn()
            status[name] = {"src": label, "ok": True}
        except Exception as e:  # noqa: BLE001
            if name in prev and prev[name].notna().any():
                cols[name] = prev[name].dropna()
                status[name] = {"src": label, "ok": False, "note": "本次抓取失败,沿用上次数据"}
            else:
                status[name] = {"src": label, "ok": False, "note": "抓取失败,无数据"}
            print(f"[警告] 泡沫体检 {name} 获取失败: {e}", file=sys.stderr)

    def safe(name, fn, label):
        try:
            take(name, fn, label)
        except RuntimeError:
            if name in prev:
                cols[name] = prev[name].dropna()
                status[name] = {"src": label, "ok": False, "note": "离线,用上次数据"}

    sh = {}

    def get_shiller():
        df, lab = shiller()
        sh["label"] = lab
        cols["sp_price"] = df["sp_price"].dropna()
        return df["cape"].dropna()

    safe("cape", get_shiller, "Shiller")
    if "label" in sh:
        status["cape"]["src"] = sh["label"]
    elif "sp_price" in prev:
        cols["sp_price"] = prev["sp_price"].dropna()
    for sid, lab in FRED_IDS.items():
        safe(sid, lambda sid=sid: fred(sid), f"FRED {sid}:{lab}")
    safe("margin", lambda: margin(out), "FINRA 券商融资余额")

    raw = pd.DataFrame(cols).sort_index()
    raw = raw[raw.index >= FIRST_MONTH]
    return raw, status


# ---------------------------------------------------------------------------
# 计算
# ---------------------------------------------------------------------------
def hp_trend_onesided(x: pd.Series, lam: float = 400_000, min_obs: int = 40) -> pd.Series:
    """单边 HP 趋势:每个时点只用到当时为止的数据(BIS 信贷缺口的做法)。"""
    x = x.dropna()
    v = x.values.astype(float)
    out = np.full(len(v), np.nan)
    for t in range(min_obs - 1, len(v)):
        n = t + 1
        k = np.zeros((n - 2, n))
        i = np.arange(n - 2)
        k[i, i], k[i, i + 1], k[i, i + 2] = 1, -2, 1
        out[t] = np.linalg.solve(np.eye(n) + lam * k.T @ k, v[:n])[-1]
    return pd.Series(out, index=x.index)


def _lag(s: pd.Series, months: int) -> pd.Series:
    s = s.dropna().copy()
    s.index = s.index + pd.DateOffset(months=months)
    return s


def features(raw: pd.DataFrame, months: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=months)

    def put(name, s, lag):
        f[name] = _lag(s, lag).reindex(months.union(_lag(s, lag).index)).ffill().reindex(months)

    def col(c):
        return raw[c].dropna() if c in raw else pd.Series(dtype=float)

    # CAPE:Shiller 数据之后的月份按“实际股价 ÷ 10 年平均实际企业利润”的变化比例延伸。
    # 价格用标普 500 月均,利润用 NIPA 企业税后利润(季度,错开 5 个月),都用 CPI 换成实际值。
    cape, px = col("cape"), col("sp_price")
    spx, cp, cpi = col("SP500"), col("CP"), col("CPIAUCSL")
    if len(cape):
        t0 = cape.index[-1]
        if len(px) and len(spx):
            tail = spx[spx.index > px.index[-1]]
            if len(tail) and px.index[-1] in spx.index:
                px = pd.concat([px, tail / spx[px.index[-1]] * px.iloc[-1]])
        ext = None
        if len(px) and len(cp) and len(cpi) and t0 in px.index:
            cpim = cpi.resample("MS").last().ffill()
            rcp = (cp / cpim.reindex(cp.index)).dropna()
            e10 = _lag(rcp.rolling(40, min_periods=40).mean(), LAG_Q)
            e10 = e10.reindex(months.union(e10.index)).ffill().reindex(months)
            cpm = cpim.reindex(months).ffill()
            rp = px.reindex(months) / cpm
            ext = cape.iloc[-1] * (rp / rp[t0]) / (e10 / e10[t0])
            ext = ext[(ext.index > t0) & (ext.index <= t0 + pd.DateOffset(months=EXTRAP_MAX))].dropna()
        if ext is not None and len(ext):
            cape = pd.concat([cape, ext])
            f.attrs["cape_ext_to"] = ext.index[-1].strftime("%Y-%m")
        put("cape", cape, 0)
        f.attrs["cape_last_actual"] = t0.strftime("%Y-%m")

    gdp = col("GDP")
    eq = col("NCBEILQ027S")
    if len(gdp) and len(eq):
        put("mcap_gdp", (eq / 1e3) / gdp * 100, LAG_Q)   # 百万美元 → 十亿
    if len(gdp) and len(col("PNFI")):
        sh = col("PNFI") / gdp * 100
        put("capex_dev", sh - sh.rolling(40, min_periods=40).mean(), LAG_Q)
        put("capex_share", sh, LAG_Q)

    cr = col("QUSPAM770A")
    if not len(cr) and len(gdp) and len(col("CMDEBT")) and len(col("BCNSDODNS")):
        cr = (col("CMDEBT") + col("BCNSDODNS")) / gdp * 100
    if len(cr):
        q = cr[cr.index.month.isin([1, 4, 7, 10])]
        put("credit_gap", q - hp_trend_onesided(q), LAG_Q)
        put("credit_gdp", q, LAG_Q)

    m = col("margin")
    if len(m):
        mm = m.resample("MS").last().ffill()
        put("margin_yoy", (mm / mm.shift(12) - 1) * 100, LAG_M)

    if len(col("BAA10YM")):
        put("baa_spread", col("BAA10YM"), LAG_M)
    ff = col("FEDFUNDS")
    if len(ff):
        ffm = ff.resample("MS").last()
        put("fedfunds", ffm, LAG_M)
        put("ff_chg24", ffm - ffm.shift(24), LAG_M)
        if len(col("GS10")):
            put("curve", (col("GS10") - ffm).dropna(), LAG_M)
    return f


def _score(x: pd.Series, lo: float, hi: float) -> pd.Series:
    return ((x - lo) / (hi - lo) * 100).clip(0, 100)


def score(f: pd.DataFrame) -> pd.DataFrame:
    s = pd.DataFrame(index=f.index)
    for k, (p, *_rest) in INDICATORS.items():
        lo, hi = _rest[2], _rest[3]
        if k in f:
            s[f"s_{k}"] = _score(f[k], lo, hi)
    for p in PILLARS:
        cs = [f"s_{k}" for k, v in INDICATORS.items() if v[0] == p and f"s_{k}" in s]
        s[p] = s[cs].mean(axis=1) if cs else np.nan
    w = pd.Series({p: v["weight"] for p, v in PILLARS.items()})
    have = s[list(PILLARS)].notna()
    num = (s[list(PILLARS)].fillna(0) * w).sum(axis=1)
    den = (have * w).sum(axis=1)
    s["pillars_used"] = have.sum(axis=1)
    s["score"] = (num / den).where(s["pillars_used"] >= MIN_PILLARS)
    return s


def level(v) -> str:
    if v is None or pd.isna(v):
        return "数据不足"
    return [n for t, n, _ in LEVELS if v >= t][-1]


def _r(v, nd=1):
    return None if v is None or pd.isna(v) or (isinstance(v, float) and math.isinf(v)) else round(float(v), nd)


def run(out: Path, offline: bool = False) -> dict | None:
    raw, status = load(out, offline)
    if raw.empty:
        return None
    raw.to_csv(out / "bubble_raw.csv", float_format="%.4f")
    today = pd.Timestamp.today().to_period("M").to_timestamp()
    months = pd.date_range(FIRST_MONTH, today, freq="MS")
    f = features(raw, months)
    s = score(f)
    hist = f.join(s)
    hist = hist[hist.notna().any(axis=1)]
    hist.to_csv(out / "bubble_history.csv", float_format="%.3f")

    valid = s["score"].dropna()
    if valid.empty:
        return None
    i = valid.index[-1]
    first = valid.index[0]

    def last_of(c):
        x = f[c].dropna() if c in f else pd.Series(dtype=float)
        return x.iloc[-1] if len(x) else None

    ind = []
    for k, (p, name, unit, lo, hi, desc) in INDICATORS.items():
        x = f[k].dropna() if k in f else pd.Series(dtype=float)
        ind.append({"key": k, "pillar": p, "name": name, "unit": unit, "lo": lo, "hi": hi, "desc": desc,
                    "value": _r(x.iloc[-1], 2) if len(x) else None,
                    "score": _r(s[f"s_{k}"].dropna().iloc[-1]) if f"s_{k}" in s and s[f"s_{k}"].notna().any() else None,
                    "since": x.index[0].strftime("%Y-%m") if len(x) else None})
    eps = []
    for ym, label in EPISODES + [(i.strftime("%Y-%m"), "现在")]:
        t = pd.Timestamp(ym + "-01")
        row = {"month": ym, "label": label, "score": _r(s["score"].get(t)),
               "cape": _r(f["cape"].get(t)) if "cape" in f else None}
        for p in PILLARS:
            row[p] = _r(s[p].get(t))
        eps.append(row)

    src = {}
    for k, st in status.items():
        src[k] = st

    data = {
        "asof": i.strftime("%Y-%m"),
        "score": _r(valid.iloc[-1]),
        "level": level(valid.iloc[-1]),
        "chg12": _r(valid.iloc[-1] - valid.get(i - pd.DateOffset(months=12), np.nan)),
        "max": {"month": valid.idxmax().strftime("%Y-%m"), "score": _r(valid.max())},
        "pctile": _r((valid <= valid.iloc[-1]).mean() * 100, 0),
        "first": first.strftime("%Y-%m"),
        "cape_last_actual": f.attrs.get("cape_last_actual"),
        "cape_ext_to": f.attrs.get("cape_ext_to"),
        "levels": [[t, n, c] for t, n, c in LEVELS],
        "pillars": {p: {**v, "now": _r(s[p].dropna().iloc[-1]) if s[p].notna().any() else None}
                    for p, v in PILLARS.items()},
        "indicators": ind,
        "episodes": eps,
        "sources": src,
        "months": [d.strftime("%Y-%m") for d in s.loc[first:].index],
        "series": {"score": [_r(v) for v in s.loc[first:, "score"]],
                   **{p: [_r(v) for v in s.loc[first:, p]] for p in PILLARS}},
        "cape": {"months": [d.strftime("%Y-%m") for d in f["cape"].dropna().index],
                 "v": [_r(v) for v in f["cape"].dropna()]} if "cape" in f else None,
    }
    (out / "bubble.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return data


def report_lines(b: dict | None) -> list[str]:
    if not b:
        return []
    L = ["## 泡沫体检(按月,0–100)", "",
         f"**{b['asof']}:{b['score']} 分({b['level']})**,一年前相比 {b['chg12']:+.1f};"
         f"{b['first']} 以来的历史分位 {b['pctile']:.0f}%,最高 {b['max']['score']}({b['max']['month']})。", "",
         "| 支柱 | 权重 | 分数 |", "| --- | --- | --- |"]
    L += [f"| {v['name']} | {int(v['weight'] * 100)}% | {v['now'] if v['now'] is not None else '—'} |"
          for v in b["pillars"].values()]
    L += ["", "| 指标 | 最新值 | 分数 |", "| --- | --- | --- |"]
    L += [f"| {x['name']} | {x['value'] if x['value'] is not None else '—'} {x['unit']} | "
          f"{x['score'] if x['score'] is not None else '—'} |" for x in b["indicators"]]
    L += ["", "阈值和权重是经验设定,没有经过严格回测。", ""]
    return L


# ---------------------------------------------------------------------------
# 页面部分:HTML 插在模板的 __BUBBLE__ 处,SCRIPT 插在主脚本之后
# ---------------------------------------------------------------------------
HTML = r"""
  <div class="card" id="bubCard">
    <h2>泡沫体检:价格是不是跑得太远了</h2>
    <div class="sub">技术是真的,破的往往是价格。这里按四个问题每月给美股打一个泡沫分(0–100):利润追不追得上价格、钱是借来的还是赚来的、利率周期、钱是不是在产业链里转。每个月都按当时能看到的数据重算,所以历史上每个月都有分数,可以和 2000、2007、2021 年对照。</div>
    <div class="hero" style="margin-top:12px">
      <div>
        <div class="sub" id="bAsof"></div>
        <div class="big" id="bNow"></div>
        <div style="margin-top:8px"><span class="badge" id="bLevel"></span></div>
      </div>
      <div class="bPil" id="bPil"></div>
    </div>
    <div class="sub" id="bLine" style="margin-top:10px"></div>
    <div class="row" style="margin-top:12px" id="bRange"></div>
    <div id="cBubble" class="chart" style="height:400px"></div>
    <div class="note">黑线是总分;点下面的图例可以加上四个支柱的分线。竖线是历史上美股大跌前的高点月份。底色:绿 = 正常(&lt;30),浅绿 = 偏热(30–45),橙 = 过热(45–60),红 = 泡沫区(≥60)。总分至少要有 3 个支柱的数据才算,所以从 <span id="bFirst"></span> 开始。</div>
    <h3 class="bH3">历史高点对照</h3>
    <div class="scroll"><table id="bEp"></table></div>
    <div class="note">1929 年只有估值数据(CAPE),所以没有总分。日本 1989 年泡沫是日本的数据,这里只看美国。</div>
    <h3 class="bH3">估值单项,回看到 1881 年(Shiller CAPE)</h3>
    <div id="cCape" class="chart" style="height:280px"></div>
    <div class="note" id="bCapeNote"></div>
    <h3 class="bH3">指标、阈值和权重</h3>
    <div class="scroll"><table id="bInd"></table></div>
    <div class="note">打分方法:每个指标在“0 分线”和“100 分线”之间线性换算,超出就封顶在 0 或 100;支柱分 = 该支柱下有数据的指标平均;总分 = 四个支柱按权重加权(缺数据的支柱按剩余权重归一)。<b>阈值和权重都是经验设定,没有经过严格回测</b>,是用来对照历史的体检表,不是择时信号。为了尽量只用当时能看到的数据,季度数据(GDP、Z.1、BIS)往后错开 5 个月,月度数据错开 1 个月;但用的是今天修订后的数值。CAPE 在 Shiller 数据之后的月份,按“标普 500 实际价格 ÷ 美国企业税后利润(NIPA)10 年实际均值”的变化比例延伸,是近似值。第 4 条的“产业链内部循环”(供应商融资、客户拿你的投资回头买你的货)没有公开的宏观数据,只能用企业投资占 GDP 是否过热来近似。利率支柱衡量的是“破裂条件”,泡沫在低利率时可以一直涨,开始加息后才危险。</div>
    <details><summary>数据来源和本次抓取状态</summary><div id="bSrc" class="note"></div></details>
  </div>
<style>
.bPil{display:grid;grid-template-columns:repeat(4,minmax(110px,1fr));gap:8px;flex:1;min-width:280px}
@media (max-width:760px){.bPil{grid-template-columns:repeat(2,1fr)}}
.bPil div{background:var(--bg);border-radius:8px;padding:8px 10px;font-size:12px;color:var(--muted)}
.bPil b{display:block;font-size:20px;color:var(--ink);font-variant-numeric:tabular-nums}
.bPil i{display:block;height:6px;border-radius:3px;margin-top:4px;background:var(--line)}
.bPil i u{display:block;height:6px;border-radius:3px}
.bH3{font-size:15px;margin:18px 0 6px}
#bInd td,#bInd th{vertical-align:top}
#bInd td.l{text-align:left;color:var(--muted);font-size:13px;min-width:220px}
</style>
"""

SCRIPT = r"""
<script>
(function(){
const card = document.getElementById("bubCard");
const B = window.__BUBBLE__;
if (!B || !window.Plotly){ if (!B) card.style.display = "none"; return; }
const $ = id => document.getElementById(id);
const C = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const f1 = v => v == null ? "—" : Number(v).toFixed(1);
const lvl = v => { let r = B.levels[0]; for (const L of B.levels) if (v != null && v >= L[0]) r = L; return r; };
const pc = {P1: "--p1", P2: "--p2", P3: "--p4", P4: "--p5"};

// 头部
$("bAsof").textContent = `${B.asof} 泡沫分(0–100)`;
$("bNow").textContent = f1(B.score);
const L0 = lvl(B.score); $("bLevel").textContent = L0[1]; $("bLevel").style.background = C(L0[2]);
$("bPil").innerHTML = Object.entries(B.pillars).map(([k, p]) =>
  `<div>${p.short} · ${Math.round(p.weight * 100)}%<b>${f1(p.now)}</b><span>${p.name}</span>` +
  `<i><u style="width:${p.now || 0}%;background:${C(pc[k])}"></u></i></div>`).join("");
$("bLine").textContent = `比一年前 ${B.chg12 == null ? "—" : (B.chg12 >= 0 ? "+" : "") + f1(B.chg12)} 分;` +
  `在 ${B.first} 以来的所有月份里排在 ${B.pctile}% 分位;历史最高 ${f1(B.max.score)} 分(${B.max.month})。`;
$("bFirst").textContent = B.first;

// 历史曲线
const x = B.months.map(m => m + "-01");
const bandShapes = B.levels.map((L, i) => ({type: "rect", xref: "paper", x0: 0, x1: 1, y0: L[0],
  y1: i + 1 < B.levels.length ? B.levels[i + 1][0] : 100, fillcolor: C(L[2]), opacity: 0.10, line: {width: 0}, layer: "below"}));
const epi = B.episodes.filter(e => e.label !== "现在" && e.month + "-01" >= x[0]);
const epShapes = epi.map(e => ({type: "line", x0: e.month + "-01", x1: e.month + "-01", yref: "paper", y0: 0, y1: 1,
  line: {color: C("--muted"), width: 1, dash: "dot"}}));
const epNotes = epi.map((e, i) => ({x: e.month + "-01", y: 1, yref: "paper", yanchor: "bottom", text: e.label.split(" ")[0],
  showarrow: false, font: {size: 11, color: C("--muted")}}));
const traces = [{x, y: B.series.score, name: "总分", type: "scatter", mode: "lines", line: {color: C("--ink"), width: 2.4},
  hovertemplate: "总分 %{y:.1f}<extra></extra>"}].concat(Object.entries(B.pillars).map(([k, p]) => ({
  x, y: B.series[k], name: `${p.short}(${Math.round(p.weight * 100)}%)`, type: "scatter", mode: "lines",
  line: {color: C(pc[k]), width: 1.2}, opacity: 0.85, visible: "legendonly", hovertemplate: `${p.short} %{y:.0f}<extra></extra>`})));
const lay = {paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: {color: C("--ink"), size: 12},
  margin: {l: 40, r: 12, t: 22, b: 30}, hovermode: "x unified", showlegend: true, legend: {orientation: "h", y: -0.12},
  xaxis: {type: "date", gridcolor: C("--grid"), linecolor: C("--line"), hoverformat: "%Y-%m"},
  yaxis: {range: [0, 100], gridcolor: C("--grid"), dtick: 20}, shapes: bandShapes.concat(epShapes), annotations: epNotes};
const cfg = {displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"]};
Plotly.newPlot("cBubble", traces, lay, cfg);
const rb = $("bRange");
[["全部", B.first], ["1990 年起", "1990-01"], ["2000 年起", "2000-01"], ["近 10 年", null]].forEach(([t, s], i) => {
  const b = document.createElement("button"); b.textContent = t; if (i === 0) b.classList.add("on");
  b.onclick = () => {
    const last = x[x.length - 1];
    const x0 = s ? s + "-01" : (Number(last.slice(0, 4)) - 10) + last.slice(4);
    Plotly.relayout("cBubble", {"xaxis.range": [x0, last]});
    [...rb.children].forEach(c => c.classList.toggle("on", c === b));
  };
  rb.appendChild(b);
});

// 历史高点对照表
const pk = Object.keys(B.pillars);
$("bEp").innerHTML = `<tr><th>时间</th><th>总分</th>${pk.map(k => `<th>${B.pillars[k].short}</th>`).join("")}<th>CAPE</th></tr>` +
  B.episodes.map(e => `<tr${e.label === "现在" ? ' style="font-weight:700"' : ""}><td>${e.label}(${e.month})</td>` +
  `<td>${f1(e.score)}</td>${pk.map(k => `<td>${f1(e[k])}</td>`).join("")}<td>${f1(e.cape)}</td></tr>`).join("");

// CAPE 长历史
if (B.cape){
  const cx = B.cape.months.map(m => m + "-01");
  const ce = B.episodes.filter(e => e.label !== "现在");
  Plotly.newPlot("cCape", [{x: cx, y: B.cape.v, type: "scatter", mode: "lines", line: {color: C("--p1"), width: 1.6},
    hovertemplate: "CAPE %{y:.1f}<extra></extra>"}],
    Object.assign({}, lay, {showlegend: false, yaxis: {gridcolor: C("--grid"), rangemode: "tozero"},
      shapes: [15, 35].map(v => ({type: "line", xref: "paper", x0: 0, x1: 1, y0: v, y1: v,
        line: {color: v === 35 ? C("--tight") : C("--loose"), width: 1, dash: "dash"}}))
        .concat(ce.map(e => ({type: "line", x0: e.month + "-01", x1: e.month + "-01", yref: "paper", y0: 0, y1: 1,
          line: {color: C("--muted"), width: 1, dash: "dot"}}))),
      annotations: ce.map(e => ({x: e.month + "-01", y: 1, yref: "paper", yanchor: "bottom", text: e.label.split(" ")[0],
        showarrow: false, font: {size: 11, color: C("--muted")}}))}), cfg);
  $("bCapeNote").textContent = `绿色虚线 15 倍 = 0 分,红色虚线 35 倍 = 100 分。Shiller 实际数据到 ${B.cape_last_actual || "—"}${B.cape_ext_to ? `,之后到 ${B.cape_ext_to} 按标普 500 价格和美国企业税后利润(NIPA)的变化比例延伸` : ""}。CAPE 只看估值,不看杠杆和利率,所以 1929 年(约 33 倍)比 2021 年(约 38 倍)低,但 1929 年借钱买股的程度远高于今天。`;
}

// 指标表
$("bInd").innerHTML = `<tr><th>指标</th><th>支柱</th><th>最新值</th><th>0 分线</th><th>100 分线</th><th>分数</th><th>起始</th><th>说明</th></tr>` +
  B.indicators.map(r => `<tr><td>${r.name}</td><td>${B.pillars[r.pillar].short} ${Math.round(B.pillars[r.pillar].weight * 100)}%</td>` +
  `<td>${r.value == null ? "—" : r.value}</td><td>${r.lo}</td><td>${r.hi}</td><td>${f1(r.score)}</td><td>${r.since || "—"}</td>` +
  `<td class="l">单位:${r.unit}。${r.desc}</td></tr>`).join("");

// 数据源
$("bSrc").innerHTML = Object.entries(B.sources).map(([k, s]) =>
  `${s.ok ? "✓" : "✗"} ${s.src}${s.note ? "(" + s.note + ")" : ""}`).join("<br>");
})();
</script>
"""


def page(out: Path) -> tuple[str, str]:
    """返回 (HTML, SCRIPT);没有数据时卡片自动隐藏。"""
    p = out / "bubble.json"
    data = p.read_text(encoding="utf-8") if p.exists() else "null"
    return HTML, f"<script>window.__BUBBLE__ = {data};</script>" + SCRIPT


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "output"))
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    b = run(Path(a.out), a.offline)
    print("\n".join(report_lines(b)) if b else "没有数据")
