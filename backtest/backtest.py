"""用历史数据校准流动性仪表盘:信号规则回测。

数据:output/raw_data.csv(仪表盘自己的衍生序列,2018 年起)+ output/market.csv(SPY/QQQ 复权价)。
防偷看:每个序列按公布滞后往后挪(pit.py),信号次日收盘执行,每次换仓扣 5BP,空仓部分拿联邦基金利率。
防过拟合:参数只取粗档;2019-03~2022-12 为样本内,2023-01 起为样本外;另给参数网格和“同等平均仓位”对照。

输出:output/backtest_summary.md、output/signals.csv、output/backtest.json(仪表盘读取)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from pit import PUB_LAG  # noqa: E402
from liquidity_monitor import score  # noqa: E402

START = "2004-01-02"          # 2026-11 起把数据补到 2003 年,规则不变,多出 2004~2018 这段“更早的样本外”
SPLIT = "2023-01-01"          # 之前样本内,之后样本外
COST = 0.0005                 # 每换 100% 仓位的成本
PERIODS = [("更早样本外 2004~2018", START, "2019-03-14"),
           ("样本内 2019-03~2022-12", "2019-03-15", "2022-12-31"),
           ("样本外 2023-01~至今", SPLIT, "2100-01-01"),
           ("全期 2004~至今", START, "2100-01-01")]

# 规则参数(事先定的粗档,不做细调):13 周 ≈ 65 个工作日,确认 20 个工作日 ≈ 一个月
NL_WIN, RATE_WIN, RATE_JUMP, CONFIRM, STEP = 65, 65, 0.75, 20, 0.25
LOW_POS = 0.5  # 推荐规则在“净流动性收缩”时的仓位
BTC_START = "2024-01-11"  # 美国现货比特币 ETF 上市首日


def persist(bad: pd.Series, good: pd.Series, n: int, bad_pos: float = 0.0) -> pd.Series:
    """条件连续 n 天成立才切换;否则维持原状态。返回 1(正常)或 bad_pos。"""
    lo = bad.astype(int).rolling(n).sum() == n
    hi = good.astype(int).rolling(n).sum() == n
    p = pd.Series(np.nan, index=bad.index)
    p[lo] = bad_pos
    p[hi] = 1.0
    return p.ffill().fillna(1.0)


def load(f: pd.DataFrame | None = None):
    if f is None:
        f = pd.read_csv(ROOT / "output" / "raw_data.csv", index_col=0, parse_dates=True)
    g = f.copy()
    for c, lag in PUB_LAG.items():
        if c in g:
            g[c] = g[c].shift(lag)
    sc = score(g)
    m = pd.read_csv(ROOT / "output" / "market.csv", index_col=0, parse_dates=True)
    spy = m["SPY"].dropna()
    idx = spy.index[spy.index >= START]
    F = g.reindex(idx, method="ffill")
    S = sc.reindex(idx, method="ffill")
    cash = ((f["real_policy_rate"] + f["core_pce_yoy"]).reindex(idx, method="ffill") / 100 / 252).fillna(0)
    rets = {k: m[k].dropna().pct_change().reindex(idx).fillna(0) for k in ["SPY", "QQQ"] if k in m}
    return idx, F, S, cash, rets, spy.reindex(idx)


def rules(idx, F, S, spy):
    li = S["liquidity_index"]
    sm = li.rolling(20).mean()
    base = li.rolling(250, min_periods=120).mean()
    nl = F["net_liq"].diff(NL_WIN)
    u5 = F["ust5"].diff(RATE_WIN)
    nl_ok = persist(nl < 0, nl > 0, CONFIRM, 0.0)
    rate_ok = persist(u5 > RATE_JUMP, u5 < 0, CONFIRM, 0.0)
    below = spy < spy.rolling(200).mean()
    R = {
        "买入持有": pd.Series(1.0, idx),
        "① 综合指数水平:20日均值<35 满20天→50%,>50 满20天→100%": persist(sm < 35, sm > 50, 20, 0.5),
        "② 综合指数趋势:20日均值低于一年均值5分 满20天→50%": persist(sm < base - 5, sm > base + 5, 20, 0.5),
        "③ 推荐:净流动性13周变化<0 满20天→50%,>0 满20天→100%": persist(nl < 0, nl > 0, CONFIRM, LOW_POS),
        "④ 净流动性 + 利率冲击(5年债13周升>0.75点),各扣25%": 1 - STEP * (1 - nl_ok) - STEP * (1 - rate_ok),
        "参考:SPY 跌破200日线 满5天→50%(纯价格趋势)": persist(below, ~below, 5, 0.5),
    }
    # 候选(2026-11-25 补 2004~2018 数据后发现 ③ 在早年失效才提出,属于看过数据后的选择,要打折看)
    R["⑤ 候选:①或③任一减仓就半仓"] = pd.concat([R[list(R)[1]], R[list(R)[3]]], axis=1).min(axis=1)
    extra = {"sm": sm, "base": base, "nl": nl, "u5": u5, "nl_ok": nl_ok, "rate_ok": rate_ok}
    return R, extra


# 页面上用的短名和说明,顺序与 rules() 一致
RULE_META = [
    ("bh", "买入持有", "一直满仓,作为基准。"),
    ("lvl", "综合指数·水平", "综合指数 20 日均值低于 35 连续 20 天 → 半仓;高于 50 连续 20 天 → 满仓。"),
    ("trd", "综合指数·趋势", "综合指数 20 日均值比自己一年均值低 5 分以上连续 20 天 → 半仓;高 5 分以上连续 20 天 → 满仓。"),
    ("nl", "净流动性(推荐)", "净流动性(联储总资产 − TGA − 逆回购)13 周变化连续 20 天为负 → 半仓;连续 20 天为正 → 满仓。"),
    ("nlr", "净流动性 + 利率冲击", "净流动性收缩扣 25%,5 年美债 13 周上升超过 0.75 个百分点再扣 25%(100% / 75% / 50%)。"),
    ("ma", "参考·200 日线", "SPY 跌破 200 日均线连续 5 天 → 半仓;站回 5 天 → 满仓。纯价格趋势,不看流动性。"),
    ("combo", "指数+净流动性(候选)", "综合指数水平规则 ① 或净流动性规则 ③ 任何一个处于减仓状态 → 半仓;两个都满仓才满仓。"
     "这是补上 2004–2018 数据、看到 ③ 在 2008 年失效之后才提出的组合,没有经过“事先定好再检验”,结果要打折看。"),
]
FWD = 63  # “之后三个月”= 63 个交易日


def simulate(pos: pd.Series, r: pd.Series, cash: pd.Series, px: pd.Series) -> dict:
    """单条规则的资金曲线、逐年收益、每段操作结果、各状态下之后三个月的表现。"""
    p = pos.shift(1).fillna(1.0)
    ret = p * r + (1 - p) * cash - p.diff().abs().fillna(0) * COST
    eq = (1 + ret).cumprod()
    yearly = {str(y): round(float((1 + g).prod() - 1), 4) for y, g in ret.groupby(ret.index.year)}
    # 每段持仓:从一次切换到下一次切换
    seg, start = [], pos.index[0]
    chg = list(pos.index[pos.diff().fillna(0) != 0]) + [None]
    for end in chg:
        sl = slice(start, end) if end is None else slice(start, pos.index[pos.index.get_loc(end) - 1])
        rr, cc = r[sl], cash[sl]
        a_ret, c_ret = float((1 + rr).prod() - 1), float((1 + cc).prod() - 1)
        level = float(pos[start])
        # 减仓段:股票跑输现金才算判断对;满仓段:股票跑赢现金才算对
        right = (a_ret < c_ret) if level < 1 else (a_ret >= c_ret)
        seg.append({"start": start.strftime("%Y-%m-%d"), "end": (pos.index[-1] if end is None else pos.index[pos.index.get_loc(end) - 1]).strftime("%Y-%m-%d"),
                    "pos": level, "asset": round(a_ret, 4), "cash": round(c_ret, 4), "right": bool(right),
                    "open": end is None})
        if end is not None:
            start = end
    fwd = px.shift(-FWD) / px - 1
    states = {}
    for lab, mask in [("full", pos >= 1), ("reduced", pos < 1)]:
        x = fwd[mask].dropna()
        states[lab] = {"days": int(mask.sum()), "avg": round(float(x.mean()), 4) if len(x) else None,
                       "up": round(float((x > 0).mean()), 4) if len(x) else None,
                       "bad": round(float((x < -0.10).mean()), 4) if len(x) else None}
    closed = [g for g in seg[1:] if not g["open"]]  # 第一段是起始默认满仓,不算一次判断
    return {"equity": [round(float(v), 4) for v in eq], "yearly": yearly, "segments": seg,
            "states": states, "hit": round(sum(g["right"] for g in closed) / len(closed), 4) if closed else None,
            "n_closed": len(closed)}


def perf(pos: pd.Series, r: pd.Series, cash: pd.Series, a: str, b: str) -> dict:
    p = pos.shift(1).fillna(1.0)
    ret = (p * r + (1 - p) * cash - p.diff().abs().fillna(0) * COST)[a:b]
    eq = (1 + ret).cumprod()
    yrs = len(ret) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = ret.std() * np.sqrt(252)
    return {"cagr": cagr, "maxdd": (eq / eq.cummax() - 1).min(), "vol": vol,
            "sharpe": (ret - cash[a:b]).mean() * 252 / vol if vol else np.nan,
            "exposure": p[a:b].mean(), "switches": int((p[a:b].diff().abs() > 0).sum())}


def const_benchmark(e: float, r, cash, a, b) -> float:
    return perf(pd.Series(e, r.index), r, cash, a, b)["cagr"]


def run(f: pd.DataFrame | None = None, out: Path = ROOT / "output") -> dict:
    idx, F, S, cash, rets, spy = load(f)
    R, X = rules(idx, F, S, spy)
    L = ["# 流动性仪表盘历史回测", "",
         f"数据 {idx[0].date()} ~ {idx[-1].date()}。信号按数据公布滞后重算,次日执行,换仓成本 5BP,空仓部分拿联邦基金利率。",
         "“同仓位对照”= 一直持有与该规则平均仓位相同比例的 SPY;规则只有跑赢它,才说明择时本身有用。", ""]
    res = {}
    for sym, r in rets.items():
        L += [f"## {sym}", ""]
        for lab, a, b in PERIODS:
            L += [f"### {lab}", "", "| 规则 | 年化 | 同仓位对照 | 最大回撤 | 夏普 | 平均仓位 | 切换次数 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
            for name, pos in R.items():
                p = perf(pos, r, cash, a, b)
                cb = const_benchmark(p["exposure"], r, cash, a, b)
                res.setdefault(sym, {}).setdefault(lab, {})[name] = {**p, "const": cb}
                L.append(f"| {name} | {p['cagr']:.1%} | {cb:.1%} | {p['maxdd']:.1%} | {p['sharpe']:.2f} | "
                         f"{p['exposure']:.0%} | {p['switches']} |")
            L.append("")
    # 参数稳健性:净流动性规则
    r = rets["SPY"]
    L += ["## 参数稳健性(SPY,规则③,年化减去同仓位对照)", "",
          "| 窗口 \\ 确认天数 | 5 | 10 | 20 | 40 |", "|---|---:|---:|---:|---:|"]
    for w in [40, 65, 90, 130]:
        c = F["net_liq"].diff(w)
        cells = []
        for n in [5, 10, 20, 40]:
            pos = persist(c < 0, c > 0, n, 0.5)
            row = []
            for _, a, b in PERIODS[:3]:
                p = perf(pos, r, cash, a, b)
                row.append(p["cagr"] - const_benchmark(p["exposure"], r, cash, a, b))
            cells.append(f"{row[0]:+.1%} / {row[1]:+.1%} / {row[2]:+.1%}")
        L.append(f"| {w} 天 | " + " | ".join(cells) + " |")
    L += ["", "格式:更早样本外 2004~2018 / 样本内 2019~2022 / 样本外 2023~。", ""]
    # 推荐规则的切换记录
    pos = R["③ 推荐:净流动性13周变化<0 满20天→50%,>0 满20天→100%"]
    ch = pos[pos.diff().fillna(0) != 0]
    L += ["## 推荐规则 ③ 的仓位切换记录", "", "| 日期 | 新仓位 | 净流动性13周变化(十亿) | 5年美债13周变化(百分点) | SPY |", "|---|---:|---:|---:|---:|"]
    for d, v in ch.items():
        L.append(f"| {d.date()} | {v:.0%} | {X['nl'][d]:+.0f} | {X['u5'][d]:+.2f} | {spy[d]:.0f} |")
    (out / "backtest_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    sig = pd.DataFrame({"spy": spy, "liquidity_index_pit": S["liquidity_index"], "index_20d": X["sm"],
                        "net_liq_13w_chg": X["nl"], "ust5_13w_chg": X["u5"],
                        "net_liq_ok": X["nl_ok"], "rate_ok": X["rate_ok"],
                        "position": pos})
    sig.to_csv(out / "signals.csv", float_format="%.3f")
    last = sig.iloc[-1]
    # 距离下一次切换还差几天:当前仓位低 → 数净流动性连续为正的天数;仓位满 → 数连续为负的天数
    want_pos = bool(last["position"] < 1)
    cond = (X["nl"] > 0) if want_pos else (X["nl"] < 0)
    streak = 0
    for v in cond[::-1]:
        if not v:
            break
        streak += 1
    j = {
        "dates": [d.strftime("%Y-%m-%d") for d in sig.index],
        "spy": [round(x, 2) for x in sig["spy"]],
        "position": [float(x) for x in sig["position"]],
        "net_liq_chg": [None if pd.isna(x) else round(float(x), 1) for x in sig["net_liq_13w_chg"]],
        "ust5_chg": [None if pd.isna(x) else round(float(x), 3) for x in sig["ust5_13w_chg"]],
        "now": {"date": sig.index[-1].strftime("%Y-%m-%d"), "position": float(last["position"]),
                "net_liq_chg": round(float(last["net_liq_13w_chg"]), 1),
                "ust5_chg": round(float(last["ust5_13w_chg"]), 2),
                "net_liq_ok": bool(last["net_liq_ok"]), "rate_ok": bool(last["rate_ok"]),
                "streak": streak, "confirm": CONFIRM, "low_pos": LOW_POS},
        "table": [{"rule": n, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}}
                  for n, m in res["SPY"]["全期 2004~至今"].items()],
        "table_oos": [{"rule": n, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}}
                      for n, m in res["SPY"]["样本外 2023-01~至今"].items()],
        "rules": [],
        "params": {"nl_win": NL_WIN, "rate_win": RATE_WIN, "rate_jump": RATE_JUMP, "confirm": CONFIRM, "step": STEP},
    }
    # 每条规则的完整模拟(SPY 和 QQQ)
    pos_list = list(R.values())
    for (key, short, desc), pos_r in zip(RULE_META, pos_list):
        item = {"key": key, "name": short, "desc": desc,
                "position": [float(x) for x in pos_r], "assets": {}}
        for sym, r in rets.items():
            px = (1 + r).cumprod()
            sim = simulate(pos_r, r, cash, px)
            sim["stats"] = {}
            for lab, a, b in PERIODS:
                pf = perf(pos_r, r, cash, a, b)
                pf["const"] = const_benchmark(pf["exposure"], r, cash, a, b)
                sim["stats"][lab] = {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in pf.items()}
            item["assets"][sym] = sim
        j["rules"].append(item)
    j["periods"] = [lab for lab, _, _ in PERIODS]
    j["split"] = SPLIT
    qqq = pd.read_csv(ROOT / "output" / "market.csv", index_col=0, parse_dates=True)["QQQ"].reindex(idx).ffill()
    j["qqq"] = [None if pd.isna(x) else round(float(x), 2) for x in qqq]
    j["index_pit"] = [None if pd.isna(x) else round(float(x), 1) for x in sig["liquidity_index_pit"]]
    # 比特币:只用现货 ETF 上市(2024-01-11)之后的数据,只给历史回放用,不进上面的成绩表
    mk = pd.read_csv(ROOT / "output" / "market.csv", index_col=0, parse_dates=True)
    if "BTC" in mk and mk["BTC"].notna().any():
        btc = mk["BTC"].reindex(idx).ffill().where(idx >= BTC_START)
        rb = btc.pct_change().fillna(0)
        on = idx >= BTC_START
        j["btc"] = [None if pd.isna(x) else round(float(x), 0) for x in btc]
        j["btc_start"] = BTC_START
        for (key, _, _), pos_r, item in zip(RULE_META, pos_list, j["rules"]):
            eq = (1 + (pos_r.shift(1).fillna(1.0) * rb + (1 - pos_r.shift(1).fillna(1.0)) * cash
                       - pos_r.diff().abs().fillna(0) * COST).where(on, 0)).cumprod()
            item["assets"]["BTC"] = {"equity": [round(float(v), 4) if o else None for v, o in zip(eq, on)]}
    try:
        import portfolio
        pf_ = portfolio.run(idx, cash, R, X.get("sm"))
        if pf_:
            j["portfolio"] = pf_
            with open(out / "backtest_summary.md", "a", encoding="utf-8") as fh:
                fh.write("\n" + "\n".join(portfolio.md(pf_)) + "\n")
    except Exception as e:  # noqa: BLE001
        print(f"[警告] 综合组合回测失败: {e}")
    (out / "backtest.json").write_text(json.dumps(j, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return j


if __name__ == "__main__":
    j = run()
    print("当前推荐仓位:", j["now"])
    print((ROOT / "output" / "backtest_summary.md").read_text())
