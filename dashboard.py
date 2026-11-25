"""
生成流动性仪表盘(单个自包含 HTML 文件,数据内嵌,图表用 Plotly)。
由 liquidity_monitor.py 在每次运行结束时调用。
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# 仪表盘里可查看的原始指标: 列名 -> (中文名, 单位, 分组, 预警线或 None)
RAW_SERIES = {
    "reserves":        ("银行准备金", "十亿美元", "A 钱的数量", 2900),
    "tga":             ("财政部 TGA", "十亿美元", "A 钱的数量", None),
    "rrp":             ("隔夜逆回购", "十亿美元", "A 钱的数量", None),
    "net_liq":         ("净流动性(总资产 − TGA − 逆回购)", "十亿美元", "A 钱的数量", None),
    "fed_bills_chg":   ("联储持有短债四周变化(含 MBS 再投资,不等于 RMP)", "十亿美元", "A 钱的数量", None),
    "sofr_iorb":       ("SOFR − IORB", "BP", "B 回购管道价格", 10),
    "sofr99_iorb":     ("SOFR99 − IORB", "BP", "B 回购管道价格", 25),
    "tgcr_rrp":        ("TGCR − 逆回购利率", "BP", "B 回购管道价格", None),
    "dvp_iorb":        ("OFR 双边回购(DVP)隔夜利率 − IORB", "BP", "B 回购管道价格", None),
    "dvp_vol":         ("OFR 双边回购隔夜成交量", "十亿美元", "B 回购管道价格", None),
    "srf":             ("SRF 等回购操作用量", "十亿美元", "B 回购管道价格", 10),
    "ccc_oas":         ("CCC 及以下 OAS", "%", "C 信用价格", 12.15),
    "baa_spread":      ("Baa 公司债 − 10 年美债利差(长历史)", "%", "C 信用价格", None),
    "ccc_cost_approx": ("CCC 借钱成本(5 年美债 + CCC OAS)", "%", "C 信用价格", None),
    "hy_oas":          ("高收益债 OAS", "%", "C 信用价格", None),
    "ig_oas":          ("投资级 OAS", "%", "C 信用价格", None),
    "real_policy_rate":("实际政策利率(EFFR − 核心 PCE)", "%", "D 利率与政策", None),
    "tips5":           ("5 年 TIPS 实际收益率", "%", "D 利率与政策", None),
    "tips10":          ("10 年 TIPS 实际收益率(观察黄金用,不计分)", "%", "D 利率与政策", None),
    "term_premium":    ("10 年期限溢价", "%", "D 利率与政策", None),
    "core_pce_yoy":    ("核心 PCE 同比", "%", "D 利率与政策", None),
    "vix":             ("VIX", "点", "E 市场压力与美元", None),
    "usdjpy":          ("美元兑日元", "日元", "E 市场压力与美元", None),
    "nfci":            ("芝加哥联储 NFCI(正 = 偏紧)", "", "E 市场压力与美元", 0),
    "ofr_fsi":         ("OFR 金融压力指数(正 = 压力)", "", "E 市场压力与美元", 0),
    "stlfsi":          ("圣路易斯联储金融压力指数(正 = 压力)", "", "E 市场压力与美元", 0),
}


def _clean(v):
    if v is None:
        return None
    try:
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
    except TypeError:
        pass
    return v


def _col(df: pd.DataFrame, c: str, nd: int):
    if c not in df:
        return [None] * len(df)
    return [_clean(round(float(x), nd)) if pd.notna(x) else None for x in df[c]]


def write_dashboard(f: pd.DataFrame, sc: pd.DataFrame, alerts: list[str], cal: pd.DataFrame,
                    path: Path, log_path: Path, bt: dict | None = None, w: dict | None = None,
                    asof: dict | None = None, rs: dict | None = None) -> None:
    from liquidity_monitor import PILLAR_NAMES, PILLAR_WEIGHTS, SCORED
    from watch import calendar_rows
    asof = asof or {}

    valid = sc.dropna(subset=["liquidity_index"])
    idx = valid.index
    ff = f.ffill().reindex(idx)

    data = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "index": _col(valid, "liquidity_index", 1),
        "pillars_used": _col(valid, "pillars_used", 0),
        "pillars": {p: {"name": PILLAR_NAMES[p], "weight": w, "z": _col(valid, p, 2)}
                    for p, w in PILLAR_WEIGHTS.items()},
        "raw": {k: {"name": n, "unit": u, "group": g, "threshold": t, "asof": asof.get(k),
                    "v": _col(ff, k, 3)}
                for k, (n, u, g, t) in RAW_SERIES.items() if k in ff},
        "contrib": [{"pillar": PILLAR_NAMES[p], "name": name,
                     "z": _clean(round(float(valid[f"z_{c}"].ffill().iloc[-1]), 2))
                     if f"z_{c}" in valid and pd.notna(valid[f"z_{c}"].ffill().iloc[-1]) else None}
                    for c, p, _, name in SCORED],
        "alerts": alerts,
        "calendar": calendar_rows(cal, w, 30),
        "watch": w,
        "research": rs,
        "bt": bt,
        "log": json.loads(pd.read_csv(log_path).to_json(orient="records")) if log_path.exists() else [],
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>美元流动性仪表盘</title>
<!-- Plotly 放在仓库 docs/ 里随页面一起发布;部分网络(如中国大陆)打不开 cdnjs,图表会空白。本地文件失败时再退回 CDN -->
<script src="plotly.min.js"></script>
<script>window.Plotly || document.write('<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"><\/script>');</script>
<script>window.Plotly || document.write('<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"><\/script>');</script>
<style>
:root{
  --bg:#f6f7f9; --card:#ffffff; --ink:#1b1f24; --muted:#5f6b7a; --line:#e3e7ec; --grid:#eef1f4;
  --accent:#2f6fde; --tight:#d64545; --semitight:#e8913a; --neutral:#8a94a3; --semiloose:#57b26a; --loose:#2e8b57;
  --band-tight:rgba(214,69,69,.10); --band-semitight:rgba(232,145,58,.10); --band-neutral:rgba(138,148,163,.08);
  --band-semiloose:rgba(87,178,106,.10); --band-loose:rgba(46,139,87,.13);
  --p1:#2f6fde; --p2:#e8913a; --p3:#d64545; --p4:#7b5cd6; --p5:#2aa198;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#111418; --card:#1a1f25; --ink:#e7ebf0; --muted:#98a3b1; --line:#2a313a; --grid:#232a32;
    --accent:#6b9cff; --tight:#ff6b6b; --semitight:#f0a35a; --neutral:#8d97a6; --semiloose:#6cc87e; --loose:#43b27a;
    --band-tight:rgba(255,107,107,.12); --band-semitight:rgba(240,163,90,.12); --band-neutral:rgba(141,151,166,.08);
    --band-semiloose:rgba(108,200,126,.12); --band-loose:rgba(67,178,122,.15);
    --p1:#6b9cff; --p2:#f0a35a; --p3:#ff6b6b; --p4:#a58bff; --p5:#3cc2b5;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif}
.wrap{max-width:1120px;margin:0 auto;padding:20px 16px 48px}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:17px;margin:0 0 12px}
.sub{color:var(--muted);font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-top:16px}
.hero{display:flex;flex-wrap:wrap;gap:20px;align-items:center}
.big{font-size:56px;font-weight:700;line-height:1;font-variant-numeric:tabular-nums}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;color:#fff;font-weight:600;font-size:14px}
.deltas{display:grid;grid-template-columns:repeat(4,minmax(72px,1fr));gap:8px;flex:1;min-width:280px}
.delta{background:var(--bg);border-radius:8px;padding:8px 10px}
.delta b{display:block;font-size:18px;font-variant-numeric:tabular-nums}
.delta span{font-size:12px;color:var(--muted)}
.alerts{margin:12px 0 0;padding:0;list-style:none}
.alerts li{padding:8px 10px;border-left:3px solid var(--tight);background:var(--band-tight);border-radius:4px;margin-top:6px;font-size:14px}
.ok{color:var(--muted);font-size:14px;margin-top:10px}
.chart{width:100%;height:360px}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:10px}
select,input[type=date],button{font:inherit;color:var(--ink);background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:6px 10px}
button{cursor:pointer}
button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600}
.pos{color:var(--loose)} .neg{color:var(--tight)}
.bar{display:inline-block;height:8px;border-radius:4px;vertical-align:middle}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media (max-width:760px){.grid2{grid-template-columns:1fr}.big{font-size:44px}.chart{height:300px}}
.note{color:var(--muted);font-size:13px;margin-top:8px}
.scroll{overflow-x:auto}
select{max-width:100%}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
@media (max-width:900px){.grid3{grid-template-columns:1fr}}
.wbox{background:var(--bg);border-radius:10px;padding:12px;font-size:14px}
.wbox h3{font-size:14px;margin:0 0 6px;color:var(--muted);font-weight:600}
.wbox .v{font-size:26px;font-weight:700;font-variant-numeric:tabular-nums}
.chk{margin:4px 0}
.verdict{border-left:3px solid var(--accent);padding:6px 10px;margin:8px 0;background:var(--bg);border-radius:4px;font-size:14px}
details{margin-top:10px} summary{cursor:pointer;color:var(--accent)}
.gbox{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between}
.gstat{font-size:22px;font-weight:700;padding:4px 12px;border-radius:8px;color:#fff}
.abar{display:flex;height:30px;border-radius:6px;overflow:hidden;margin:10px 0 4px;font-size:12px;color:#fff;font-weight:600}
.abar div{display:flex;align-items:center;justify-content:center;white-space:nowrap;overflow:hidden}
.alab{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:13px}
.alab i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}
</style>
</head>
<body>
<div class="wrap">
  <h1>美元流动性仪表盘</h1>
  <div class="sub" id="asof"></div>

  <div class="card" id="guideCard">
    <h2>操作指引(综合组合:比特币 + 美股 + 美债 + 现金)</h2>
    <div class="gbox"><div><div class="sub" id="gdDate"></div><div style="margin-top:4px"><span class="gstat" id="gdStat"></span></div></div>
      <div class="sub" id="gdEq" style="text-align:right"></div></div>
    <div class="abar" id="gdBar"></div>
    <div class="alab" id="gdLab"></div>
    <div class="sub" id="gdWhy" style="margin-top:6px"></div>
    <div class="row" style="margin-top:8px"><button data-gr="etf" class="on">2024 ETF 上市后</button><button data-gr="all">2004 年至今(看历史上的宽松加码)</button></div>
    <div id="cGuide" class="chart" style="height:380px;margin-top:6px"></div>
    <div class="note">把鼠标移到图上任意一天(手机上点一下),上面就显示那一天的操作建议;移开或在最右端就是今天的建议。上图:流动性综合指数 20 日均值,红区 = 低于 35,绿区 = 高于 50,深绿 = 高于 70。下图:每天各资产的建议比例。三档规则:① 20 日均值低于 35 连续 20 天 → 减半(比特币 20%,股票减半,其余放现金),高于 50 连续 20 天 → 恢复正常(比特币 40%);② 美联储宣布 QE/大规模购债的第二天 → 宽松加码(比特币 70%、股票 25%、美债 5%),持有约一年。减半信号优先:2025-12 联储宣布购买短债时,指数已处在减半状态,所以没有加码;点“2004 年至今”可以看 2009、2010–11、2012–13、2020 年等 QE 期规则会怎么建议(紫色底色 = 加码,红色底色 = 减半)。2024 年前的比特币比例只是规则输出,不算收益。比例是假设的,只是规则输出,不是投资建议。</div>
  </div>

  <div class="card">
    <div class="hero">
      <div>
        <div class="sub">综合流动性指数(0–100,50 = 三年中性)</div>
        <div class="big" id="now"></div>
        <div style="margin-top:8px"><span class="badge" id="regime"></span></div>
      </div>
      <div class="deltas" id="deltas"></div>
    </div>
    <ul class="alerts" id="alerts"></ul>
  </div>

  <div class="card" id="btCard">
    <h2>当前操作信号(净流动性规则)</h2>
    <div class="hero">
      <div>
        <div class="sub">建议股票仓位</div>
        <div class="big" id="btPos"></div>
      </div>
      <div style="flex:1;min-width:260px" id="btNow"></div>
    </div>
    <div id="cNl" class="chart" style="height:220px;margin-top:8px"></div>
    <div class="note">净流动性(联储总资产 − TGA − 逆回购)13 周变化,十亿美元。红色底色 = 规则处于半仓的时段。规则:13 周变化连续 20 个交易日为负 → 半仓;连续 20 个交易日为正 → 满仓。为什么选它,见下面的历史模拟。</div>
  </div>

  <div class="card" id="replayCard">
    <h2>历史回放:每个阶段仪表盘怎么说、指引怎么变、结果如何</h2>
    <div class="sub">从上到下:① 当时能看到的综合指数(按数据公布滞后重算);② 净流动性 13 周变化;③ 规则给出的建议仓位;④ 按指引操作 vs 一直满仓的资金曲线。点图上任意一天,或在下面选日期,看那一天的完整读数。</div>
    <div class="row" style="margin-top:10px" id="rpAsset"></div>
    <div class="row" id="rpRule"></div>
    <div id="cReplay" class="chart" style="height:680px"></div>
    <div class="row" style="margin-top:6px"><label>回放日期 <input type="date" id="rpDate"></label>
      <button data-rp="2008-09-15">2008 雷曼</button><button data-rp="2015-08-24">2015 人民币</button><button data-rp="2018-12-24">2018 缩表</button>
      <button data-rp="2020-03-16">2020 疫情</button><button data-rp="2022-06-13">2022 加息</button><button data-rp="2025-04-08">2025 关税</button></div>
    <div class="grid2"><div class="scroll"><table id="rpSnap"></table></div><div class="scroll"><table id="rpAfter"></table></div></div>
  </div>

  <div class="card" id="pfCard">
    <h2>综合组合:比特币 + 美股 + 美债 + 现金,一个流动性信号统一调仓</h2>
    <div class="sub" id="pfRule"></div>
    <div id="pfToday" class="verdict" style="margin-top:8px"></div>
    <h3 style="margin:14px 0 4px">2024-01-11 起(含比特币),1 元起步</h3>
    <div id="cPf" class="chart" style="height:300px"></div>
    <div class="scroll"><table id="pfTbl"></table></div>
    <div class="scroll" style="margin-top:8px"><table id="pfSingle"></table></div>
    <h3 style="margin:14px 0 4px">2004 年起(不含比特币):股债部分长期是否站得住</h3>
    <div class="sub" id="pfLongW"></div>
    <div class="scroll"><table id="pfLong"></table></div>
    <h3 style="margin:14px 0 4px">宽松加码档在历史上什么时候出现、结果如何</h3>
    <div class="scroll"><table id="pfLoose"></table></div>
    <div class="scroll" style="margin-top:8px"><table id="pfRef"></table></div>
    <div class="note">第二张表用的是 ETF 上市前的比特币,那时它还不太受美元流动性驱动,所以只作参考。2021 年加码之后正好赶上 2022 年大跌,这就是加码档的代价:等指数确认“真宽松”,价格往往已经涨了很多。</div>
    <h3 style="margin:14px 0 4px">未来一年推演(不是预测)</h3>
    <div class="scroll"><table id="pfBoot"></table></div>
    <div class="note">把历史每日收益按 63 天一段打乱重抽,模拟 5000 个“未来一年”,看结果的范围。它假设未来和抽样那段历史相似;如果出现历史上没有的冲击,真实结果会落在范围之外。</div>
  </div>

  <div class="card" id="btcCard">
    <h2>比特币与流动性(只看 2024-01 现货 ETF 上市之后)</h2>
    <div class="sub">机构通过 ETF 大举买入之后,比特币才更像一项受美元流动性影响的资产;更早的“小圈子时代”不纳入。样本不到 3 年,只能当参考。</div>
    <div id="cBtc" class="chart" style="height:320px;margin-top:8px"></div>
    <div class="note">蓝线 = 比特币价格(对数,右轴);柱 = 净流动性 13 周变化(十亿美元,左轴);红色底色 = 净流动性规则 ③ 半仓的时段。</div>
    <h3 style="margin:14px 0 4px">按流动性择时 vs 一直满仓(2024-01-11 起,1 元起步)</h3>
    <div id="cBtcEq" class="chart" style="height:300px"></div>
    <div class="scroll"><table id="btcPerf"></table></div>
    <div class="note">信号次日执行,每次调仓扣 0.05%,空仓部分拿联邦基金利率。择时增益 = 年化 − 一直持有同等平均仓位的年化。只有 3–5 次调仓,运气成分很大。</div>
    <div id="cBtcCorr" class="chart" style="height:200px"></div>
    <div class="note">比特币与纳指 100 的 60 日收益相关性(2020 年起,虚线 = ETF 上市)。相关性越高,比特币越像“高波动的科技股”,越受同一股流动性驱动。</div>
    <div id="btcVerdict"></div>
  </div>

  <div class="card" id="watchCard">
    <h2>管道与砍仓观察(未经回测验证,不进仓位规则)</h2>
    <div class="grid3">
      <div class="wbox" id="wPipe"></div>
      <div class="wbox" id="wReb"></div>
      <div class="wbox" id="wCross"></div>
    </div>
    <div class="grid2" style="margin-top:12px">
      <div><div class="sub">SOFR − IORB 近一年(BP),背景 = 分档</div><div id="cPipe" class="chart" style="height:220px"></div></div>
      <div><div class="sub">黄金(GLD)与纳指(QQQ)20 日收益相关性</div><div id="cCorr" class="chart" style="height:220px"></div></div>
    </div>
    <div class="note" id="wNote"></div>
  </div>

  <div class="card" id="simCard">
    <h2>历史模拟:用不同数据指导操作,结果如何</h2>
    <div class="sub">2004 年起,假如每天按某条规则调整美股仓位。规则是用 2019–2022 的数据定的,2004–2018 和 2023 年以后都是它没见过的数据。信号按数据实际公布时间滞后计算,第二天执行,每次调仓扣 0.05% 成本,没买股票的钱拿联邦基金利率。</div>
    <div class="row" style="margin-top:12px" id="simAsset"></div>
    <div class="row" id="simPeriod"></div>
    <div class="row" id="simRules"></div>
    <div id="simDesc" class="note" style="margin-top:0"></div>
    <h2 style="margin-top:14px;font-size:15px" id="simPxTitle"></h2>
    <div id="cSimPx" class="chart"></div>
    <div class="note">红色底色 = 所选规则减仓的时段(颜色越深仓位越低)。虚线 = 样本内 / 样本外分界(2019-03 和 2023-01)。</div>
    <h2 style="margin-top:14px;font-size:15px">资金曲线(起点 = 1)</h2>
    <div id="cSimEq" class="chart"></div>
    <div class="note">粗线 = 所选规则。点击图例可隐藏或显示某条线。</div>
    <h2 style="margin-top:18px;font-size:15px">各规则成绩对比</h2>
    <div class="scroll"><table id="simTable"></table></div>
    <div class="note">“同仓位对照”= 一直持有和该规则平均仓位相同比例的股票。“择时增益”= 年化 − 同仓位对照,只有它为正才说明这套数据真的帮上了忙,而不只是少买了股票。“判断正确率”= 每次调仓之后到下一次调仓之间:减仓段股票跑输现金、加仓段股票跑赢现金,算判断正确。</div>
    <div class="grid2" style="margin-top:14px">
      <div>
        <h2 style="font-size:15px">所选规则:之后三个月的概率</h2>
        <div class="scroll"><table id="simStates"></table></div>
        <div class="note">按规则处于满仓或减仓状态的每一天,统计之后 63 个交易日股票的涨跌。减仓时的上涨概率和平均涨幅如果明显低于满仓时,说明这套数据能区分好坏时期;反过来就说明它常在不该减仓的时候减仓。</div>
      </div>
      <div>
        <h2 style="font-size:15px">所选规则:逐年收益</h2>
        <div class="scroll"><table id="simYears"></table></div>
      </div>
    </div>
    <h2 style="margin-top:18px;font-size:15px">所选规则:每一次调仓</h2>
    <div class="scroll"><table id="simSegs"></table></div>
    <div class="note">历史表现不代表未来。2020 年疫情、2025 年 4 月关税这类突发暴跌,流动性数据事先看不出来。完整报告见仓库 output/backtest_summary.md。</div>
  </div>

  <div class="card" id="resCard">
    <h2>新指标历史检验</h2>
    <div class="sub" id="resSub"></div>
    <div id="resBody"></div>
  </div>

  <div class="card">
    <h2>综合指数走势</h2>
    <div class="row" id="rangeBtns"></div>
    <div id="cIndex" class="chart"></div>
    <div class="note">背景色:红 = 紧张(&lt;30),橙 = 偏紧(30–45),灰 = 中性(45–55),浅绿 = 偏松(55–70),绿 = 宽松(≥70)。</div>
  </div>

  <div class="card">
    <h2>五大支柱(z 分数,正 = 宽松)</h2>
    <div id="cPillars" class="chart"></div>
    <div class="note">点击图例可隐藏或显示某条线。</div>
  </div>

  <div class="card">
    <h2>单项指标</h2>
    <div class="row"><select id="rawSel"></select><span class="sub" id="rawNote"></span></div>
    <div id="cRaw" class="chart"></div>
  </div>

  <div class="card">
    <h2>任意两天对比</h2>
    <div class="row">
      <label>日期 A <input type="date" id="dA"></label>
      <label>日期 B <input type="date" id="dB"></label>
      <button data-back="5">B = A 前一周</button>
      <button data-back="21">前一月</button>
      <button data-back="252">前一年</button>
    </div>
    <div class="scroll"><table id="cmp"></table></div>
  </div>

  <div class="grid2">
    <div class="card">
      <h2>今日各序列贡献</h2>
      <div class="scroll"><table id="contrib"></table></div>
    </div>
    <div class="card">
      <h2>每日运行记录</h2>
      <div class="scroll"><table id="log"></table></div>
      <h2 style="margin-top:18px">未来 30 天日历</h2>
      <div class="scroll"><table id="cal"></table></div>
      <div class="note" id="calNote"></div>
    </div>
  </div>

  <div class="note" style="margin-top:16px">数据来源:FRED(圣路易斯联储)、纽约联储、美国财政部。ICE 信用利差在 FRED 上只有约 3 年历史,所以较早年份的综合指数没有信用支柱,用其余支柱按权重归一计算。仅用于研究,不构成投资建议。</div>
</div>

<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const N = D.dates.length;
const last = i => { for (let k = i; k >= 0; k--) if (D.index[k] != null) return k; return -1; };
const li = last(N - 1);
const fmt = (v, nd = 1) => v == null ? "—" : Number(v).toLocaleString("zh-CN", {minimumFractionDigits: nd, maximumFractionDigits: nd});
function regime(v){
  if (v == null) return ["数据不足", css("--neutral")];
  if (v >= 70) return ["宽松", css("--loose")];
  if (v >= 55) return ["偏松", css("--semiloose")];
  if (v >= 45) return ["中性", css("--neutral")];
  if (v >= 30) return ["偏紧", css("--semitight")];
  return ["紧张", css("--tight")];
}
function idxOnOrBefore(dstr){
  let lo = 0, hi = N - 1, ans = -1;
  while (lo <= hi){ const m = (lo + hi) >> 1; if (D.dates[m] <= dstr){ ans = m; lo = m + 1; } else hi = m - 1; }
  return ans;
}

// 头部
document.getElementById("asof").textContent = `数据截至 ${D.dates[li]} · 页面生成于 ${D.generated}`;
const nowV = D.index[li];
document.getElementById("now").textContent = fmt(nowV);
const [rn, rc] = regime(nowV);
const badge = document.getElementById("regime"); badge.textContent = rn; badge.style.background = rc;
document.getElementById("deltas").innerHTML = [["一周前",5],["四周前",20],["三个月前",63],["一年前",252]].map(([t,n]) => {
  const k = li - n, v = k >= 0 ? D.index[k] : null, d = v == null ? null : nowV - v;
  const cls = d == null ? "" : d >= 0 ? "pos" : "neg";
  return `<div class="delta"><span>${t}</span><b>${fmt(v)}</b><span class="${cls}">${d == null ? "" : (d >= 0 ? "+" : "") + fmt(d)}</span></div>`;
}).join("");
document.getElementById("alerts").outerHTML = D.alerts.length
  ? `<ul class="alerts">${D.alerts.map(a => `<li>${a}</li>`).join("")}</ul>`
  : `<div class="ok">当前没有触发硬预警。</div>`;

// Plotly 通用布局
function baseLayout(extra = {}){
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: {color: css("--ink"), size: 12},
    margin: {l: 48, r: 16, t: 10, b: 36},
    xaxis: {gridcolor: css("--grid"), linecolor: css("--line"), type: "date"},
    yaxis: {gridcolor: css("--grid"), zerolinecolor: css("--line")},
    hovermode: "x unified", showlegend: false,
    legend: {orientation: "h", y: -0.15},
  }, extra);
}
const cfg = {displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d","select2d"]};

// 综合指数
const bands = [[0,30,"--band-tight"],[30,45,"--band-semitight"],[45,55,"--band-neutral"],[55,70,"--band-semiloose"],[70,100,"--band-loose"]]
  .map(([a,b,c]) => ({type:"rect", xref:"paper", x0:0, x1:1, y0:a, y1:b, fillcolor: css(c), line:{width:0}, layer:"below"}));
Plotly.newPlot("cIndex", [{x: D.dates, y: D.index, type: "scatter", mode: "lines", line: {color: css("--accent"), width: 2},
  name: "综合指数", hovertemplate: "%{y:.1f}<extra></extra>"}],
  baseLayout({shapes: bands, yaxis: {range: [0,100], gridcolor: css("--grid"), dtick: 10}}), cfg);

// 支柱
const pk = Object.keys(D.pillars), pc = ["--p1","--p2","--p3","--p4","--p5"];
Plotly.newPlot("cPillars", pk.map((p,i) => ({x: D.dates, y: D.pillars[p].z, type:"scatter", mode:"lines",
  name: `${D.pillars[p].name}(${Math.round(D.pillars[p].weight*100)}%)`, line:{color: css(pc[i]), width:1.6},
  hovertemplate: "%{y:.2f}"})),
  baseLayout({showlegend: true, yaxis: {range: [-3.2,3.2], gridcolor: css("--grid"), zerolinecolor: css("--muted")}}), cfg);

// 时间范围按钮,同时作用于三张图
const ranges = [["1年",252],["3年",756],["5年",1260],["10年",2520],["全部",0]];
const rb = document.getElementById("rangeBtns");
function setRange(n){
  const x0 = n ? D.dates[Math.max(0, N - n)] : D.dates[0], x1 = D.dates[N-1];
  ["cIndex","cPillars","cRaw"].forEach(id => Plotly.relayout(id, {"xaxis.range": [x0, x1]}));
  [...rb.children].forEach(b => b.classList.toggle("on", +b.dataset.n === n));
}
ranges.forEach(([t,n]) => { const b = document.createElement("button"); b.textContent = t; b.dataset.n = n; b.onclick = () => setRange(n); rb.appendChild(b); });

// 单项指标
const sel = document.getElementById("rawSel");
let lastGroup = null, og = null;
Object.entries(D.raw).forEach(([k, s]) => {
  if (s.group !== lastGroup){ og = document.createElement("optgroup"); og.label = s.group; sel.appendChild(og); lastGroup = s.group; }
  const o = document.createElement("option"); o.value = k; o.textContent = s.name; og.appendChild(o);
});
function drawRaw(){
  const s = D.raw[sel.value];
  const shapes = s.threshold == null ? [] : [{type:"line", xref:"paper", x0:0, x1:1, y0:s.threshold, y1:s.threshold, line:{color: css("--tight"), dash:"dash", width:1}}];
  const xr = document.getElementById("cIndex").layout.xaxis.range;
  Plotly.react("cRaw", [{x: D.dates, y: s.v, type:"scatter", mode:"lines", line:{color: css("--accent"), width:1.8},
    hovertemplate: `%{y:,.2f} ${s.unit}<extra></extra>`}],
    baseLayout({shapes, xaxis: {gridcolor: css("--grid"), type:"date", range: xr}, yaxis: {gridcolor: css("--grid"), title: {text: s.unit}}}), cfg);
  const lv = s.v[li];
  document.getElementById("rawNote").textContent = `最新 ${fmt(lv, 2)} ${s.unit}` + (s.asof ? `(数据日期 ${s.asof})` : "") + (s.threshold == null ? "" : ` · 红虚线 = 预警线 ${s.threshold}`);
}
sel.onchange = drawRaw;
drawRaw();
setRange(756);

// 历史校准信号
if (!D.bt){ document.getElementById("btCard").style.display = "none"; document.getElementById("simCard").style.display = "none"; }
else {
  const B = D.bt, now = B.now;
  document.getElementById("btPos").textContent = Math.round(now.position * 100) + "%";
  const low = now.position < 1, need = Math.max(0, now.confirm - now.streak);
  const turn = low
    ? (now.net_liq_chg > 0 ? `净流动性 13 周变化已连续 ${now.streak} 个交易日为正,再 ${need} 天仍为正即恢复满仓。` : `净流动性仍在收缩,维持半仓。`)
    : (now.net_liq_chg < 0 ? `净流动性 13 周变化已连续 ${now.streak} 个交易日为负,再 ${need} 天仍为负即减到半仓。` : `净流动性在扩张,维持满仓。`);
  document.getElementById("btNow").innerHTML =
    `<div>截至 ${now.date}:净流动性 13 周变化 <b class="${now.net_liq_chg >= 0 ? "pos" : "neg"}">${now.net_liq_chg >= 0 ? "+" : ""}${fmt(now.net_liq_chg, 0)}</b> 十亿美元。</div>` +
    `<div style="margin-top:6px">${turn}</div>` +
    (() => { const lv = B.rules.find(r => r.key === "lvl"), cb = B.rules.find(r => r.key === "combo"); const L = x => Math.round(x.position[x.position.length - 1] * 100) + "%";
      return lv && cb ? `<div class="sub" style="margin-top:6px">对照:综合指数规则 ① 当前 ${L(lv)};候选组合(①或③任一减仓就半仓)当前 ${L(cb)}。2004–2018 的检验显示 ③ 单独使用在 2008 年失效,见下面历史模拟里的“更早样本外”。</div>` : ""; })() +
    `<div class="sub" style="margin-top:6px">辅助观察:5 年美债 13 周变化 ${now.ust5_chg >= 0 ? "+" : ""}${fmt(now.ust5_chg, 2)} 个百分点${now.rate_ok ? "" : "(利率冲击中,注意风险)"}。</div>`;
  const bandsOf = pos => {
    const out = []; let s0 = null;
    for (let i = 0; i <= pos.length; i++){
      const p = i < pos.length ? pos[i] : 1;
      if (s0 != null && (p !== pos[s0] || i === pos.length)){ out.push({x0: B.dates[s0], x1: B.dates[Math.min(i, pos.length - 1)], p: pos[s0]}); s0 = null; }
      if (s0 == null && p < 1 && i < pos.length) s0 = i;
    }
    return out.map(g => ({type:"rect", xref:"x", yref:"paper", x0:g.x0, x1:g.x1, y0:0, y1:1, line:{width:0}, layer:"below",
      fillcolor: css("--tight"), opacity: g.p <= 0.5 ? 0.16 : 0.08}));
  };
  const nlRule = B.rules.find(r => r.key === "nl");
  Plotly.newPlot("cNl", [{x: B.dates, y: B.net_liq_chg, type:"bar",
    marker:{color: B.net_liq_chg.map(v => v == null ? css("--neutral") : v >= 0 ? css("--loose") : css("--tight"))},
    hovertemplate: "%{y:+,.0f} 十亿<extra></extra>"}],
    baseLayout({shapes: bandsOf(nlRule.position), bargap: 0, yaxis: {gridcolor: css("--grid"), zerolinecolor: css("--muted")}}), cfg);

  // ---- 历史模拟
  const RC = {bh: "--muted", lvl: "--p2", trd: "--p4", nl: "--p1", nlr: "--p5", ma: "--p3", combo: "--ink"};
  const st = {asset: "SPY", period: B.periods.find(p => p.startsWith("全期")) || B.periods[0], rule: "nl"};
  const pct = (v, nd = 1, sign = false) => v == null ? "—" : (sign && v >= 0 ? "+" : "") + (v * 100).toFixed(nd) + "%";
  const lastD = B.dates[B.dates.length - 1];
  const periodRange = lab => lab.startsWith("更早") ? [B.dates[0], "2019-03-14"] : lab.startsWith("样本内") ? ["2019-03-15", "2022-12-31"] : lab.startsWith("样本外") ? [B.split, lastD] : [B.dates[0], lastD];
  const chips = (id, items, key) => {
    const el = document.getElementById(id); el.innerHTML = "";
    items.forEach(([v, t, col]) => { const b = document.createElement("button"); b.textContent = t; b.dataset.v = v;
      if (col) b.style.borderLeft = `4px solid ${css(col)}`;
      b.onclick = () => { st[key] = v; draw(); }; el.appendChild(b); });
  };
  chips("simAsset", [["SPY","标普 500(SPY)"],["QQQ","纳斯达克 100(QQQ)"]], "asset");
  chips("simPeriod", B.periods.map(p => [p, p]), "period");
  chips("simRules", B.rules.map(r => [r.key, r.name, RC[r.key]]), "rule");
  const mark = () => [["simAsset","asset"],["simPeriod","period"],["simRules","rule"]].forEach(([id,k]) =>
    [...document.getElementById(id).children].forEach(b => b.classList.toggle("on", b.dataset.v === st[k])));
  function draw(){
    mark();
    const R = B.rules.find(r => r.key === st.rule), A = R.assets[st.asset];
    const [x0, x1] = periodRange(st.period);
    const i0 = Math.max(0, B.dates.findIndex(d => d >= x0)), i1 = B.dates.length - 1 - [...B.dates].reverse().findIndex(d => d <= x1);
    document.getElementById("simDesc").textContent = R.desc;
    document.getElementById("simPxTitle").textContent = `${st.asset} 走势与「${R.name}」的仓位`;
    const px = st.asset === "SPY" ? B.spy : B.qqq;
    const split = [B.split, "2019-03-15"].map(x => ({type:"line", xref:"x", yref:"paper", x0:x, x1:x, y0:0, y1:1, line:{color: css("--muted"), dash:"dot", width:1}}));
    const posTxt = R.position.map(p => Math.round(p * 100) + "%");
    Plotly.react("cSimPx", [{x: B.dates, y: px, type:"scatter", mode:"lines", line:{color: css("--ink"), width:1.5},
      customdata: posTxt, hovertemplate: `${st.asset} %{y:,.0f} · 仓位 %{customdata}<extra></extra>`}],
      baseLayout({shapes: [...bandsOf(R.position), ...split], xaxis: {type:"date", range:[x0, x1], gridcolor: css("--grid")},
        yaxis: {type:"log", gridcolor: css("--grid")}}), cfg);
    const traces = B.rules.map(r => {
      const e = r.assets[st.asset].equity, base = i0 > 0 ? e[i0 - 1] : 1;
      const sel = r.key === st.rule;
      return {x: B.dates.slice(i0, i1 + 1), y: e.slice(i0, i1 + 1).map(v => v / base), type:"scatter", mode:"lines", name: r.name,
        line:{color: css(RC[r.key]), width: sel ? 3 : 1.2, dash: r.key === "bh" ? "dot" : "solid"}, opacity: sel || r.key === "bh" ? 1 : 0.55,
        hovertemplate: "%{y:.2f}"};
    });
    Plotly.react("cSimEq", traces, baseLayout({showlegend: true, shapes: split, xaxis: {type:"date", range:[x0, x1], gridcolor: css("--grid")},
      yaxis: {type:"log", gridcolor: css("--grid")}}), cfg);
    // 对比表
    document.getElementById("simTable").innerHTML =
      `<tr><th>规则</th><th>年化</th><th>同仓位对照</th><th>择时增益</th><th>最大回撤</th><th>夏普</th><th>平均仓位</th><th>调仓次数</th><th>判断正确率</th></tr>` +
      B.rules.map(r => {
        const s = r.assets[st.asset].stats[st.period], g = s.cagr - s.const, sel = r.key === st.rule;
        return `<tr style="${sel ? "background:var(--bg);font-weight:600" : ""};cursor:pointer" data-k="${r.key}"><td><span class="bar" style="width:10px;background:${css(RC[r.key])}"></span> ${r.name}</td>` +
          `<td>${pct(s.cagr)}</td><td>${pct(s.const)}</td><td class="${r.key === "bh" ? "" : g >= 0 ? "pos" : "neg"}">${r.key === "bh" ? "—" : pct(g, 1, true)}</td>` +
          `<td class="neg">${pct(s.maxdd)}</td><td>${fmt(s.sharpe, 2)}</td><td>${pct(s.exposure, 0)}</td><td>${s.switches}</td>` +
          `<td>${r.assets[st.asset].hit == null ? "—" : pct(r.assets[st.asset].hit, 0) + `(${r.assets[st.asset].n_closed} 次)`}</td></tr>`;
      }).join("");
    document.querySelectorAll("#simTable tr[data-k]").forEach(tr => tr.onclick = () => { st.rule = tr.dataset.k; draw(); });
    // 概率
    const S = A.states, row = (t, x) => x.days ? `<tr><td>${t}</td><td>${x.days}</td><td class="${x.avg >= 0 ? "pos" : "neg"}">${pct(x.avg, 1, true)}</td><td>${pct(x.up, 0)}</td><td>${pct(x.bad, 0)}</td></tr>` : `<tr><td>${t}</td><td>0</td><td colspan="3" class="sub">从未出现</td></tr>`;
    document.getElementById("simStates").innerHTML = `<tr><th>规则状态</th><th>天数</th><th>之后 3 个月平均</th><th>上涨概率</th><th>跌超 10% 概率</th></tr>` +
      row("满仓时", S.full) + row("减仓时", S.reduced);
    // 逐年
    const bhY = B.rules[0].assets[st.asset].yearly;
    document.getElementById("simYears").innerHTML = `<tr><th>年份</th><th>${R.name}</th><th>买入持有</th><th>差</th></tr>` +
      Object.keys(A.yearly).map(y => { const d = A.yearly[y] - bhY[y];
        return `<tr><td>${y}${y === B.dates[0].slice(0,4) ? "(3 月起)" : y === B.dates[B.dates.length-1].slice(0,4) ? "(至今)" : ""}</td><td>${pct(A.yearly[y])}</td><td>${pct(bhY[y])}</td><td class="${d >= 0 ? "pos" : "neg"}">${pct(d, 1, true)}</td></tr>`; }).join("");
    // 每段
    const segs = A.segments.slice(1);
    document.getElementById("simSegs").innerHTML = segs.length
      ? `<tr><th>开始</th><th>结束</th><th>仓位</th><th>期间 ${st.asset}</th><th>期间现金</th><th>判断</th></tr>` +
        [...segs].reverse().map(g => `<tr><td>${g.start}</td><td>${g.open ? "进行中" : g.end}</td><td>${Math.round(g.pos * 100)}%</td>` +
          `<td class="${g.asset >= 0 ? "pos" : "neg"}">${pct(g.asset, 1, true)}</td><td>${pct(g.cash, 1, true)}</td>` +
          `<td>${g.open ? "—" : g.right ? "✓ 正确" : "✗ 错误"}</td></tr>`).join("")
      : `<tr><td class="sub">这条规则从不调仓。</td></tr>`;
  }
  draw();
}

// 历史回放
if (!D.bt || !D.bt.index_pit){ document.getElementById("replayCard").style.display = "none"; }
else {
  const B = D.bt, nl = B.rules.find(r => r.key === "nl"), bh = B.rules.find(r => r.key === "bh");
  const rp = {asset: "SPY", date: null, rule: "nl"};
  const cur = () => B.rules.find(r => r.key === rp.rule) || nl;
  const rpRules = document.getElementById("rpRule");
  B.rules.filter(r => ["nl","lvl","combo"].includes(r.key)).forEach(r => { const b = document.createElement("button"); b.textContent = r.name; b.dataset.v = r.key; b.onclick = () => { rp.rule = r.key; snap(); }; rpRules.appendChild(b); });
  const bi = dstr => { let lo = 0, hi = B.dates.length - 1, a = 0; while (lo <= hi){ const m = (lo + hi) >> 1; if (B.dates[m] <= dstr){ a = m; lo = m + 1; } else hi = m - 1; } return a; };
  const pc = (v, s = true) => v == null || isNaN(v) ? "—" : (s && v >= 0 ? "+" : "") + (v * 100).toFixed(1) + "%";
  const rpChips = document.getElementById("rpAsset");
  [["SPY","标普 500(SPY)"],["QQQ","纳斯达克 100(QQQ)"], ...(B.btc ? [["BTC","比特币(ETF 上市后)"]] : [])].forEach(([v,t]) => { const b = document.createElement("button"); b.textContent = t; b.dataset.v = v;
    b.onclick = () => { rp.asset = v; if (v === "BTC" && (!rp.date || rp.date < B.btc_start)) rp.date = B.dates[B.dates.length - 1]; snap(); }; rpChips.appendChild(b); });
  function drawReplay(){
    [...rpChips.children].forEach(b => b.classList.toggle("on", b.dataset.v === rp.asset));
    [...rpRules.children].forEach(b => b.classList.toggle("on", b.dataset.v === rp.rule));
    const nl = cur();
    const eR = nl.assets[rp.asset].equity, eB = bh.assets[rp.asset].equity;
    const dom = [[0.78,1],[0.53,0.74],[0.30,0.49],[0,0.26]];
    const yax = (i, extra = {}) => Object.assign({domain: dom[i], gridcolor: css("--grid"), zerolinecolor: css("--muted")}, extra);
    const bandsY = [[0,30,"--band-tight"],[30,45,"--band-semitight"],[45,55,"--band-neutral"],[55,70,"--band-semiloose"],[70,100,"--band-loose"]]
      .map(([a,b,c]) => ({type:"rect", xref:"paper", yref:"y", x0:0, x1:1, y0:a, y1:b, fillcolor: css(c), line:{width:0}, layer:"below"}));
    const lowBands = []; let s0 = null;
    nl.position.forEach((p, i) => { if (p < 1 && s0 == null) s0 = i; if ((p >= 1 || i === nl.position.length - 1) && s0 != null){ lowBands.push({type:"rect", xref:"x", yref:"paper", x0:B.dates[s0], x1:B.dates[i], y0:0, y1:1, fillcolor: css("--tight"), opacity:0.07, line:{width:0}, layer:"below"}); s0 = null; } });
    const vline = rp.date ? [{type:"line", xref:"x", yref:"paper", x0:rp.date, x1:rp.date, y0:0, y1:1, line:{color: css("--ink"), width:1, dash:"dot"}}] : [];
    const tr = [
      {x: B.dates, y: B.index_pit, type:"scatter", mode:"lines", line:{color: css("--accent"), width:1.4}, name:"综合指数", yaxis:"y", hovertemplate:"指数 %{y:.1f}<extra></extra>"},
      {x: B.dates, y: B.net_liq_chg, type:"bar", marker:{color: B.net_liq_chg.map(v => v == null ? css("--neutral") : v >= 0 ? css("--loose") : css("--tight"))}, name:"净流动性 13 周变化", yaxis:"y2", hovertemplate:"净流动性 13 周 %{y:+,.0f} 十亿<extra></extra>"},
      {x: B.dates, y: nl.position.map(p => p * 100), type:"scatter", mode:"lines", line:{color: css("--p2"), width:2, shape:"hv"}, name:"建议仓位", yaxis:"y3", hovertemplate:"建议仓位 %{y:.0f}%<extra></extra>"},
      {x: B.dates, y: eB, type:"scatter", mode:"lines", line:{color: css("--muted"), width:1.2, dash:"dot"}, name:"一直满仓", yaxis:"y4", hovertemplate:"满仓 %{y:.2f}<extra></extra>"},
      {x: B.dates, y: eR, type:"scatter", mode:"lines", line:{color: css("--p1"), width:2}, name:"按指引操作", yaxis:"y4", hovertemplate:"按指引 %{y:.2f}<extra></extra>"},
    ];
    Plotly.react("cReplay", tr, baseLayout({showlegend: true, legend:{orientation:"h", y:-0.06}, bargap:0, margin:{l:56, r:16, t:10, b:30},
      shapes: [...bandsY, ...lowBands, ...vline],
      yaxis: yax(0, {range:[0,100], title:{text:"指数"}}), yaxis2: yax(1, {title:{text:"十亿"}}),
      yaxis3: yax(2, {range:[-5,105], title:{text:"仓位%"}}), yaxis4: yax(3, {type:"log", title:{text:"资金"}}),
      xaxis: {gridcolor: css("--grid"), type:"date", range: rp.asset === "BTC" ? [B.btc_start, B.dates[B.dates.length - 1]] : [B.dates[0], B.dates[B.dates.length - 1]]}}), cfg);
  }
  function snap(){
    const dI = document.getElementById("rpDate");
    if (!rp.date) rp.date = B.dates[B.dates.length - 1];
    dI.value = rp.date;
    const nl = cur();
    const i = bi(rp.date), d = B.dates[i], k = idxOnOrBefore(d), last = B.dates.length - 1;
    const px = rp.asset === "SPY" ? B.spy : rp.asset === "QQQ" ? B.qqq : B.btc, eR = nl.assets[rp.asset].equity, eB = bh.assets[rp.asset].equity;
    const [rn] = regime(B.index_pit[i]);
    let h = `<tr><th colspan="2" style="text-align:left">${d} 当天仪表盘读数</th></tr>`;
    h += `<tr><td>综合指数(当时可见)</td><td>${fmt(B.index_pit[i])}(${rn})</td></tr>`;
    if (k >= 0) pk.forEach(p => h += `<tr><td>${D.pillars[p].name}</td><td class="${D.pillars[p].z[k] >= 0 ? "pos" : "neg"}">${fmt(D.pillars[p].z[k], 2)}</td></tr>`);
    h += `<tr><td>净流动性 13 周变化</td><td class="${B.net_liq_chg[i] >= 0 ? "pos" : "neg"}">${fmt(B.net_liq_chg[i], 0)} 十亿</td></tr>`;
    h += `<tr><td><b>建议仓位(${nl.name})</b></td><td><b>${Math.round(nl.position[i] * 100)}%</b></td></tr>`;
    let st = i; while (st > 0 && nl.position[st - 1] === nl.position[i]) st--;
    let en = i; while (en < last && nl.position[en + 1] === nl.position[i]) en++;
    h += `<tr><td>这个仓位从哪天持续到哪天</td><td>${B.dates[st]} → ${en === last ? "至今" : B.dates[en]}(${en - st + 1} 个交易日)</td></tr>`;
    document.getElementById("rpSnap").innerHTML = h;
    const fw = n => { const j = Math.min(last, i + n); return j > i ? [eR[j] / eR[i] - 1, eB[j] / eB[i] - 1] : [null, null]; };
    let a = `<tr><th style="text-align:left">从 ${d} 起</th><th>按指引</th><th>一直满仓</th><th>差</th></tr>`;
    [["之后 1 个月",21],["之后 3 个月",63],["之后 1 年",252],["一直到今天", last - i]].forEach(([t, n]) => {
      const [r, b] = fw(n); a += `<tr><td>${t}</td><td>${pc(r)}</td><td>${pc(b)}</td><td class="${r == null ? "" : r - b >= 0 ? "pos" : "neg"}">${r == null ? "—" : pc(r - b)}</td></tr>`; });
    a += `<tr><td colspan="4" class="sub" style="text-align:left">${rp.asset} 价格 ${fmt(px[i], 0)}。“差”为正说明按指引操作比一直满仓好。</td></tr>`;
    document.getElementById("rpAfter").innerHTML = a;
    drawReplay();
  }
  document.getElementById("rpDate").onchange = e => { rp.date = e.target.value; snap(); };
  document.querySelectorAll("button[data-rp]").forEach(b => b.onclick = () => { rp.date = b.dataset.rp; snap(); });
  snap();
  document.getElementById("cReplay").on("plotly_click", ev => { if (ev.points && ev.points[0]){ rp.date = String(ev.points[0].x).slice(0, 10); snap(); } });
}



// 首页操作指引
if (!D.bt || !D.bt.portfolio || !D.bt.portfolio.guide){ document.getElementById("guideCard").style.display = "none"; }
else {
  const G = D.bt.portfolio.guide, names = Object.keys(G.alloc), n = G.dates.length;
  const AC = {"比特币": "--p2", "美股 VTI": "--p1", "海外股 VXUS": "--p5", "美债 IEF": "--p4", "现金 SGOV": "--neutral"};
  const col = k => css(AC[k] || "--muted");
  const show = i => {
    const full = G.sig[i] >= 1, loose = G.sig[i] >= 2, sm = G.sm[i];
    document.getElementById("gdDate").textContent = i === n - 1 ? `今天的建议(数据到 ${G.dates[i]})` : `历史上 ${G.dates[i]} 的建议`;
    const st = document.getElementById("gdStat");
    st.textContent = loose ? "宽松加码(比特币和股票加仓)" : full ? "正常持有" : "风险资产减半";
    st.style.background = loose ? css("--p4") : full ? css("--loose") : css("--tight");
    document.getElementById("gdBar").innerHTML = names.map(k => { const v = G.alloc[k][i]; return v > 0.005 ? `<div style="width:${v*100}%;background:${col(k)}">${v >= 0.08 ? (v*100).toFixed(0) + "%" : ""}</div>` : ""; }).join("");
    document.getElementById("gdLab").innerHTML = names.map(k => `<span><i style="background:${col(k)}"></i>${k} <b>${(G.alloc[k][i]*100).toFixed(0)}%</b></span>`).join("");
    const smt = sm == null ? "—" : sm.toFixed(1);
    let run = 0; for (let j = i; j >= 0 && G.sm[j] != null && (loose ? G.sm[j] < 60 : full ? G.sm[j] < 35 : G.sm[j] > 50); j--) run++;
    let up = 0; if (full && !loose) for (let j = i; j >= 0 && G.sm[j] != null && G.sm[j] > 70; j--) up++;
    const D_ = G.mode === "D";
    let ev = null; if (D_) for (const [d, nm] of G.qe_events){ const k = G.dates.findIndex(x => x > d); if (k >= 0 && k <= i && i - k < G.qe_hold) ev = [d, nm, G.qe_hold - (i - k)]; }
    document.getElementById("gdWhy").textContent = D_ && loose && ev
      ? `美联储 ${ev[0]} ${ev[1]} → 第二天起宽松加码,持有约一年(还剩 ${ev[2]} 个交易日)。期间综合指数 20 日均值若连续 20 天低于 35,直接减半。`
      : D_ && !loose && full ? `流动性综合指数 20 日均值 ${smt}。美联储一旦宣布 QE/大规模购债,第二天就加码;连续 20 天低于 35 则减半。`
      : D_ && !full ? `流动性综合指数 20 日均值 ${smt},之前连续 20 天低于 35 → 减半(减半优先于加码)。` + (run > 0 ? `已连续 ${run} 天高于 50,再 ${Math.max(20 - run, 0)} 天恢复。` : `要连续 20 天高于 50 才恢复。`) + (ev ? `美联储 ${ev[0]} 已宣布${ev[1]},恢复后直接进入加码。` : "")
      : loose
      ? `流动性综合指数 20 日均值 ${smt},之前连续 20 天高于 70 → 宽松加码。` + (run > 0 ? `已连续 ${run} 天低于 60,再 ${Math.max(20 - run, 0)} 天就退回正常;` : `跌回 60 以下连续 20 天才退回正常;`) + `连续 20 天低于 35 则直接减半。`
      : full && up > 0 ? `流动性综合指数 20 日均值 ${smt},已连续 ${up} 天高于 70,再 ${Math.max(20 - up, 0)} 天就进入宽松加码。`
      : full
      ? `流动性综合指数 20 日均值 ${smt}。` + (run > 0 ? `已连续 ${run} 天低于 35,再 ${Math.max(20 - run, 0)} 天仍低于 35 就减仓。` : `连续 20 天低于 35 才会减仓。`)
      : `流动性综合指数 20 日均值 ${smt},之前连续 20 天低于 35 → 减半。` + (run > 0 ? `已连续 ${run} 天高于 50,再 ${Math.max(20 - run, 0)} 天就恢复满仓。` : `要连续 20 天高于 50 才恢复满仓。`);
    const t = G.eq_timed[i], h = G.eq_hold[i];
    document.getElementById("gdEq").innerHTML = t == null ? `比特币 ETF 上市(2024-01-11)前:<br>这里只显示规则当时会给出的比例,不算收益`
      : `2024-01-11 投入 1 元,到这天<br>按指引 <b>${t.toFixed(2)}</b> 元 · 一直持有 <b>${h.toFixed(2)}</b> 元`;
  };
  const zone = (a, b, c) => ({type:"rect", xref:"paper", yref:"y", x0:0, x1:1, y0:a, y1:b, fillcolor: css(c), line:{width:0}, layer:"below"});
  const lowB = []; let s0 = null;
  G.sig.forEach((p, i) => { if (p < 1 && s0 == null) s0 = i; if ((p >= 1 || i === n - 1) && s0 != null){ lowB.push({type:"rect", xref:"x", yref:"paper", x0:G.dates[s0], x1:G.dates[i], y0:0, y1:1, fillcolor: css("--tight"), opacity:0.07, line:{width:0}, layer:"below"}); s0 = null; } });
  const hiB = []; let h0 = null;
  G.sig.forEach((p, i) => { if (p >= 2 && h0 == null) h0 = i; if ((p < 2 || i === n - 1) && h0 != null){ hiB.push({type:"rect", xref:"x", yref:"paper", x0:G.dates[h0], x1:G.dates[i], y0:0, y1:1, fillcolor: css("--p4"), opacity:0.08, line:{width:0}, layer:"below"}); h0 = null; } });
  const tr = [{x: G.dates, y: G.sm, type:"scatter", mode:"lines", line:{color: css("--ink"), width:1.8}, name:"综合指数 20 日均值", hovertemplate:"指数均值 %{y:.1f}<extra></extra>"}];
  names.forEach(k => tr.push({x: G.dates, y: G.alloc[k], type:"scatter", mode:"lines", stackgroup:"a", line:{width:0.5, color: col(k)}, fillcolor: col(k), name:k, xaxis:"x", yaxis:"y2", hovertemplate: k + " %{y:.0%}<extra></extra>"}));
  Plotly.newPlot("cGuide", tr, baseLayout({grid:{rows:2, columns:1, roworder:"top to bottom"}, shapes:[zone(0,35,"--band-tight"), zone(50,100,"--band-loose"), zone(70,100,"--band-loose"), ...lowB, ...hiB],
    yaxis:{domain:[0.62,1], range:[5,95], gridcolor: css("--grid"), title:{text:"指数"}}, yaxis2:{domain:[0,0.56], range:[0,1], tickformat:".0%", gridcolor: css("--grid"), title:{text:"建议比例"}},
    xaxis:{type:"date", gridcolor: css("--grid"), anchor:"y2"}, showlegend:false, margin:{l:56, r:16, t:6, b:30}}), cfg);
  show(n - 1);
  const gr = k => { Plotly.relayout("cGuide", {"xaxis.range": [k === "all" ? G.dates[0] : G.btc_start, G.dates[n - 1]]});
    document.querySelectorAll("[data-gr]").forEach(b => b.classList.toggle("on", b.dataset.gr === k)); };
  document.querySelectorAll("[data-gr]").forEach(b => b.onclick = () => gr(b.dataset.gr));
  gr("etf");
  const el = document.getElementById("cGuide");
  el.on("plotly_hover", ev => { const x = String(ev.points[0].x).slice(0, 10); const i = G.dates.indexOf(x); if (i >= 0) show(i); });
  el.on("plotly_unhover", () => show(n - 1));
}

// 综合组合
if (!D.bt || !D.bt.portfolio){ document.getElementById("pfCard").style.display = "none"; }
else {
  const P = D.bt.portfolio, E = P.btc_era, pc = (v, s=true) => v == null ? "—" : (s && v > 0 ? "+" : "") + (v*100).toFixed(1) + "%";
  const wtxt = o => Object.entries(o).map(([k, v]) => `${k} ${(v*100).toFixed(0)}%`).join("、");
  document.getElementById("pfRule").textContent = `目标比例:${wtxt(P.weights)}。规则(事先定好):综合指数 20 日均值低于 35 连续 20 天 → 比特币和股票各减一半,腾出的钱放现金;高于 50 连续 20 天 → 恢复。美债不动。宽松加码 A:高于 70 连续 20 天 → ${wtxt(P.loose_weights)},跌回 60 以下连续 20 天退出。加码 B:联储总资产 13 周增长超过 5% 连续 10 天 → 同样加码,扩表停止(13 周变化转负)连续 20 天退出。信号次日执行,调仓扣 0.05%。③、⑤ 只作对照,不按结果挑。`;
  const T = P.today;
  document.getElementById("pfToday").innerHTML = `<b>今天(${T.date})主方案:${T.signal >= 2 ? "宽松加码" : T.signal >= 1 ? "正常持有" : "风险资产减半"}</b>。对应比例:${wtxt(T.alloc)}。`;
  const cols = [css("--muted"), css("--p1"), css("--p2"), css("--ink")];
  Plotly.newPlot("cPf", Object.entries(E.equity).map(([n, y], i) => ({x: E.dates, y, type:"scatter", mode:"lines", name:n,
      line:{color: cols[i % 4], width: i === 1 ? 2.4 : 1.4, dash: i === 0 ? "dot" : "solid"}, hovertemplate: n + " %{y:.2f}<extra></extra>"})),
    baseLayout({showlegend:true, legend:{orientation:"h", y:-0.15}, margin:{l:56, r:16, t:10, b:60}, yaxis:{gridcolor: css("--grid"), title:{text:"资金(起点 = 1)"}}}), cfg);
  const tbl = (id, h, rows) => document.getElementById(id).innerHTML = "<tr>" + h.map(x => `<th>${x}</th>`).join("") + "</tr>" + rows.map(r => "<tr>" + r.map(x => `<td>${x}</td>`).join("") + "</tr>").join("");
  tbl("pfTbl", ["方案", "总收益", "年化", "最大回撤", "波动", "夏普", "调仓"], E.rows.map(r => [r[0], pc(r[1]), pc(r[2], false), pc(r[3]), pc(r[4], false), r[5] == null ? "—" : r[5].toFixed(2), r[6]]));
  tbl("pfSingle", ["对照:单一资产一直持有", "总收益", "年化", "最大回撤", "夏普"], E.single.map(r => [r[0], pc(r[1]), pc(r[2], false), pc(r[3]), r[4] == null ? "—" : r[4].toFixed(2)]));
  const L = P.long;
  document.getElementById("pfLongW").textContent = `把比特币那份按比例分给其他资产:${wtxt(L.weights)}。`;
  tbl("pfLong", ["方案", ...L.periods.flatMap(x => [`${x} 年化`, "回撤", "夏普"])], L.rows.map(r => [r[0], ...L.periods.flatMap((_, i) => [pc(r[1+3*i], false), pc(r[2+3*i]), r[3+3*i] == null ? "—" : r[3+3*i].toFixed(2)])]));
  if (P.loose_segments) tbl("pfLoose", [`宽松加码 ${(P.guide && P.guide.mode) || "A"} 开始`, "结束", "天数", "期间美股 VTI", "期间 VTI 最大回撤"], P.loose_segments.map(r => [r[0], r[1], r[2], pc(r[3]), pc(r[4])]));
  if (P.ref_2020) tbl("pfRef", ["参考:2020–2023(ETF 前比特币)", "总收益", "年化", "最大回撤", "夏普"], P.ref_2020.map(r => [r[0], pc(r[1]), pc(r[2], false), pc(r[3]), r[4] == null ? "—" : r[4].toFixed(2)]));
  const br = [];
  [["2024 起(含 BTC)", E.boot], ["2004 起(股债)", L.boot]].forEach(([lab, b]) => {
    Object.entries(b).filter(([k]) => !k.startsWith("_")).forEach(([k, v]) => br.push([lab, k, pc(v.p10), pc(v.p50), pc(v.p90), pc(v.dd50), (v.loss20*100).toFixed(0) + "%"]));
    if (b._beat) Object.entries(b._beat).forEach(([k, v]) => br.push([lab, `${k} 一年后胜过一直持有的概率`, (v*100).toFixed(0) + "%", "", "", "", ""]));
  });
  tbl("pfBoot", ["样本", "方案", "差(10%)", "中间(50%)", "好(90%)", "一年内回撤中位数", "亏超 20% 概率"], br);
}

// 比特币
if (!D.bt || !D.bt.btc){ document.getElementById("btcCard").style.display = "none"; }
else {
  const B = D.bt, i0 = B.dates.findIndex(d => d >= B.btc_start), nlR = B.rules.find(r => r.key === "nl");
  const ds = B.dates.slice(i0), sl = a => a.slice(i0);
  const lowB = []; let s0 = null; const ps = sl(nlR.position);
  ps.forEach((p, i) => { if (p < 1 && s0 == null) s0 = i; if ((p >= 1 || i === ps.length - 1) && s0 != null){ lowB.push({type:"rect", xref:"x", yref:"paper", x0:ds[s0], x1:ds[i], y0:0, y1:1, fillcolor: css("--tight"), opacity:0.08, line:{width:0}, layer:"below"}); s0 = null; } });
  Plotly.newPlot("cBtc", [
    {x: ds, y: sl(B.net_liq_chg), type:"bar", marker:{color: sl(B.net_liq_chg).map(v => v == null ? css("--neutral") : v >= 0 ? css("--loose") : css("--tight"))}, opacity:0.6, name:"净流动性 13 周变化", hovertemplate:"%{y:+,.0f} 十亿<extra></extra>"},
    {x: ds, y: sl(B.btc), type:"scatter", mode:"lines", line:{color: css("--accent"), width:2}, yaxis:"y2", name:"比特币", hovertemplate:"BTC %{y:,.0f}<extra></extra>"}],
    baseLayout({shapes: lowB, bargap:0, yaxis:{gridcolor: css("--grid"), zerolinecolor: css("--muted"), title:{text:"十亿美元"}},
      yaxis2:{overlaying:"y", side:"right", type:"log", showgrid:false, title:{text:"BTC"}}, margin:{l:56, r:56, t:10, b:36}}), cfg);
  const eqT = [["bh","一直满仓", css("--muted"), "dot"], ["nl","③ 净流动性", css("--p1"), "solid"], ["lvl","① 综合指数水平", css("--p2"), "solid"], ["combo","⑤ ①或③任一减仓", css("--ink"), "solid"]]
    .map(([k, n, c, dsh]) => { const r = B.rules.find(x => x.key === k); if (!r || !r.assets || !r.assets.BTC) return null;
      const e = sl(r.assets.BTC.equity), b0 = e.find(v => v != null); return {x: ds, y: e.map(v => v == null ? null : v / b0), type:"scatter", mode:"lines", line:{color:c, width: k === "bh" ? 1.5 : 2, dash:dsh}, name:n, hovertemplate: n + " %{y:.2f}<extra></extra>"}; }).filter(Boolean);
  Plotly.newPlot("cBtcEq", eqT, baseLayout({yaxis:{gridcolor: css("--grid"), title:{text:"资金(起点 = 1)"}}, showlegend:true, legend:{orientation:"h", y:-0.15}, margin:{l:56, r:16, t:10, b:60}}), cfg);
  const sec = D.research && D.research.sections.find(s => s.btc_qqq_corr60);
  if (sec && sec.detail){
    const h = sec.detail.headers, rows = sec.detail.rows.filter(r => r[0] === "BTC" || r[0] === "MSTR");
    document.getElementById("btcPerf").innerHTML = "<tr>" + h.map(x => `<th>${x}</th>`).join("") + "</tr>" + rows.map(r => "<tr>" + r.map(x => `<td>${x}</td>`).join("") + "</tr>").join("");
  }
  if (sec){
    const C = sec.btc_qqq_corr60;
    Plotly.newPlot("cBtcCorr", [{x: C.dates, y: C.v, type:"scatter", mode:"lines", line:{color: css("--p2"), width:1.5}, hovertemplate:"%{y:.2f}<extra></extra>"}],
      baseLayout({shapes:[{type:"line", xref:"x", yref:"paper", x0:B.btc_start, x1:B.btc_start, y0:0, y1:1, line:{color: css("--muted"), dash:"dot", width:1}}],
        yaxis:{range:[-0.6,1], gridcolor: css("--grid"), zerolinecolor: css("--muted")}}), cfg);
    if (sec.verdict) document.getElementById("btcVerdict").innerHTML = `<div class="verdict">${sec.verdict}</div><div class="note">完整数字见下面“新指标历史检验”第 6 节。</div>`;
  } else document.getElementById("cBtcCorr").style.display = "none";
}

// 管道与砍仓观察
const W = D.watch;
if (!W){ document.getElementById("watchCard").style.display = "none"; }
else {
  const P = W.pipe;
  document.getElementById("wPipe").innerHTML = `<h3>回购管道 SOFR − IORB</h3>` +
    `<div class="v">${P.bp == null ? "—" : fmt(P.bp, 0) + " BP"}</div>` +
    `<div style="margin:6px 0"><span class="badge" style="background:${css(P.color)}">${P.tier}</span></div>` +
    `<div class="sub">数据日期 ${P.date || "—"} · 已连续 ${P.streak} 天高于 IORB</div>` +
    `<div class="sub">SOFR99 − IORB ${fmt(P.sofr99, 0)} BP · SRF 近 5 日最高 ${fmt(P.srf5, 1)} 十亿</div>` +
    `<div class="sub">分档:≤0 宽松 / 0–3 正常 / 3–5 偏紧 / 5–10 紧张 / ≥10 瘫痪</div>`;
  const RB = W.rebound;
  document.getElementById("wReb").innerHTML = `<h3>反抽体检(三条都满足才算“没恶化”)</h3>` +
    `<div class="v">${RB.passed} / ${RB.total}</div>` +
    RB.conds.map(c => `<div class="chk"><b class="${c.ok == null ? "" : c.ok ? "pos" : "neg"}">${c.ok == null ? "—" : c.ok ? "✓" : "✗"}</b> ${c.name}<br><span class="sub">${c.value} · ${c.date || "无数据"}</span></div>`).join("");
  const X = W.cross;
  document.getElementById("wCross").innerHTML = X
    ? `<h3>跨资产砍仓检测(${X.date})</h3>` +
      `<div class="v ${X.alarm ? "neg" : ""}">${X.quadrant}</div>` +
      `<div class="sub" style="margin:4px 0">${X.quadrant_note}</div>` +
      `<div>近 5 日 QQQ <b class="${X.qqq5 >= 0 ? "pos" : "neg"}">${(X.qqq5*100).toFixed(1)}%</b> · GLD <b class="${X.gld5 >= 0 ? "pos" : "neg"}">${(X.gld5*100).toFixed(1)}%</b></div>` +
      `<div>20 日相关 <b>${fmt(X.corr20, 2)}</b> · 近 20 日同日双跌 <b>${X.dd20}</b> 天</div>` +
      (X.tips10_5d_bp == null ? "" : `<div class="sub">10 年 TIPS 5 日变化 ${X.tips10_5d_bp > 0 ? "+" : ""}${X.tips10_5d_bp} BP${X.gold_rate_diverge ? ",<b class='neg'>黄金逆着实际利率下跌,疑似被砍仓</b>" : ""}</div>`) +
      (X.alarm ? `<div class="neg" style="margin-top:4px"><b>砍仓警报</b></div>` : "")
    : `<h3>跨资产砍仓检测</h3><div class="sub">缺少 GLD 价格数据</div>`;
  const vl = W.vix_latest, mg = W.margin;
  document.getElementById("wNote").innerHTML =
    (vl ? `最新 VIX ${fmt(vl.value, 2)}(${vl.date},FRED 滞后时用 Yahoo 当日收盘补)。` : "") +
    (mg ? ` 券商融资余额(FINRA,月度):${mg.month} 为 ${fmt(mg.value_bn, 0)} 十亿美元,同比 ${(mg.yoy*100).toFixed(1)}%,距 ${mg.peak_month} 峰值 ${(mg.from_peak*100).toFixed(1)}%。` : " 券商融资余额暂未获取。") +
    " 这些指标是和外部分析对比后加入的,历史检验见下面“新指标历史检验”。";
  const tb = [[-100,0,"--band-loose"],[0,3,"--band-neutral"],[3,5,"--band-semitight"],[5,10,"--band-tight"],[10,1000,"--band-tight"]]
    .map(([a,b,c]) => ({type:"rect", xref:"paper", x0:0, x1:1, y0:a, y1:b, fillcolor: css(c), line:{width:0}, layer:"below"}));
  const ys = P.hist.bp, ylo = Math.min(-5, ...ys) - 2, yhi = Math.max(12, ...ys) + 2;
  Plotly.newPlot("cPipe", [{x: P.hist.dates, y: ys, type:"scatter", mode:"lines", line:{color: css("--accent"), width:1.6, shape:"hv"},
    hovertemplate: "%{y:.0f} BP<extra></extra>"}], baseLayout({shapes: tb, yaxis:{range:[ylo, yhi], gridcolor: css("--grid"), zerolinecolor: css("--muted")}}), cfg);
  if (X) Plotly.newPlot("cCorr", [{x: X.hist.dates, y: X.hist.corr20, type:"scatter", mode:"lines", line:{color: css("--p2"), width:1.6},
    hovertemplate: "%{y:.2f}<extra></extra>"}], baseLayout({shapes:[{type:"line", xref:"paper", x0:0, x1:1, y0:0.5, y1:0.5, line:{color: css("--tight"), dash:"dash", width:1}}],
    yaxis:{range:[-1,1], gridcolor: css("--grid"), zerolinecolor: css("--muted")}}), cfg);
  else document.getElementById("cCorr").style.display = "none";
}

// 新指标历史检验
const RS = D.research;
if (!RS){ document.getElementById("resCard").style.display = "none"; }
else {
  document.getElementById("resSub").textContent = `数据截至 ${RS.asof}。只做研究,不进仓位规则;样本小的结论只能当参考。完整版见仓库 output/research_summary.md。`;
  const tbl = t => `<div class="scroll"><table><tr>${t.headers.map(h => `<th>${h}</th>`).join("")}</tr>` +
    t.rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join("")}</tr>`).join("") + `</table></div>`;
  document.getElementById("resBody").innerHTML = RS.sections.map(s =>
    `<h2 style="margin-top:16px;font-size:15px">${s.title}</h2>` +
    (s.verdict ? `<div class="verdict">${s.verdict}</div>` : "") +
    `<div class="note" style="margin-top:0">${s.note}</div>` + tbl(s) +
    (s.summary ? `<div class="sub" style="margin-top:8px">汇总</div>` + tbl(s.summary) : "") +
    (s.diverge ? `<div class="note">黄金在 10 年 TIPS 实际利率 5 日下降时 5 日跌超 2%:共 ${s.diverge.days} 天(${s.diverge.episodes} 段),之后 GLD 20 日 ${s.diverge.gld20}、60 日 ${s.diverge.gld60}(所有日子平均 ${s.diverge.base20} / ${s.diverge.base60})。</div>` : "") +
    (s.detail ? `<details${s.detail_title ? " open" : ""}><summary>${s.detail_title || "最近 15 次大涨明细"}</summary>${tbl(s.detail)}</details>` : "")
  ).join("");
}

// 两天对比
const dA = document.getElementById("dA"), dB = document.getElementById("dB");
dA.min = dB.min = D.dates[0]; dA.max = dB.max = D.dates[N-1];
dA.value = D.dates[li]; dB.value = D.dates[Math.max(0, li - 252)];
function cmp(){
  const a = idxOnOrBefore(dA.value), b = idxOnOrBefore(dB.value);
  if (a < 0 || b < 0){ document.getElementById("cmp").innerHTML = "<tr><td>所选日期早于数据起点</td></tr>"; return; }
  const row = (name, va, vb, nd, unit = "") => {
    const d = va == null || vb == null ? null : va - vb;
    const cls = d == null ? "" : "";
    return `<tr><td>${name}</td><td>${fmt(va, nd)}${unit}</td><td>${fmt(vb, nd)}${unit}</td><td>${d == null ? "—" : (d >= 0 ? "+" : "") + fmt(d, nd)}</td></tr>`;
  };
  let h = `<tr><th>项目</th><th>A:${D.dates[a]}</th><th>B:${D.dates[b]}</th><th>A − B</th></tr>`;
  h += row("<b>综合流动性指数</b>", D.index[a], D.index[b], 1);
  pk.forEach(p => h += row(D.pillars[p].name + "(z)", D.pillars[p].z[a], D.pillars[p].z[b], 2));
  Object.values(D.raw).forEach(s => h += row(s.name, s.v[a], s.v[b], 2, s.unit && s.unit !== "点" ? " " + s.unit : ""));
  document.getElementById("cmp").innerHTML = h;
}
dA.onchange = dB.onchange = cmp;
document.querySelectorAll("button[data-back]").forEach(b => b.onclick = () => {
  const a = idxOnOrBefore(dA.value); dB.value = D.dates[Math.max(0, a - +b.dataset.back)]; cmp();
});
cmp();

// 贡献
document.getElementById("contrib").innerHTML = `<tr><th>支柱 / 序列</th><th>z</th><th style="width:40%"></th></tr>` +
  D.contrib.map(c => {
    const z = c.z, w = z == null ? 0 : Math.min(Math.abs(z), 3) / 3 * 50;
    const bar = z == null ? "" : `<span style="display:inline-block;width:50%;text-align:right">${z < 0 ? `<span class="bar" style="width:${w*2}%;background:${css("--tight")}"></span>` : ""}</span><span style="display:inline-block;width:50%;text-align:left">${z > 0 ? `<span class="bar" style="width:${w*2}%;background:${css("--loose")}"></span>` : ""}</span>`;
    return `<tr><td><span class="sub">${c.pillar}</span><br>${c.name}</td><td class="${z == null ? "" : z >= 0 ? "pos" : "neg"}">${fmt(z, 2)}</td><td>${bar}</td></tr>`;
  }).join("");

// 运行记录
document.getElementById("log").innerHTML = `<tr><th>运行日</th><th>数据日</th><th>指数</th><th>状态</th></tr>` +
  [...D.log].reverse().slice(0, 30).map(r => `<tr><td>${r.run_date}</td><td>${r.data_date}</td><td>${fmt(r.liquidity_index)}</td><td>${r.regime}</td></tr>`).join("");

// 结算日历
document.getElementById("cal").innerHTML = D.calendar.length
  ? `<tr><th>日期</th><th>结算毛额</th><th>到期</th><th>净额</th><th>事件 / 品种</th></tr>` + D.calendar.map(c =>
      `<tr><td>${c.date}</td><td>${c.pending ? "待公布" : fmt(c.gross_bn)}</td><td>${fmt(c.mature_bn)}</td>` +
      `<td class="${c.net_bn == null ? "" : c.net_bn > 0 ? "neg" : "pos"}">${fmt(c.net_bn)}</td>` +
      `<td style="text-align:left">${[...c.events.map(e => `<b>${e}</b>`), c.detail].filter(Boolean).join(" · ")}</td></tr>`).join("")
  : `<tr><td class="sub">暂无数据</td></tr>`;
document.getElementById("calNote").textContent = "金额单位十亿美元。净额 = 结算毛额 − 到期(约数),正数(红)表示从市场抽走现金。";
</script>
</body>
</html>
"""
