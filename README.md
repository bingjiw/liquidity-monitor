# 美元流动性综合指标

一个 Python 脚本,从 FRED、纽约联储、美国财政部的免费公开数据计算 0–100 的每日流动性指数(50 = 三年中性,越低越紧),外加硬预警和未来 30 天结算日历。不调用任何大模型。设计说明见项目文档《流动性跟踪指标:视频观点分析与设计方案》。

## 安装与运行

```bash
pip3 install pandas requests
python3 liquidity_monitor.py
```

第一次运行会下载 2018 年以来的历史数据(约 1 分钟),之后每次几秒钟。输出在 `output/`:

- `report_latest.md`:当日报告(综合指数、五大支柱、关键读数、各序列贡献、预警、结算日历)
- `liquidity_history.csv`:每日指数历史,可以导入 Excel 画图
- `raw_data.csv`:对齐后的原始数据
- `cache/`:下载缓存,某个数据源临时失败时会自动用缓存

## 每天自动运行

本仓库用 GitHub Actions(`.github/workflows/daily.yml`)在每个工作日 UTC 23:30(北京时间次日 7:30)自动运行,结果提交到 `output/`。最新报告见 [output/report_latest.md](output/report_latest.md)。在 Actions 页面点 Run workflow 可随时手动运行。

### 也可以在本机运行(Mac)

`crontab -e` 加一行(北京时间每个工作日早上 7:30,对应美东前一天傍晚,数据已更新):

```
30 7 * * 2-6 cd /path/to/liquidity && /usr/bin/python3 liquidity_monitor.py >> output/run.log 2>&1
```

## 调整

权重、序列、预警阈值都在脚本开头的配置区:`PILLAR_WEIGHTS`、`SCORED`、`ALERTS`。某个 FRED 序列下载失败时脚本会跳过它,并按剩余权重重新归一。

仅用于研究,不构成投资建议。
