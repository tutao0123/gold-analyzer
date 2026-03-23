# 📅 黄金期货定时分析任务

## 定时任务时间表（北京时间 UTC+8）

| 任务 | 时间 | 说明 |
|------|------|------|
| **🌅 盘前盘点** | 08:30 | 隔夜市场回顾、亚市开盘分析、当日关键事件 |
| **☀️ 午间复盘** | 12:30 | 上午交易总结、美市开盘前策略调整 |
| **🌆 美市盘前预测** | 20:30 | 美市开盘前分析、最佳交易时机预测 |
| **🌙 盘后总结** | 23:30 | 全天交易复盘、持仓建议、明日展望 |

---

## 📋 安装步骤

### 1. 运行安装脚本

```bash
cd /home/gold_analyzer
./scripts/install_cron.sh
```

安装脚本会：
- 检查虚拟环境和脚本
- 创建日志目录
- 添加 cron 任务
- 显示确认提示

### 2. 验证安装

```bash
# 查看已安装的 cron 任务
crontab -l

# 查看日志目录
ls -la /home/gold_analyzer/logs/
```

---

## 🔧 手动运行

也可以手动运行特定分析：

```bash
cd /home/gold_analyzer

# 盘前盘点
.venv/bin/python scripts/scheduled_analysis.py pre_market

# 午间复盘
.venv/bin/python scripts/scheduled_analysis.py midday

# 美市盘前预测
.venv/bin/python scripts/scheduled_analysis.py pre_us

# 盘后总结
.venv/bin/python scripts/scheduled_analysis.py post_market
```

---

## 📊 查看报告

报告保存在 `reports/` 目录：

```bash
# 查看最新报告
ls -lt reports/ | head

# 查看今日所有报告
ls reports/*_$(date +%Y%m%d)*.txt

# 查看特定报告
cat reports/pre_market_20260310_083000.txt
```

---

## 📝 查看日志

```bash
# 实时查看盘前分析日志
tail -f logs/pre_market.log

# 查看最近的错误
grep -i error logs/*.log | tail -20

# 清空所有日志
> logs/pre_market.log && > logs/midday.log && > logs/pre_us.log && > logs/post_market.log
```

---

## 🛑 卸载任务

```bash
# 编辑 crontab
crontab -e

# 删除所有包含 gold_analyzer 的行
# 或者清空整个 crontab（谨慎！）
crontab -r
```

---

## ⚠️ 注意事项

1. **交易日运行** - 任务仅在周一至周五运行（排除周末）
2. **节假日** - cron 不会自动排除节假日，需要手动管理
3. **API Key** - 确保 `.env` 文件中的 `DASHSCOPE_API_KEY` 有效
4. **日志轮转** - 建议定期清理日志文件，避免占用过多磁盘空间
5. **假期休市** - 美国假期期间（如感恩节、圣诞节），市场休市，报告可能无意义

---

## 📈 报告示例

每份报告包含：

```
黄金期货分析报告
生成时间：2026 年 03 月 10 日 20:49:37
============================================================

分析问题：[具体问题]

────────────────────────────────────────────────────────────
各领域专家报告
────────────────────────────────────────────────────────────

【技术面研究员】
[技术分析内容]

【宏观基本面分析师】
[宏观分析内容]

【量化工程师】
[量化指标]

【跨市场联动分析师】
[跨市场分析]

【情绪分析师】
[市场情绪分析]

============================================================
【首席策略官 最终决议】
[交易决策]

────────────────────────────────────────────────────────────
【风控合规官 审查结论】
[风险审查意见]
```

---

## 🎯 定制分析

如需修改分析问题或添加新的分析模式，编辑 `scripts/scheduled_analysis.py` 中的 `ANALYSIS_MODES` 配置。

---

*最后更新：2026 年 3 月 10 日*
