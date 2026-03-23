#!/bin/bash
# 黄金期货定时分析任务安装脚本

set -e

PROJECT_DIR="/home/gold_analyzer"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
SCRIPT="$PROJECT_DIR/scripts/scheduled_analysis.py"

echo "======================================"
echo "  黄金期货定时分析任务安装"
echo "======================================"

# 检查虚拟环境
if [ ! -f "$VENV_PYTHON" ]; then
    echo "错误：虚拟环境未找到：$VENV_PYTHON"
    exit 1
fi

# 检查脚本
if [ ! -f "$SCRIPT" ]; then
    echo "错误：分析脚本未找到：$SCRIPT"
    exit 1
fi

# 创建日志目录
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# 创建 cron 任务
CRON_FILE="/tmp/gold_analyzer_cron_$$.tmp"

cat > "$CRON_FILE" << EOF
# 黄金期货定时分析任务
# 盘前盘点 - 每个交易日 08:30 (北京时间)
30 0 * * 1-5 $VENV_PYTHON $SCRIPT pre_market >> $LOG_DIR/pre_market.log 2>&1

# 午间复盘 - 每个交易日 12:30 (北京时间)
30 4 * * 1-5 $VENV_PYTHON $SCRIPT midday >> $LOG_DIR/midday.log 2>&1

# 美市盘前预测 - 每个交易日 20:30 (北京时间)
30 12 * * 1-5 $VENV_PYTHON $SCRIPT pre_us >> $LOG_DIR/pre_us.log 2>&1

# 盘后总结 - 每个交易日 23:30 (北京时间)
30 15 * * 1-5 $VENV_PYTHON $SCRIPT post_market >> $LOG_DIR/post_market.log 2>&1
EOF

echo ""
echo "即将安装的 cron 任务："
echo "--------------------------------------"
cat "$CRON_FILE"
echo "--------------------------------------"
echo ""

# 确认安装
read -p "是否继续安装？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "安装已取消"
    rm -f "$CRON_FILE"
    exit 0
fi

# 安装 cron 任务
(crontab -l 2>/dev/null || true; cat "$CRON_FILE") | crontab -

echo ""
echo "✅ cron 任务安装成功！"
echo ""
echo "查看已安装的任务：crontab -l"
echo "查看日志：tail -f $LOG_DIR/pre_market.log"
echo ""
echo "======================================"

rm -f "$CRON_FILE"
