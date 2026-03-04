#!/usr/bin/env python3
"""
黄金价格分析 Agent - 主入口
Usage: python main.py [--update] [--report]
"""

import argparse
import logging
import sys
from datetime import datetime

from core.config import LOG_FILE, LOG_FORMAT, LOG_LEVEL, REPORT_DIR
from core.data_fetcher import GoldDataFetcher
from core.analyzer import GoldPriceAnalyzer
from core.report_generator import ReportGenerator
from core.llm_analyzer import analyze_with_llm, LLMAnalysisResult

# 配置日志
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='黄金价格分析 Agent')
    parser.add_argument('--update', '-u', action='store_true', help='更新数据')
    parser.add_argument('--report', '-r', action='store_true', help='生成报表')
    parser.add_argument('--auto', '-a', action='store_true', help='自动模式：更新+报表')
    args = parser.parse_args()

    # 默认执行全部
    if not any([args.update, args.report, args.auto]):
        args.auto = True

    try:
        # 1. 获取/更新数据
        if args.update or args.auto:
            logger.info("=" * 50)
            logger.info("开始更新金价数据...")
            fetcher = GoldDataFetcher()
            data = fetcher.update_data()
            logger.info(f"数据更新完成，共 {len(data)} 条记录")
        else:
            # 加载已有数据
            fetcher = GoldDataFetcher()
            data = fetcher.load_data()
            if not data:
                logger.warning("没有历史数据，先执行更新...")
                data = fetcher.update_data()

        # 2. 分析数据
        logger.info("=" * 50)
        logger.info("开始分析数据...")
        analyzer = GoldPriceAnalyzer(data)
        result = analyzer.analyze()
        logger.info("技术分析完成")

        # 3. LLM 智能分析
        logger.info("=" * 50)
        logger.info("开始 LLM 智能分析...")
        llm_result = analyze_with_llm(result)
        logger.info("LLM 智能分析完成")

        # 4. 生成报表
        if args.report or args.auto:
            logger.info("=" * 50)
            logger.info("生成分析报表...")
            generator = ReportGenerator()
            report_path = generator.generate(result, data, llm_result)
            logger.info(f"报表已保存: {report_path}")

        logger.info("=" * 50)
        logger.info("任务完成！")
        
        # 打印关键信息
        print(f"\n📊 当前金价: ${result.current_price:.2f}")
        print(f"📈 30日涨跌: {result.price_change_30d:+.2f}%")
        print(f"📉 趋势: {result.trend.description}")
        print(f"💡 建议: {result.recommendation}")
        
        return 0

    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
