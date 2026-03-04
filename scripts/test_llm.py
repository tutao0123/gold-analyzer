import os
import sys

# 添加根目录方便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm_analyzer import LLMAnalyzer

def test_search():
    print("=== 测试联网增强功能 ===")
    analyzer = LLMAnalyzer()
    
    # 构造假数据测 search
    class MockResult:
        current_price = 2300.5
        price_change_1d = 0.5
        price_change_7d = 1.0
        price_change_30d = 2.0
        trend = type('obj', (object,), {'direction': 'up', 'strength': 0.8, 'description': '上涨'})()
        volatility = type('obj', (object,), {'volatility_level': 'high', 'annualized_volatility': 0.15, 'daily_volatility': 0.01})()
        rsi = 65.5
        ma_analysis = {'MA5': 2250, 'MA10': 2240}
        support_resistance = type('obj', (object,), {'support_levels': [2200], 'resistance_levels': [2350]})()
        recommendation = "买入"
    
    mock = MockResult()
    result = analyzer.analyze_with_search(mock)
    print(result.detailed_analysis[:500] + "...\n")

def test_doc_understanding():
    print("=== 测试长文档理解功能 ===")
    analyzer = LLMAnalyzer()
    
    # 建一个假的研报txt做测试
    test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_report.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("据最新华尔街日报称，美联储将在本月降低利率25个基点。黄金应声大涨突破2380美元/盎司，市场分析师认为牛市并未结束。")
    
    res = analyzer.analyze_document(test_file, query="根据这份文本，黄金还会涨吗？")
    print("文档分析结果:\n", res)

if __name__ == "__main__":
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("需要设置 DASHSCOPE_API_KEY 环境变量来进行真实测试。")
        sys.exit(1)
        
    test_search()
    test_doc_understanding()
