import os
import sys

# 添加根目录方便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.engine import GoldMultimodalRAG

def test_multimodal_rag():
    print("=== 测试多模态 RAG 构建与查询 ===")
    
    # 初始化（自动创建本地目录 .rag_db）
    rag = GoldMultimodalRAG()
    
    if not rag.api_key:
        print("需要设置 DASHSCOPE_API_KEY 才能测试 Dashscope Embedding。")
        return
        
    print("1. 插入多模态知识...")
    # 模拟知识 1：纯文本（上个月的非农数据报告摘要）
    text1 = "2025年上个月非农就业数据大幅不及预期，失业率飙升，美联储被迫召开紧急会议降息。黄金一日暴涨 50 美元。"
    try:
         rag.add_knowledge(knowledge_id="nfp_report_2025", text=text1, metadata={"type": "macro_news", "date": "2025-02-15"})
         print("   加入文本知识点 1 成功")
    except Exception as e:
         print(f"写入知识 1 失败：{e}")
         
    # 模拟知识 2：图文历史记忆（带网络K线图片）
    text2 = "典型的头肩底突破形态，伴随MACD金叉，这是强烈的看涨技术形态。"
    image2_url = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/256_1.png" # 用公网图测
    try:
         rag.add_knowledge(knowledge_id="tech_pattern_hs", text=text2, image_path=image2_url, metadata={"type": "tech_chart"})
         print("   加入图文知识点 2 成功")
    except Exception as e:
         print(f"写入知识 2 失败：{e}")
         
    print("\n2. 测试多模态检索...")
    # 查询 1：用自然语言查
    print("   -> 查询: '就业数据不好黄金会涨吗？'")
    res1 = rag.search(query_text="就业数据不好黄金会涨吗？", top_k=1)
    if res1:
         print(f"   命中知识: [{res1[0]['id']}] 距离: {res1[0]['distance']:.4f}\n      内容: {res1[0]['text']}")
         
    # 查询 2：用包含图表特征的文本查询
    print("   -> 查询: '有没有看到过类似头肩底的形态？'")
    res2 = rag.search(query_text="有没有看到过类似头肩底的形态？", top_k=1)
    if res2:
         print(f"   命中知识: [{res2[0]['id']}] 距离: {res2[0]['distance']:.4f}\n      内容: {res2[0]['text']}\n      附图: {res2[0]['image']}")

if __name__ == "__main__":
    test_multimodal_rag()
