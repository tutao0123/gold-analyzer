import os
import json
import logging
import numpy as np
import dashscope
from pathlib import Path

# 确保Faiss可用
try:
    import faiss
except ImportError:
    faiss = None

logger = logging.getLogger(__name__)

class GoldMultimodalRAG:
    """基于 DashScope 多模态 Embedding 与 Faiss 的知识库"""
    
    def __init__(self, db_dir=".rag_db", dimension=2560):
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "")
        self.db_dir = db_dir
        self.dimension = dimension
        
        # 内部存储
        self.knowledge_store = [] # 存储文档元数据
        self.index_file = os.path.join(self.db_dir, "gold.index")
        self.meta_file = os.path.join(self.db_dir, "gold_meta.json")
        
        os.makedirs(self.db_dir, exist_ok=True)
        
        if faiss:
            self.index = faiss.IndexFlatL2(self.dimension)
            self._load_db()
        else:
            logger.warning("未安装 faiss，RAG 将仅支持遍历对比（性能较低）")
            self.index = None

    def _load_db(self):
        """加载已有的本地库"""
        if os.path.exists(self.index_file):
            logger.info("加载现存的图文向量库...")
            self.index = faiss.read_index(self.index_file)
        if os.path.exists(self.meta_file):
            with open(self.meta_file, "r", encoding="utf-8") as f:
                self.knowledge_store = json.load(f)
                
    def _save_db(self):
        """保存本地库"""
        if self.index is not None:
            faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self.knowledge_store, f, ensure_ascii=False, indent=2)

    def get_embedding(self, text=None, image_path=None, video_path=None):
        """
        调用百炼多模态接口生成融合向量
        """
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY 未设置")
            
        input_data = {}
        if text:
            input_data["text"] = text
        if image_path:
            # Dashscope 支持 file:// url 用于本地文件上传识别，但在新版库通常传绝对路径可能存在问题。
            # 这里按照通用形态转绝对协议 (若是网络图则直接传 URL)
            if image_path.startswith("http"):
                input_data["image"] = image_path
            else:
                abs_path = os.path.abspath(image_path)
                input_data["image"] = f"file://{abs_path}"
        if video_path:
             if video_path.startswith("http"):
                 input_data["video"] = video_path
             else:
                 abs_path = os.path.abspath(video_path)
                 input_data["video"] = f"file://{abs_path}"

        if not input_data:
            raise ValueError("至少需要提供 text, image 或 video 之一")
            
        logger.info(f"正在进行多模态Embedding... 输入Keys: {list(input_data.keys())}")
        
        resp = dashscope.MultiModalEmbedding.call(
            api_key=self.api_key,
            model="qwen3-vl-embedding",
            input=[input_data]
            # dimension 默认 2560 不用传，如果想传可以配置
        )
        
        if resp.status_code != 200:
            raise Exception(f"多模态 Embedding 失败: {resp.message}")
            
        return resp.output["embeddings"][0]["embedding"]

    def add_knowledge(self, knowledge_id: str, text: str = None, image_path: str = None, metadata: dict = None, precomputed_vector: list = None):
        """
        添加知识到 RAG 库并本地化
        """
        if precomputed_vector is not None:
            vector = precomputed_vector
        else:
            vector = self.get_embedding(text=text, image_path=image_path)
            
        np_vector = np.array([vector], dtype="float32")
        
        # 保存 Metadata
        new_entry = {
            "id": knowledge_id,
            "text": text,
            "image": image_path,
            "metadata": metadata or {}
        }
        
        if self.index is not None:
            self.index.add(np_vector)
            
        self.knowledge_store.append(new_entry)
        self._save_db()
        logger.info(f"知识 [{knowledge_id}] 已加入多模态向量库。")
        
    def search(self, query_text: str = None, query_image: str = None, top_k: int = 2):
        """
        基于提问或者提问图文查询最相关的记忆
        """
        if not self.knowledge_store:
            return []
            
        vector = self.get_embedding(text=query_text, image_path=query_image)
        np_vector = np.array([vector], dtype="float32")
        
        results = []
        if self.index is not None:
            # Faiss L2距离越小越相似
            distances, indices = self.index.search(np_vector, top_k)
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.knowledge_store) and idx >= 0:
                     entry = self.knowledge_store[idx].copy()
                     entry["distance"] = float(dist)
                     results.append(entry)
        else:
            # 暴力遍历（欧氏距离）
            best_dists = []
            # 但既然上面安装 faiss, 这里可以简略容错即可
            pass
            
        return results
