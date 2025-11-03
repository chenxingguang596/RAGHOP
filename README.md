# 🧠 Medical RAG 多跳推理问答系统

本项目是一个基于多跳推理的 RAG（Retrieval-Augmented Generation）系统，结合本地知识库 + 在线搜索 + LLM 推理，实现更强的医疗问答能力。

---

## 🚀 功能特性
- 支持 **PDF 文档知识库构建**  
- 自动 **语义分块 + 向量化 + FAISS 索引**  
- 支持 **多跳推理（Multi-hop Reasoning）**  
- 可选 **联网搜索**（通过 Bing）  
- 支持 **Markdown 表格化回答输出**  
- 使用 **OpenAI 兼容接口（DashScope / Qwen 系列模型）**

---

## 📦 环境配置

### 1. 安装依赖
```bash
pip install -r requirements.txt
