import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

class Config:
    # retrievor 参数
    topd = 3
    topt = 6
    maxlen = 128
    topk = 5
    bert_path = '/workspace/model/embedding/tao-8k'
    recall_way = 'embed'  # keyword, embed

    # generator 参数
    max_source_length = 767
    max_target_length = 256
    model_max_length = 1024

    # embedding API 参数
    use_api = True
    api_key = os.getenv("API_KEY", "sk-xx")
    base_url = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model_name = "text-embedding-v3"
    dimensions = 1024
    batch_size = 10

    # LLM API 参数
    llm_api_key = os.getenv("LLM_API_KEY", api_key)
    llm_base_url = os.getenv("BASE_URL", base_url)
    llm_model = "qwen-plus"

    # 知识库配置
    kb_base_dir = "knowledge_bases"
    default_kb = "default"

    # 输出目录配置
    output_dir = "output_files"
