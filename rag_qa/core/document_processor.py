# -*- coding:utf-8 -*-
# 说明：统一的“文档加载 + 分层切分”模块
# 系统定位：
# - 这是 RAG 系统中【离线文档处理阶段】的核心组件
# - 负责把多源文档（PDF / Word / PPT / Markdown / 图片等）
#   → 统一加载为 LangChain Document
#   → 再按“父块 / 子块”两级结构进行切分
# - 其输出结果通常用于：向量化、入库（Milvus / FAISS 等）

# ⚠️ 注释中特别说明：
# 当前讲义中的“代码架构图”未体现该模块的存在，
# 实际上它处于【数据准备层 / 文档预处理层】，
# 是 RAG 系统不可或缺的一环，需要在架构图中补充体现。

import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter
from datetime import datetime
import sys

# ========== 路径处理：保证项目内模块可被正确导入 ==========
# 获取当前文件所在目录（例如 rag_qa/core）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取 rag_qa 目录的绝对路径
rag_qa_path = os.path.dirname(current_dir)
# 将 rag_qa 加入模块搜索路径
sys.path.insert(0, rag_qa_path)
# 获取项目根目录路径
project_root = os.path.dirname(rag_qa_path)
# 将项目根目录加入模块搜索路径
sys.path.insert(0, project_root)

# 导入自定义 OCR 文档加载器
# 用于处理 PDF / Word / PPT / 图片等非纯文本文件
from document_loaders import (
    OCRPDFLoader,
    OCRDOCLoader,
    OCRPPTLoader,
    OCRIMGLoader
)

# 导入中文递归文本切分器（适合中文长文本）
from text_splitter import ChineseRecursiveTextSplitter
# 导入全局日志与配置
from base import logger, Config

# 初始化全局配置对象
conf = Config()

# ========== 文档加载器注册表 ==========
# 定义支持的文件类型与其对应的加载器
# 作用：
# - 通过文件后缀自动选择合适的文档加载策略
# - 屏蔽不同文件格式的解析差异
document_loaders = {
    ".txt": TextLoader,                         # 纯文本文件
    ".pdf": OCRPDFLoader,                       # PDF（OCR）
    ".docx": OCRDOCLoader,                      # Word（OCR）
    ".ppt": OCRPPTLoader,                       # PPT（OCR）
    ".pptx": OCRPPTLoader,                      # PPTX（OCR）
    ".jpg": OCRIMGLoader,                       # 图片（OCR）
    ".png": OCRIMGLoader,                       # 图片（OCR）
    ".md": UnstructuredMarkdownLoader           # Markdown 文档
}

def load_documents_from_directory(directory_path):
    """
    从指定目录递归加载所有支持类型的文档，并统一转换为 Document 对象。

    功能职责：
    - 遍历目录及子目录
    - 根据文件后缀选择对应 Loader
    - 加载文档内容
    - 为每个 Document 注入统一的元数据（source / file_path / timestamp）

    参数：
    - directory_path：学科数据目录路径（如 ai_data / java_data）

    返回：
    - documents：List[Document]
    """
    #初始化 documents列表
    documents = []

    # 支持的文件后缀集合
    supported_extensions = document_loaders.keys()

    # 从目录名中提取“学科标签”
    # 例如：ai_data → ai
    source = os.path.basename(directory_path).replace("_data", "")

    # 递归遍历目录
    for root, _, files in os.walk(directory_path):
        for file in files:
            # 构造文件完整路径
            # print(f"root----->{root}")
            # print(f"file----->{file}")
            file_path = os.path.join(root, file)
            # 通过.切割获取文件扩展名（统一转小写）
            file_extension = os.path.splitext(file_path)[1].lower()
            # print(f"file_extension----->{file_extension}")

            # 若文件类型受支持
            if file_extension in supported_extensions:
                try:
                    # 根据扩展名选择对应加载器
                    loader_class = document_loaders[file_extension]

                    # txt 文件需要显式指定编码
                    if file_extension == ".txt":
                        loader = loader_class(file_path, encoding="utf-8")
                    else:
                        loader = loader_class(file_path)

                    # 执行加载，返回 Document 列表
                    loaded_docs = loader.load()

                    # 为每个 Document 添加统一元数据
                    for doc in loaded_docs:
                        doc.metadata["source"] = source #学科信息
                        doc.metadata["file_path"] = file_path#文件路径
                        doc.metadata["timestamp"] = datetime.now().isoformat()#时间戳
                        """
                                            [
                        Document(正文1, 元数据1),
                        Document(正文2, 元数据2),
                        Document(正文3, 元数据3),
                        ...
                        ]

                        
                        
                        """


                    documents.extend(loaded_docs)
                    logger.info(f"成功加载文件: {file_path}")

                except Exception as e:
                    # 单文件加载失败不影响整体流程
                    logger.error(f"加载文件 {file_path} 失败: {str(e)}")
            else:
                # 非支持类型文件仅记录告警
                logger.warning(f"不支持的文件类型: {file_path}")

    return documents


def process_documents(
    directory_path,                             # 学科数据目录
    parent_chunk_size=conf.PARENT_CHUNK_SIZE,   # 父块长度
    child_chunk_size=conf.CHILD_CHUNK_SIZE,     # 子块长度
    chunk_overlap=conf.CHUNK_OVERLAP            # 块重叠长度
):
    """
    文档处理主流程：加载 → 父块切分 → 子块切分。

    设计思想：
    - 父块（Parent Chunk）：保留完整上下文，用于最终拼接
    - 子块（Child Chunk）：用于向量化与检索，提高召回精度
    - 检索命中子块 → 回溯父块 → 构建上下文

    参数：
    - directory_path：学科数据目录
    - parent_chunk_size：父块长度
    - child_chunk_size：子块长度
    - chunk_overlap：切分重叠长度

    返回：
    - child_chunks：包含父子关系元数据的子块 Document 列表
    """
    #使用load_documents_from_directory()加载原始文档
    documents = load_documents_from_directory(directory_path)
    logger.info(f"加载的文档数量: {len(documents)}")

    # 初始化中文递归切分器（用于非 Markdown 文档）
    #ChineseRecursiveTextSplitter 是递归式切分器：它会尽量在“语义单元”处切分（段落、句子），保证切块自然。
    # parent_splitter → 用于生成父块（较大块，保留完整上下文）
    # child_splitter → 用于生成子块（较小块，方便向量化检索）

    parent_splitter = ChineseRecursiveTextSplitter(
        chunk_size=parent_chunk_size,#父块长度
        chunk_overlap=chunk_overlap#块重叠长度
    )
    child_splitter = ChineseRecursiveTextSplitter(
        chunk_size=child_chunk_size,#子块长度
        chunk_overlap=chunk_overlap#块重叠长度
    )

    # 初始化 Markdown 专用切分器（保留 Markdown 结构）
    markdown_parent_splitter = MarkdownTextSplitter(
        chunk_size=parent_chunk_size,#父块长度
        chunk_overlap=chunk_overlap#块重叠长度
    )
    markdown_child_splitter = MarkdownTextSplitter(
        chunk_size=child_chunk_size,#子块长度
        chunk_overlap=chunk_overlap#块重叠长度
    )


    # 初始化子块列表
    child_chunks = []
    logger.info("开始处理文档")
    # 遍历每一篇原始文档

    for i, doc in enumerate(documents):#enumerate 给每篇文档一个索引 i，用于生成唯一 id。
        # 根据文件后缀判断是否为 Markdown
        file_extension = os.path.splitext(
            doc.metadata.get("file_path", "")
        )[1].lower()

        is_markdown = (file_extension == ".md")

        # 按文档类型选择切分器
        parent_splitter_to_use = (# 选择父块切分器
            markdown_parent_splitter if is_markdown else parent_splitter
        )
        child_splitter_to_use = (# 选择子块切分器
            markdown_child_splitter if is_markdown else child_splitter
        )

        logger.info(
            f"处理文档: {doc.metadata['file_path']}, "
            f"使用切分器: {'Markdown' if is_markdown else 'ChineseRecursive'}"
        )

        # 先进行父块切分
        parent_docs = parent_splitter_to_use.split_documents([doc])#加中括号是因为迭代器要求输入写法

        # 对每个父块继续切分子块
        for j, parent_doc in enumerate(parent_docs):
            # 构造父块唯一 ID
            parent_id = f"doc_{i}_parent_{j}"

            # 对父块进行子块切分
            sub_chunks = child_splitter_to_use.split_documents([parent_doc])

            for k, sub_chunk in enumerate(sub_chunks):
                # 建立父子关系元数据
                sub_chunk.metadata["parent_id"] = parent_id                   # 子块的父块ID
                sub_chunk.metadata["parent_content"] = parent_doc.page_content# 子块的父块内容
                sub_chunk.metadata["id"] = f"{parent_id}_child_{k}"           # 子块的ID

                #将子块添加到子块列表中
                child_chunks.append(sub_chunk)

    logger.info(f"子块数量: {len(child_chunks)}")
    # print(f"child_chunks：{child_chunks}")
    return child_chunks
#最后返回的 child_chunks，就是“所有文档 → 所有父块 → 所有子块”的
#全集合（flatten 后的一维列表）。





"""
Document(
    page_content="这是被切分出来的子块文本（用于向量检索）",
    metadata={
        "source": "ai",
        "file_path": "xxx/xxx.md",
        "timestamp": "2026-01-03T09:30:21",

        # ↓↓↓ 父子关系核心 ↓↓↓
        "parent_id": "doc_0_parent_2",
        "parent_content": "这是完整父块文本（用于最终上下文）",
        "id": "doc_0_parent_2_child_1"
    }
)


"""
if __name__ == '__main__':
    directory_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples"))

    # 获取所有子块
    child_chunks = process_documents(directory_path)
    # print(child_chunks)
    print(child_chunks[0])
    print('-'*28)
    print(child_chunks[1])
    print('-'*28)
    print(child_chunks[2])
    #
    #
    # if not child_chunks:
    #     print("未生成任何子块")
    #     exit(0)
    #
    # # 取第一个子块
    # chunk = child_chunks[0]
    #
    # print("=" * 80)
    # print("子块 ID:")
    # print(chunk.metadata.get("id"))
    #
    # print("\n父块 ID:")
    # print(chunk.metadata.get("parent_id"))
    #
    # print("\n子块内容:")
    # print(chunk.page_content)
    #
    # print("\n父块内容:")
    # print(chunk.metadata.get("parent_content"))
