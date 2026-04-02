# -*- coding:utf-8 -*-
# 导入 BGE-M3 嵌入函数，用于生成文档和查询的向量表示
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch.cuda
from milvus_model.hybrid import BGEM3EmbeddingFunction
# 导入 Milvus 相关类，用于操作向量数据库
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 Document 类，用于创建文档对象
from langchain.docstore.document import Document
# 导入 CrossEncoder，用于重排序和 NLI 判断
from sentence_transformers import CrossEncoder
# 导入 hashlib 模块，用于生成唯一 ID 的哈希值
import hashlib
import sys, os
# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(f'current_dir--》{current_dir}')
# 获取core文件所在的目录的绝对路径
rag_qa_path = os.path.dirname(current_dir)
# print(f'rag_qa_path--》{rag_qa_path}')
core_path = os.path.join(rag_qa_path, 'core')
sys.path.insert(0, core_path)
sys.path.insert(0, rag_qa_path)
# 获取根目录文件所在的绝对位置
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)
from document_processor import *
from base import logger, Config


conf = Config()


# core/vector_store.py
# 定义 VectorStore 类，封装向量存储和检索功能
class VectorStore:
    # 初始化方法，设置向量存储的基本参数，链接Milvus数据库
    def __init__(self,
                 collection_name=conf.MILVUS_COLLECTION_NAME,#milvus集合名称
                 host=conf.MILVUS_HOST,                      #Milvus主机地址
                 port=conf.MILVUS_PORT,                      #Milvus 端口
                 database=conf.MILVUS_DATABASE_NAME):        #Milvus数据库名称
        logger.info(f"初始化 VectorStore中...")
        # 设置 Milvus 集合名称
        self.collection_name = collection_name
        # 设置 Milvus 主机地址
        self.host = host
        # 设置 Milvus 端口号
        self.port = port
        # 设置 Milvus 数据库名称
        self.database = database


        # 检查CUDA是否可用
        self.device ='cuda' if torch.cuda.is_available() else 'cpu'
        # 日志提醒使用的是什么设备
        logger.info(f"使用设置：{self.device}")

        # 初始化 BGE-Reranker 模型，用于重排序检索结果
        reranker_path = os.path.join(rag_qa_path, 'models', 'bge-reranker-large')
        # print(f'reranker_path--》{reranker_path}')
        self.reranker = CrossEncoder(reranker_path, device=self.device)

        # 初始化 BGE-M3 嵌入函数，使用 CPU 设备，不启用 FP16
        m3_path = os.path.join(rag_qa_path, 'models', 'bge-m3')
        logger.info(f"初始化 BGE-M3 嵌入函数中...")
        self.embedding_function = BGEM3EmbeddingFunction(model_name_or_path=m3_path, use_fp16=(self.device == 'cuda'), device=self.device)
        # 获取稠密向量的维度# 1024
        self.dense_dim = self.embedding_function.dim["dense"]#创建表的时候要指定向量维度---83行
        logger.info(f"稠密向量维度：{self.dense_dim}")

        # 初始化 Milvus 客户端对象，连接到指定主机和数据库
        logger.info(f"连接到 Milvus 服务器 ing...")
        # self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)#指明加载数据库
        self.client = MilvusClient(uri=f"tcp://{self.host}:{self.port}", db_name=self.database)
        logger.info(f"Milvus 连接成功！")
        # 调用方法创建或加载 Milvus 集合
        self._create_or_load_collection()




    # 类私有方法--# 检查指定集合是否已经存在，若无则创建集合
    def _create_or_load_collection(self):
        # 检查指定集合是否已经存在
        logger.info(f"检查集合 {self.collection_name} 是否存在ing...")
        if not self.client.has_collection(self.collection_name):
            logger.info(f"集合 {self.collection_name} 不存在，创建集合中...")
            # 创建集合 Schema，禁用自动 ID，启用动态字段
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 添加 ID 字段，作为主键，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 添加文本字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 添加稠密向量字段，FLOAT_VECTOR 类型，维度由嵌入函数指定
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            # 添加稀疏向量字段，SPARSE_FLOAT_VECTOR 类型
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 添加父块 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 添加父块内容字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 添加学科类别字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 添加时间戳字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            # 创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 为稠密向量字段添加 IVF_FLAT 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}#添加簇中心
            )
            # 为稀疏向量字段添加 SPARSE_INVERTED_INDEX 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}#在指定的搜索过程中忽略多大比例的小向量值来微调搜索性能(忽略20%小向量)
            )

            # 创建 Milvus 集合，应用定义的 Schema 和索引参数
            self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                          index_params=index_params)
            # 记录创建集合的日志
            logger.info(f"已创建集合 {self.collection_name}")
        # 如果集合已存在
        else:
            # 记录加载集合的日志
            logger.info(f"已加载集合 {self.collection_name}")
        # 将集合加载到内存，确保可立即查询
        self.client.load_collection(self.collection_name)






    # 定义方法，向向量存储添加文档
    def add_documents(self, documents):
        # print(f'documents--》{documents[0]}')
        # 提取所有文档的内容列表
        texts = [doc.page_content for doc in documents]
        print(f'texts-type--》{type( texts)}')
        print(f'texts--》{texts}')


        # 使用 BGE-M3 嵌入函数生成文档的嵌入
        embeddings = self.embedding_function(texts)
        print(f'embeddings--》{embeddings}')
        print(f'embeddings--》{embeddings.keys()}')
        # 初始化空列表，存储插入的数据
        data = []

        """
        这个循环的作用是：
将 BGE-M3 模型为“每个文档子块”生成的稀疏向量表示（indices + values），
转换为 Milvus 所要求的稀疏向量存储格式（{维度ID: 权重} 的字典），
并与对应的 dense 向量及元数据一起写入向量库。
        """
        # # 遍历每个文档，带上索引i
        for i, doc in enumerate(documents):
            # 生成文档内容的哈希值作为唯一的ID
            text_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            # print(f'text_hash--》{text_hash}')
            # print(f'text_hash--》{type(text_hash)}')
            # 初始化一个稀疏向量的字典（Milvus要求存储稀疏向量的格式）
            sparse_vector = {}
            # 获取第i行对应的稀疏向量数据[0.4, 0.2, 0, 0, 0.1]
            # row = embeddings["sparse"][ i]
            row = embeddings["sparse"][i: i + 1]

            # row = embeddings["sparse"][i]:新版本milvus-model，支持这种获取稀疏向量的形式
            indics = row. indices
            print(f'row--》{row}')
            print(f'row-shape-》{row.shape}')
            # 获取稀疏向量的非零值的索引
            indics = row.indices
            print(f'indics--》{indics}')
            # 获取稀疏向量的非零值
            values = row.data
            # 将索引和值进行配对，存储到字典中
            # for idx, value in zip(indics, values):
            #     sparse_vector[idx] = value（修改）
            sparse_vector = {
                int(idx): float(val)
                for idx, val in zip(row.indices, values)
            }


            # print(f'sparse_vector--》{sparse_vector}')
            # print(f'sparse_vector--》{len(sparse_vector)}')
            # print(embeddings["dense"][i])
            # print(embeddings["dense"][i].shape)
            # 创建数据字典，包含所有字段
            data.append({
                "id": text_hash,
                "text": doc.page_content,
                "dense_vector": embeddings["dense"][i],
                "sparse_vector": sparse_vector,
                "parent_id": doc.metadata["parent_id"],
                "parent_content": doc.metadata["parent_content"],
                "source": doc.metadata.get("source", "unknown"),
                "timestamp": doc.metadata.get("timestamp", "unknown")
            })
        # 检查是否有数据需要插入
        if data:
            # 使用 upsert 操作插入数据，覆盖重复 ID
            self.client.upsert(collection_name=self.collection_name, data=data)
            # 记录插入或更新的文档数量日志
            logger.info(f"已插入或更新 {len(data)} 个文档")

        """
Milvus 里存的每一条 entity，都是一个「子块（chunk）」
检索阶段：用子块向量做匹配
返回阶段：通过 parent_id 回溯并聚合父块（或原文档）

        """

    # 定义方法：执行混合检索（稠密 + 稀疏向量）并对父文档进行重排序
                        #参数含义 query: 输入的查询文本；k: 检索返回数量；source_filter: 源过滤器(通过科目进行分类)
    def hybrid_search_with_rerank(self, query, k=conf.RETRIEVAL_K, source_filter=None):
        """
        这个方法的作用：
        1. 对输入查询 query 生成稠密向量和稀疏向量表示
        2. 使用稠密向量和稀疏向量在向量数据库中进行混合搜索
        3. 将搜索结果封装为 Document 子块
        4. 去重合并成父文档
        5. 如果父文档数量 >= 2，使用 BGE-Reranker 对父文档进行重排序
        6. 返回前 conf.CANDIDATE_M 个排序后的父文档
        """

        # ------------------- 1. 生成查询向量 -------------------
        # 使用嵌入模型生成查询的稠密和稀疏向量
        logger.info("正在生成查询向量ing...")
        query_embeddings = self.embedding_function([query])

        # print(f'query_embeddings--》{query_embeddings}')
        logger.info("提取查询向量的稀疏稠密向量用于搜索向量库ing...")
        # 提取稠密向量（用于向量搜索）
        dense_query_vector = query_embeddings["dense"][0]
        # dense_query_vector = query_embeddings["dense"][1]报错  -----因为query只有一句，这里的dense_query_vector只有一个值

        # 提取稀疏向量，并转换为字典格式 {维度ID: 权重}，用于向量搜索
        sparse_query_vector = {}

        # 兼容 csr_array / csr_matrix：用切片取 1×N 行
        row = query_embeddings["sparse"][0:1]

        indices = row.indices
        values = row.data

        sparse_query_vector = {int(i): float(v) for i, v in zip(indices, values)}

        # row = query_embeddings["sparse"].getrow(0)  # 稀疏矩阵的第 0 行
        # # print(f'row--》{row}')
        # indices = row.indices  # 非零索引
        # values = row.data  # 非零值
        # sparse_query_vector = {
        #     int(idx): float(val)
        #     for idx, val in zip(row.indices, values)
        # }
        # for idx, value in zip(indices, values):
        #     sparse_query_vector[idx] = value
            #将问题格式也从(Key列表，Data列表)转换为(Data列表:Key列表)字典形式，用于稀疏向量匹配

        # ------------------- 2. 构建过滤条件 -------------------
        # 如果指定 source_filter，只匹配该来源文档；否则不过滤
        logger.info(f"正在构建过滤条件ing...")
        filter_expr = f"source == '{source_filter}'" if source_filter else ""
        #
        # # ------------------- 3. 构建向量搜索请求 -------------------
        # 稠密向量搜索请求
        logger.info("构建稀疏，稠密向量搜索请求ing...")
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,
            expr=filter_expr
        )

        # 稀疏向量搜索请求
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=k,
            expr=filter_expr
        )

        # ------------------- 4. 执行混合搜索 -------------------
        # 创建加权排序器：稠密向量权重1.0，稀疏向量权重0.7
        ranker = WeightedRanker(1.0, 0.7)
        logger.info("正在执行混合向量检索ing...")
        # 使用客户端执行混合向量检索
        results = self.client.hybrid_search(
            collection_name=self.collection_name,# 向量库名称
            reqs=[dense_request, sparse_request],# 向量搜索请求
            ranker=ranker,# 加权排序器
            limit=k,# 返回数量
            output_fields=["text", "parent_id", "parent_content", "source", "timestamp"]# 返回字段
        )[0]

        print(f"命中返回子块数量:{len(results)}")


        # ------------------- 5. 封装为子块 Document -------------------
        # 将每条检索结果转换为 Document 对象（子块）
        logger.info("将每条结果封装为子块document对象ing...")
        sub_chunks = [self._doc_from_hit(hit["entity"]) for hit in results]#只提取实体内容
        # print(repr(f'sub_chunks---=============》{sub_chunks}'))
        # print(type(f"sub_chunks-type{type(sub_chunks)}"))
        """
        results是混合检索用向量化的query和milvus库中的entity进行多种评估后返回的前K个entity
        然后通过sub_chunks依次遍历这些entity
        将他们封装成多个document对象(自定义格式封装)
        """
        logger.info(f"一共封装{len(sub_chunks)}个子块document对象")



        # ------------------- 6. 去重合并父文档 -------------------
        # 从命中milvus中的子块中提取去重的父文档
        parent_docs = self._get_unique_parent_docs(sub_chunks)
        print(f"命中父文档数量----->{len(parent_docs)}")
        # ------------------- 7. 如果父文档少于 2 个，直接返回 -------------------
        if len(parent_docs) < 2:
            # 直接返回前 conf.CANDIDATE_M 个文档，跳过重排序
            return parent_docs[:conf.CANDIDATE_M]

        # ------------------- 8. 使用 BGE-Reranker 对父文档重排序 -------------------
        if parent_docs:
            # 将查询与父文档内容配对，构建用于重排序的输入
            pairs = [[query, doc.page_content] for doc in parent_docs]

            # 使用 BGE-Reranker 计算每个文档的相关性得分
            scores = self.reranker.predict(pairs)

            # 根据得分从高到低排序父文档
            ranked_parent_docs = [doc for _, doc in sorted(zip(scores, parent_docs), reverse=True)]
        else:
            # 如果没有父文档，返回空列表
            ranked_parent_docs = []

        # ------------------- 9. 返回最终结果 -------------------
        # 返回前 conf.CANDIDATE_M 个重排序后的父文档
        return ranked_parent_docs[:conf.CANDIDATE_M]


    """
        Milvus 向量库存储的是每个文档的“子块”（chunk）
        检索阶段：用子块向量进行匹配
        返回阶段：通过 parent_id 回溯聚合成父文档（原文档）
        该方法整合了稠密+稀疏检索，并用 BGE-Reranker 提升父文档排序质量
    """

        # ------------------- 检索流程回顾 -------------------
# 1. 生成查询向量
#    - dense_query_vector: 用于语义匹配（向量检索）
#    - sparse_query_vector: 用于关键词匹配（稀疏向量检索）
#
# 2. 构建搜索请求
#    - dense_request / sparse_request: 分别描述稠密向量和稀疏向量的搜索
#    - self.client.hybrid_search: 将两个请求组合，按加权（Dense 1.0, Sparse 0.7）执行混合搜索
#
# 3. 得到匹配的子块
#    - results: 最相关的子块列表（每个子块是 Document，对应向量库中的一个 chunk）
#
# 4. 聚合父文档
#    - self._get_unique_parent_docs(sub_chunks): 使用 parent_id 去重，将属于同一文档的子块合并
#    - 得到父文档列表
#
# 5. 父文档重排序
#    - 如果父文档数量 >= 2，使用 BGE-Reranker 对父文档打分排序
#    - 最终返回前 conf.CANDIDATE_M 个父文档
#
# ------------------- 返回示例 -------------------
# 返回的列表示例：
# [
#     Document(metadata={'parent_id': 'doc_1_parent_0', ...}),
#     Document(metadata={'parent_id': 'doc_1_parent_4', ...})
# ]
# 这两个 Document 对象就是最相关的父文档，每个父文档可能包含多个子块
# 子块用于向量匹配，但最终返回的是父文档，便于用户理解或直接使用
#
# 总结：
# 该方法最终返回的是最相关的父文档，而不是子块本身，子块只是中间匹配单位


#流程概括：向量化query，使用混合检索，返回最相关的父文档



    # 定义方法，从子块中获取唯一的父文档
    def _get_unique_parent_docs(self, sub_chunks):
        # 初始化集合，用于存储已处理的父块内容（去重）

        parent_contents = set()
        # 初始化列表，用于存储唯一父文档
        unique_docs = []
        # 遍历所有子块
        for chunk in sub_chunks:

            # 获取子块的父块内容，默认为子块内容
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            # 检查父块内容是否非空且未重复
            if parent_content and parent_content not in parent_contents:
                # 创建新的 Document 对象，包含父块内容和元数据
                unique_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                # 将父块内容添加到去重集合
                parent_contents.add(parent_content)
            # 返回去重后的父文档列表
        return unique_docs


    # 定义类似私有方法，从 Milvus 查询结果创建 Document 对象
    # 创建并返回 Document 对象，填充内容和元数据
    def _doc_from_hit(self, hit):
        # 创建并返回 Document 对象，填充内容和元数据
        return Document(
            page_content=hit.get("text"),
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp")
            }
        )





if __name__ == "__main__":
    vector_store = VectorStore()
    # directory_path = r'.\data\ai_data'
    # print(f"embedding_function.dim--》{vector_store.embedding_function.dim}")
    # documents = process_documents(directory_path)
    # vector_store.add_documents(documents)
    query = "AI相关内容是什么"
    results = vector_store.hybrid_search_with_rerank(query, source_filter='ai')
    print(f'命中父文档内容:results-->{results}')
    print(f'命中父文档个数:{len(results)}')
