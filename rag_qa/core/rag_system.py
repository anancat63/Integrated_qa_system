# -*-coding:utf-8-*-
# core/rag_system.py 源码
import sys, os
# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(f'current_dir--》{current_dir}')
# 获取core文件所在的目录的绝对路径
rag_qa_path = os.path.dirname(current_dir)
# print(f'rag_qa_path--》{rag_qa_path}')
sys.path.insert(0, rag_qa_path)
# 获取根目录文件所在的绝对位置
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)
from prompts import RAGPrompts
#   导入 time 模块，用于计算时间
import time
from base import logger, Config
from query_classifier import QueryClassifier  #   导入查询分类器
from strategy_selector import StrategySelector  #   导入策略选择器
from vector_store import VectorStore # 导入向量数据库对象
from query_normalizer import QueryNormalizer

conf = Config()

#   定义 RAGSystem 类，封装 RAG 系统的核心逻辑
class RAGSystem:
    #   初始化方法，设置 RAG 系统的基本参数
    def __init__(self, vector_store, llm):
        #   设置向量数据库对象
        self.vector_store = vector_store
        #   设置大语言模型调用函数
        self.llm = llm
        #   获取 RAG 提示模板
        self.rag_prompt = RAGPrompts.rag_prompt()#使用基础莫态的模板
        #   初始化查询分类器模型
        classifier_path = os.path.join(rag_qa_path, 'core', 'bert_query_classifier_1')
        self.query_classifier = QueryClassifier(model_path=classifier_path)
        #   初始化策略选择器
        self.strategy_selector = StrategySelector()
        #   定义方法，生成答案
        self.query_normalizer = QueryNormalizer()




    #   定义类似私有方法，使用回溯问题进行检索 （注意讲义中没有加source_filter参数，这里补齐了）
    def _retrieve_with_backtracking(self, query, source_filter):
        logger.info(f"使用回溯问题策略进行检索 (查询: '{query}')")
        #   获取回溯问题生成的 Prompt 模板
        backtrack_prompt_template = RAGPrompts.backtracking_prompt()  # 使用 template 后缀区分
        try:
            #   调用大语言模型生成回溯问题
            simplified_query = self.llm(backtrack_prompt_template.format(query=query)).strip()
            logger.info(f"生成的回溯问题: '{simplified_query}'")
            #   使用回溯问题进行检索，并返回检索结果

            #进行查询milvus向量数据库并进行混合检索
            return self.vector_store.hybrid_search_with_rerank(
                simplified_query,#经过回溯问题生成的问题
                k=conf.RETRIEVAL_K,# 使用 K
                source_filter=source_filter  #传入学科筛选
            )
        except Exception as e:
            logger.error(f"回溯问题策略执行失败: {e}")
            return []

    #   定义类似私有方法，使用子查询进行检索（注意讲义中没有加source_filter参数，这里补齐了）
    def _retrieve_with_subqueries(self, query, source_filter):
        logger.info(f"使用子查询策略进行检索 (查询: '{query}')")
        #   获取子查询生成的 Prompt 模板
        subquery_prompt_template = RAGPrompts.subquery_prompt() # 使用 template 后缀区分
        try:
            #   调用大语言模型生成子查询列表
            subqueries_text = self.llm(subquery_prompt_template.format(query=query)).strip()
            # print(f'subqueries_text--》{subqueries_text}')
            subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
            logger.info(f"生成的子查询: {subqueries}")
            if not subqueries:
                 logger.warning("未能生成有效的子查询")
                 return []
            #   初始化空列表，用于存储所有子查询的检索结果
            all_docs = []
            #   遍历每个子查询
            for sub_q in subqueries:
                #   使用子查询进行检索，并将结果添加到列表中
                #   这里对每个子查询都执行了 hybrid search + rerank，开销可能较大
                # 这里面的k是conf.CANDIDATE_M//2 onf.CANDIDATE_M是它的一半
                docs = self.vector_store.hybrid_search_with_rerank(
                    sub_q, k=conf.CANDIDATE_M//2, source_filter=source_filter # 使用 K
                )
                all_docs.extend(docs)
                logger.info(f"子查询 '{sub_q}' 检索到 {len(docs)} 个文档")
            # print(f'all_docs-->{len(all_docs)}')
            # print(f'all_docs[0]-->{all_docs[0]}')
            #   对所有检索结果进行去重 (基于对象内存地址，如果 Document 内容相同但对象不同则无法去重)
            #   更可靠的去重方式是基于文档内容或 ID
            unique_docs_dict = {doc.page_content: doc for doc in all_docs} # 基于内容去重
            unique_docs = list(unique_docs_dict.values())
        #合并与截断：子查询会合并多路检索结果并去重；最终按配置截断 top-M 父文档作为上下文。
            logger.info(f"所有子查询共检索到 {len(all_docs)} 个文档, 去重后剩 {len(unique_docs)} 个")
            return unique_docs # 返回所有唯一文档，让 retrieve_and_merge 处理数量
        except Exception as e:
            logger.error(f'子查询存在错误：{e}')
            return []

    #   定义私有方法，使用假设文档进行检索（HyDE）
    def _retrieve_with_hyde(self, query, source_filter):
        logger.info(f"使用 HyDE 策略进行检索 (查询: '{query}')")
        #   获取假设问题生成的 Prompt 模板
        hyde_prompt_template = RAGPrompts.hyde_prompt() # 实例化模板
        #   调用大语言模型生成假设答案
        try:
            hypo_answer = self.llm(hyde_prompt_template.format(query=query)).strip()#将问题填入模板
            logger.info(f"HyDE 生成的假设答案: '{hypo_answer}'")
            #   使用假设答案进行检索，并返回检索结果
            return self.vector_store.hybrid_search_with_rerank(
                query = hypo_answer, k=conf.RETRIEVAL_K, source_filter=source_filter # 使用 K 而非 M
            )
        except Exception as e:
            logger.error(f"HyDE 策略执行失败: {e}")
            return []





#策略选择器选择
    def retrieve_and_merge(self, query, source_filter=None, strategy=None):

        # 对查询进行预处理:轻量标准化
        norm = self.query_normalizer.normalize(query)
        norm_query = norm.normalized_query
        print(f'query--->{query}')
        print(f'norm_query--->{norm_query}')


        #   如果未指定检索策略，则使用策略选择器选择
        if not strategy:#是否指定检索策略
            strategy = self.strategy_selector.select_strategy(norm_query)
        # 根据检索策略选择不同的检索方式
        #通过模型选择检索策略，之后根据某策略生成检索问题，然后使用milvus-向量数据库进行检索得到命中的父文档列表
        #根据配置文件选取命中父文档中前m的文档作为最终入选文档，选作上下文(返回一个列表--->这俩)
        ranked_chunks = [] # 初始化检索结果列表
        if strategy == "回溯问题检索":
            ranked_chunks = self._retrieve_with_backtracking(norm_query, source_filter)
        elif strategy == '子查询检索':
            ranked_chunks = self._retrieve_with_subqueries(norm_query, source_filter)
        elif strategy == "假设问题检索":
            ranked_chunks = self._retrieve_with_hyde(norm_query, source_filter)
        else:
            # 直接检索：
            logger.info(f"使用直接检索策略 (查询: '{norm_query}')")
            ranked_chunks = self.vector_store.hybrid_search_with_rerank(
                norm_query, k=conf.RETRIEVAL_K, source_filter=source_filter
            )  # 注意 hybrid_search_with_rerank 返回的是 rerank 后的父文档
            # print(f'ranked_chunks--》{ranked_chunks}')

        logger.info(f"策略 '{strategy}' 检索到 {len(ranked_chunks)} 个候选文档 (命中的父文档)")
        final_context_docs = ranked_chunks[:conf.CANDIDATE_M]#最终入选数量M，选作上下文
        logger.info(f"最终选取 {len(final_context_docs)} 个文档作为上下文")
        return final_context_docs



#处理查询
    def generate_answer(self, query, source_filter=None):

        #   记录查询开始时间
        start_time = time.time()
        logger.info(f"开始处理查询: '{query}', 学科过滤: {source_filter}")


        #   判断查询类型==================================
        query_category = self.query_classifier.predict_category(query)#预训练模型判断是否是通用知识
        logger.info(f"查询分类结果：{query_category} (查询: '{query}')")
        #   如果查询属于“通用知识”类别，则直接使用 LLM 回答
        if query_category == "通用知识":
            logger.info("查询为通用知识，直接调用 LLM")
            prompt_input = self.rag_prompt.format(
                context="", question=query, phone=conf.CUSTOMER_SERVICE_PHONE
            )  #   不使用上下文，直接使用通用 Prompt
            try:
                logger.info(f"直接调用 LLM (查询: '{query}')")
                answer = self.llm(prompt_input)
                logger.info(f"直接调用 LLM 结果: '{answer}'")
            except Exception as e:
                logger.error(f"直接调用 LLM 失败: {e}")
                answer = f"抱歉，处理您的通用知识问题时出错。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"
            processing_time = time.time() - start_time
            logger.info(
                f"通用知识查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')"
            )
            return answer



        #   否则，进行 RAG 检索并生成答案
        logger.info("查询为专业咨询，执行 RAG 流程")
        #   选择检索策略
        strategy = self.strategy_selector.select_strategy(query)
        # print(f'strategy--{strategy}')
        #  检索相关文档(查询后得到的命中前2的父文档)
        context_docs = self.retrieve_and_merge(query, source_filter=source_filter, strategy=strategy)  # 传递 strategy
        # print(f'context_docs--》{context_docs}')
        print(f'context_docs--》{context_docs[0]}')
        #context_docs[0] - -》page_content = '人工智能就业课 【课程大纲】--->父文档标题

        #   准备上下文
        if context_docs:
            context = "\n\n".join([doc.page_content for doc in context_docs])  # 使用换行符分隔文档
            logger.info(f"构建上下文完成，包含 {len(context_docs)} 个文档块")
            # logger.debug(f"上下文内容:\n{context[:500]}...") # Debug 日志可以打印部分上下文
        else:
            context = ""
            logger.info("未检索到相关文档，上下文为空")


        #   使用检索结果构造 Prompt，调用大语言模型生成答案
        logger.info(f"使用检索结果(命中返回2个父文档)构建 Prompt 进行查询)")
        prompt_input = self.rag_prompt.format(
            context=context, # 使用上下文
            question=query, # 问题
            phone=conf.CUSTOMER_SERVICE_PHONE# 客服电话
        )
        # logger.debug(f"最终生成的 Prompt:\n{prompt_input}") # Debug 日志
        logger.info("提示词已生成,调用 LLM 生成答案")
        try:
            answer = self.llm(prompt_input)
            logger.info("LLM 已生成答案")
        except Exception as e:
            logger.error(f"调用 LLM 生成最终答案失败: {e}")
            answer = f"抱歉，处理您的专业咨询问题时出错。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"
        #   记录查询处理完成的日志
        processing_time = time.time() - start_time
        logger.info(f"查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')")
        return answer

if __name__ == '__main__':
    vector_store = VectorStore()#实例化milvus数据库操作对象
    llm = StrategySelector().call_dashscope#实例化大语言模型对象
    # print(llm(prompt="什么是AI"))
    rag_system = RAGSystem(vector_store, llm)#实例化rag系统对象

    # result0 = rag_system._retrieve_with_backtracking(query="AI和JAVA的区别是什么？，人工智能呢？", source_filter="ai")  #测试回溯问题检索
    # print(result0)
    # result1 =  rag_system._retrieve_with_subqueries(query="AI和JAVA的区别是什么？，人工智能呢？", source_filter="ai")#测试子查询
    # print(result1)
    # print(len(result1))
    # result2 = rag_system._retrieve_with_hyde(query="AI课程的NLP的技术有哪些?",source_filter="ai")#测试假设问题检索
    # print(result2)
    # result3 = rag_system.retrieve_and_merge(query="AI学科的课程大纲内容有什么",source_filter="ai")#测试直接检索
    # print(result3)
    answer = rag_system.generate_answer(query="AI学科课程都学点啥？",source_filter="ai")#带决策选择的测试
    print(answer)