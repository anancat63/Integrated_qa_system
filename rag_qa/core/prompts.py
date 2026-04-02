# core/prompts.py
# 导入 PromptTemplate 类，用于创建 Prompt 模板
from langchain.prompts import PromptTemplate


# 定义 RAGPrompts 类，用于管理所有 Prompt 模板
class RAGPrompts:
    # 定义 RAG 提示模板
    @staticmethod
    def rag_prompt():
        # 创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""
            你是一个智能助手，帮助用户回答问题。
            如果提供了上下文，请基于上下文回答；如果没有上下文，请直接根据你的知识回答。
            如果答案来源于检索到的文档，请在回答中说明。

            上下文: {context}
            问题: {question}
            如果问起你是什么模型，请回答是黑马独立开发的模型--HMSBFQA-2.0
            如果信息不太够回复：“若有其他问题请联系人工客服，电话：{phone}。”
        
            """,
            #   定义输入变量
            input_variables=["context", "question", "phone"],
        )




    #     # 定义假设问题生成的 Prompt 模板
    # @staticmethod
    # def rag_prompt():
    #     '''添加了历史记录，注意用在：new_rag_system'''
    #     return PromptTemplate(
    #         template="""
    #     你是一个智能助手，负责帮助用户回答问题。请按照以下步骤处理：
    #
    #     1. **分析问题和上下文**：
    #        - 基于提供的上下文（如果有）和你的知识回答问题。
    #        - 如果答案来源于检索到的文档，请在回答中明确说明，例如：“根据提供的文档，……”。
    #
    #     2. **评估对话历史**：
    #        - 检查对话历史是否与当前问题相关（例如，是否涉及相同的话题、实体或问题背景）。
    #        - 如果对话历史与问题相关，请结合历史信息生成更准确的回答。
    #        - 如果对话历史无关（例如，仅包含问候或不相关的内容），忽略历史，仅基于上下文和问题回答。
    #
    #     3. **生成回答**：
    #        - 提供清晰、准确的回答，避免无关信息。
    #        - 如果上下文和历史消息均不足以回答问题，请回复：“信息不足，无法回答，请联系人工客服，电话：{phone}。”
    #     **对话历史**:
    #      {history}
    #     **上下文**:
    #      {context}
    #     **问题**:
    #      {question}
    #
    #     **回答**:
    #     """,
    #         input_variables=["context", "history", "question", "phone"],
    #     )

    @staticmethod
    def hyde_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
               假设你是用户，想了解以下问题，请生成一个简短的假设答案：  
               问题: {query}  
               假设答案:  
               """,
            #   定义输入变量
            input_variables=["query"],
        )

    #   定义子查询生成的 Prompt 模板
    @staticmethod
    def subquery_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
               将以下复杂查询分解为多个简单子查询，每行一个子查询，最多生成两个子查询（只保留子查询问题，其他的文本都不需要）：
               eg: 
               用户原始query："Milvus 和 Zilliz Cloud 在功能上有什么不同？
               子查询："Milvus 有哪些功能？"，"Zilliz Cloud 有哪些功能？"
               
               查询: {query}  
               子查询:  
               """,
            #   定义输入变量
            input_variables=["query"],
        )

    #   定义回溯问题生成的 Prompt 模板
    @staticmethod
    def backtracking_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
               将以下复杂查询简化为一个更简单的问题：  
               查询: {query}  
               简化问题:  
               """,
            #   定义输入变量
            input_variables=["query"],
        )



if __name__ == '__main__':

    print("========== 测试 rag_prompt ==========")
    rag_prompt = RAGPrompts.rag_prompt()
    rag_result = rag_prompt.format(
        context="Milvus 是一款开源向量数据库，主要用于存储和检索高维向量数据。",
        question="Milvus 是做什么的？",
        phone="400-123-4567"
    )
    print(rag_result)


    print("\n========== 测试 hyde_prompt ==========")
    hyde_prompt = RAGPrompts.hyde_prompt()
    hyde_result = hyde_prompt.format(
        query="Milvus 和 FAISS 有什么区别？"
    )
    print(hyde_result)


    print("\n========== 测试 subquery_prompt ==========")
    subquery_prompt = RAGPrompts.subquery_prompt()
    subquery_result = subquery_prompt.format(
        query="Milvus 和 Zilliz Cloud 在功能和部署方式上有什么不同？"
    )
    print(subquery_result)


    print("\n========== 测试 backtracking_prompt ==========")
    backtracking_prompt = RAGPrompts.backtracking_prompt()
    backtracking_result = backtracking_prompt.format(
        query="在 RAG 系统中，如何结合向量检索与倒排索引以提升召回率？"
    )
    print(backtracking_result)
