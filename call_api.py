# -*- coding: utf-8 -*-
"""
简单的API调用脚本
用于测试和演示如何调用问答系统API
"""

import requests
import json
import time
from typing import Optional, Dict, Any

# API 服务器地址
API_BASE_URL = "http://localhost:8000"

class QAAPIClient:
    """问答系统API客户端"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        """
        初始化API客户端
        
        Args:
            base_url: API服务器地址，默认为本地8000端口
        """
        self.base_url = base_url
        self.session_id = None
        self.session = requests.Session()
    
    def query(
        self, 
        query: str, 
        source_filter: Optional[str] = None,
        use_stream: bool = True
    ) -> Dict[str, Any]:
        """
        发送查询请求到API
        
        Args:
            query: 用户问题
            source_filter: 学科过滤条件（可选）
            use_stream: 是否使用流式响应
        
        Returns:
            API响应结果
        """
        # 构建请求URL
        url = f"{self.base_url}/query"
        
        # 构建请求体
        payload = {
            "query": query,
            "source_filter": source_filter,
            "session_id": self.session_id
        }
        
        print(f"\n📤 发送查询: {query}")
        print(f"🔗 请求URL: {url}")
        print(f"📋 请求体: {json.dumps(payload, ensure_ascii=False, indent=2)}")
        
        try:
            # 发送POST请求
            response = self.session.post(
                url,
                json=payload,
                timeout=30,
                stream=use_stream
            )
            
            # 检查响应状态码
            if response.status_code != 200:
                print(f"❌ 错误: HTTP {response.status_code}")
                print(f"📝 错误信息: {response.text}")
                return {"error": response.text, "status_code": response.status_code}
            
            # 处理流式响应
            if use_stream:
                return self._handle_stream_response(response)
            else:
                return response.json()
        
        except requests.exceptions.ConnectionError:
            print("❌ 连接失败: 无法连接到API服务器")
            print(f"   请确保服务器运行在 {self.base_url}")
            return {"error": "连接失败"}
        
        except requests.exceptions.Timeout:
            print("❌ 请求超时: API服务器响应超时")
            return {"error": "请求超时"}
        
        except Exception as e:
            print(f"❌ 发生错误: {str(e)}")
            return {"error": str(e)}
    
    def _handle_stream_response(self, response) -> Dict[str, Any]:
        """
        处理流式响应
        
        Args:
            response: requests响应对象
        
        Returns:
            完整的响应数据
        """
        print("\n📥 接收流式响应:")
        print("-" * 50)
        
        full_response = {
            "answer": "",
            "sources": [],
            "session_id": None,
            "metadata": {}
        }
        
        try:
            for line in response.iter_lines():
                if line:
                    # 解码行数据
                    line_str = line.decode('utf-8') if isinstance(line, bytes) else line
                    
                    # 处理SSE格式的数据
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]  # 移除 "data: " 前缀
                        
                        try:
                            data = json.loads(data_str)
                            
                            # 处理不同类型的数据
                            if "token" in data:
                                # 流式token
                                print(data["token"], end="", flush=True)
                                full_response["answer"] += data["token"]
                            
                            elif "sources" in data:
                                # 来源信息
                                full_response["sources"] = data["sources"]
                            
                            elif "session_id" in data:
                                # 会话ID
                                self.session_id = data["session_id"]
                                full_response["session_id"] = data["session_id"]
                            
                            elif "metadata" in data:
                                # 元数据
                                full_response["metadata"] = data["metadata"]
                        
                        except json.JSONDecodeError:
                            print(f"\n⚠️  无法解析JSON: {data_str}")
            
            print("\n" + "-" * 50)
            return full_response
        
        except Exception as e:
            print(f"\n❌ 流式响应处理错误: {str(e)}")
            return {"error": str(e)}
    
    def multi_turn_conversation(self, queries: list):
        """
        进行多轮对话
        
        Args:
            queries: 问题列表
        """
        print("\n" + "=" * 60)
        print("🤖 开始多轮对话")
        print("=" * 60)
        
        for i, query in enumerate(queries, 1):
            print(f"\n【第 {i} 轮】")
            result = self.query(query)
            
            if "error" in result:
                print(f"❌ 查询失败: {result['error']}")
                break
            
            # 显示结果摘要
            if "answer" in result:
                answer_preview = result["answer"][:100] + "..." if len(result["answer"]) > 100 else result["answer"]
                print(f"\n✅ 答案: {answer_preview}")
            
            if "sources" in result and result["sources"]:
                print(f"📚 来源数量: {len(result['sources'])}")
            
            if "session_id" in result:
                print(f"🔑 会话ID: {result['session_id']}")
            
            # 等待一下，避免请求过快
            time.sleep(1)


def main():
    """主函数 - 演示API调用"""
    
    # 创建API客户端
    client = QAAPIClient()
    
    # 示例1: 单个查询
    print("\n" + "=" * 60)
    print("示例1: 单个查询")
    print("=" * 60)
    result = client.query("什么是人工智能？")
    
    # 示例2: 带学科过滤的查询
    print("\n" + "=" * 60)
    print("示例2: 带学科过滤的查询")
    print("=" * 60)
    result = client.query("Python是什么？", source_filter="ai")
    
    # 示例3: 多轮对话
    print("\n" + "=" * 60)
    print("示例3: 多轮对话")
    print("=" * 60)
    queries = [
        "什么是机器学习？",
        "机器学习有哪些应用？",
        "如何学习机器学习？"
    ]
    client.multi_turn_conversation(queries)
    
    print("\n" + "=" * 60)
    print("✅ 所有示例执行完成")
    print("=" * 60)


if __name__ == "__main__":
    # 运行主函数
    main()
    
    # 或者交互式使用
    # client = QAAPIClient()
    # while True:
    #     query = input("\n请输入问题 (输入 'quit' 退出): ").strip()
    #     if query.lower() == 'quit':
    #         break
    #     result = client.query(query)
