import os
import json
import time
import requests
from typing import TypeVar, Type, Optional, Dict, Any, List
from pydantic import BaseModel, ValidationError
import warnings
warnings.filterwarnings('ignore')

T = TypeVar('T', bound=BaseModel)

class LLMAPIError(Exception):
    """LLM API调用错误"""
    pass

class LLMAPIClient:
    """通用的LLM API客户端，支持结构化输出"""
    
    def __init__(self, api_base: str = "http://127.0.0.1:9019/v1", 
                 model_name: str = "/tcci_mnt/shihao/models/Qwen3-8B"):
        """
        初始化LLM API客户端
        
        Args:
            api_base: API基础URL
            model_name: 模型名称或路径
        """
        self.api_base = api_base
        self.model_name = model_name
        self.headers = {
            "Content-Type": "application/json",
        }
        
        # 验证API连接
        self.test_connection()
    
    def test_connection(self) -> bool:
        """测试API连接"""
        try:
            response = requests.get(f"{self.api_base}/models", headers=self.headers, timeout=10)
            if response.status_code == 200:
                print(f"✓ API连接成功: {self.api_base}")
                print(f"✓ 模型: {self.model_name}")
                return True
            else:
                print(f"⚠️ API连接警告: 状态码 {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API连接失败: {e}")
            print("请确保LLM模型服务正在运行")
            return False
    
    def _call_raw_api(self, messages: List[Dict[str, str]], 
                     temperature: float = 0.1, 
                     max_tokens: int = 200,
                     top_p: float = 0.9,
                     max_retries: int = 3,
                     retry_delay: float = 1.0) -> Optional[str]:
        """
        调用原始API接口
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: top_p参数
            max_retries: 最大重试次数
            retry_delay: 重试延迟
            
        Returns:
            API响应的文本内容，失败时返回None
        """
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=self.headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        content = result['choices'][0]['message']['content'].strip()
                        return content
                    else:
                        print(f"API响应格式错误: {result}")
                        
                else:
                    print(f"API请求失败: 状态码 {response.status_code}")
                    print(f"响应内容: {response.text}")
                    
            except Exception as e:
                print(f"API调用异常 (尝试 {attempt + 1}/{max_retries}): {e}")
                
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
        
        return None
    
    def generate_structured_prompt(self, user_message: str, 
                                 output_schema: Type[T], 
                                 system_message: Optional[str] = None) -> str:
        """
        生成结构化输出的提示词
        
        Args:
            user_message: 用户消息内容
            output_schema: Pydantic模型类
            system_message: 可选的系统消息
            
        Returns:
            完整的提示词
        """
        # 获取schema的JSON格式
        schema_json = output_schema.model_json_schema()
        
        # 构建提示词
        prompt_parts = []
        
        if system_message:
            prompt_parts.append(f"系统指令: {system_message}")
        
        prompt_parts.extend([
            f"用户请求: {user_message}",
            "",
            "请严格按照以下JSON格式输出结果:",
            json.dumps(schema_json, indent=2, ensure_ascii=False),
            "",
            "要求:",
            "1. 只输出JSON格式的结果，不要包含任何其他文字",
            "2. JSON必须完整且格式正确",
            "3. 所有字段都必须包含",
            "4. 字符串值不能为空",
            "",
            "JSON输出:"
        ])
        
        return "\n".join(prompt_parts)
    
    def call_with_structured_output(self, 
                                  user_message: str,
                                  output_schema: Type[T],
                                  system_message: Optional[str] = None,
                                  temperature: float = 0.1,
                                  max_tokens: int = 200,
                                  max_retries: int = 3) -> Optional[T]:
        """
        调用API并返回结构化输出
        
        Args:
            user_message: 用户消息内容
            output_schema: Pydantic模型类
            system_message: 可选的系统消息
            temperature: 温度参数
            max_tokens: 最大token数
            max_retries: 最大重试次数
            
        Returns:
            解析后的Pydantic模型实例，失败时返回None
        """
        # 生成结构化提示词
        prompt = self.generate_structured_prompt(user_message, output_schema, system_message)
        
        messages = [{"role": "user", "content": prompt}]
        
        # 调用API
        response_text = self._call_raw_api(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries
        )
        
        if not response_text:
            print("API调用失败")
            return None
        
        # 尝试解析JSON
        try:
            # 清理响应文本，提取JSON部分
            cleaned_response = self._extract_json_from_response(response_text)
            
            if not cleaned_response:
                print(f"无法从响应中提取JSON: {response_text}")
                return None
            
            # 解析JSON
            json_data = json.loads(cleaned_response)
            
            # 创建Pydantic模型实例
            result = output_schema(**json_data)
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"原始响应: {response_text}")
            return None
        except ValidationError as e:
            print(f"Pydantic验证错误: {e}")
            print(f"解析的数据: {json_data}")
            return None
        except Exception as e:
            print(f"解析结构化输出时发生未知错误: {e}")
            print(f"原始响应: {response_text}")
            return None
    
    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        """
        从响应文本中提取JSON部分
        
        Args:
            response_text: API响应文本
            
        Returns:
            提取的JSON字符串，失败时返回None
        """
        # 尝试直接解析
        try:
            json.loads(response_text.strip())
            return response_text.strip()
        except:
            pass
        
        # 查找JSON块
        import re
        
        # 匹配 ```json ... ``` 格式
        json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_block_pattern, response_text, re.DOTALL)
        if match:
            return match.group(1)
        
        # 查找第一个完整的JSON对象
        start_idx = response_text.find('{')
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            
            for i, char in enumerate(response_text[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            potential_json = response_text[start_idx:end_idx]
            try:
                json.loads(potential_json)
                return potential_json
            except:
                pass
        
        return None
    
    def call_simple(self, 
                   user_message: str,
                   system_message: Optional[str] = None,
                   temperature: float = 0.1,
                   max_tokens: int = 200,
                   max_retries: int = 3) -> Optional[str]:
        """
        简单的API调用，返回原始文本
        
        Args:
            user_message: 用户消息内容
            system_message: 可选的系统消息
            temperature: 温度参数
            max_tokens: 最大token数
            max_retries: 最大重试次数
            
        Returns:
            API响应的文本内容，失败时返回None
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})
        
        return self._call_raw_api(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries
        ) 