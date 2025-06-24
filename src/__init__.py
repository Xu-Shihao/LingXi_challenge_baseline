"""
LLM API 和诊断模型包
"""

from .llm_api import LLMAPIClient, LLMAPIError
from .diagnosis_models import DiagnosisResult, DiagnosisCategory, DiagnosisAnalysis

__version__ = "1.0.0"
__all__ = [
    "LLMAPIClient", 
    "LLMAPIError",
    "DiagnosisResult", 
    "DiagnosisCategory", 
    "DiagnosisAnalysis"
] 