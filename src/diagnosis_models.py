from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum

class DiagnosisCategory(str, Enum):
    """诊断类别枚举"""
    DEPRESSION = "抑郁相关"
    BIPOLAR = "双相情感障碍"
    ANXIETY = "焦虑相关"
    ADHD = "多动障碍"

class DiagnosisResult(BaseModel):
    """诊断结果模型"""
    category: DiagnosisCategory = Field(
        description="诊断类别，必须是以下之一：抑郁相关、双相情感障碍、焦虑相关、多动障碍"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="诊断置信度，范围0-1"
    )
    reasoning: str = Field(
        min_length=10,
        description="诊断推理过程，说明为什么选择该诊断类别"
    )
    key_symptoms: list[str] = Field(
        min_items=1,
        description="关键症状列表，至少包含一个症状"
    )

class DiagnosisAnalysis(BaseModel):
    """详细诊断分析模型"""
    primary_diagnosis: DiagnosisResult = Field(
        description="主要诊断结果"
    )
    alternative_diagnoses: Optional[list[DiagnosisResult]] = Field(
        default=None,
        description="可能的替代诊断，按置信度排序"
    )
    severity_assessment: Literal["轻度", "中度", "重度"] = Field(
        description="严重程度评估"
    )
    risk_factors: list[str] = Field(
        default_factory=list,
        description="识别的风险因素"
    )
    recommendations: list[str] = Field(
        min_items=1,
        description="治疗建议或进一步评估建议"
    ) 