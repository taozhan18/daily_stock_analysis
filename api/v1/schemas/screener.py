# -*- coding: utf-8 -*-
"""
股票筛选器请求/响应模型
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ScreenerRequest(BaseModel):
    """筛选请求"""
    top_n: int = Field(30, ge=1, le=200, description="返回数量")
    min_price: Optional[float] = Field(None, description="最低价格")
    min_turnover_rate: Optional[float] = Field(None, description="最低换手率(%)")
    pe_min: Optional[float] = Field(None, description="最低市盈率")
    pe_max: Optional[float] = Field(None, description="最高市盈率")
    min_market_cap_yi: Optional[float] = Field(None, description="最低市值(亿元)")
    exclude_st: Optional[bool] = Field(None, description="是否排除ST股")
    exclude_kcb: Optional[bool] = Field(None, description="是否排除科创板(688)")
    layer2_pool_size: Optional[int] = Field(None, ge=50, le=2000, description="Layer2候选池大小")
    max_workers: Optional[int] = Field(None, ge=1, le=20, description="并发拉取线程数")


class ScoredStockResult(BaseModel):
    """单只评分股票"""
    code: str = Field(..., description="股票代码")
    name: str = Field("", description="股票名称")
    score: float = Field(..., description="综合得分(0-100)")
    match_reasons: List[str] = Field(default_factory=list, description="匹配原因")
    indicators: Dict[str, Any] = Field(default_factory=dict, description="各维度得分")


class ScreenerResponse(BaseModel):
    """筛选结果"""
    total_market: int = Field(0, description="全市场股票数")
    layer1_passed: int = Field(0, description="粗筛通过数")
    layer2_passed: int = Field(0, description="精筛通过数")
    top_stocks: List[ScoredStockResult] = Field(default_factory=list, description="Top N 股票")
    elapsed_seconds: float = Field(0.0, description="耗时(秒)")
