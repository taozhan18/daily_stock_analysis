# -*- coding: utf-8 -*-
"""
股票筛选器接口

职责：
1. POST /api/v1/screener/run   — 执行筛选
2. GET  /api/v1/screener/result — 获取最近筛选结果
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from api.v1.schemas.screener import ScreenerRequest, ScreenerResponse, ScoredStockResult
from src.services.screener_service import ScreenerService

logger = logging.getLogger(__name__)

router = APIRouter()

_service: Optional[ScreenerService] = None


def _get_service() -> ScreenerService:
    global _service
    if _service is None:
        _service = ScreenerService()
    return _service


@router.post(
    "/run",
    response_model=ScreenerResponse,
    summary="执行全市场筛选",
    description="纯数值计算的两层漏斗筛选，零 AI 调用，预计 1-5 分钟完成。",
)
def run_screener(req: ScreenerRequest) -> ScreenerResponse:
    svc = _get_service()
    try:
        result = svc.run_screener(
            top_n=req.top_n,
            overrides=req.model_dump(exclude_none=True, exclude={"top_n", "layer2_pool_size", "max_workers"}),
            layer2_pool_size=req.layer2_pool_size,
            max_workers=req.max_workers,
        )
    except Exception as exc:
        logger.error("[Screener API] 筛选执行失败: %s", exc)
        raise HTTPException(status_code=500, detail=f"筛选执行失败: {exc}")

    return ScreenerResponse(
        total_market=result.total_market,
        layer1_passed=result.layer1_passed,
        layer2_passed=result.layer2_passed,
        top_stocks=[
            ScoredStockResult(
                code=s.code,
                name=s.name,
                score=s.score,
                match_reasons=s.match_reasons,
                indicators=s.indicators,
            )
            for s in result.top_stocks
        ],
        elapsed_seconds=result.elapsed_seconds,
    )


@router.get(
    "/result",
    response_model=Optional[ScreenerResponse],
    summary="获取最近筛选结果",
    description="返回今天最近一次筛选的结果，如果没有则返回 null。",
)
def get_screener_result() -> Optional[ScreenerResponse]:
    svc = _get_service()
    result = svc.get_last_result()
    if result is None:
        return None
    return ScreenerResponse(
        total_market=result.total_market,
        layer1_passed=result.layer1_passed,
        layer2_passed=result.layer2_passed,
        top_stocks=[
            ScoredStockResult(
                code=s.code, name=s.name, score=s.score,
                match_reasons=s.match_reasons, indicators=s.indicators,
            )
            for s in result.top_stocks
        ],
        elapsed_seconds=result.elapsed_seconds,
    )
