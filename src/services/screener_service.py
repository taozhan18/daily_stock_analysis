# -*- coding: utf-8 -*-
"""
股票筛选器服务层
"""

import json
import logging
import os
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from src.core.screener import ScoredStock, ScreenerResult, StockScreener

logger = logging.getLogger(__name__)


class ScreenerService:
    """封装筛选器，提供结果缓存与通知能力。"""

    def __init__(self):
        self._last_result: Optional[ScreenerResult] = None
        self._last_run_date: Optional[date] = None

    def run_screener(
        self,
        top_n: int = 30,
        overrides: Optional[Dict[str, Any]] = None,
        layer2_pool_size: Optional[int] = None,
        max_workers: Optional[int] = None,
    ) -> ScreenerResult:
        layer1_overrides = {}
        if overrides:
            for key in (
                "min_price", "min_turnover_rate", "pe_min", "pe_max",
                "min_market_cap_yi", "volume_ratio_min", "volume_ratio_max",
                "change_pct_min", "change_pct_max", "max_amplitude", "exclude_st",
                "exclude_kcb",
            ):
                if key in overrides and overrides[key] is not None:
                    layer1_overrides[key] = overrides[key]

        pool_size = layer2_pool_size or int(os.environ.get("SCREENER_LAYER2_POOL_SIZE", "500"))
        workers = max_workers or int(os.environ.get("SCREENER_MAX_WORKERS", "5"))

        screener = StockScreener(
            layer1_overrides=layer1_overrides or None,
            layer2_pool_size=pool_size,
            max_workers=workers,
        )
        result = screener.run(top_n=top_n)
        self._last_result = result
        self._last_run_date = date.today()
        return result

    def get_last_result(self) -> Optional[ScreenerResult]:
        if self._last_result and self._last_run_date == date.today():
            return self._last_result
        return None

    def format_report(self, result: ScreenerResult) -> str:
        """将筛选结果格式化为 Markdown 报告。"""
        lines = [
            f"## 全市场筛选报告",
            f"",
            f"- 全市场: **{result.total_market}** 只",
            f"- 粗筛通过: **{result.layer1_passed}** 只",
            f"- 精筛通过: **{result.layer2_passed}** 只",
            f"- 耗时: **{result.elapsed_seconds:.1f}s**",
            f"",
            f"### Top {len(result.top_stocks)} 候选股",
            f"",
            f"| # | 代码 | 名称 | 得分 | 匹配原因 |",
            f"|---|------|------|------|----------|",
        ]
        for i, s in enumerate(result.top_stocks, 1):
            reasons = "、".join(s.match_reasons[:3]) if s.match_reasons else "-"
            lines.append(f"| {i} | {s.code} | {s.name} | {s.score:.1f} | {reasons} |")
        return "\n".join(lines)
