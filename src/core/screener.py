# -*- coding: utf-8 -*-
"""
===================================
全市场股票筛选器 (Stock Screener)
===================================

纯数值计算的两层漏斗筛选，零 AI 调用：
  Layer 1 — 实时数据粗筛（全市场一次 API 拉取，DataFrame 过滤）
  Layer 2 — 技术指标精筛打分（批量拉日K线，pandas 向量化计算）

使用方式：
    from src.core.screener import StockScreener
    screener = StockScreener()
    result = screener.run(top_n=30)
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────

@dataclass
class ScoredStock:
    code: str
    name: str
    score: float
    indicators: Dict[str, Any] = field(default_factory=dict)
    match_reasons: List[str] = field(default_factory=list)


@dataclass
class ScreenerResult:
    total_market: int = 0
    layer1_passed: int = 0
    layer2_passed: int = 0
    top_stocks: List[ScoredStock] = field(default_factory=list)
    elapsed_seconds: float = 0.0


# ── Layer 1 默认过滤条件 ──────────────────────────────────────

_LAYER1_DEFAULTS = {
    "min_price": 3.0,
    "min_turnover_rate": 0.5,
    "pe_min": 0.0,
    "pe_max": 200.0,
    "min_market_cap_yi": 50.0,
    "volume_ratio_min": 0.5,
    "volume_ratio_max": 5.0,
    "change_pct_min": -8.0,
    "change_pct_max": 8.0,
    "max_amplitude": 10.0,
    "exclude_st": True,
    "exclude_kcb": True,
}

# ── Layer 2 默认权重 ──────────────────────────────────────────

_WEIGHT_DEFAULTS = {
    "ma_alignment": 25,
    "golden_cross": 15,
    "macd": 15,
    "rsi": 10,
    "bollinger": 10,
    "volume": 15,
    "bias": 10,
}


# ── 核心类 ────────────────────────────────────────────────────

class StockScreener:
    """两层漏斗全市场筛选器。"""

    def __init__(
        self,
        layer1_overrides: Optional[Dict[str, Any]] = None,
        weight_overrides: Optional[Dict[str, int]] = None,
        layer2_pool_size: int = 500,
        max_workers: int = 5,
    ):
        self.layer1_cfg = {**_LAYER1_DEFAULTS, **(layer1_overrides or {})}
        total_w = sum({**_WEIGHT_DEFAULTS, **(weight_overrides or {})}.values())
        self.weights = {**_WEIGHT_DEFAULTS, **(weight_overrides or {})}
        if total_w > 0:
            self.weights = {k: v / total_w for k, v in self.weights.items()}
        self.layer2_pool_size = max(50, min(layer2_pool_size, 2000))
        self.max_workers = max(1, min(max_workers, 20))

    # ── 主入口 ────────────────────────────────────────────────

    def run(self, top_n: int = 30) -> ScreenerResult:
        t0 = time.time()
        result = ScreenerResult()

        # Layer 1
        df_all = self._fetch_batch_realtime()
        if df_all is None or df_all.empty:
            logger.warning("[Screener] 未能获取全市场实时数据")
            return result
        result.total_market = len(df_all)

        df_filtered = self._layer1_filter(df_all)
        result.layer1_passed = len(df_filtered)
        logger.info("[Screener] Layer 1: %d → %d", result.total_market, result.layer1_passed)

        if df_filtered.empty:
            return result

        # Layer 2
        scored = self._layer2_score(df_filtered)
        result.top_stocks = scored[:top_n]
        result.layer2_passed = len(scored)
        result.elapsed_seconds = time.time() - t0
        logger.info(
            "[Screener] Layer 2: %d → Top %d, 耗时 %.1fs",
            result.layer1_passed, len(result.top_stocks), result.elapsed_seconds,
        )
        return result

    # ── Layer 1: 实时数据粗筛 ─────────────────────────────────

    def _fetch_batch_realtime(self) -> Optional[pd.DataFrame]:
        """通过多种数据源拉全市场实时行情 DataFrame（带重试）。

        数据源优先级：
          - SCREENER_SOURCE_PRIORITY 环境变量可控制，默认 "tushare,efinance,akshare"
          - 有 TUSHARE_TOKEN 时 Tushare 优先（不受东方财富封IP影响）
        """
        max_retries = 3

        # 解析数据源优先级
        priority_str = os.environ.get("SCREENER_SOURCE_PRIORITY", "tushare,efinance,akshare")
        source_order = [s.strip().lower() for s in priority_str.split(",")]

        for source in source_order:
            if source == "tushare":
                df = self._fetch_via_tushare()
                if df is not None and not df.empty:
                    return df

            elif source == "efinance":
                # 尝试复用 EfinanceFetcher 的缓存机制
                try:
                    from data_provider.efinance_fetcher import _realtime_cache
                    current_time = time.time()
                    if (
                        _realtime_cache.get('data') is not None
                        and current_time - _realtime_cache.get('timestamp', 0) < _realtime_cache.get('ttl', 600)
                    ):
                        cached_df = _realtime_cache['data']
                        if cached_df is not None and not cached_df.empty:
                            logger.info("[Screener] 复用 efinance 缓存: %d 只股票", len(cached_df))
                            return cached_df
                except Exception:
                    pass

                for attempt in range(1, max_retries + 1):
                    try:
                        import efinance as ef
                        from data_provider.efinance_fetcher import _ef_call_with_timeout
                        logger.info("[Screener] 调用 ef.stock.get_realtime_quotes()... (第%d次)", attempt)
                        df = _ef_call_with_timeout(ef.stock.get_realtime_quotes)
                        if df is not None and not df.empty:
                            logger.info("[Screener] efinance 返回 %d 只股票", len(df))
                            return df
                        logger.warning("[Screener] efinance 返回空数据")
                    except Exception as exc:
                        logger.warning("[Screener] efinance 批量获取失败 (第%d次): %s", attempt, exc)
                    if attempt < max_retries:
                        time.sleep(5 * attempt)

            elif source == "akshare":
                for attempt in range(1, max_retries + 1):
                    try:
                        import akshare as ak
                        logger.info("[Screener] 调用 ak.stock_zh_a_spot_em()... (第%d次)", attempt)
                        df = ak.stock_zh_a_spot_em()
                        if df is not None and not df.empty:
                            logger.info("[Screener] akshare 返回 %d 只股票", len(df))
                            return df
                        logger.warning("[Screener] akshare 返回空数据")
                    except Exception as exc:
                        logger.warning("[Screener] akshare 批量获取失败 (第%d次): %s", attempt, exc)
                    if attempt < max_retries:
                        time.sleep(5 * attempt)

        logger.error("[Screener] 所有数据源重试均失败")
        return None

    def _fetch_via_tushare(self) -> Optional[pd.DataFrame]:
        """通过 Tushare Pro 按交易日拉全市场日线数据（不走东方财富，不受IP封禁影响）。

        使用 daily(trade_date=...) 一次获取所有股票当日数据，
        再用 stk_factorpro 获取 PE/市值/换手率等。
        """
        tushare_token = os.environ.get("TUSHARE_TOKEN", "").strip()
        if not tushare_token:
            logger.info("[Screener] Tushare: 未配置 TUSHARE_TOKEN，跳过")
            return None

        try:
            from data_provider.tushare_fetcher import _TushareHttpClient
            client = _TushareHttpClient(token=tushare_token, timeout=30)
        except Exception as exc:
            logger.warning("[Screener] Tushare 客户端初始化失败: %s", exc)
            return None

        # 确定最近交易日
        from datetime import datetime, timedelta
        today = datetime.now()
        # Tushare daily 在盘后才更新，盘中可能没有当天数据，往前尝试3天
        for days_back in range(0, 4):
            check_date = today - timedelta(days=days_back)
            trade_date = check_date.strftime("%Y%m%d")
            try:
                logger.info("[Screener] Tushare: 尝试 trade_date=%s", trade_date)
                df = client.query("daily", trade_date=trade_date)
                if df is None or df.empty:
                    continue

                logger.info("[Screener] Tushare daily 返回 %d 条记录", len(df))

                # 获取股票基本信息（名称）
                stock_info = None
                try:
                    stock_info = client.query("stock_basic", exchange="", list_status="L", fields="ts_code,name,industry")
                except Exception:
                    pass

                # 尝试获取 PE/市值/换手率（stk_factorpro 需要较高积分）
                factor_df = None
                try:
                    factor_df = client.query("stk_factorpro", trade_date=trade_date)
                except Exception:
                    # 降级: 尝试 stk_factor
                    try:
                        factor_df = client.query("stk_factor", trade_date=trade_date)
                    except Exception:
                        pass

                # 合并数据
                df = self._merge_tushare_data(df, stock_info, factor_df)
                if df is not None and not df.empty:
                    logger.info("[Screener] Tushare 合并后 %d 只股票", len(df))
                    return df

            except Exception as exc:
                logger.warning("[Screener] Tushare trade_date=%s 失败: %s", trade_date, exc)

        logger.warning("[Screener] Tushare 所有日期均失败")
        return None

    @staticmethod
    def _merge_tushare_data(daily_df: pd.DataFrame,
                            stock_info: Optional[pd.DataFrame],
                            factor_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """将 Tushare daily / stock_basic / stk_factorpro 合并为统一格式。"""
        # daily 返回: ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount
        result = daily_df.copy()
        result["code"] = result["ts_code"].str.replace(r"\.(SH|SZ|BJ)$", "", regex=True)

        if stock_info is not None and not stock_info.empty:
            name_map = dict(zip(stock_info["ts_code"], stock_info["name"]))
            result["name"] = result["ts_code"].map(name_map)
        else:
            result["name"] = ""

        # 重命名列以匹配 _normalize_columns
        result = result.rename(columns={
            "close": "price",
            "pct_chg": "change_pct",
            "vol": "volume",
            "open": "open_price",
        })

        # amount 单位是千元，转元
        if "amount" in result.columns:
            result["amount"] = result["amount"] * 1000

        # 合并 factor 数据（PE/市值/换手率）
        if factor_df is not None and not factor_df.empty and "ts_code" in factor_df.columns:
            merge_cols = ["ts_code"]
            factor_rename = {}
            if "pe_ttm" in factor_df.columns:
                factor_rename["pe_ttm"] = "pe_ratio"
            elif "pe" in factor_df.columns:
                factor_rename["pe"] = "pe_ratio"
            if "total_mv" in factor_df.columns:
                factor_rename["total_mv"] = "total_mv"
            if "turnover_rate" in factor_df.columns:
                factor_rename["turnover_rate"] = "turnover_rate"
            if "volume_ratio" in factor_df.columns:
                factor_rename["volume_ratio"] = "volume_ratio"

            if factor_rename:
                factor_subset = factor_df[merge_cols + list(factor_rename.keys())].copy()
                factor_subset = factor_subset.rename(columns=factor_rename)
                result = result.merge(factor_subset, on="ts_code", how="left")

        # 总市值单位是万元，转亿元
        if "total_mv" in result.columns:
            result["total_mv"] = result["total_mv"] / 10000

        return result

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """统一中文/英文列名为标准英文名。"""
        col_map = {
            "股票代码": "code", "代码": "code", "stock_code": "code",
            "股票名称": "name", "名称": "name",
            "最新价": "price", "close": "price",
            "涨跌幅": "change_pct", "pct_chg": "change_pct", "涨跌幅(%)": "change_pct",
            "成交量": "volume", "vol": "volume",
            "成交额": "amount",
            "换手率": "turnover_rate", "换手率(%)": "turnover_rate",
            "振幅": "amplitude", "振幅(%)": "amplitude",
            "最高": "high",
            "最低": "low",
            "开盘": "open_price", "开盘价": "open_price", "open": "open_price",
            "量比": "volume_ratio",
            "市盈率": "pe_ratio", "市盈率-动态": "pe_ratio",
            "总市值": "total_mv",
            "流通市值": "circ_mv",
            "昨收": "pre_close", "昨收价": "pre_close",
            "涨跌额": "change_amount",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        return df

    def _layer1_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Layer 1: 基于实时数据的规则过滤。"""
        df = self._normalize_columns(df.copy())
        cfg = self.layer1_cfg

        # 排除 ST
        if cfg["exclude_st"] and "name" in df.columns:
            before = len(df)
            df = df[~df["name"].str.contains("ST|\\*ST|退", case=False, na=False)]
            logger.debug("[Screener] 排除ST: %d → %d", before, len(df))

        # 排除科创板（代码以 688 开头）
        if cfg["exclude_kcb"] and "code" in df.columns:
            before = len(df)
            df = df[~df["code"].astype(str).str.startswith("688")]
            logger.debug("[Screener] 排除科创板: %d → %d", before, len(df))

        # 价格
        if "price" in df.columns and cfg["min_price"] is not None:
            df = df[pd.to_numeric(df["price"], errors="coerce") >= cfg["min_price"]]

        # 换手率
        if "turnover_rate" in df.columns and cfg["min_turnover_rate"] is not None:
            df = df[pd.to_numeric(df["turnover_rate"], errors="coerce") >= cfg["min_turnover_rate"]]

        # PE
        if "pe_ratio" in df.columns:
            pe = pd.to_numeric(df["pe_ratio"], errors="coerce")
            mask = pd.Series(True, index=df.index)
            if cfg["pe_min"] is not None:
                mask &= (pe >= cfg["pe_min"])
            if cfg["pe_max"] is not None:
                mask &= (pe <= cfg["pe_max"])
            df = df[mask]

        # 市值（亿）
        if "total_mv" in df.columns and cfg["min_market_cap_yi"] is not None:
            mv = pd.to_numeric(df["total_mv"], errors="coerce")
            threshold = cfg["min_market_cap_yi"] * 1e8
            df = df[mv >= threshold]

        # 量比
        if "volume_ratio" in df.columns:
            vr = pd.to_numeric(df["volume_ratio"], errors="coerce")
            vr_mask = pd.Series(True, index=df.index)
            if cfg["volume_ratio_min"] is not None:
                vr_mask &= (vr >= cfg["volume_ratio_min"])
            if cfg["volume_ratio_max"] is not None:
                vr_mask &= (vr <= cfg["volume_ratio_max"])
            df = df[vr_mask]

        # 涨跌幅
        if "change_pct" in df.columns:
            cp = pd.to_numeric(df["change_pct"], errors="coerce")
            cp_mask = pd.Series(True, index=df.index)
            if cfg["change_pct_min"] is not None:
                cp_mask &= (cp >= cfg["change_pct_min"])
            if cfg["change_pct_max"] is not None:
                cp_mask &= (cp <= cfg["change_pct_max"])
            df = df[cp_mask]

        # 振幅
        if "amplitude" in df.columns and cfg["max_amplitude"] is not None:
            df = df[pd.to_numeric(df["amplitude"], errors="coerce") <= cfg["max_amplitude"]]

        df = df.dropna(subset=["code", "price"])
        return df

    # ── Layer 2: 技术指标精筛打分 ─────────────────────────────

    def _layer2_score(self, df_filtered: pd.DataFrame) -> List[ScoredStock]:
        """对 Layer 1 通过的股票拉日K线，计算技术指标并打分。"""
        codes = df_filtered["code"].tolist()
        code_name_map = dict(zip(df_filtered["code"], df_filtered.get("name", pd.Series([str(c) for c in codes]))))

        # Layer 1 可能返回数千只，先用实时数据排序取前 N 减少日K线拉取量
        if len(codes) > self.layer2_pool_size:
            df_sorted = df_filtered.copy()
            if "change_pct" in df_sorted.columns:
                df_sorted["_sort_key"] = pd.to_numeric(df_sorted["change_pct"], errors="coerce").abs()
                df_sorted = df_sorted.sort_values("_sort_key", ascending=False)
            codes = df_sorted["code"].tolist()[:self.layer2_pool_size]
            logger.info("[Screener] Layer 2 pool reduced: %d → %d (pool_size=%d)", len(df_filtered), len(codes), self.layer2_pool_size)

        # 批量获取日K线
        histories = self._fetch_daily_histories(codes)

        scored: List[ScoredStock] = []
        success_count = 0
        for code, hist_df in histories.items():
            if hist_df is None or len(hist_df) < 20:
                continue
            success_count += 1
            try:
                indicators, reasons = self._calc_all_indicators(hist_df)
                score = self._weighted_score(indicators)
                scored.append(ScoredStock(
                    code=code,
                    name=str(code_name_map.get(code, code)),
                    score=round(score, 1),
                    indicators=indicators,
                    match_reasons=reasons,
                ))
            except Exception as exc:
                logger.debug("[Screener] %s 指标计算异常: %s", code, exc)

        logger.info("[Screener] 日K线获取成功: %d/%d", success_count, len(codes))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored

    def _fetch_daily_histories(self, codes: List[str]) -> Dict[str, Optional[pd.DataFrame]]:
        """批量拉取日K线。优先尝试 Tushare 批量接口，失败降级到逐股并发。"""
        # 1. 尝试 Tushare 批量获取（一次 API 调用获取所有股票的 60 天数据）
        tushare_result = self._fetch_histories_via_tushare(codes)
        if tushare_result is not None:
            missing = [c for c in codes if c not in tushare_result or tushare_result[c] is None]
            if len(missing) < len(codes) * 0.3:  # 成功率 > 70% 就用 Tushare 结果
                if missing:
                    logger.info("[Screener] Tushare 批量K线缺少 %d 只，逐股补充", len(missing))
                    fallback = self._fetch_histories_via_manager(missing)
                    for code, df in fallback.items():
                        if df is not None:
                            tushare_result[code] = df
                return tushare_result

        # 2. 降级：逐股并发获取
        return self._fetch_histories_via_manager(codes)

    def _fetch_histories_via_tushare(self, codes: List[str]) -> Optional[Dict[str, Optional[pd.DataFrame]]]:
        """通过 Tushare daily 批量获取 60 天K线（按日期逐天拉全市场）。"""
        tushare_token = os.environ.get("TUSHARE_TOKEN", "").strip()
        if not tushare_token:
            return None

        try:
            from data_provider.tushare_fetcher import _TushareHttpClient
            client = _TushareHttpClient(token=tushare_token, timeout=30)
        except Exception:
            return None

        from datetime import datetime, timedelta

        # 找到最近的交易日，往前拉 60 天
        # 先获取交易日历
        try:
            today_str = datetime.now().strftime("%Y%m%d")
            cal_df = client.query("trade_cal", exchange="SSE", start_date=(datetime.now() - timedelta(days=120)).strftime("%Y%m%d"), end_date=today_str, is_open="1")
            if cal_df is None or cal_df.empty:
                return None
            trade_dates = sorted(cal_df["cal_date"].tolist())[-60:]
        except Exception as exc:
            logger.warning("[Screener] Tushare 获取交易日历失败: %s", exc)
            return None

        # Tushare 每分钟调用次数有限，批量拉取每天的全市场数据
        code_set = set(codes)
        all_data: Dict[str, List[dict]] = {}

        for i, trade_date in enumerate(trade_dates):
            try:
                if i > 0 and i % 200 == 0:
                    time.sleep(60)  # Tushare 频率限制：每分钟 200 次
                df = client.query("daily", trade_date=trade_date)
                if df is None or df.empty:
                    continue
                for _, row in df.iterrows():
                    code = row["ts_code"].split(".")[0] if "." in str(row["ts_code"]) else str(row["ts_code"])
                    if code in code_set:
                        if code not in all_data:
                            all_data[code] = []
                        all_data[code].append(row.to_dict())
            except Exception as exc:
                logger.warning("[Screener] Tushare daily trade_date=%s 失败: %s", trade_date, exc)
                continue

        if not all_data:
            return None

        # 转换为 DataFrame
        results: Dict[str, Optional[pd.DataFrame]] = {}
        for code in codes:
            if code in all_data and len(all_data[code]) >= 20:
                df = pd.DataFrame(all_data[code])
                df = df.sort_values("trade_date")
                # 标准化列名
                if "close" not in df.columns:
                    pass
                else:
                    if "vol" in df.columns:
                        df["volume"] = df["vol"]
                    if "amount" in df.columns:
                        df["amount"] = df["amount"] * 1000  # 千元→元
                    results[code] = df
            else:
                results[code] = None

        fetched = sum(1 for v in results.values() if v is not None)
        logger.info("[Screener] Tushare 批量K线: %d/%d 成功", fetched, len(codes))
        return results

    def _fetch_histories_via_manager(self, codes: List[str]) -> Dict[str, Optional[pd.DataFrame]]:
        """逐股并发拉取日K线，通过 data_provider 管理器。"""
        try:
            from data_provider.base import DataFetcherManager
            manager = DataFetcherManager()
        except Exception as exc:
            logger.error("[Screener] DataFetcherManager 初始化失败: %s", exc)
            return {code: None for code in codes}

        def _fetch_one(code: str) -> tuple:
            try:
                df, _source = manager.get_daily_data(code, days=60)
                return code, df if df is not None and len(df) >= 20 else None
            except Exception:
                return code, None

        results: Dict[str, Optional[pd.DataFrame]] = {}
        total = len(codes)
        logger.info("[Screener] 开始并发拉取日K线: %d 只, max_workers=%d", total, self.max_workers)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_fetch_one, code): code for code in codes}
            done_count = 0
            for future in as_completed(futures):
                code, df = future.result()
                results[code] = df
                done_count += 1
                if done_count % 100 == 0:
                    fetched = sum(1 for v in results.values() if v is not None)
                    logger.info("[Screener] 日K线进度: %d/%d (成功 %d)", done_count, total, fetched)

        return results

    # ── 技术指标计算 ───────────────────────────────────────────

    def _calc_all_indicators(self, df: pd.DataFrame) -> tuple:
        """计算全部技术指标，返回 (indicators_dict, reasons_list)。"""
        close = df["close"].astype(float)
        volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(dtype=float)

        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()

        indicators: Dict[str, Any] = {}
        reasons: List[str] = []

        # 1. 均线排列
        ma_score, ma_reason = self._score_ma_alignment(close, ma5, ma10, ma20)
        indicators["ma_score"] = ma_score
        if ma_reason:
            reasons.append(ma_reason)

        # 2. 金叉
        gc_score, gc_reason = self._score_golden_cross(ma5, ma10, ma20)
        indicators["golden_cross_score"] = gc_score
        if gc_reason:
            reasons.append(gc_reason)

        # 3. MACD
        macd_score, macd_reason = self._score_macd(close)
        indicators["macd_score"] = macd_score
        if macd_reason:
            reasons.append(macd_reason)

        # 4. RSI
        rsi_score, rsi_reason = self._score_rsi(close)
        indicators["rsi_score"] = rsi_score
        if rsi_reason:
            reasons.append(rsi_reason)

        # 5. 布林带
        boll_score, boll_reason = self._score_bollinger(close, ma20)
        indicators["bollinger_score"] = boll_score
        if boll_reason:
            reasons.append(boll_reason)

        # 6. 量价
        vol_score, vol_reason = self._score_volume(close, volume)
        indicators["volume_score"] = vol_score
        if vol_reason:
            reasons.append(vol_reason)

        # 7. 乖离率
        bias_score, bias_reason = self._score_bias(close, ma5)
        indicators["bias_score"] = bias_score
        if bias_reason:
            reasons.append(bias_reason)

        return indicators, reasons

    def _weighted_score(self, indicators: Dict[str, Any]) -> float:
        key_map = {
            "ma_alignment": "ma_score",
            "golden_cross": "golden_cross_score",
            "macd": "macd_score",
            "rsi": "rsi_score",
            "bollinger": "bollinger_score",
            "volume": "volume_score",
            "bias": "bias_score",
        }
        total = 0.0
        for wkey, ikey in key_map.items():
            total += self.weights.get(wkey, 0) * indicators.get(ikey, 0)
        return total * 100

    # ── 各维度评分函数（0~1）──────────────────────────────────

    @staticmethod
    def _score_ma_alignment(close, ma5, ma10, ma20) -> tuple:
        c, m5, m10, m20 = close.iloc[-1], ma5.iloc[-1], ma10.iloc[-1], ma20.iloc[-1]
        if any(pd.isna(v) for v in [m5, m10, m20]):
            return 0.3, ""
        if m5 > m10 > m20:
            if c > m5:
                return 1.0, "多头排列"
            return 0.85, "多头排列(接近MA5)"
        if m5 > m10:
            return 0.6, "短多头"
        if m10 > m20:
            return 0.4, "中多头"
        if m5 < m10 < m20:
            return 0.05, "空头排列"
        return 0.2, ""

    @staticmethod
    def _score_golden_cross(ma5, ma10, ma20) -> tuple:
        if len(ma5) < 4:
            return 0.3, ""
        for i in range(1, 4):
            prev5, curr5 = ma5.iloc[-i - 1], ma5.iloc[-i]
            prev10, curr10 = ma10.iloc[-i - 1], ma10.iloc[-i]
            if pd.notna(prev5) and pd.notna(curr5) and pd.notna(prev10) and pd.notna(curr10):
                if prev5 <= prev10 and curr5 > curr10:
                    return 1.0, f"MA5×MA10金叉({i}日前)"
        for i in range(1, 4):
            prev10, curr10 = ma10.iloc[-i - 1], ma10.iloc[-i]
            prev20, curr20 = ma20.iloc[-i - 1], ma20.iloc[-i]
            if pd.notna(prev10) and pd.notna(curr10) and pd.notna(prev20) and pd.notna(curr20):
                if prev10 <= prev20 and curr10 > curr20:
                    return 0.8, f"MA10×MA20金叉({i}日前)"
        if pd.notna(ma5.iloc[-1]) and pd.notna(ma10.iloc[-1]) and ma5.iloc[-1] > ma10.iloc[-1]:
            return 0.4, "MA5>MA10"
        return 0.2, ""

    @staticmethod
    def _score_macd(close) -> tuple:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_bar = (dif - dea) * 2
        if len(macd_bar) < 3:
            return 0.3, ""
        cur_dif, cur_dea = dif.iloc[-1], dea.iloc[-1]
        prev_bar = macd_bar.iloc[-2]
        curr_bar = macd_bar.iloc[-1]
        if pd.isna(cur_dif) or pd.isna(cur_dea):
            return 0.3, ""
        if prev_bar <= 0 < curr_bar:
            if cur_dif > 0:
                return 1.0, "MACD零轴上方金叉"
            return 0.9, "MACD金叉"
        if cur_dif > 0 and cur_dea > 0:
            return 0.6, "MACD零轴上方"
        if cur_dif > cur_dea:
            return 0.5, "DIF>DEA"
        if cur_dif < 0 and cur_dea < 0:
            return 0.1, "MACD零轴下方"
        return 0.3, ""

    @staticmethod
    def _score_rsi(close, period=14) -> tuple:
        if len(close) < period + 1:
            return 0.3, ""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        cur_rsi = rsi.iloc[-1]
        if pd.isna(cur_rsi):
            return 0.3, ""
        if 30 <= cur_rsi <= 50:
            return 1.0, f"RSI超卖回升({cur_rsi:.0f})"
        if 50 < cur_rsi <= 65:
            return 0.7, f"RSI中性偏强({cur_rsi:.0f})"
        if 20 <= cur_rsi < 30:
            return 0.8, f"RSI超卖({cur_rsi:.0f})"
        if 65 < cur_rsi <= 75:
            return 0.4, f"RSI偏强({cur_rsi:.0f})"
        if cur_rsi > 75:
            return 0.1, f"RSI超买({cur_rsi:.0f})"
        return 0.5, f"RSI({cur_rsi:.0f})"

    @staticmethod
    def _score_bollinger(close, ma20) -> tuple:
        if len(close) < 20:
            return 0.3, ""
        std20 = close.rolling(20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        c = close.iloc[-1]
        u, l, m = upper.iloc[-1], lower.iloc[-1], ma20.iloc[-1]
        if any(pd.isna(v) for v in [u, l, m]):
            return 0.3, ""
        bandwidth = u - l
        if bandwidth == 0:
            return 0.3, ""
        pos = (c - l) / bandwidth
        if pos <= 0.15:
            return 1.0, "触及布林下轨"
        if pos <= 0.3:
            return 0.8, "接近布林下轨"
        if 0.4 <= pos <= 0.6:
            return 0.6, "布林中轨附近"
        if pos >= 0.9:
            return 0.1, "触及布林上轨"
        return 0.4, ""

    @staticmethod
    def _score_volume(close, volume) -> tuple:
        if len(volume) < 6 or volume.iloc[-5:].sum() == 0:
            return 0.3, ""
        avg5 = volume.iloc[-6:-1].mean()
        cur_vol = volume.iloc[-1]
        if avg5 == 0:
            return 0.3, ""
        ratio = cur_vol / avg5
        price_up = close.iloc[-1] > close.iloc[-2] if len(close) >= 2 else False

        if ratio < 0.6 and not price_up:
            return 1.0, "缩量回踩"
        if ratio < 0.8 and not price_up:
            return 0.8, "温和缩量"
        if 0.8 <= ratio <= 1.5 and price_up:
            return 0.7, "温和放量上涨"
        if 1.5 < ratio <= 2.5 and price_up:
            return 0.6, "放量上涨"
        if ratio > 3.0:
            return 0.2, "巨量"
        return 0.4, ""

    @staticmethod
    def _score_bias(close, ma5) -> tuple:
        if len(close) < 5 or pd.isna(ma5.iloc[-1]) or ma5.iloc[-1] == 0:
            return 0.3, ""
        bias = (close.iloc[-1] - ma5.iloc[-1]) / ma5.iloc[-1] * 100
        if abs(bias) <= 1:
            return 1.0, f"紧贴MA5(乖离{bias:+.1f}%)"
        if abs(bias) <= 3:
            return 0.7, f"接近MA5(乖离{bias:+.1f}%)"
        if -5 <= bias < -3:
            return 0.8, f"回踩MA5(乖离{bias:+.1f}%)"
        if 3 < bias <= 5:
            return 0.4, f"偏离MA5(乖离{bias:+.1f}%)"
        if bias > 5:
            return 0.1, f"远离MA5(乖离{bias:+.1f}%)"
        if bias < -5:
            return 0.5, f"超跌MA5(乖离{bias:+.1f}%)"
        return 0.3, ""
