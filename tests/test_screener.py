# -*- coding: utf-8 -*-
"""Unit tests for stock screener."""

import unittest

import numpy as np
import pandas as pd

from src.core.screener import StockScreener, ScoredStock


def _make_realtime_df(rows):
    """构造模拟全市场实时行情 DataFrame。"""
    return pd.DataFrame(rows)


def _make_daily_history(closes, volumes=None):
    """构造模拟日K线 DataFrame。"""
    n = len(closes)
    data = {"close": closes}
    if volumes is not None:
        data["volume"] = volumes
    else:
        data["volume"] = [1000000] * n
    data["open"] = closes
    data["high"] = [c * 1.01 for c in closes]
    data["low"] = [c * 0.99 for c in closes]
    return pd.DataFrame(data)


class Layer1FilterTest(unittest.TestCase):
    """测试 Layer 1 粗筛逻辑。"""

    def setUp(self):
        self.screener = StockScreener()

    def test_exclude_st(self):
        df = _make_realtime_df([
            {"code": "000001", "name": "平安银行", "price": 15, "turnover_rate": 1.0,
             "pe_ratio": 8, "total_mv": 2e11, "volume_ratio": 1.0,
             "change_pct": 1.0, "amplitude": 3.0},
            {"code": "000002", "name": "*ST万科", "price": 2, "turnover_rate": 1.0,
             "pe_ratio": 8, "total_mv": 2e11, "volume_ratio": 1.0,
             "change_pct": 1.0, "amplitude": 3.0},
            {"code": "000003", "name": "ST退市", "price": 1, "turnover_rate": 0.5,
             "pe_ratio": 5, "total_mv": 1e10, "volume_ratio": 1.0,
             "change_pct": -1, "amplitude": 2.0},
        ])
        result = self.screener._layer1_filter(df)
        codes = result["code"].tolist()
        self.assertIn("000001", codes)
        self.assertNotIn("000002", codes)
        self.assertNotIn("000003", codes)

    def test_filter_by_price(self):
        df = _make_realtime_df([
            {"code": "600001", "name": "A股", "price": 50, "turnover_rate": 2.0,
             "pe_ratio": 20, "total_mv": 1e11, "volume_ratio": 1.0,
             "change_pct": 1.0, "amplitude": 3.0},
            {"code": "600002", "name": "B股", "price": 1.5, "turnover_rate": 2.0,
             "pe_ratio": 20, "total_mv": 1e11, "volume_ratio": 1.0,
             "change_pct": 1.0, "amplitude": 3.0},
        ])
        result = self.screener._layer1_filter(df)
        self.assertIn("600001", result["code"].tolist())
        self.assertNotIn("600002", result["code"].tolist())

    def test_filter_by_pe(self):
        df = _make_realtime_df([
            {"code": "300001", "name": "高PE", "price": 30, "turnover_rate": 1.0,
             "pe_ratio": 300, "total_mv": 1e11, "volume_ratio": 1.0,
             "change_pct": 1.0, "amplitude": 3.0},
            {"code": "300002", "name": "亏损", "price": 30, "turnover_rate": 1.0,
             "pe_ratio": -5, "total_mv": 1e11, "volume_ratio": 1.0,
             "change_pct": 1.0, "amplitude": 3.0},
            {"code": "300003", "name": "正常", "price": 30, "turnover_rate": 1.0,
             "pe_ratio": 25, "total_mv": 1e11, "volume_ratio": 1.0,
             "change_pct": 1.0, "amplitude": 3.0},
        ])
        result = self.screener._layer1_filter(df)
        codes = result["code"].tolist()
        self.assertNotIn("300001", codes)
        self.assertNotIn("300002", codes)
        self.assertIn("300003", codes)

    def test_filter_by_market_cap(self):
        df = _make_realtime_df([
            {"code": "002001", "name": "大盘", "price": 20, "turnover_rate": 1.0,
             "pe_ratio": 15, "total_mv": 2e11, "volume_ratio": 1.0,
             "change_pct": 1.0, "amplitude": 3.0},
            {"code": "002002", "name": "小盘", "price": 20, "turnover_rate": 1.0,
             "pe_ratio": 15, "total_mv": 1e9, "volume_ratio": 1.0,
             "change_pct": 1.0, "amplitude": 3.0},
        ])
        result = self.screener._layer1_filter(df)
        codes = result["code"].tolist()
        self.assertIn("002001", codes)
        self.assertNotIn("002002", codes)

    def test_custom_overrides(self):
        screener = StockScreener(layer1_overrides={"min_price": 1.0, "pe_max": 500})
        df = _make_realtime_df([
            {"code": "000001", "name": "正常", "price": 2, "turnover_rate": 1.0,
             "pe_ratio": 300, "total_mv": 1e11, "volume_ratio": 1.0,
             "change_pct": 1.0, "amplitude": 3.0},
        ])
        result = screener._layer1_filter(df)
        self.assertEqual(len(result), 1)


class Layer2ScoringTest(unittest.TestCase):
    """测试 Layer 2 技术指标评分。"""

    def test_ma_bullish_alignment(self):
        closes = list(range(10, 35))
        df = _make_daily_history(closes)
        score, reason = StockScreener._score_ma_alignment(
            pd.Series(closes),
            pd.Series(closes).rolling(5).mean(),
            pd.Series(closes).rolling(10).mean(),
            pd.Series(closes).rolling(20).mean(),
        )
        self.assertEqual(score, 1.0)
        self.assertIn("多头", reason)

    def test_ma_bearish_alignment(self):
        closes = list(range(35, 10, -1))
        df = _make_daily_history(closes)
        score, reason = StockScreener._score_ma_alignment(
            pd.Series(closes),
            pd.Series(closes).rolling(5).mean(),
            pd.Series(closes).rolling(10).mean(),
            pd.Series(closes).rolling(20).mean(),
        )
        self.assertLess(score, 0.2)

    def test_golden_cross(self):
        # MA5 crosses above MA10 within last 3 days
        closes = [10, 9, 8, 7, 6, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        ma5 = pd.Series(closes, dtype=float).rolling(5).mean()
        ma10 = pd.Series(closes, dtype=float).rolling(10).mean()
        ma20 = pd.Series(closes, dtype=float).rolling(20).mean()
        score, reason = StockScreener._score_golden_cross(ma5, ma10, ma20)
        self.assertGreaterEqual(score, 0.4)

    def test_macd_golden_cross(self):
        # Strictly ascending — MACD should be positive (zero axis above)
        closes = list(range(10, 35))
        score, reason = StockScreener._score_macd(pd.Series(closes, dtype=float))
        self.assertGreater(score, 0.5)

    def test_rsi_oversold(self):
        np.random.seed(42)
        closes = list(np.cumsum(np.random.randn(30) - 0.5) + 50)
        score, reason = StockScreener._score_rsi(pd.Series(closes))
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_bollinger_lower_band(self):
        closes = [20] * 20 + [15]
        close_s = pd.Series(closes, dtype=float)
        ma20 = close_s.rolling(20).mean()
        score, reason = StockScreener._score_bollinger(close_s, ma20)
        self.assertGreater(score, 0.7)

    def test_bias_close_to_ma5(self):
        closes = pd.Series([10.0, 10.1, 10.0, 10.1, 10.05, 10.0])
        ma5 = closes.rolling(5).mean()
        score, reason = StockScreener._score_bias(closes, ma5)
        self.assertGreater(score, 0.5)

    def test_volume_shrink_pullback(self):
        # Price drops + volume shrinks → shrink pullback signal
        closes = pd.Series([26, 25, 24, 23, 22, 21], dtype=float)
        volumes = pd.Series([100, 100, 100, 100, 100, 40], dtype=float)
        score, reason = StockScreener._score_volume(closes, volumes)
        self.assertGreater(score, 0.5)


class WeightedScoreTest(unittest.TestCase):
    """测试综合评分。"""

    def test_weighted_score_range(self):
        screener = StockScreener()
        indicators = {
            "ma_score": 1.0,
            "golden_cross_score": 1.0,
            "macd_score": 1.0,
            "rsi_score": 1.0,
            "bollinger_score": 1.0,
            "volume_score": 1.0,
            "bias_score": 1.0,
        }
        score = screener._weighted_score(indicators)
        self.assertAlmostEqual(score, 100.0, places=0)

    def test_weighted_score_zero(self):
        screener = StockScreener()
        indicators = {
            "ma_score": 0, "golden_cross_score": 0, "macd_score": 0,
            "rsi_score": 0, "bollinger_score": 0, "volume_score": 0,
            "bias_score": 0,
        }
        score = screener._weighted_score(indicators)
        self.assertAlmostEqual(score, 0.0, places=0)

    def test_custom_weights(self):
        screener = StockScreener(weight_overrides={"ma_alignment": 100})
        # Only ma_alignment has weight 100, others keep defaults → ma_alignment should dominate
        self.assertGreater(screener.weights["ma_alignment"], 0.5)


if __name__ == "__main__":
    unittest.main()
