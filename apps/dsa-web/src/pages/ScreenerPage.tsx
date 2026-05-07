import type React from 'react';
import { useState, useCallback } from 'react';
import { Search, Loader2, TrendingUp, Filter, ChevronDown, ChevronUp } from 'lucide-react';
import { screenerApi } from '../api/screener';
import { getParsedApiError } from '../api/error';
import type { ParsedApiError } from '../api/error';
import { ApiErrorAlert, Badge } from '../components/common';
import type { ScreenerResponse } from '../types/screener';

const INPUT_CLASS =
  'input-surface input-focus-glow h-10 w-full rounded-xl border bg-transparent px-3 py-2 text-sm transition-all focus:outline-none disabled:cursor-not-allowed disabled:opacity-60';

function scoreBadge(score: number) {
  if (score >= 70) return <Badge variant="success" glow>{score.toFixed(1)}</Badge>;
  if (score >= 50) return <Badge variant="warning">{score.toFixed(1)}</Badge>;
  return <Badge variant="default">{score.toFixed(1)}</Badge>;
}

function dimensionCell(value: number | undefined) {
  if (value == null) return '--';
  const pct = Math.round(value * 100);
  let color = 'text-secondary-text';
  if (pct >= 70) color = 'text-green-500';
  else if (pct >= 40) color = 'text-yellow-500';
  else color = 'text-red-400';
  return <span className={color}>{pct}%</span>;
}

const ScreenerPage: React.FC = () => {
  const [result, setResult] = useState<ScreenerResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ParsedApiError | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const [topN, setTopN] = useState(30);
  const [filters, setFilters] = useState({ minPrice: '', peMax: '', minMarketCap: '', poolSize: '', maxWorkers: '' });
  const [excludeKcb, setExcludeKcb] = useState(true);

  const runScreener = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const req: Record<string, unknown> = { top_n: topN };
      if (filters.minPrice) req.min_price = Number(filters.minPrice);
      if (filters.peMax) req.pe_max = Number(filters.peMax);
      if (filters.minMarketCap) req.min_market_cap_yi = Number(filters.minMarketCap);
      if (filters.poolSize) req.layer2_pool_size = Number(filters.poolSize);
      if (filters.maxWorkers) req.max_workers = Number(filters.maxWorkers);
      req.exclude_kcb = excludeKcb;
      const data = await screenerApi.run(req as any);
      setResult(data);
    } catch (e) {
      setError(getParsedApiError(e));
    } finally {
      setLoading(false);
    }
  }, [topN, filters]);

  return (
    <div className="min-h-full flex flex-col rounded-[1.5rem] bg-transparent">
      <header className="flex-shrink-0 border-b border-white/5 px-3 py-3">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Search className="h-5 w-5 text-[hsl(var(--primary))]" />
            <h1 className="text-lg font-semibold text-foreground">全市场选股</h1>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-secondary-text">Top</span>
            <input
              type="number"
              value={topN}
              onChange={(e) => setTopN(Number(e.target.value) || 30)}
              className={`${INPUT_CLASS} !w-16 text-center`}
              min={1}
              max={200}
              disabled={loading}
            />
            <button
              type="button"
              onClick={() => setShowFilters(!showFilters)}
              className="flex items-center gap-1 rounded-xl border border-border/50 px-3 py-2 text-xs text-secondary-text transition-all hover:bg-hover hover:text-foreground"
            >
              <Filter className="h-3.5 w-3.5" />
              筛选条件
              {showFilters ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            </button>
            <button
              type="button"
              onClick={runScreener}
              disabled={loading}
              className="btn-primary flex items-center gap-2 !rounded-xl px-4 py-2 text-sm"
            >
              {loading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <TrendingUp className="h-4 w-4" />
              )}
              {loading ? '筛选中...' : '开始筛选'}
            </button>
          </div>
        </div>

        {showFilters && (
          <div className="mt-3 flex flex-wrap items-center gap-3 rounded-xl bg-surface/50 p-3">
            <label className="flex items-center gap-2 text-xs text-secondary-text">
              <span>最低价(元)</span>
              <input
                type="number"
                value={filters.minPrice}
                onChange={(e) => setFilters({ ...filters, minPrice: e.target.value })}
                className={`${INPUT_CLASS} !w-20`}
                placeholder="3"
                disabled={loading}
              />
            </label>
            <label className="flex items-center gap-2 text-xs text-secondary-text">
              <span>最大PE</span>
              <input
                type="number"
                value={filters.peMax}
                onChange={(e) => setFilters({ ...filters, peMax: e.target.value })}
                className={`${INPUT_CLASS} !w-20`}
                placeholder="200"
                disabled={loading}
              />
            </label>
            <label className="flex items-center gap-2 text-xs text-secondary-text">
              <span>最小市值(亿)</span>
              <input
                type="number"
                value={filters.minMarketCap}
                onChange={(e) => setFilters({ ...filters, minMarketCap: e.target.value })}
                className={`${INPUT_CLASS} !w-24`}
                placeholder="50"
                disabled={loading}
              />
            </label>
            <label className="flex items-center gap-2 text-xs text-secondary-text">
              <span>候选池</span>
              <input
                type="number"
                value={filters.poolSize}
                onChange={(e) => setFilters({ ...filters, poolSize: e.target.value })}
                className={`${INPUT_CLASS} !w-20`}
                placeholder="500"
                min={50}
                max={2000}
                disabled={loading}
              />
            </label>
            <label className="flex items-center gap-2 text-xs text-secondary-text">
              <span>并发数</span>
              <input
                type="number"
                value={filters.maxWorkers}
                onChange={(e) => setFilters({ ...filters, maxWorkers: e.target.value })}
                className={`${INPUT_CLASS} !w-16`}
                placeholder="5"
                min={1}
                max={20}
                disabled={loading}
              />
            </label>
            <label className="flex items-center gap-2 text-xs text-secondary-text cursor-pointer select-none">
              <input
                type="checkbox"
                checked={excludeKcb}
                onChange={(e) => setExcludeKcb(e.target.checked)}
                className="h-4 w-4 rounded border-border"
                disabled={loading}
              />
              <span>排除科创板</span>
            </label>
          </div>
        )}
      </header>

      <main className="flex min-h-0 flex-1 flex-col gap-3 overflow-auto p-3">
        {error && <ApiErrorAlert error={error} />}

        {result && (
          <>
            <div className="flex items-center gap-4 text-xs text-secondary-text">
              <span>全市场 <strong className="text-foreground">{result.totalMarket}</strong> 只</span>
              <span>→ 粗筛 <strong className="text-foreground">{result.layer1Passed}</strong> 只</span>
              <span>→ 精筛 <strong className="text-foreground">{result.layer2Passed}</strong> 只</span>
              <span>→ Top <strong className="text-[hsl(var(--primary))]">{result.topStocks.length}</strong></span>
              <span className="ml-auto">耗时 {result.elapsedSeconds.toFixed(1)}s</span>
            </div>

            {result.topStocks.length === 0 ? (
              <div className="flex flex-1 items-center justify-center text-sm text-secondary-text">
                未筛选到符合条件的股票
              </div>
            ) : (
              <div className="overflow-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border/50 text-xs text-secondary-text">
                      <th className="py-2 text-left font-medium">#</th>
                      <th className="py-2 text-left font-medium">代码</th>
                      <th className="py-2 text-left font-medium">名称</th>
                      <th className="py-2 text-center font-medium">得分</th>
                      <th className="py-2 text-center font-medium">均线</th>
                      <th className="py-2 text-center font-medium">金叉</th>
                      <th className="py-2 text-center font-medium">MACD</th>
                      <th className="py-2 text-center font-medium">RSI</th>
                      <th className="py-2 text-center font-medium">布林</th>
                      <th className="py-2 text-center font-medium">量价</th>
                      <th className="py-2 text-center font-medium">乖离</th>
                      <th className="py-2 text-left font-medium">匹配原因</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.topStocks.map((stock, i) => (
                      <tr
                        key={stock.code}
                        className="border-b border-border/20 transition-colors hover:bg-hover/50"
                      >
                        <td className="py-2 text-secondary-text">{i + 1}</td>
                        <td className="py-2 font-mono text-foreground">{stock.code}</td>
                        <td className="py-2 text-foreground">{stock.name}</td>
                        <td className="py-2 text-center">{scoreBadge(stock.score)}</td>
                        <td className="py-2 text-center">{dimensionCell(stock.indicators?.maScore)}</td>
                        <td className="py-2 text-center">{dimensionCell(stock.indicators?.goldenCrossScore)}</td>
                        <td className="py-2 text-center">{dimensionCell(stock.indicators?.macdScore)}</td>
                        <td className="py-2 text-center">{dimensionCell(stock.indicators?.rsiScore)}</td>
                        <td className="py-2 text-center">{dimensionCell(stock.indicators?.bollingerScore)}</td>
                        <td className="py-2 text-center">{dimensionCell(stock.indicators?.volumeScore)}</td>
                        <td className="py-2 text-center">{dimensionCell(stock.indicators?.biasScore)}</td>
                        <td className="py-2 text-xs text-secondary-text max-w-[200px] truncate">
                          {stock.matchReasons?.slice(0, 3).join('、') || '-'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}

        {!result && !loading && !error && (
          <div className="flex flex-1 flex-col items-center justify-center gap-4 text-secondary-text">
            <Search className="h-12 w-12 opacity-30" />
            <p className="text-sm">点击「开始筛选」从全市场中选出候选股</p>
            <p className="text-xs opacity-60">
              两层漏斗筛选：实时数据粗筛 → 技术指标精筛打分，预计 1-5 分钟
            </p>
          </div>
        )}

        {loading && !result && (
          <div className="flex flex-1 flex-col items-center justify-center gap-4 text-secondary-text">
            <Loader2 className="h-10 w-10 animate-spin text-[hsl(var(--primary))]" />
            <p className="text-sm">正在筛选全市场股票...</p>
            <p className="text-xs opacity-60">Layer 1 粗筛 → Layer 2 技术指标打分</p>
          </div>
        )}
      </main>
    </div>
  );
};

export default ScreenerPage;
