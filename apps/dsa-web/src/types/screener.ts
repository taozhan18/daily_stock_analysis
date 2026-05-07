export interface ScreenerRequest {
  topN?: number;
  minPrice?: number;
  minTurnoverRate?: number;
  peMin?: number;
  peMax?: number;
  minMarketCapYi?: number;
  excludeSt?: boolean;
  excludeKcb?: boolean;
  layer2PoolSize?: number;
  maxWorkers?: number;
}

export interface ScoredStockResult {
  code: string;
  name: string;
  score: number;
  matchReasons: string[];
  indicators: Record<string, number>;
}

export interface ScreenerResponse {
  totalMarket: number;
  layer1Passed: number;
  layer2Passed: number;
  topStocks: ScoredStockResult[];
  elapsedSeconds: number;
}
