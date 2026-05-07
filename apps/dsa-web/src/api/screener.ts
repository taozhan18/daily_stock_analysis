import apiClient from './index';
import { toCamelCase } from './utils';
import type { ScreenerRequest, ScreenerResponse } from '../types/screener';

export const screenerApi = {
  run: async (params: ScreenerRequest = {}): Promise<ScreenerResponse> => {
    const requestData: Record<string, unknown> = {};
    if (params.topN) requestData.top_n = params.topN;
    if (params.minPrice != null) requestData.min_price = params.minPrice;
    if (params.minTurnoverRate != null) requestData.min_turnover_rate = params.minTurnoverRate;
    if (params.peMin != null) requestData.pe_min = params.peMin;
    if (params.peMax != null) requestData.pe_max = params.peMax;
    if (params.minMarketCapYi != null) requestData.min_market_cap_yi = params.minMarketCapYi;
    if (params.excludeSt != null) requestData.exclude_st = params.excludeSt;
    if (params.excludeKcb != null) requestData.exclude_kcb = params.excludeKcb;
    if (params.layer2PoolSize != null) requestData.layer2_pool_size = params.layer2PoolSize;
    if (params.maxWorkers != null) requestData.max_workers = params.maxWorkers;

    const response = await apiClient.post<Record<string, unknown>>(
      '/api/v1/screener/run',
      requestData,
      { timeout: 600000 },
    );
    return toCamelCase<ScreenerResponse>(response.data);
  },

  getResult: async (): Promise<ScreenerResponse | null> => {
    const response = await apiClient.get<Record<string, unknown> | null>(
      '/api/v1/screener/result',
    );
    if (!response.data) return null;
    return toCamelCase<ScreenerResponse>(response.data);
  },
};
