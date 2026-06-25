import time
from collections import defaultdict
from typing import Any

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MetricsCollector:
    def __init__(self):
        self._start = time.time()
        self.total_requests = 0
        self.endpoint_counts: dict[str, int] = defaultdict(int)
        self.endpoint_latencies: dict[str, list[float]] = defaultdict(list)
        self.error_counts: dict[str, int] = defaultdict(int)
        self.total_errors = 0

    def record_request(self, endpoint: str, latency_ms: float, status_code: int) -> None:
        self.total_requests += 1
        self.endpoint_counts[endpoint] += 1
        self.endpoint_latencies[endpoint].append(latency_ms)
        if status_code >= 400:
            self.error_counts[endpoint] += 1
            self.total_errors += 1

    def get_metrics(self) -> dict[str, Any]:
        endpoints = {}
        for ep in sorted(self.endpoint_counts):
            latencies = self.endpoint_latencies.get(ep, [])
            endpoints[ep] = {
                "requests": self.endpoint_counts[ep],
                "errors": self.error_counts.get(ep, 0),
                "latency_ms_avg": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
                "latency_ms_min": round(min(latencies), 2) if latencies else 0.0,
                "latency_ms_max": round(max(latencies), 2) if latencies else 0.0,
                "latency_ms_p99": self._percentile(sorted(latencies), 99) if latencies else 0.0,
            }

        return {
            "uptime_seconds": round(time.time() - self._start),
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate_pct": round(self.total_errors / max(self.total_requests, 1) * 100, 2),
            "endpoints": endpoints,
        }

    @staticmethod
    def _percentile(sorted_data: list[float], p: float) -> float:
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * p / 100.0
        f = int(k)
        c = f + 1
        if c >= len(sorted_data):
            return sorted_data[-1]
        return round(sorted_data[f] * (c - k) + sorted_data[c] * (k - f), 2)


metrics_collector = MetricsCollector()
