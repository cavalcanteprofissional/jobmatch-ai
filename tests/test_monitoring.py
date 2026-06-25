"""
Testes para o módulo de monitoramento/métricas.
"""

import time

from src.monitoring.metrics import MetricsCollector


class TestMetricsCollector:
    def setup_method(self):
        self.collector = MetricsCollector()

    def test_record_and_get_metrics(self):
        self.collector.record_request("GET /health", 10.5, 200)
        self.collector.record_request("POST /predict", 250.3, 200)
        self.collector.record_request("POST /predict", 300.1, 500)

        metrics = self.collector.get_metrics()

        assert metrics["total_requests"] == 3
        assert metrics["total_errors"] == 1
        assert metrics["error_rate_pct"] == pytest.approx(33.33, rel=0.1)
        assert "GET /health" in metrics["endpoints"]
        assert "POST /predict" in metrics["endpoints"]

    def test_no_requests(self):
        metrics = self.collector.get_metrics()
        assert metrics["total_requests"] == 0
        assert metrics["total_errors"] == 0
        assert metrics["error_rate_pct"] == 0.0
        assert metrics["endpoints"] == {}

    def test_endpoint_metrics(self):
        self.collector.record_request("GET /test", 100.0, 200)
        self.collector.record_request("GET /test", 200.0, 200)

        metrics = self.collector.get_metrics()
        ep = metrics["endpoints"]["GET /test"]
        assert ep["requests"] == 2
        assert ep["latency_ms_avg"] == 150.0
        assert ep["latency_ms_min"] == 100.0
        assert ep["latency_ms_max"] == 200.0
        assert ep["errors"] == 0

    def test_error_tracking(self):
        self.collector.record_request("GET /error", 50.0, 404)
        self.collector.record_request("GET /error", 30.0, 500)

        metrics = self.collector.get_metrics()
        ep = metrics["endpoints"]["GET /error"]
        assert ep["errors"] == 2
        assert metrics["total_errors"] == 2

    def test_uptime(self):
        time.sleep(0.01)
        metrics = self.collector.get_metrics()
        assert metrics["uptime_seconds"] >= 0

    def test_latency_p99(self):
        for i in range(100):
            self.collector.record_request("GET /test", float(i), 200)

        metrics = self.collector.get_metrics()
        ep = metrics["endpoints"]["GET /test"]
        assert ep["latency_ms_p99"] > 95.0


import pytest
