"""외부 API 지연·결과 지표 — 고정 버킷 히스토그램(Prometheus 형식으로 내보낸다).

## 왜 필요한가

이 시스템의 지연은 네트워크가 아니라 **증권사 유량 한도**가 지배한다(모의 실효 초당 약 1건).
그런데 그것을 "느낌"이 아니라 숫자로 보려면 API(TR)별로 왕복 지연과 실패를 따로 세야 한다.
평균은 꼬리를 숨기므로 **분위수(p50/p95/p99)**를 낸다.

## 왜 고정 버킷인가

모든 표본을 들고 있으면 장중 수만 건이 쌓인다. Prometheus 히스토그램처럼 **고정 경계의 누적
카운트**만 들고 분위수는 버킷 경계로 근사한다 — 메모리 O(버킷 수), 기록 O(log 버킷 수).
정확도는 버킷 간격만큼이고, 지연 판단(수십 ms vs 수백 ms vs 초)에는 충분하다.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Dict, List

# 초 단위 상한 경계 — 10ms ~ 10s (KIS 왕복은 수십~수백 ms, 타임아웃은 10s)
BUCKETS: List[float] = [0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]


@dataclass
class _Series:
    counts: List[int] = field(default_factory=lambda: [0] * (len(BUCKETS) + 1))
    total: int = 0
    sum_s: float = 0.0
    max_s: float = 0.0
    outcomes: Dict[str, int] = field(default_factory=dict)

    def observe(self, seconds: float, outcome: str) -> None:
        self.counts[bisect.bisect_left(BUCKETS, seconds)] += 1
        self.total += 1
        self.sum_s += seconds
        self.max_s = max(self.max_s, seconds)
        self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1

    def quantile(self, q: float) -> float:
        """버킷 상한으로 근사한 분위수(초). 마지막 버킷(10s 초과)은 관측 최대값."""
        if not self.total:
            return 0.0
        rank = q * self.total
        acc = 0
        for i, c in enumerate(self.counts):
            acc += c
            if acc >= rank:
                # 버킷 상한이 관측 최대보다 크면 최대로 자른다(표시가 실제보다 커지지 않게)
                return min(BUCKETS[i], self.max_s) if i < len(BUCKETS) else self.max_s
        return self.max_s


class LatencyStats:
    """키(TR ID 등)별 지연 히스토그램. 스레드가 아니라 asyncio 단일 루프에서 쓴다(락 불필요)."""

    def __init__(self) -> None:
        self.series: Dict[str, _Series] = {}

    def observe(self, key: str, seconds: float, outcome: str = "ok") -> None:
        self.series.setdefault(key, _Series()).observe(max(0.0, float(seconds)), outcome)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for key, s in self.series.items():
            out[key] = {"count": s.total,
                        "p50_ms": round(s.quantile(0.50) * 1000, 1),
                        "p95_ms": round(s.quantile(0.95) * 1000, 1),
                        "p99_ms": round(s.quantile(0.99) * 1000, 1),
                        "max_ms": round(s.max_s * 1000, 1),
                        "errors": sum(v for k, v in s.outcomes.items() if k != "ok")}
        return out

    def prometheus(self, metric: str = "kis_http_request_seconds") -> List[str]:
        """Prometheus 텍스트 형식(히스토그램 + 결과별 카운터)."""
        lines = [f"# TYPE {metric} histogram"]
        for key, s in sorted(self.series.items()):
            acc = 0
            for le, c in zip(BUCKETS + [float("inf")], s.counts):
                acc += c
                le_s = "+Inf" if le == float("inf") else f"{le:g}"
                lines.append(f'{metric}_bucket{{tr="{key}",le="{le_s}"}} {acc}')
            lines.append(f'{metric}_sum{{tr="{key}"}} {s.sum_s:.6f}')
            lines.append(f'{metric}_count{{tr="{key}"}} {s.total}')
        lines.append("# TYPE kis_http_requests_total counter")
        for key, s in sorted(self.series.items()):
            for outcome, n in sorted(s.outcomes.items()):
                lines.append(f'kis_http_requests_total{{tr="{key}",outcome="{outcome}"}} {n}')
        return lines
