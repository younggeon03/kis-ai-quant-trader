"""표본 가중치 — 중첩 라벨이 만드는 편향을 제거한다.

## 왜 필요한가

삼중배리어 라벨은 시점 t에서 t+1..t+H 구간을 본다. H=5면 연속된 5개 샘플이 **같은 미래 구간을
공유**한다. 이 상태로 학습하면 모델은 서로 독립인 샘플 N개가 아니라 훨씬 적은 수의 독립 정보를
N번 본 것처럼 취급한다. 결과는 두 가지다.

  1. 중첩이 심한 구간(가격이 오래 정체한 구간)의 표본이 과대 대표된다.
  2. 검증 점수가 부풀려진다. 모델이 실제보다 많은 정보를 본 것처럼 평가된다.

표준 대응은 López de Prado가 정리한 세 가지 가중치다.

  · **고유도(uniqueness)** — 각 라벨이 다른 라벨과 겹치지 않는 정도. 겹침의 역수.
  · **수익 귀속(return attribution)** — 고유도 × 구간 수익률 절대값. 큰 움직임을 만든
    표본에 더 큰 비중을 준다. 작은 노이즈 구간에 모델이 끌려다니지 않게 한다.
  · **시간 감쇠(time decay)** — 오래된 표본의 비중을 낮춘다. 시장 구조가 변하기 때문이다.

세 가지를 곱해 최종 가중치를 만들고, **합이 표본 수와 같도록 정규화**한다. 그래야 가중치를
넣기 전후로 정규화 강도가 달라지지 않는다.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def concurrency(t_start: np.ndarray, t_end: np.ndarray, n_bars: Optional[int] = None) -> np.ndarray:
    """각 봉에서 동시에 열려 있는 라벨 수를 센다.

    Args:
        t_start: 샘플별 라벨 시작 봉 인덱스
        t_end:   샘플별 라벨 종료 봉 인덱스(포함)
    Returns:
        길이 `n_bars`의 배열. c[i] = 봉 i를 관측 구간에 포함하는 라벨 수
    """
    t_start = np.asarray(t_start, dtype=np.int64)
    t_end = np.asarray(t_end, dtype=np.int64)
    if len(t_start) == 0:
        return np.zeros(0, dtype=np.float64)
    size = int(n_bars if n_bars is not None else t_end.max() + 1)
    # 차분 배열로 O(N) 누적 — 샘플마다 구간을 순회하면 O(N·H)가 된다
    diff = np.zeros(size + 1, dtype=np.float64)
    np.add.at(diff, np.clip(t_start, 0, size - 1), 1.0)
    np.add.at(diff, np.clip(t_end + 1, 0, size), -1.0)
    return np.cumsum(diff)[:size]


def average_uniqueness(t_start: np.ndarray, t_end: np.ndarray,
                       n_bars: Optional[int] = None) -> np.ndarray:
    """샘플별 평균 고유도 (0, 1]. 1이면 어떤 라벨과도 겹치지 않는다."""
    t_start = np.asarray(t_start, dtype=np.int64)
    t_end = np.asarray(t_end, dtype=np.int64)
    if len(t_start) == 0:
        return np.zeros(0, dtype=np.float64)

    size = int(n_bars if n_bars is not None else t_end.max() + 1)
    conc = concurrency(t_start, t_end, size)
    inv = np.divide(1.0, conc, out=np.zeros_like(conc), where=conc > 0)
    cum = np.concatenate([[0.0], np.cumsum(inv)])   # 구간 합을 O(1)로 얻기 위한 누적합

    lo = np.clip(t_start, 0, size - 1)
    hi = np.clip(t_end, 0, size - 1)
    span = (hi - lo + 1).astype(np.float64)
    total = cum[hi + 1] - cum[lo]
    return np.divide(total, span, out=np.ones_like(total), where=span > 0)


def return_attribution(t_start: np.ndarray, t_end: np.ndarray, log_returns: np.ndarray,
                       n_bars: Optional[int] = None) -> np.ndarray:
    """라벨 구간의 수익률을 동시성으로 나눠 귀속시킨 가중치(절대값).

    같은 봉을 여러 라벨이 공유하면 그 봉의 수익률을 나눠 갖는다.
    """
    t_start = np.asarray(t_start, dtype=np.int64)
    t_end = np.asarray(t_end, dtype=np.int64)
    log_returns = np.asarray(log_returns, dtype=np.float64)
    if len(t_start) == 0:
        return np.zeros(0, dtype=np.float64)

    size = int(n_bars if n_bars is not None else max(t_end.max() + 1, len(log_returns)))
    conc = concurrency(t_start, t_end, size)
    ret = np.zeros(size, dtype=np.float64)
    ret[: len(log_returns)] = np.nan_to_num(log_returns[:size], nan=0.0, posinf=0.0, neginf=0.0)

    share = np.divide(ret, conc, out=np.zeros_like(ret), where=conc > 0)
    cum = np.concatenate([[0.0], np.cumsum(share)])

    lo = np.clip(t_start, 0, size - 1)
    hi = np.clip(t_end, 0, size - 1)
    return np.abs(cum[hi + 1] - cum[lo])


def time_decay(weights: np.ndarray, last_weight: float = 0.5) -> np.ndarray:
    """오래된 표본의 가중치를 선형 감쇠한다.

    `last_weight`는 **가장 오래된** 표본이 받는 비율이다.
      1.0 → 감쇠 없음
      0.5 → 가장 오래된 표본이 최신 표본의 절반
      0.0 → 가장 오래된 표본은 무시
    누적 고유도를 기준으로 감쇠하므로, 단순 순번이 아니라 정보량 기준으로 줄어든다.
    """
    weights = np.asarray(weights, dtype=np.float64)
    n = len(weights)
    if n == 0:
        return weights
    last_weight = float(np.clip(last_weight, 0.0, 1.0))
    if last_weight >= 1.0:
        return np.ones(n, dtype=np.float64)

    cum = np.cumsum(weights)
    if cum[-1] <= 0:
        return np.ones(n, dtype=np.float64)
    cum = cum / cum[-1]                       # 0~1로 정규화된 누적 정보량
    slope = (1.0 - last_weight) / 1.0
    decay = last_weight + slope * cum
    return np.clip(decay, 0.0, 1.0)


def compute_sample_weights(
    t_start: np.ndarray,
    t_end: np.ndarray,
    log_returns: np.ndarray,
    n_bars: Optional[int] = None,
    use_return: bool = True,
    decay_last: float = 0.5,
) -> Tuple[np.ndarray, dict]:
    """최종 표본 가중치와 진단값을 반환한다.

    가중치 합은 표본 수와 같도록 정규화한다(평균 1). 이렇게 해야 가중치를 넣기 전후로
    정규화 강도와 학습률의 의미가 유지된다.
    """
    t_start = np.asarray(t_start, dtype=np.int64)
    t_end = np.asarray(t_end, dtype=np.int64)
    n = len(t_start)
    if n == 0:
        return np.zeros(0), {"samples": 0}

    uniq = average_uniqueness(t_start, t_end, n_bars)
    base = uniq.copy()
    if use_return:
        attr = return_attribution(t_start, t_end, log_returns, n_bars)
        # 수익 귀속이 전부 0이면(수익률 정보가 없으면) 고유도만 쓴다
        if attr.sum() > 0:
            base = uniq * attr

    decay = time_decay(uniq, decay_last)
    weights = base * decay

    total = weights.sum()
    if total <= 0:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = weights * (n / total)       # 평균 1로 정규화

    info = {
        "samples": int(n),
        "mean_uniqueness": float(uniq.mean()),
        "min_uniqueness": float(uniq.min()),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
        "effective_n": float(weights.sum() ** 2 / np.square(weights).sum()),
    }
    return weights, info


def effective_sample_size(weights: np.ndarray) -> float:
    """유효 표본 수. 가중치가 고르면 N에, 한쪽에 몰리면 작아진다.

    중첩 라벨로 실제 독립 정보가 얼마나 되는지 가늠하는 값이다.
    """
    weights = np.asarray(weights, dtype=np.float64)
    denom = np.square(weights).sum()
    return float(weights.sum() ** 2 / denom) if denom > 0 else 0.0
