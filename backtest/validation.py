"""백테스트 과적합 검증 — CPCV, Deflated Sharpe, PBO.

## 왜 필요한가

백테스트를 한 번 돌려 Sharpe 1.7이 나왔다고 해서 그 전략이 좋은 것은 아니다. 파라미터를
여러 번 바꿔가며 가장 좋은 조합을 고르는 순간, 그 Sharpe에는 **선택 편향**이 섞인다.
무작위 전략이라도 20번 시도하면 그중 하나는 우연히 좋아 보인다.

이 모듈은 세 가지로 그것을 걸러낸다.

**CPCV (Combinatorial Purged Cross-Validation)** — 하나의 시계열에서 여러 개의 서로 다른
백테스트 경로를 만든다. 단일 경로의 Sharpe가 아니라 **경로들의 분포**를 본다. 분포가 넓으면
그 전략의 성과는 기간 선택에 크게 의존한다는 뜻이다.

**PSR / DSR (Probabilistic / Deflated Sharpe Ratio)** — 관측된 Sharpe가 운이 아닐 확률.
DSR은 여기에 더해 **몇 번 시도했는지**를 보정한다. 시도가 많을수록 기준선이 올라간다.

**PBO (Probability of Backtest Overfitting)** — 파라미터를 여러 개 시험했을 때, 학습 구간에서
가장 좋았던 조합이 검증 구간에서 중간 이하로 떨어질 확률. 0.5에 가까우면 그 탐색은 무의미하다.

수익률을 올리는 도구가 아니라, **올랐다고 착각하는 것을 막는 도구**다.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

EULER_MASCHERONI = 0.5772156649015329


# ======================================================================== CPCV
@dataclass
class CPCVSplit:
    """조합 하나. 테스트 그룹들과 그에 대응하는 학습 인덱스를 담는다."""

    split_id: int
    test_groups: Tuple[int, ...]
    train_idx: np.ndarray
    test_idx: np.ndarray
    group_slices: Dict[int, Tuple[int, int]]   # 그룹 → (시작, 끝) 봉 인덱스


class CombinatorialPurgedCV:
    """시계열을 N개 그룹으로 나누고, 그중 k개를 테스트로 쓰는 모든 조합을 만든다.

    일반 K-Fold는 N개의 경로만 주지만, CPCV는 C(N, k)개의 분할에서
    **k·C(N,k)/N 개의 서로 다른 백테스트 경로**를 재구성할 수 있다.
    N=6, k=2면 분할 15개에서 경로 5개가 나온다.

    학습 구간에서는 테스트 구간과 라벨이 겹치는 부분을 제거(purge)하고
    경계에 embargo를 둔다. `PurgedKFold`와 같은 규칙이다.
    """

    def __init__(self, n_groups: int = 6, n_test: int = 2,
                 horizon: int = 5, embargo: int = 5):
        if n_test >= n_groups:
            raise ValueError("n_test는 n_groups보다 작아야 합니다")
        self.n_groups = int(n_groups)
        self.n_test = int(n_test)
        self.horizon = max(1, int(horizon))
        self.embargo = max(0, int(embargo))

    @property
    def n_splits(self) -> int:
        return math.comb(self.n_groups, self.n_test)

    @property
    def n_paths(self) -> int:
        """재구성 가능한 백테스트 경로 수."""
        return self.n_test * self.n_splits // self.n_groups

    def group_bounds(self, n_bars: int) -> Dict[int, Tuple[int, int]]:
        """봉 인덱스를 N개 연속 구간으로 나눈다."""
        edges = np.linspace(0, n_bars, self.n_groups + 1).astype(int)
        return {g: (int(edges[g]), int(edges[g + 1])) for g in range(self.n_groups)}

    def split(self, n_bars: int) -> Iterator[CPCVSplit]:
        bounds = self.group_bounds(n_bars)
        all_idx = np.arange(n_bars)
        for split_id, combo in enumerate(
                itertools.combinations(range(self.n_groups), self.n_test)):
            test_mask = np.zeros(n_bars, dtype=bool)
            for g in combo:
                lo, hi = bounds[g]
                test_mask[lo:hi] = True

            # purge + embargo: 테스트 구간 경계에서 horizon+embargo 만큼 학습을 비운다
            block = self.horizon + self.embargo
            purge_mask = test_mask.copy()
            for g in combo:
                lo, hi = bounds[g]
                purge_mask[max(0, lo - block): min(n_bars, hi + block)] = True

            train_idx = all_idx[~purge_mask]
            test_idx = all_idx[test_mask]
            if len(train_idx) < 50 or len(test_idx) < 20:
                continue
            yield CPCVSplit(split_id, combo, train_idx, test_idx, bounds)

    def assemble_paths(self, group_returns: Dict[Tuple[int, int], np.ndarray]
                       ) -> List[np.ndarray]:
        """(split_id, group) → 수익률 배열 맵에서 전 구간 경로들을 재구성한다.

        각 그룹은 여러 분할에서 테스트된다. 경로 하나는 **그룹마다 서로 다른 분할의
        결과를 하나씩** 골라 이어 붙인 것이다.
        """
        by_group: Dict[int, List[np.ndarray]] = {g: [] for g in range(self.n_groups)}
        for (split_id, group), returns in sorted(group_returns.items()):
            if returns is not None and len(returns):
                by_group[group].append(np.asarray(returns, dtype=float))

        usable = [g for g in range(self.n_groups) if by_group[g]]
        if not usable:
            return []

        depth = min(len(by_group[g]) for g in usable)
        paths: List[np.ndarray] = []
        for k in range(depth):
            pieces = [by_group[g][k] for g in usable]
            paths.append(np.concatenate(pieces))
        return paths


# ======================================================== Sharpe 계열 통계
def sharpe_ratio(returns: Sequence[float], periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std(ddof=1) < 1e-12:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def _moments(returns: np.ndarray) -> Tuple[float, float]:
    """(왜도, 첨도). 정규분포면 (0, 3)."""
    r = returns[np.isfinite(returns)]
    n = len(r)
    if n < 4:
        return 0.0, 3.0
    mu, sd = r.mean(), r.std(ddof=1)
    if sd < 1e-12:
        return 0.0, 3.0
    z = (r - mu) / sd
    return float((z ** 3).mean()), float((z ** 4).mean())


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """표준정규 분위수. scipy 없이도 동작하도록 근사식을 쓴다."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    try:
        from scipy.stats import norm

        return float(norm.ppf(p))
    except Exception:
        # Acklam 근사
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]
        plow, phigh = 0.02425, 1 - 0.02425
        if p < plow:
            q = math.sqrt(-2 * math.log(p))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        if p > phigh:
            q = math.sqrt(-2 * math.log(1 - p))
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def probabilistic_sharpe_ratio(returns: Sequence[float], benchmark_sr: float = 0.0,
                               periods_per_year: int = 252) -> float:
    """관측 Sharpe가 기준값보다 진짜로 높을 확률.

    수익률의 왜도와 첨도를 반영한다. 팻테일 분포에서는 같은 Sharpe라도 신뢰도가 낮다.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 8:
        return 0.0
    sr = sharpe_ratio(r, periods_per_year) / math.sqrt(periods_per_year)   # 주기당 SR
    bench = benchmark_sr / math.sqrt(periods_per_year)
    skew, kurt = _moments(r)
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom <= 0:
        return 0.0
    return _norm_cdf((sr - bench) * math.sqrt(n - 1) / math.sqrt(denom))


def expected_max_sharpe(trial_sharpes: Sequence[float]) -> float:
    """N번 시도했을 때 순전히 운으로 기대되는 최대 Sharpe.

    시도 횟수가 많을수록 이 기준선이 올라간다. DSR은 관측 Sharpe를 이 값과 비교한다.
    """
    s = np.asarray(trial_sharpes, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < 2:
        return 0.0
    var = float(s.std(ddof=1))
    if var <= 0:
        return 0.0
    e = math.e
    return var * ((1 - EULER_MASCHERONI) * _norm_ppf(1 - 1.0 / n)
                  + EULER_MASCHERONI * _norm_ppf(1 - 1.0 / (n * e)))


def deflated_sharpe_ratio(returns: Sequence[float], trial_sharpes: Sequence[float],
                          periods_per_year: int = 252) -> Dict[str, float]:
    """다중 검정 선택편향을 보정한 Sharpe 신뢰도.

    **`trial_sharpes`에는 "서로 다른 전략/파라미터 조합"의 Sharpe를 넣어야 한다.**
    같은 전략을 여러 기간으로 평가한 값(CPCV 분할별 Sharpe)을 넣으면 안 된다.
    그것은 선택의 폭이 아니라 기간에 따른 표본 변동이라, 기준선만 부풀리고
    DSR을 의미 없이 낮춘다. 단일 전략의 유의성은 PSR로 본다.

    Returns:
        observed_sr   관측 Sharpe (연율)
        threshold_sr  운으로 기대되는 최대 Sharpe (연율)
        dsr           보정 후 "운이 아닐 확률". 0.95 이상이면 통계적으로 유의하다고 본다
        psr           기준 0 대비 확률적 Sharpe
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    observed = sharpe_ratio(r, periods_per_year)

    trials = np.asarray(trial_sharpes, dtype=float)
    trials = trials[np.isfinite(trials)]
    # 시도들의 Sharpe는 주기당 단위로 환산해 기준선을 계산한다
    sr0_periodic = expected_max_sharpe(trials / math.sqrt(periods_per_year))
    sr0 = sr0_periodic * math.sqrt(periods_per_year)

    return {
        "observed_sr": float(observed),
        "threshold_sr": float(sr0),
        "dsr": float(probabilistic_sharpe_ratio(r, sr0, periods_per_year)),
        "psr": float(probabilistic_sharpe_ratio(r, 0.0, periods_per_year)),
        "n_trials": int(len(trials)),
    }


# ======================================================================== PBO
def pbo_cscv(returns_matrix: np.ndarray, n_splits: int = 8) -> Dict[str, float]:
    """조합 대칭 교차검증으로 백테스트 과적합 확률을 구한다.

    Args:
        returns_matrix: (T, N) — 시점 × 전략(파라미터 조합)별 수익률
        n_splits: 시계열을 나눌 블록 수(짝수)

    Returns:
        pbo            학습 구간 1등이 검증 구간에서 중앙값 이하로 떨어질 확률
        median_oos_rank 검증 구간 순위의 중앙값(0~1, 1이 최고)
        n_combinations  평가한 조합 수
    """
    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim != 2 or R.shape[1] < 2:
        return {"pbo": float("nan"), "median_oos_rank": float("nan"), "n_combinations": 0}

    T, N = R.shape
    n_splits = max(4, n_splits - n_splits % 2)
    edges = np.linspace(0, T, n_splits + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(n_splits)]
    half = n_splits // 2

    ranks: List[float] = []
    for combo in itertools.combinations(range(n_splits), half):
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(n_splits) if i not in combo])
        if len(is_idx) < 8 or len(oos_idx) < 8:
            continue

        is_sr = np.array([sharpe_ratio(R[is_idx, j]) for j in range(N)])
        oos_sr = np.array([sharpe_ratio(R[oos_idx, j]) for j in range(N)])
        best = int(np.nanargmax(is_sr))
        # 검증 구간에서 그 전략의 상대 순위 (1이 최고)
        order = np.argsort(np.argsort(oos_sr))
        ranks.append(float(order[best]) / max(1, N - 1))

    if not ranks:
        return {"pbo": float("nan"), "median_oos_rank": float("nan"), "n_combinations": 0}

    arr = np.array(ranks)
    return {
        "pbo": float((arr < 0.5).mean()),
        "median_oos_rank": float(np.median(arr)),
        "n_combinations": int(len(arr)),
    }


# ======================================================================== 리포트
@dataclass
class ValidationReport:
    path_sharpes: List[float] = field(default_factory=list)
    path_mdds: List[float] = field(default_factory=list)
    dsr: Dict[str, float] = field(default_factory=dict)
    pbo: Dict[str, float] = field(default_factory=dict)
    n_groups: int = 0
    n_splits: int = 0
    split_sharpes: List[float] = field(default_factory=list)

    #: 파라미터 조합을 몇 개나 시험했는지. 0이면 DSR을 적용하지 않는다
    n_configs: int = 0

    @property
    def verdict(self) -> str:
        psr = self.dsr.get("psr", 0.0)
        paths = np.array(self.path_sharpes, dtype=float) if self.path_sharpes else np.array([0.0])
        negative = int((paths < 0).sum())

        if self.n_configs > 0:
            d = self.dsr.get("dsr", 0.0)
            if d >= 0.95:
                return f"유의 — 조합 {self.n_configs}개를 시험한 것을 보정해도 운으로 보기 어렵다"
            if d >= 0.80:
                return f"경계 — 조합 {self.n_configs}개 보정 후 DSR {d:.2f}. 표본을 늘려야 한다"
            return f"과적합 의심 — 조합 {self.n_configs}개 보정 시 DSR {d:.2f}"

        # 단일 설정의 유의성은 PSR과 경로 분포로 판단한다
        if psr >= 0.95 and negative == 0:
            return "0보다 유의 — 다만 실제 기대치는 단일 백테스트가 아니라 경로 중앙값이다"
        if psr >= 0.95:
            return f"0보다 유의하나 경로 {negative}개가 음수 — 기간 의존성이 크다"
        return "0보다 유의하다고 보기 어렵다"

    def pretty(self) -> str:
        s = np.array(self.path_sharpes, dtype=float) if self.path_sharpes else np.array([0.0])
        m = np.array(self.path_mdds, dtype=float) if self.path_mdds else np.array([0.0])
        lines = [
            "════════════════ 과적합 검증 ════════════════",
            f" CPCV          : {self.n_groups}그룹 · 분할 {self.n_splits}개 · 경로 {len(self.path_sharpes)}개",
            "───────────── 경로별 Sharpe 분포 ─────────────",
            f" 중앙값        : {np.median(s):>8.3f}",
            f" 평균 ± 표준편차: {s.mean():>8.3f} ± {s.std(ddof=1) if len(s) > 1 else 0:.3f}",
            f" 최저 / 최고   : {s.min():>8.3f} / {s.max():.3f}",
            f" 음수 경로     : {int((s < 0).sum())} / {len(s)}",
            "───────────── 경로별 최대낙폭 ─────────────",
            f" 중앙값 / 최악 : {np.median(m) * 100:>7.2f}% / {m.max() * 100:.2f}%",
            "───────────── 통계적 유의성 ─────────────",
            f" 관측 Sharpe   : {self.dsr.get('observed_sr', 0):>8.3f}  (전 경로 통합)",
            f" PSR (vs 0)    : {self.dsr.get('psr', 0):>8.3f}  ← 단일 설정의 유의성",
        ]
        if self.n_configs > 0:
            lines += [
                f" 시험한 조합   : {self.n_configs}개",
                f" 운의 기준선   : {self.dsr.get('threshold_sr', 0):>8.3f}",
                f" DSR           : {self.dsr.get('dsr', 0):>8.3f}  ← 선택편향 보정 후",
            ]
        else:
            lines.append("  (DSR은 --trial-sharpes로 시험한 조합들의 Sharpe를 줘야 계산한다)")
        if self.pbo and not math.isnan(self.pbo.get("pbo", float("nan"))):
            lines += [
                "───────────── 과적합 확률 ─────────────",
                f" PBO           : {self.pbo['pbo']:>8.3f}"
                f"  (조합 {self.pbo['n_combinations']}개)",
                f" 검증 순위 중앙: {self.pbo['median_oos_rank']:>8.3f}",
            ]
        lines += [
            "──────────────────────────────────────────",
            f" 판정: {self.verdict}",
            "══════════════════════════════════════════",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, object]:
        return {
            "path_sharpes": [round(x, 4) for x in self.path_sharpes],
            "path_mdds": [round(x, 5) for x in self.path_mdds],
            "dsr": {k: round(v, 5) for k, v in self.dsr.items()},
            "pbo": {k: (round(v, 5) if isinstance(v, float) else v)
                    for k, v in self.pbo.items()},
            "split_sharpes": [round(x, 4) for x in self.split_sharpes],
            "n_groups": self.n_groups, "n_splits": self.n_splits,
            "n_configs": self.n_configs, "verdict": self.verdict,
        }


def max_drawdown_from_returns(returns: Sequence[float]) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 0.0
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    return float(np.max((peak - equity) / np.where(peak > 0, peak, 1.0)))
