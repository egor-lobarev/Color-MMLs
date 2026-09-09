"""F-критерий сравнения двух STRESS (Мелгоса).

Стандартный инструмент колориметрии для ответа на вопрос «действительно ли
формула A согласуется с человеческими данными лучше, чем формула B, или разница
в пределах случайности». Описан в García P.A., Huertas R., Melgosa M., Cui G.
Measurement of the relationship between perceived and computed color
differences. J Opt Soc Am A. 2007. V. 24. № 7. P. 1823-1829.

Критерий:

    F = (STRESS_A / STRESS_B)^2,   df1 = df2 = N - 1

где N — число пар, на которых обе величины посчитаны (обязательно одних и тех
же). При двустороннем тесте уровня alpha:

    F < F_{alpha/2, N-1, N-1}    -> A значимо лучше B
    F > F_{1-alpha/2, N-1, N-1}  -> B значимо лучше A
    иначе                        -> различие незначимо

Критерий инвариантен к единицам STRESS (доли или проценты): в F входит только
отношение.

⚠️ Ограничение. Критерий выведен для STRESS, посчитанного один раз на N парах.
Если подставить среднее по фолдам кросс-валидации, результат становится
приближением: разброс между фолдами в F не входит. Для величин с большим
разбросом (на Leeds sigma достигает 0.05 при STRESS 0.23) вывод следует
перепроверять на объединённых по фолдам предсказаниях — см. ftest_from_residuals.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import numpy as np
from scipy import stats


@dataclass
class FTestResult:
    stress_a: float
    stress_b: float
    n_pairs: int
    f: float
    df: int
    p_value: float
    alpha: float
    f_crit_low: float
    f_crit_high: float
    verdict: str            # "A", "B" или "ns"
    better: Optional[str]   # имя лучшей стороны либо None

    def as_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}

    def __str__(self) -> str:
        rel = {"A": "A значимо лучше", "B": "B значимо лучше",
               "ns": "различие незначимо"}[self.verdict]
        return (f"STRESS_A={self.stress_a:.4f} STRESS_B={self.stress_b:.4f} "
                f"N={self.n_pairs} F={self.f:.4f} p={self.p_value:.4f} -> {rel}")


def stress_ftest(stress_a: float, stress_b: float, n_pairs: int,
                 alpha: float = 0.05,
                 name_a: str = "A", name_b: str = "B") -> FTestResult:
    """F-критерий Мелгосы для двух STRESS, посчитанных на одних и тех же N парах.

    Меньший STRESS = лучшее согласие с человеком, поэтому «A лучше» отвечает
    F < 1 при достаточной значимости.
    """
    if not (stress_a > 0 and stress_b > 0):
        raise ValueError("STRESS должен быть строго положителен")
    if n_pairs < 3:
        raise ValueError("нужно не менее 3 пар")

    df = int(n_pairs) - 1
    f = float(stress_a) ** 2 / float(stress_b) ** 2
    dist = stats.f(df, df)

    f_low = dist.ppf(alpha / 2.0)
    f_high = dist.ppf(1.0 - alpha / 2.0)
    cdf = dist.cdf(f)
    p = 2.0 * min(cdf, 1.0 - cdf)

    if f < f_low:
        verdict, better = "A", name_a
    elif f > f_high:
        verdict, better = "B", name_b
    else:
        verdict, better = "ns", None

    return FTestResult(stress_a=float(stress_a), stress_b=float(stress_b),
                       n_pairs=int(n_pairs), f=f, df=df, p_value=float(p),
                       alpha=alpha, f_crit_low=float(f_low),
                       f_crit_high=float(f_high), verdict=verdict, better=better)


def stress(pred: Sequence[float], target: Sequence[float]) -> float:
    """STRESS в долях единицы: ||k*x - y|| / ||y||, k = argmin ||k*x - y||.

    Та же нормировка, что и во всех расчётах статьи. У García et al. величина
    выражается в процентах — отличается множителем 100, на F-критерий не влияет.
    """
    x = np.asarray(pred, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"формы не совпадают: {x.shape} против {y.shape}")
    denom = float(x @ x)
    if denom <= 0:
        raise ValueError("нулевой вектор предсказаний")
    k = float(x @ y) / denom
    return float(np.linalg.norm(k * x - y) / np.linalg.norm(y))


def ftest_from_residuals(pred_a: Sequence[float], pred_b: Sequence[float],
                         target: Sequence[float], alpha: float = 0.05,
                         name_a: str = "A", name_b: str = "B") -> FTestResult:
    """Строгий вариант: STRESS считается по по-парным предсказаниям.

    Предпочтителен там, где предсказания доступны: N здесь — фактическое число
    пар, а не оценка, и приближение «среднее по фолдам вместо одного STRESS»
    не используется.
    """
    y = np.asarray(target, dtype=float)
    sa = stress(pred_a, y)
    sb = stress(pred_b, y)
    return stress_ftest(sa, sb, n_pairs=len(y), alpha=alpha,
                        name_a=name_a, name_b=name_b)
