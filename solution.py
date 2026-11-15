# Round 3: teams compete for the same reward (exponential, mean 10). If two or
# more teams take the same reward, all of them are locked out and nobody gets it.
#
# Greedy rule: other teams go for the rewards above the solo-optimal threshold,
# so leave those to them while any of them is free, and grab everything from
# GREED * that threshold upwards that they let go. When every other team is
# locked, nobody can collide with us, so take anything above the greedy bar.

import math

MEAN = 10.0  # rewards are exponential with this mean
GREED = 0.2  # take rewards above GREED * solo threshold; lower is greedier

cache = None  # (lockout, T, C) where C[n] = solo-optimal threshold with n rounds left


def solo_thresholds(lockout, T):
    # Exact solo DP for exponential rewards: V[n] = V[n - 1] + MEAN * exp(-C[n] / MEAN).
    V, C = [0.0] * (T + 2), [0.0] * (T + 2)
    for n in range(1, T + 2):
        C[n] = V[n - 1] - (V[n - 1 - lockout] if n - 1 - lockout > 0 else 0.0)
        V[n] = V[n - 1] + MEAN * math.exp(-C[n] / MEAN)
    return C


def play(reward: float, lockout: int, t: int, T: int, your_score: float, other_scores: list[tuple[float, bool]], is_locked: bool) -> bool:
    global cache
    reward = float(reward)
    if is_locked or not math.isfinite(reward) or reward <= 0:
        return False
    lockout = max(int(lockout), 0)
    if cache is None or cache[:2] != (lockout, T):
        cache = (lockout, T, solo_thresholds(lockout, T))
    solo = cache[2][min(max(T - t + 1, 1), T + 1)]

    if reward <= GREED * solo:
        return False
    others_free = any(not locked for _, locked in other_scores)
    return bool(not others_free or reward <= solo)
