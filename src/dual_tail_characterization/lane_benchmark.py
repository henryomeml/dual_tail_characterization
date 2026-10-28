"""Authoritative corrected independent-lane PAM4 benchmark, not endpoint sigmas."""
ORIGINAL_ROWS = (
    (0, 0, 0, 0, .2), (0, 1, .03, 1, .2), (0, 2, .06, 2, .2), (0, 3, .09, 2.91, .2),
    (1, 0, 1, .03, .1), (1, 1, 1, 1, .1), (1, 2, 1, 2, .1), (1, 3, 1, 2.94, .1),
    (2, 0, 2, .06, .1), (2, 1, 2, 1, .1), (2, 2, 2, 2.97, .1), (2, 3, 2, 2.97, .1),
    (3, 0, 2.91, .09, .2), (3, 1, 2.94, 1, .2), (3, 2, 2.97, 2, .2), (3, 3, 3, 3, .2),
)


def original_lane_records():
    pairs = [(row[0], row[1]) for row in ORIGINAL_ROWS]
    if len(pairs) != 16 or set(pairs) != {(a, b) for a in range(4) for b in range(4)}:
        raise ValueError("original benchmark must contain exactly 16 unique ordered PAM4 pairs")
    return [dict(zip(("start", "end", "mean_start", "mean_end", "sigma"), row)) for row in ORIGINAL_ROWS]
