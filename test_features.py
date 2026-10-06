import numpy as np
from features import normalise, MIDDLE_MCP

rng = np.random.default_rng(0)
hand = rng.uniform(100, 500, size=(21, 2))  # fake hand in pixels


def test_wrist_is_origin_and_scale_is_one():
    f = normalise(hand).reshape(21, 2)
    assert np.allclose(f[0], 0)
    assert np.isclose(np.linalg.norm(f[MIDDLE_MCP]), 1.0)


def test_translation_invariant():
    assert np.allclose(normalise(hand), normalise(hand + [250, -80]))


def test_scale_invariant():
    assert np.allclose(normalise(hand), normalise(hand * 3.7))


def test_rotation_is_NOT_removed():
    # Rotating by 180 degrees (thumbs up -> thumbs down) must change features
    flipped = -hand
    assert not np.allclose(normalise(hand), normalise(flipped))


def test_degenerate_hand_returns_none():
    assert normalise(np.zeros((21, 2))) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("all tests passed")