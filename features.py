import numpy as np

WRIST = 0
MIDDLE_MCP = 9  # base of the middle finger


def landmarks_to_array(hand_landmarks, width, height):
    """Convert MediaPipe landmarks to a (21, 2) array in pixel units."""
    return np.array(
        [[lm.x * width, lm.y * height] for lm in hand_landmarks.landmark],
        dtype=np.float64,
    )


def normalise(points):
    """(21, 2) pixel points -> 42 numbers, or None if the hand is degenerate.

    Translates to the wrist and scales by hand size. Does NOT rotate,
    because thumbs up vs thumbs down depends on orientation.
    """
    pts = np.asarray(points, dtype=np.float64)
    shifted = pts - pts[WRIST]
    scale = np.linalg.norm(shifted[MIDDLE_MCP])
    if scale < 1e-6:
        return None
    return (shifted / scale).flatten()