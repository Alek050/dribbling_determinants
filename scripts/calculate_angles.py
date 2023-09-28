import numpy as np

def get_angles(A: list, B: list, format: str = "degree") -> np.ndarray:
    """
    Calculates the angle between 2D vectors.

    Args:
        A: A list of 2D vectors represented as a list of lists [[x1, y1], ..., [xn, yn]].
        B: A list of 2D vectors represented as a list of lists [[x1, y1], ..., [xn, yn]].
        format: A string representing the format of the returned angles. Valid values are "degree" and "radian". Default is "degree".

    Returns:
        A numpy array with angles between the two vectors.

    Raises:
        AssertionError: If A and B do not contain the same amount of vectors.
        AssertionError: If format is not "degree" or "radian".
    """

    assert len(A) == len(B), "A and B should contain the same amount of vectors"
    assert (
        format == "degree" or format == "radian"
    ), f"input 'format' must be 'degree' or 'radian', not '{format}'."

    angles = []
    for a, b in zip(A, B):
        angles.append(get_angle(a, b, format=format))

    return np.array(angles)


def get_angle(a: list, b: list, format: str = "degree") -> float:
    """
    Calculates the angle between two 2D vectors.

    Args:
        a: A list representing the first 2D vector.
        b: A list representing the second 2D vector.
        format: A string representing the format of the returned angle. Valid values are "degree" and "radian". Default is "degree".

    Returns:
        A float representing the angle between the two vectors.

    Raises:
        AssertionError: If a and b are not of length 2.
        AssertionError: If format is not "degree" or "radian".
    """

    assert len(a) == len(a) == 2, "a and b should be of len(2)"
    assert (
        format == "degree" or format == "radian"
    ), f"input 'format' must be 'degree' or 'radian', not '{format}'."

    inner = np.inner(a, b)
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    cos = np.inf if norms == 0 else inner / norms

    rad = np.arccos(np.clip(cos, -1.0, 1.0))
    if format == "radian":
        return rad
    else:
        return np.rad2deg(rad)
