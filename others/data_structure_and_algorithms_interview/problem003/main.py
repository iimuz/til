import sys
from typing import List


def search_min_max(arr: List[int]) -> (int, int):
    """ Search min and max value without using python function.

    Parameters
    ---
    arr : List[int]
        input array

    Returns:
    ---
    min_val : int
        minimum value contained arr

    max_val : int
        maximum value contained arr
    """
    min_val = sys.maxsize
    max_val = -sys.maxsize
    for val in arr:
        min_val = val if val < min_val else min_val
        max_val = val if val > max_val else max_val

    return (min_val, max_val)


if __name__ == "__main__":
    print(search_min_max([3, -1, 5, 2]))
