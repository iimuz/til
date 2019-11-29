from typing import List, Set


def missing_number(arr: List[int], n: int) -> Set[int]:
    """ 与えられたリストの中から 1 <= idx <= n までの値でかけている番号のセットを返す

    Parameters
    ---
    arr : List[int]
        欠けのある番号リスト
    n : int
        最大番号

    Returns
    ---
    missing_set : Set[int]
        欠けている番号のセット

    Notes
    ---
    arr がソートされている必要はない。
    arr 内に重複があってもよい。
    """
    unique_set = set(arr)
    all_set = {num for num in range(1, n + 1)}
    diff_set = all_set - unique_set
    return diff_set


if __name__ == "__main__":
    print(missing_number([1, 2, 4], 5))
