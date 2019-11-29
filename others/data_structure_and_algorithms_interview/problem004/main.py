from typing import List


def search_pair(arr: List[int], sum_val: int) -> List[List[int]]:
    """ 与えられたリストの中から合計した値が sum_val に等しくなるペアを返す

    Parameters
    ---
    arr : List[int]
        入力リスト
    sum_val : int
        合計値

    Returns
    ---
    pair_list : List[List[int]]
        合計値が sum_val となるペアリスト
    """
    pair_list = []
    for idx, first in enumerate(arr[:-1]):
        for second in arr[idx + 1:]:
            val = first + second
            if val != sum_val:
                continue
            pair_list.append([first, second])

    return pair_list


if __name__ == "__main__":
    print(search_pair([2, 6, 3, 9, 11], 9))
