from typing import List


def search_switching_pos(arr: List[int], pivot: int) -> (int, int):
    """ リストの中から pivot を起点として変換候補のインデックスを返す

    Parameters
    ---
    arr : List[int]
        探索するリスト
    pivot : int
        起点とする中心座標

    Returns
    ---
    left_index : int
        左側の座標 (pivot 値よりも最初に大きい値となった位置)
    right_index : int
        右側の座標 (pivot 値よりも最初に小さい値となった位置)
    """
    left_index = pivot
    for idx, val in enumerate(arr[:pivot]):
        if val > arr[pivot]:
            left_index = idx
            break

    right_index = pivot
    for idx, val in enumerate(arr[pivot:][::-1]):
        if val < arr[pivot]:
            right_index = len(arr) - idx - 1
            break

    return (left_index, right_index)


def switch_arr(arr: List[int]) -> None:
    pivot_pos = len(arr) // 2
    left_index = 0
    right_index = len(arr)
    while (pivot_pos != left_index) and (pivot_pos != right_index):
        left, right = search_switching_pos(arr[left_index:right_index], pivot_pos)
        left_index += left
        right_index = right
        temp_val = arr[left_index]
        arr[left_index] = arr[right_index]
        arr[right_index] = temp_val


def quick_sort(arr: List[int]) -> List[int]:
    switch_arr(arr)
    return arr


if __name__ == "__main__":
    print(quick_sort([0, 4, 5, 1, 3]))
