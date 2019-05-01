from typing import List


def reverse_list(arr: List[int]) -> List[int]:
    center_pos = len(arr) // 2
    for idx, val in enumerate(arr[:center_pos]):
        switch_pos = -(idx + 1)
        arr[idx] = arr[switch_pos]
        arr[switch_pos] = val

    return arr


if __name__ == "__main__":
    print(reverse_list([1, 2, 3, 4, 5]))
