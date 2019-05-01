from typing import List


def remove_duplicate(arr: List[int]) -> List[int]:
    sorted_arr = sorted(arr)
    prev_val = sorted_arr[0] - 1
    unique_arr = []
    for val in sorted_arr:
        if val == prev_val:
            continue
        unique_arr.append(val)
        prev_val = val

    return unique_arr


if __name__ == "__main__":
    print(remove_duplicate([1, 2, 3, 1]))
