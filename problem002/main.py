from typing import List, Set


def search_duplicate_number(arr: List[int]) -> Set[int]:
    unique_set = set()
    duplicate_set = set()
    for val in arr:
        if val in unique_set:
            duplicate_set.add(val)
        unique_set.add(val)

    return duplicate_set


if __name__ == "__main__":
    print(search_duplicate_number([0, 1, 2, 2]))
