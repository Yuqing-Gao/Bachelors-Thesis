def w1_calculation(lst):
    total_sum = 0
    for k in range(len(lst) - 1):
        inner_sum = 0
        for j in range(len(lst) - 1):
            inner_sum += lst[k] - lst[j]
        total_sum += (lst[k + 1] - lst[k]) * inner_sum
    return total_sum


def w2_calculation(lst):
    total_sum = 0
    for k in range(len(lst) - 1):
        inner_sum = 0
        for j in range(len(lst) - 1):
            inner_sum += lst[j] ** 2 - lst[j] * lst[k]
        total_sum += (lst[k + 1] - lst[k]) * inner_sum
    return total_sum


result1 = w1_calculation([1, 5, 8, 3])  # -91
result2 = w2_calculation([1, 5, 8, 3])  # 474
print(result1, result2)
