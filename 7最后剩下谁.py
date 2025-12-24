def josephus(n, k):
    people = list(range(1, n + 1))  # 编号从 1 到 n
    idx = 0  # 起始位置

    while len(people) > 1:
        # 找到第 k 个报数的人（k 是从 1 开始计数）
        idx = (idx + k - 1) % len(people)
        people.pop(idx)

    return people[0]

result = josephus(233, 3)
print(f"最后剩下的是第 {result} 号")