if __name__ == "__main__":
    lst = list(range(1000))
    for idx in range(len(lst)-1, -1, -1):  # 从后往前遍历
        if lst[idx] % 2 == 1:
            lst.pop(idx)
    print(lst)  # 输出所有偶数