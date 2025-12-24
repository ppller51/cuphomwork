
count = 0
results = []

for i in range(1, 5):      # 百位
    for j in range(1, 5):  # 十位
        for k in range(1, 5):  # 个位
            if i != j and i != k and j != k:
                num = i * 100 + j * 10 + k
                results.append(num)
                count += 1

print(f"共能组成 {count} 个互不相同的三位数：")
print(results)