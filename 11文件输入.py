import random


with open('data1.txt', 'w') as f:
    for _ in range(10):
        row = [random.randint(1, 100) for _ in range(3)]
        f.write(','.join(map(str, row)) + '\n')

# 2. 读取第二列并计算
second_col = []
with open('data1.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 2:
            second_col.append(int(parts[1]))

# 计算结果
max_val = max(second_col)
min_val = min(second_col)
avg_val = sum(second_col) / len(second_col)
median_val = sorted(second_col)[len(second_col)//2]  # 中位数（奇数长度）

print(f"最大值: {max_val}")
print(f"最小值: {min_val}")
print(f"平均值: {avg_val:.2f}")
print(f"中位数: {median_val}")