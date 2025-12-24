import random

with open('data.txt', 'w') as f:
    for _ in range(100000):
        num = random.randint(1, 100)
        f.write(str(num) + '\n')

print("已成功生成 data.txt 文件，包含 100000 个随机数。")