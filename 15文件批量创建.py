import os
import random
import string

n = int(input("要创建多少个文件？"))
m = int(input("每个文件写入多少个随机字符？"))

# 创建目录
os.makedirs('test', exist_ok=True)

# 生成随机字符
chars = string.ascii_letters + string.digits

# 创建文件
for i in range(1, n + 1):
    filename = f'file{i}.txt'
    filepath = os.path.join('test', filename)

    with open(filepath, 'w') as f:
        rand_chars = ''.join(random.choice(chars) for _ in range(m))
        f.write(rand_chars)

# 修改所有文件：重命名 + 内容
for filename in os.listdir('test'):
    if filename.endswith('.txt'):
        old_path = os.path.join('test', filename)
        new_filename = filename.replace('.txt', '-python.txt')
        new_path = os.path.join('test', new_filename)

        # 重命名
        os.rename(old_path, new_path)

        # 追加内容
        with open(new_path, 'a') as f:
            f.write('\n-python')

print("批量文件创建与修改完成！")