content = input("请输入文件内容（ASCII标准字符）: ")

# 写入原文件
with open('test.txt', 'w') as f:
    f.write(content)

# 复制文件
with open('test.txt', 'r') as src, open('copy_test.txt', 'w') as dst:
    dst.write(src.read())

print("文件复制成功！")