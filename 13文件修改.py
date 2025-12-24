
with open('test.txt', 'r') as f:
    content = f.read()

# 修改内容
new_content = "python\n" + content + "\npython"

# 写回文件
with open('test.txt', 'w') as f:
    f.write(new_content)

print("文件已修改，前后添加了 'python'。")