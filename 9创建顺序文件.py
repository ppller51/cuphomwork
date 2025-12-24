from pathlib import Path
import string
import random

# 创建目录
img_dir = Path('img')
img_dir.mkdir(exist_ok=True)

# 生成100个不同的文件名
for i in range(100):
    # 随机生成4个字母+数字组合
    chars = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    filename = f"{chars}.png"
    filepath = img_dir / filename


    filepath.touch()  # 创建空文件

print("已成功创建 img 目录及 100 个 .png 文件。")