from pathlib import Path
import random

img_dir = Path('img')
png_files = list(img_dir.glob('*.png'))  # 获取所有 png 文件

# 随机选择 50 个文件
selected_files = random.sample(png_files, min(50, len(png_files)))

for file in selected_files:
    new_name = file.stem + '.jpg'  # 去掉扩展名，加上 jpg
    file.rename(new_name)

print(f"已将 {len(selected_files)} 个文件从 .png 改为 .jpg。")