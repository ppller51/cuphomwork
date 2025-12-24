def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    diff_lines = []
    max_len = max(len(lines1), len(lines2))

    for i in range(max_len):
        line1 = lines1[i].strip() if i < len(lines1) else ""
        line2 = lines2[i].strip() if i < len(lines2) else ""
        if line1 != line2:
            diff_lines.append(i + 1)  # 行号从1开始

    return diff_lines

diffs = compare_files('test.txt', 'copy_test.txt')
if diffs:
    print("不同行的编号：", diffs)
else:
    print("两个文件完全相同。")