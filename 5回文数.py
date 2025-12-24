n = int(input("请输入一个正整数: "))
s = str(n)

if s == s[::-1]:
    print(f"{n} 是回文数")
else:
    print(f"{n} 不是回文数")