n = int(input("请输入一个正整数: "))

num = n
count = 0
digits = []

while num > 0:
    digit = num % 10
    digits.append(digit)
    num //= 10
    count += 1

print(f"该数是 {count} 位数")
print("逆序打印各位数字：", end="")
for d in digits:
    print(d, end=" ")
print()