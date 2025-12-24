
n = 20
fib = []

a, b = 0, 1
for _ in range(n):
    fib.append(a)
    a, b = b, a + b

print("斐波那契数列前20项：")
print(fib)