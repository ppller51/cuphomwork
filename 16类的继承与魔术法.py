# 1. 定义 Person 类
class Person:
    def __init__(self, name, age, gender):
        self.name = name
        self.age = age
        self.gender = gender

    def personInfo(self):
        print(f"姓名: {self.name}, 年龄: {self.age}, 性别: {self.gender}")

    def __str__(self):
        return f"Person(name='{self.name}', age={self.age}, gender='{self.gender}')"


# 2. 定义 Student 类，继承 Person
class Student(Person):
    def __init__(self, name, age, gender, college, class_name):
        super().__init__(name, age, gender)  # 调用父类构造函数
        self.college = college
        self.class_name = class_name

    def personInfo(self):
        super().personInfo()
        print(f"学院: {self.college}, 班级: {self.class_name}")

    def __str__(self):
        return f"Student(name='{self.name}', age={self.age}, gender='{self.gender}', college='{self.college}', class='{self.class_name}')"

if __name__ == "__main__":
    # 创建 Person 对象
    p = Person("张三", 20, "男")
    p.personInfo()
    print(p)  # 输出 __str__ 结果

    print("-" * 40)

    # 创建 Student 对象
    s = Student("李四", 19, "女", "计算机学院", "软件工程2班")
    s.personInfo()
    print(s)  # 输出 __str__ 结果