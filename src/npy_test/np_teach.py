import numpy as np

class Student:
    def __init__(self, name, age):
        self.name = name
        self.age: float = age

    def __repr__(self):
        return f'{{"name": "{self.name}", "age": {self.age}}}'

def circle(radius: float) -> float:
    return 3.14 * radius * radius
radius: float = 3.0


s: float = circle(radius)
print(f"面积s= {s}, 半径: {radius}")
ages: list = [21,22,23,24,25,26,27,28,29]

st1: Student = Student('zhangsan', 20)
st2 = Student('lisi', 22)

students: list[Student] = [st1, st2 ]
print(students)
print(type(students[0]))
print(st1)

print('===========================')
ages = []
for s in students:
    ages.append(s.age)
print(ages)
print('===============================')

np.asarray([1,2,3])
v = np.asarray(list(s.age for s in students))
print(v)

# x = 1
#
# a:list[int] = [0.1,2,3, 'ddd']
# print(a)
# print(type(a))
#
# b = np.array(a)
# print(b)
# print(type(b))
# c = b * 2
# print(c)
