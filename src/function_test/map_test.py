# 测试 Python 的 map 和 reduce
# 测试列表推导式/列表生成式
from functools import reduce

from numpy import arange


def to_low(s: str):
    return s.upper()
def link(s1, s2):
    return s1+s2

if __name__ == '__main__':
    # map函数接收一个函数和一个迭代器，迭代器可以是 list或者 str
    names = ['zhangsan', 'lisi', 'wangwu']

    names = map(to_low, names)
    names = map(lambda x: x.lower(), names)
    namestr = reduce(link, names)
    print(list(names))
    print(namestr)

    a = [x for x in arange(1, 10) if x % 3 == 0]
    print(a)

    b = map(lambda x: x.upper(), [x for x in 'ahkj12h' if 'A' <= x <= 'z'])
    print(type(b))
    print(list(b))