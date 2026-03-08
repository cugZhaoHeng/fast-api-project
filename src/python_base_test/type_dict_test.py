# 类型校验测试

from typing import TypedDict

class User(TypedDict):
    name: str
    age: int

def test_user():
    # TypedDict 创建字典对象，使用字典语法访问
    user: User = {'name': 'Alice', 'age': 30}
    assert user['name'] == 'Alice'
    assert user['age'] == 30

    # TypedDict 主要用于静态类型检查（如 mypy），不会在运行时进行类型验证
    # 以下代码在运行时不会报错，但类型检查器会提示类型错误
    user2: User = {'name': 'Bob', 'age': 'thirty'}  # 类型检查器会警告这里
    print(f"user2: {user2}")

    # 如果需要运行时类型验证，应该使用 Pydantic 或 dataclasses
    print("TypedDict test passed - note: TypedDict is for static type checking only")

if __name__ == "__main__":
    test_user()
    print("All tests passed.")