import ctypes

# 加载 .dll 文件
lib = ctypes.CDLL('./engines.dll')

# 定义函数原型
lib.value_vector.argtypes = []  # 没有参数
lib.value_vector.restype = ctypes.c_int  # 返回类型为 int

lib.redirect_darts_output.argtypes = [ctypes.c_char_p]  # 参数类型为 const char*
lib.redirect_darts_output.restype = None  # 没有返回值

# 调用函数
result = lib.value_vector()
print(f"Result of value_vector: {result}")

lib.redirect_darts_output(b"Hello, DARTS!")