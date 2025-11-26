
import time

# 记录开始时间
start_time = time.perf_counter()
print(start_time)

# ========== 你要测量的代码段 ==========
# 示例：模拟一些计算任务
total = 0
for i in range(1000000):
    total += i
# =====================================

# 记录结束时间
end_time = time.perf_counter()
print(end_time)

# 计算耗时（秒）
elapsed_time = end_time - start_time

print(f"程序耗时: {elapsed_time:.4f} 秒")

import time

start_time = time.time()
print(start_time)

# 你的代码
total = 0
for i in range(1000000):
    total += i

end_time = time.time()
print(end_time)

elapsed_time = end_time - start_time
print(f"程序耗时: {elapsed_time:.4f} 秒")