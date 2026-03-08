import matplotlib.pyplot as plt

# ============== 1. 数据准备 ==============
years = ['2021', '2022', '2023', '2024', '2025']
product_a = [100, 120, 130, 140, 150]
product_b = [80, 90, 100, 110, 120]
product_c = [50, 60, 80, 100, 130]

# ============== 2. 设置中文显示 ==============
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============== 3. 创建图表 ==============
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

# ============== 4. 绘制堆叠面积图 ==============
ax.stackplot(years, product_a, product_b, product_c,
             labels=['产品A', '产品B', '产品C'],
             colors=['#3498db', '#e74c3c', '#2ecc71'],
             alpha=0.8,
             edgecolor='white',
             linewidth=1)

# ============== 5. 添加图表元素 ==============
ax.set_title('2021-2025年三类产品销售额变化趋势', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('年份', fontsize=12)
ax.set_ylabel('销售额（万元）', fontsize=12)

ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.3)

# ============== 6. 添加数据标签（修复版） ==============
# 计算每个产品在最后一年的累计高度
last_idx = len(years) - 1
cumulative = [0, product_a[last_idx], product_a[last_idx] + product_b[last_idx]]

for i, (data, label) in enumerate(zip([product_a, product_b, product_c], 
                                       ['产品A', '产品B', '产品C'])):
    # y位置 = 该层底部 + 该层厚度的一半
    y_pos = cumulative[i] + data[last_idx] / 2
    ax.text(last_idx, y_pos, 
            f'{data[last_idx]}', 
            fontsize=9, color='white', fontweight='bold',
            ha='center', va='center')

# ============== 7. 调整布局并显示 ==============
plt.tight_layout()
plt.show()