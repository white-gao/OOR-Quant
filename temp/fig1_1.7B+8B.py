import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

# 1. 准备数据 (从表格中提取 R@5 指标)
categories = ['AD', 'Product', 'Video']

# OpenOneRec-1.7B R@5
wo_think_1_7B = [1.7825, 1.9522, 1.5016]
with_think_1_7B = [1.1400, 1.4524, 1.1596]

# OpenOneRec-8B R@5
wo_think_8B = [2.3267, 2.4274, 1.1805]
with_think_8B = [1.2301, 1.7072, 1.1596]

# 2. 设置柱子的位置和宽度
x = np.arange(len(categories))
width = 0.2

# 3. 创建画布和坐标轴
fig, ax = plt.subplots(figsize=(10, 6))

# 修改配色方案：应用参考图风格 (蓝/红)，通过深浅区分 1.7B 和 8B
# 参考色：蓝色系 (w/o Think), 红色系 (w/ Think)
# 1.7B 用浅色，8B 用深色
color_17b_wo = '#a8c6e2' # 浅蓝填充
edge_17b_wo  = '#355c8c' # 浅蓝边框 (相对更深一点以显示)

color_8b_wo  = '#6b98c6' # 深蓝填充 (比1.7b深)
edge_8b_wo   = '#224068' # 深蓝边框

color_17b_w  = '#f4b0ab' # 浅红/粉填充
edge_17b_w   = '#b22625' # 浅红边框 (使用参考图的深红)

color_8b_w   = '#d96c68' # 深红填充 (比1.7b深)
edge_8b_w    = '#8b1919' # 更深的红边框

color_text_red = '#b22625' # 'w/ Think' 的数据标签文字颜色

# 4. 绘制柱状图 (共4组) - 使用新配色和线宽
rects1 = ax.bar(x - 1.5*width, wo_think_1_7B, width, label='1.7B w/o Think', 
                color=color_17b_wo, edgecolor=edge_17b_wo, linewidth=1.5, zorder=3)
rects2 = ax.bar(x - 0.5*width, with_think_1_7B, width, label='1.7B w/ Think', 
                color=color_17b_w, edgecolor=edge_17b_w, linewidth=1.5, zorder=3)
rects3 = ax.bar(x + 0.5*width, wo_think_8B, width, label='8B w/o Think', 
                color=color_8b_wo, edgecolor=edge_8b_wo, linewidth=1.5, zorder=3) 
rects4 = ax.bar(x + 1.5*width, with_think_8B, width, label='8B w/ Think', 
                color=color_8b_w, edgecolor=edge_8b_w, linewidth=1.5, zorder=3) 

# 5. 在柱子上方添加数据标签
def autolabel(rects, t_color='black'):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), # 稍微调高一点防重叠
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold', 
                    rotation=45, color=t_color) 

# w/o Think 用黑色，w/ Think 用红色 (对应图中的红色柱子)
autolabel(rects1, t_color='black')
autolabel(rects2, t_color='black')
autolabel(rects3, t_color='black')
autolabel(rects4, t_color='black')

# 6. 设置坐标轴和标签 (增大字体)
ax.set_ylabel('Recall@5', fontsize=14) 
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=13) 
ax.tick_params(axis='y', labelsize=11) 

# 7. 调整Y轴上限 (留出足够空间显示倾斜标签)
max_val = max(max(wo_think_1_7B), max(with_think_1_7B), max(wo_think_8B), max(with_think_8B))
ax.set_ylim(bottom=0, top=max_val * 1.25) # 稍微加高防止标签重叠
ax.set_yticks(np.arange(0, 3.1, 0.5)) # 设置明确的刻度

# 8. 添加图例 
ax.legend(prop={'size': 11, 'weight': 'bold'}, frameon=True, edgecolor='#333333', loc='upper right')

# 9. 网格线与边框样式调整 (匹配新风格)
ax.grid(axis='y', linestyle='--', color='#cccccc', linewidth=1.2, zorder=0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_color('#333333')
ax.spines['bottom'].set_color('#333333')

plt.tight_layout()

# 保存图片
plt.savefig('fig1_1.7B+8B.png', dpi=300, bbox_inches='tight')
plt.close()