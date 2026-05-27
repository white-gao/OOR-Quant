import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

# 1. 准备数据 (沿用 1.7B 模型的 R@5 指标)
categories = ['AD', 'Product', 'Video']

wo_think_1_7B = [1.7825, 1.9522, 1.5016]
with_think_1_7B = [1.1400, 1.4524, 1.1596]

# 2. 设置柱子的位置和宽度
x = np.arange(len(categories))
width = 0.35  

# 3. 创建画布和坐标轴
fig, ax = plt.subplots(figsize=(7, 5))

# 提取参考图中的配色方案
color_wo_fill = '#a8c6e2' # 浅蓝色填充
color_wo_edge = '#355c8c' # 深蓝色边框
color_w_fill  = '#f4b0ab' # 浅红色/粉色填充
color_w_edge  = '#b22625' # 深红色边框
color_text_red = '#b22625' # 红色字体

# 4. 绘制柱状图 (调整线宽 linewidth=1.5 增加边框的明显程度)
rects1 = ax.bar(x - width/2, wo_think_1_7B, width, label='1.7B w/o Think', 
                color=color_wo_fill, edgecolor=color_wo_edge, linewidth=1.5, zorder=3)
rects2 = ax.bar(x + width/2, with_think_1_7B, width, label='1.7B w/ Think', 
                color=color_w_fill, edgecolor=color_w_edge, linewidth=1.5, zorder=3)

# 5. 在柱子上方添加数据标签 (参考图中第二根柱子的文字为红色)
def autolabel(rects, t_color='black'):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold', color=t_color) 

autolabel(rects1, t_color='black')
autolabel(rects2, t_color='black') # w/ Think 的数据标签使用红色

# 6. 设置坐标轴和标签
ax.set_ylabel('Recall@5', fontsize=14) 
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=13) 
ax.tick_params(axis='y', labelsize=11) 

# 7. 调整Y轴刻度和上限
max_val = max(max(wo_think_1_7B), max(with_think_1_7B))
ax.set_ylim(bottom=0, top=max_val * 1.2) 
ax.set_yticks(np.arange(0, 2.6, 0.5))

# 8. 添加图例 
ax.legend(prop={'size': 11, 'weight': 'bold'}, frameon=True, edgecolor='black', loc='upper right')

# 9. 网格线与边框样式调整 (参考图的网格线是粗一点的灰色虚线，且顶部和右侧无边框)
ax.grid(axis='y', linestyle='--', color='#cccccc', linewidth=1.2, zorder=0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_color('#333333')
ax.spines['bottom'].set_color('#333333')

plt.tight_layout()

# 保存图片 (加了一个极浅的灰白背景色 facecolor='#fafafa' 以贴合原图质感)
plt.savefig('fig1_1.7B_only.png', dpi=300, bbox_inches='tight')
plt.close()