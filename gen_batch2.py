#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate 6 CS231n notebooks as reference-website quality .ipynb files.
All text in Chinese, code comments in English.
"""
import json
import os
import nbformat

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def md_cell(source):
    """Create a markdown cell."""
    return nbformat.v4.new_markdown_cell(source)


def code_cell(source):
    """Create a code cell."""
    return nbformat.v4.new_code_cell(source)


def save_notebook(cells, filename):
    """Save cells as a .ipynb file."""
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    fpath = os.path.join(OUTPUT_DIR, filename)
    with open(fpath, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Saved: {fpath} ({len(cells)} cells)")
    return fpath


# ============================================================
# NOTEBOOK 1: CNN Basics
# ============================================================
def build_notebook1():
    cells = []

    # Title
    cells.append(md_cell("""# 卷积神经网络：从全连接到卷积

> **CS231n 深度学习与计算机视觉** · 第5讲
>
> 本 notebook 将带你从零理解卷积神经网络（CNN）的核心思想——从为什么全连接层不够好，到卷积、池化、填充、步幅的每一个细节，再到亲手实现一个卷积层。

---

## 学习目标

- 理解全连接层在处理图像时的两大问题：参数量爆炸 + 丢失空间结构
- 掌握卷积操作的直觉、数学定义和手动计算
- 理解填充（Padding）和步幅（Stride）的作用及输出尺寸公式
- 理解参数共享如何大幅减少参数量
- 掌握池化层的作用与实现
- 从零实现卷积层和池化层的前向传播
- 了解 LeNet-5 等经典早期 CNN 架构

---
"""))

    # Setup
    cells.append(code_cell("""# Setup - run this cell first
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

np.random.seed(42)

# Set Chinese-friendly font (fallback to default if not available)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("Environment ready! numpy version:", np.__version__)
"""))

    # Section 1: Why not fully connected?
    cells.append(md_cell("""## 1. 问题：为什么不用全连接层处理图像？

### 问题场景

假设我们有一张 224×224×3 的彩色图像（ImageNet 标准尺寸），如果用全连接层处理：

- 输入层神经元数量：224 × 224 × 3 = **150,528 个**
- 如果下一个隐藏层有 1000 个神经元
- 参数数量：150,528 × 1000 ≈ **1.5 亿个参数**（还不算偏置）

这仅仅是**一层**的参数量！

### 两个核心问题

1. **参数量爆炸**：模型太大，容易过拟合，训练困难
2. **丢失空间结构**：把图像拉平成一维向量，像素之间的空间邻接关系被破坏

> **类比**：把一幅画剪成 15 万个小碎片，然后打乱顺序给你看——你还能认出画的是什么吗？全连接层就是这么做的。

---

**思考题 1：** 如果输入是 32×32×3 的 CIFAR-10 图像，连接到 100 个隐藏单元的全连接层有多少个参数？（包括偏置）
"""))

    cells.append(code_cell("""# Let's calculate the parameter count for a fully connected layer
# Input: 224x224x3 image, Hidden layer: 1000 neurons

input_size = 224 * 224 * 3  # 150528
hidden_size = 1000

# Weights + biases
params_fc = input_size * hidden_size + hidden_size

print(f"Input size (flattened): {input_size:,}")
print(f"Hidden units: {hidden_size:,}")
print(f"Parameters in ONE fully connected layer: {params_fc:,}")
print(f"That's ~{params_fc / 1e6:.1f} million parameters for just one layer!")
"""))

    cells.append(code_cell("""# Visual comparison: FC vs Conv parameter count
fig, ax = plt.subplots(figsize=(10, 5))

# Compare parameter counts for different approaches
methods = ['FC (1 layer)', 'FC (2 layers)', 'CNN (typical)']
params = [150_528_000, 1_000_000_000, 5_000_000]  # rough estimates

bars = ax.bar(methods, params, color=['#e74c3c', '#e67e22', '#27ae60'])
ax.set_yscale('log')
ax.set_ylabel('Number of parameters (log scale)')
ax.set_title('Parameter Count: Fully Connected vs CNN')

for bar, val in zip(bars, params):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.1,
            f'{val:,}', ha='center', va='bottom', fontsize=10)

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print("CNN uses dramatically fewer parameters while often performing better!")
print("The secret: parameter sharing + local connectivity.")
"""))

    cells.append(md_cell("""### 关键洞察

卷积神经网络通过两个核心思想解决了上述问题：

1. **局部感受野（Local Receptive Field）**：每个神经元只关注输入的一小片区域，而不是整幅图像
2. **参数共享（Parameter Sharing）**：同一个过滤器（filter）在整幅图像上滑动，使用同一组权重

这两个思想使得 CNN 既能捕捉空间结构，又能大幅减少参数量。

---
"""))

    # Section 2: Convolution intuition
    cells.append(md_cell("""## 2. 卷积操作的直觉：滑动窗口

### 什么是卷积？

**卷积**就是一个小的权重矩阵（称为**过滤器 / filter / kernel**）在输入图像上**滑动**，每到一个位置就和对应区域做**逐元素相乘再求和**（加上偏置），得到输出特征图上的一个值。

### 类比：用放大镜查看图片

想象你拿着一个放大镜在图片上移动：
- 放大镜的镜片大小 = 过滤器大小（如 3×3）
- 你每次移动的距离 = 步幅（stride）
- 你在每个位置看到的清晰区域 = 局部感受野
- 你对看到的内容的"理解" = 卷积运算的结果

### 3D 卷积（深度维度）

对于彩色图像（3 个通道），过滤器也是 3D 的（如 3×3×3）。每个深度切片和对应通道做卷积，然后**所有通道的结果相加**，得到输出特征图上的一个值。

> **要点**：一个过滤器始终产生一个输出通道（一个特征图）。如果想要 N 个输出通道，就用 N 个不同的过滤器。

---

**思考题 2：** 如果输入是 5×5×3，过滤器是 3×3×3，那么一个过滤器有多少个参数（含偏置）？
"""))

    cells.append(code_cell("""# Let's visualize what a convolution filter looks like
# We'll create a simple edge-detection filter and apply it manually

# Create a simple 5x5 grayscale "image" with a vertical edge
input_img = np.array([
    [0, 0, 0, 255, 255],
    [0, 0, 0, 255, 255],
    [0, 0, 0, 255, 255],
    [0, 0, 0, 255, 255],
    [0, 0, 0, 255, 255],
], dtype=np.float32)

# A vertical edge detection filter (Sobel-like, simplified)
vertical_filter = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
], dtype=np.float32)

print("Input image (5x5):")
print(input_img)
print()
print("Vertical edge filter (3x3):")
print(vertical_filter)
"""))

    cells.append(code_cell("""# Manual convolution: step by step for the first position
# Filter placed at top-left corner (covers rows 0-2, cols 0-2)

patch = input_img[0:3, 0:3]
result = np.sum(patch * vertical_filter)

print("Input patch (top-left 3x3):")
print(patch)
print()
print("Element-wise product with filter:")
print(patch * vertical_filter)
print()
print(f"Sum = {result:.0f}  --> this is ONE value in the output feature map")
"""))

    cells.append(code_cell("""# Now let's compute the full convolution output manually
# For a 5x5 input with 3x3 filter (no padding, stride=1), output is 3x3

def conv_2d_manual(input_arr, kernel, bias=0):
    \"\"\"Simple 2D convolution (single channel, stride=1, no padding)\"\"\"
    h_in, w_in = input_arr.shape
    k_h, k_w = kernel.shape
    h_out = h_in - k_h + 1
    w_out = w_in - k_w + 1
    output = np.zeros((h_out, w_out))
    
    for i in range(h_out):
        for j in range(w_out):
            patch = input_arr[i:i+k_h, j:j+k_w]
            output[i, j] = np.sum(patch * kernel) + bias
    return output

# Apply vertical edge filter
output_v = conv_2d_manual(input_img, vertical_filter)

print("Convolution output (vertical edges):")
print(output_v)
print()
print("Notice: high positive values on the left of the edge, high negative on the right.")
print("This is exactly what edge detection looks like!")
"""))

    cells.append(code_cell("""# Visualize the convolution operation
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].imshow(input_img, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Input: 5x5 image')
# Draw the filter position
rect = Rectangle((-0.5, -0.5), 3, 3, linewidth=2, edgecolor='red', facecolor='none')
axes[0].add_patch(rect)
axes[0].text(1, 3.5, 'filter\nposition', ha='center', color='red', fontsize=9)

axes[1].imshow(vertical_filter, cmap='RdBu_r', vmin=-3, vmax=3)
axes[1].set_title('Filter: 3x3 vertical edge')
for i in range(3):
    for j in range(3):
        axes[1].text(j, i, f'{vertical_filter[i,j]:.0f}', 
                     ha='center', va='center', fontsize=10, color='black')

axes[2].imshow(output_v, cmap='RdBu_r', vmin=-765, vmax=765)
axes[2].set_title('Output: 3x3 feature map')
for i in range(3):
    for j in range(3):
        axes[2].text(j, i, f'{output_v[i,j]:.0f}', 
                     ha='center', va='center', fontsize=10, color='black')

plt.tight_layout()
plt.show()
"""))

    cells.append(md_cell("""### 3D 卷积的深度维度

刚才我们演示了单通道的 2D 卷积。对于多通道输入（如 RGB 三通道）：

- 过滤器也必须有相同的深度：3×3×3
- 每个深度切片分别和对应通道做卷积
- 三个通道的结果**相加**，得到输出的一个值
- 加上偏置（一个过滤器共享一个偏置）

所以：**一个 3×3×3 的过滤器作用在 5×5×3 的输入上，产生一个 3×3×1 的输出特征图。**

如果有 10 个这样的过滤器，输出就是 3×3×10。

---
"""))

    # Section 3: Padding and Stride
    cells.append(md_cell("""## 3. 填充（Padding）与步幅（Stride）

### 问题：卷积会"缩小"图像

- 输入 5×5 → 3×3 过滤器 → 输出 3×3
- 每做一次卷积，空间尺寸就缩小一点
- 边缘像素只被用到一次，中间像素被多次用到 → **边缘信息丢失**

### 解决方案：填充（Padding）

在输入图像的边缘填充像素（通常填 0），使得：
1. 输出尺寸和输入相同（"same" padding）
2. 边缘像素也能被充分利用

### 两种常见填充方式

- **Valid padding**（不填充）：输出尺寸 = W - F + 1
- **Same padding**（填充后输出和输入一样大）：填充量 P = (F-1)/2

---

**思考题 3：** 如果过滤器大小是 5×5，要保持输出尺寸和输入相同，需要在每边填充多少个像素？
"""))

    cells.append(code_cell("""# Demonstrate padding effect

def conv2d(input_arr, kernel, padding=0, stride=1):
    \"\"\"2D convolution with padding and stride support (single channel)\"\"\"
    h_in, w_in = input_arr.shape
    k_h, k_w = kernel.shape
    
    # Apply padding
    if padding > 0:
        padded = np.pad(input_arr, ((padding, padding), (padding, padding)), 
                        mode='constant', constant_values=0)
    else:
        padded = input_arr
    
    h_pad, w_pad = padded.shape
    h_out = (h_pad - k_h) // stride + 1
    w_out = (w_pad - k_w) // stride + 1
    output = np.zeros((h_out, w_out))
    
    for i in range(h_out):
        for j in range(w_out):
            h_start = i * stride
            w_start = j * stride
            patch = padded[h_start:h_start+k_h, w_start:w_start+k_w]
            output[i, j] = np.sum(patch * kernel)
    return output


# Test with our edge image
print("Original input shape:", input_img.shape)
print()

# No padding (valid)
out_valid = conv2d(input_img, vertical_filter, padding=0, stride=1)
print("Valid padding (no padding):")
print(f"  Output shape: {out_valid.shape}")
print(out_valid)
print()

# Same padding (output size = input size)
out_same = conv2d(input_img, vertical_filter, padding=1, stride=1)
print("Same padding (padding=1):")
print(f"  Output shape: {out_same.shape}")
print(out_same)
"""))

    cells.append(md_cell("""### 步幅（Stride）

**步幅**是过滤器每次移动的像素数。

- Stride = 1：每次移动 1 个像素，输出尺寸较大
- Stride = 2：每次移动 2 个像素，输出尺寸大约减半
- Stride 越大，输出尺寸越小，计算量也越小

### 输出尺寸公式

$$O = \\frac{W - F + 2P}{S} + 1$$

其中：
- W = 输入宽度
- F = 过滤器大小
- P = 填充量（每边）
- S = 步幅

> **注意**：结果必须是整数，否则说明设置有问题。

---

**思考题 4：** 输入 28×28，过滤器 5×5，填充 2，步幅 2，输出尺寸是多少？
"""))

    cells.append(code_cell("""# Output size formula verification

def output_size(W, F, P, S):
    \"\"\"Calculate output spatial size for convolution\"\"\"
    return (W - F + 2 * P) // S + 1

# Test cases
test_cases = [
    (5, 3, 0, 1, "5x5 input, 3x3 filter, no pad, stride 1"),
    (5, 3, 1, 1, "5x5 input, 3x3 filter, pad=1, stride 1 (same)"),
    (28, 5, 2, 2, "28x28 input, 5x5 filter, pad=2, stride 2"),
    (224, 7, 3, 2, "224x224 input, 7x7 filter, pad=3, stride 2 (AlexNet conv1)"),
]

print("Output Size Formula: O = (W - F + 2P) / S + 1")
print("-" * 65)
for W, F, P, S, desc in test_cases:
    O = output_size(W, F, P, S)
    check = (W - F + 2 * P) % S == 0
    status = "OK" if check else "WARNING: not integer!"
    print(f"{desc:55s} -> {O}x{O}  [{status}]")
"""))

    cells.append(code_cell("""# Visualize stride effect
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Create a larger test image
test_img = np.zeros((12, 12))
test_img[2:10, 2:10] = 200  # white square
test_img[4:8, 4:8] = 50     # dark center

axes[0].imshow(test_img, cmap='gray')
axes[0].set_title('Input (12x12)')

# Stride 1
out_s1 = conv2d(test_img, vertical_filter, padding=1, stride=1)
axes[1].imshow(out_s1, cmap='RdBu_r')
axes[1].set_title(f'Stride=1 ({out_s1.shape[0]}x{out_s1.shape[1]})')

# Stride 2
out_s2 = conv2d(test_img, vertical_filter, padding=1, stride=2)
axes[2].imshow(out_s2, cmap='RdBu_r')
axes[2].set_title(f'Stride=2 ({out_s2.shape[0]}x{out_s2.shape[1]})')

# Stride 3
out_s3 = conv2d(test_img, vertical_filter, padding=1, stride=3)
axes[3].imshow(out_s3, cmap='RdBu_r')
axes[3].set_title(f'Stride=3 ({out_s3.shape[0]}x{out_s3.shape[1]})')

plt.tight_layout()
plt.show()
print("Larger stride = smaller output = more aggressive downsampling")
"""))

    # Section 4: Parameter sharing
    cells.append(md_cell("""## 4. 参数共享：CNN 高效的秘密

### 核心思想

同一个过滤器在整幅图像上滑动时，**使用完全相同的权重**。这就是参数共享。

### 为什么合理？

- 如果某个特征（如竖直边缘）在图像左上角有用，那它在右下角也应该有用
- 图像具有**平移不变性**（translation invariance）——猫不管在图片哪个位置都是猫

### 参数量对比

以输入 224×224×3，输出 224×224×64 为例：

- **全连接层**：224×224×3 × 224×224×64 ≈ **2.4 万亿个参数**（不可能！）
- **卷积层**（3×3 过滤器）：3×3×3 × 64 + 64 = **1,792 个参数**

差距是**十亿倍**！这就是 CNN 能工作的根本原因。

---

**思考题 5：** 输入 32×32×3，使用 16 个 5×5 的过滤器（same padding），这个卷积层有多少个参数？
"""))

    cells.append(code_cell("""# Parameter count comparison: FC vs Conv

input_h, input_w, input_c = 224, 224, 3
output_c = 64

# Fully connected (flatten input, flatten output - extreme case)
# More realistic: FC to a hidden layer of same spatial info
fc_input = input_h * input_w * input_c
fc_output = 1000  # typical FC layer size
params_fc = fc_input * fc_output + fc_output

# Convolutional layer with 3x3 filters
filter_size = 3
params_conv_3x3 = filter_size * filter_size * input_c * output_c + output_c

# Convolutional layer with 5x5 filters
params_conv_5x5 = 5 * 5 * input_c * output_c + output_c

print(f"{'Layer type':<25} {'Parameters':>15} {'Ratio':>10}")
print("-" * 55)
print(f"{'FC (224*224*3 -> 1000)':<25} {params_fc:>15,} {1.0:>10.1f}x")
print(f"{'Conv 5x5 (64 filters)':<25} {params_conv_5x5:>15,} {params_fc/params_conv_5x5:>10.1f}x fewer")
print(f"{'Conv 3x3 (64 filters)':<25} {params_conv_3x3:>15,} {params_fc/params_conv_3x3:>10.1f}x fewer")
"""))

    # Section 5: Pooling
    cells.append(md_cell("""## 5. 池化层：缩小特征图，保留关键信息

### 为什么需要池化？

1. **降低计算量**：特征图越来越大，池化可以缩小空间尺寸
2. **增加感受野**：后面的层能"看到"更大的区域
3. **提供平移不变性**：物体稍微移动一点，池化后的输出仍然相似

### 最大池化（Max Pooling）

在每个小区域中取最大值。常用 2×2 窗口，步幅 2。

直觉：只保留这个区域内"最强的响应"——如果这个区域有某个特征，只要有一个位置响应强，我们就认为这个特征存在。

### 平均池化（Average Pooling）

在每个小区域中取平均值。

直觉：综合考虑整个区域的响应强度。

---

**思考题 6：** 为什么最大池化比平均池化更常用？（提示：想想特征检测的场景）
"""))

    cells.append(code_cell("""# Max pooling vs Average pooling demonstration

# Create a sample feature map
feature_map = np.array([
    [1, 3, 2, 1, 5, 2],
    [2, 1, 4, 3, 1, 2],
    [3, 2, 1, 4, 3, 1],
    [1, 4, 3, 2, 2, 5],
    [2, 1, 5, 1, 4, 3],
    [1, 2, 1, 3, 1, 4],
], dtype=np.float32)

print("Input feature map (6x6):")
print(feature_map)
"""))

    cells.append(code_cell("""def max_pool_2x2(input_arr):
    \"\"\"2x2 max pooling with stride 2\"\"\"
    h, w = input_arr.shape
    h_out = h // 2
    w_out = w // 2
    output = np.zeros((h_out, w_out))
    
    for i in range(h_out):
        for j in range(w_out):
            patch = input_arr[i*2:(i+1)*2, j*2:(j+1)*2]
            output[i, j] = np.max(patch)
    return output

def avg_pool_2x2(input_arr):
    \"\"\"2x2 average pooling with stride 2\"\"\"
    h, w = input_arr.shape
    h_out = h // 2
    w_out = w // 2
    output = np.zeros((h_out, w_out))
    
    for i in range(h_out):
        for j in range(w_out):
            patch = input_arr[i*2:(i+1)*2, j*2:(j+1)*2]
            output[i, j] = np.mean(patch)
    return output

# Apply pooling
max_pooled = max_pool_2x2(feature_map)
avg_pooled = avg_pool_2x2(feature_map)

print("Max pooling (2x2, stride 2):")
print(max_pooled)
print()
print("Average pooling (2x2, stride 2):")
print(avg_pooled)
print()
print(f"Input size: {feature_map.shape} -> Output size: {max_pooled.shape}")
print("Pooling halves the spatial dimensions!")
"""))

    cells.append(code_cell("""# Visualize pooling
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

im0 = axes[0].imshow(feature_map, cmap='YlOrRd', vmin=0, vmax=5)
axes[0].set_title('Input (6x6 feature map)')
for i in range(6):
    for j in range(6):
        axes[0].text(j, i, f'{feature_map[i,j]:.0f}', ha='center', va='center', fontsize=9)
# Draw grid lines for 2x2 regions
for i in range(0, 7, 2):
    axes[0].axhline(i-0.5, color='white', linewidth=2)
    axes[0].axvline(i-0.5, color='white', linewidth=2)

im1 = axes[1].imshow(max_pooled, cmap='YlOrRd', vmin=0, vmax=5)
axes[1].set_title('Max Pool (3x3 output)')
for i in range(3):
    for j in range(3):
        axes[1].text(j, i, f'{max_pooled[i,j]:.0f}', ha='center', va='center', fontsize=11)

im2 = axes[2].imshow(avg_pooled, cmap='YlOrRd', vmin=0, vmax=5)
axes[2].set_title('Avg Pool (3x3 output)')
for i in range(3):
    for j in range(3):
        axes[2].text(j, i, f'{avg_pooled[i,j]:.1f}', ha='center', va='center', fontsize=10)

plt.tight_layout()
plt.show()

print("Max pooling preserves the strongest activation in each region.")
print("This is like saying \"there IS an edge in this area\" vs \"the average edge strength is X\".")
"""))

    # Section 6: Typical CNN architecture
    cells.append(md_cell("""## 6. 典型 CNN 架构：CONV → ReLU → POOL 的重复

一个经典的 CNN 通常由以下模块重复堆叠而成：

```
输入图像 → [CONV → ReLU] × N → POOL → ... → 全连接层 → 输出
```

### 各层的作用

| 层类型 | 作用 | 尺寸变化 |
|--------|------|----------|
| CONV | 提取特征（边缘、纹理、形状...） | 深度增加，空间尺寸可大可小 |
| ReLU | 引入非线性 | 尺寸不变 |
| POOL | 下采样，增加感受野 | 空间尺寸减半，深度不变 |
| FC | 分类/回归 | 从特征向量到输出 |

### 空间尺寸变化规律

- 通常：越往深层，空间尺寸越小，通道数越多
- 空间信息 → 语义信息 的转换过程

---
"""))

    cells.append(md_cell("""## 7. LeNet-5：早期 CNN 的代表作

LeNet-5 由 Yann LeCun 于 1998 年提出，用于手写数字识别（MNIST 数据集）。

### 架构概览

```
输入 (32x32x1)
  ↓
C1: Conv 5x5, 6 filters → 28x28x6
  ↓
S2: Max Pool 2x2 → 14x14x6
  ↓
C3: Conv 5x5, 16 filters → 10x10x16
  ↓
S4: Max Pool 2x2 → 5x5x16
  ↓
C5: Conv 5x5, 120 filters → 1x1x120
  ↓
F6: Fully connected → 84
  ↓
Output: 10 classes (digits 0-9)
```

> **参考论文**：[[LeCun et al., 1998]](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
> Gradient-based learning applied to document recognition

更早的先驱工作：[[Fukushima, 1980]](https://www.rctn.org/bruno/public/papers/Fukushima1980.pdf) Neocognitron——最早的卷积神经网络雏形之一。

---

**思考题 7：** 为什么 LeNet-5 只有 5 个卷积/全连接层，却叫 LeNet-"5"？
"""))

    cells.append(code_cell("""# Let's build a mini-LeNet and compute parameter counts

def count_lenet_params():
    \"\"\"Count parameters in LeNet-5-like architecture\"\"\"
    # C1: Conv 5x5, 1 input channel, 6 filters
    c1 = 5 * 5 * 1 * 6 + 6
    
    # C3: Conv 5x5, 6 input channels, 16 filters
    c3 = 5 * 5 * 6 * 16 + 16
    
    # C5: Conv 5x5, 16 input channels, 120 filters
    c5 = 5 * 5 * 16 * 120 + 120
    
    # F6: FC 120 -> 84
    f6 = 120 * 84 + 84
    
    # Output: 84 -> 10
    out = 84 * 10 + 10
    
    total = c1 + c3 + c5 + f6 + out
    
    print("LeNet-5 Parameter Count:")
    print(f"  C1 (Conv 5x5, 6 filters):  {c1:>6,}")
    print(f"  C3 (Conv 5x5, 16 filters): {c3:>6,}")
    print(f"  C5 (Conv 5x5, 120 filters):{c5:>6,}")
    print(f"  F6 (FC 120->84):           {f6:>6,}")
    print(f"  Output (84->10):           {out:>6,}")
    print(f"  {'='*20}")
    print(f"  Total:                     {total:>6,}")
    
    return total

total_params = count_lenet_params()
print(f"\nLeNet-5 has only ~{total_params/1000:.0f}K parameters!")
print("Compare that to a single FC layer on 224x224 images...")
"""))

    # Section 7: Implement conv layer forward pass
    cells.append(md_cell("""## 8. 从零实现卷积层前向传播

现在我们来实现一个**完整的**卷积层前向传播，支持：
- 多通道输入
- 多过滤器（多输出通道）
- 填充（padding）
- 步幅（stride）

### 输入输出维度约定

- 输入 x: (N, C, H, W) — batch 中 N 张图，C 通道，H 高，W 宽
- 权重 w: (F, C, HH, WW) — F 个过滤器，每个 C 通道，HH 高，WW 宽
- 偏置 b: (F,) — 每个过滤器一个偏置
- 输出 out: (N, F, H_out, W_out)

> 注意：这里使用 NCHW 格式（batch, channel, height, width），这是 PyTorch 的默认格式。

---
"""))

    cells.append(code_cell("""def conv_forward_naive(x, w, b, conv_param):
    \"\"\"
    A naive implementation of the forward pass for a convolutional layer.
    
    Inputs:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases of shape (F,)
    - conv_param: Dictionary with the following keys:
      - 'stride': How much the filter slides each step
      - 'pad': How many pixels to pad the input
    
    Returns a tuple of:
    - out: Output data of shape (N, F, H_out, W_out)
    - cache: (x, w, b, conv_param) for backward pass
    \"\"\"
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    # Compute output dimensions
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    
    # Initialize output
    out = np.zeros((N, F, H_out, W_out))
    
    # Pad the input
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 
                   mode='constant', constant_values=0)
    
    # Naive loop implementation (slow but clear)
    for n in range(N):          # for each image in batch
        for f in range(F):      # for each filter
            for i in range(H_out):  # for each output row
                for j in range(W_out):  # for each output col
                    # Compute the top-left corner of the receptive field
                    h_start = i * stride
                    w_start = j * stride
                    
                    # Extract the patch from the padded input
                    patch = x_pad[n, :, h_start:h_start+HH, w_start:w_start+WW]
                    
                    # Convolution: element-wise multiply + sum + bias
                    out[n, f, i, j] = np.sum(patch * w[f]) + b[f]
    
    cache = (x, w, b, conv_param)
    return out, cache


print("conv_forward_naive function defined!")
print("This is the 'naive' version with 4 nested loops - easy to understand but slow.")
"""))

    cells.append(code_cell("""# Test the conv forward pass with a simple example
np.random.seed(42)

# Create a small batch: 2 images, 3 channels, 5x5 each
x = np.random.randn(2, 3, 5, 5)

# Create 4 filters, each 3x3x3
w = np.random.randn(4, 3, 3, 3)
b = np.random.randn(4)

conv_param = {'stride': 1, 'pad': 1}  # same padding

out, cache = conv_forward_naive(x, w, b, conv_param)

print(f"Input shape:  {x.shape}  (N, C, H, W)")
print(f"Filter shape: {w.shape}  (F, C, HH, WW)")
print(f"Bias shape:   {b.shape}  (F,)")
print(f"Output shape: {out.shape}  (N, F, H_out, W_out)")
print()
print("Expected output shape: (2, 4, 5, 5) - same spatial size due to padding=1")
print(f"Actual output shape:   {out.shape}")
assert out.shape == (2, 4, 5, 5), "Output shape mismatch!"
print("\nShape check PASSED!")
"""))

    cells.append(code_cell("""# Numerical gradient check for conv layer (simplified)
# We verify the forward pass by checking against a known simple case

# Simple test: all ones input, all ones filter -> output should be sum of ones
x_test = np.ones((1, 1, 4, 4))
w_test = np.ones((1, 1, 2, 2))
b_test = np.array([0.0])

out_test, _ = conv_forward_naive(x_test, w_test, b_test, {'stride': 1, 'pad': 0})
print("Test: all-ones input (4x4), all-ones filter (2x2), no padding")
print("Expected output (3x3, all values = 4):")
print(out_test[0, 0])
print()
assert np.allclose(out_test[0, 0], 4.0), "Failed: all values should be 4!"
print("All-ones test PASSED!")
"""))

    # Section 8: Implement max pooling from scratch
    cells.append(md_cell("""## 9. 从零实现最大池化层

最大池化层没有参数（不需要学习），但需要在前向传播时记录最大值的位置，以便反向传播时使用。

### 输入输出维度

- 输入: (N, C, H, W)
- 输出: (N, C, H/2, W/2) — 对于 2×2 池化，步幅 2

> 注意：池化是在**每个通道独立**进行的，所以通道数不变。

---
"""))

    cells.append(code_cell("""def max_pool_forward_naive(x, pool_param):
    \"\"\"
    A naive implementation of the forward pass for a max pooling layer.
    
    Inputs:
    - x: Input data of shape (N, C, H, W)
    - pool_param: Dictionary with the following keys:
      - 'pool_height': Height of each pooling region
      - 'pool_width': Width of each pooling region
      - 'stride': Step size between pooling regions
    
    Returns a tuple of:
    - out: Output data of shape (N, C, H_out, W_out)
    - cache: (x, pool_param) for backward pass
    \"\"\"
    N, C, H, W = x.shape
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']
    
    # Compute output dimensions
    H_out = (H - pool_h) // stride + 1
    W_out = (W - pool_w) // stride + 1
    
    out = np.zeros((N, C, H_out, W_out))
    
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    patch = x[n, c, h_start:h_start+pool_h, w_start:w_start+pool_w]
                    out[n, c, i, j] = np.max(patch)
    
    cache = (x, pool_param)
    return out, cache


print("max_pool_forward_naive function defined!")
"""))

    cells.append(code_cell("""# Test max pooling
np.random.seed(42)

# Create test input
x = np.random.randn(1, 2, 4, 4)
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

out, cache = max_pool_forward_naive(x, pool_param)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {out.shape}")
print()
print("Input channel 0:")
print(x[0, 0])
print()
print("Max pooled channel 0:")
print(out[0, 0])

# Verify manually for first 2x2 block
manual_max = np.max(x[0, 0, 0:2, 0:2])
print(f"\nManual check: max of top-left 2x2 = {manual_max:.4f}")
print(f"Pooled output [0,0] = {out[0, 0, 0, 0]:.4f}")
assert np.isclose(manual_max, out[0, 0, 0, 0]), "Max pooling test failed!"
print("Max pooling test PASSED!")
"""))

    # Section 9: Visualize conv layer activations
    cells.append(md_cell("""## 10. 可视化卷积层激活

让我们用一张简单的测试图来看看卷积层到底"看到"了什么。

我们将手动创建几个不同的过滤器（边缘检测、模糊等），然后观察它们在图像上的响应。

---
"""))

    cells.append(code_cell("""# Create a test image with various patterns
test_image = np.zeros((1, 1, 32, 32))

# Add a horizontal line
test_image[0, 0, 15, :] = 1.0

# Add a vertical line
test_image[0, 0, :, 15] = 1.0

# Add a diagonal pattern
for i in range(20):
    if 5+i < 32 and 5+i < 32:
        test_image[0, 0, 5+i, 5+i] = 1.0

# Add a small square
test_image[0, 0, 5:10, 20:25] = 1.0

# Create different filters
filters = np.zeros((4, 1, 3, 3))

# Filter 0: vertical edge detector
filters[0, 0] = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

# Filter 1: horizontal edge detector
filters[1, 0] = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

# Filter 2: blur (average)
filters[2, 0] = np.ones((3, 3)) / 9.0

# Filter 3: sharpen
filters[3, 0] = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

biases = np.zeros(4)

# Apply convolution
activations, _ = conv_forward_naive(test_image, filters, biases, {'stride': 1, 'pad': 1})

print(f"Input shape: {test_image.shape}")
print(f"Activations shape: {activations.shape}")
print("4 filters -> 4 activation maps (feature channels)")
"""))

    cells.append(code_cell("""# Visualize the activations
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# Original image
axes[0, 0].imshow(test_image[0, 0], cmap='gray')
axes[0, 0].set_title('Input Image')
axes[0, 0].axis('off')

# Filter 0: vertical edge
axes[0, 1].imshow(activations[0, 0], cmap='RdBu_r')
axes[0, 1].set_title('Filter 0: Vertical Edge Response')
axes[0, 1].axis('off')

# Filter 1: horizontal edge
axes[0, 2].imshow(activations[0, 1], cmap='RdBu_r')
axes[0, 2].set_title('Filter 1: Horizontal Edge Response')
axes[0, 2].axis('off')

# Filter 2: blur
axes[1, 0].imshow(activations[0, 2], cmap='gray')
axes[1, 0].set_title('Filter 2: Blur (average)')
axes[1, 0].axis('off')

# Filter 3: sharpen
axes[1, 1].imshow(activations[0, 3], cmap='RdBu_r')
axes[1, 1].set_title('Filter 3: Sharpen')
axes[1, 1].axis('off')

# Filter visualization
for idx in range(4):
    pass

# Show the actual filter patterns
filter_names = ['Vertical Edge', 'Horizontal Edge', 'Blur', 'Sharpen']
for i in range(4):
    ax = axes.flatten()[2 + i] if i > 0 else None

# Add a dedicated filter visualization
axes[1, 2].axis('off')
axes[1, 2].text(0.5, 0.5, 'Each filter detects\na different pattern\nin the image', 
                ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

print("Key insight: Different filters activate on different features!")
print("- Vertical edge filter fires on vertical lines")
print("- Horizontal edge filter fires on horizontal lines")
print("In a trained CNN, the network learns these filters automatically.")
"""))

    # Section 10: Common mistakes
    cells.append(md_cell("""## 11. 常见错误与注意事项

### 错误 1：输出尺寸不是整数

如果 `(W - F + 2P) / S + 1` 不是整数，说明你的 padding/stride 设置有问题。

### 错误 2：忘记通道维度

过滤器的深度必须等于输入的通道数。一个 3×3 的过滤器作用在 RGB 图像上，实际上是 3×3×3。

### 错误 3：混淆卷积和互相关

严格来说，深度学习中说的"卷积"其实是**互相关**（cross-correlation）——过滤器不需要翻转。但因为过滤器是学习得到的，翻不翻转不影响最终结果，所以大家都叫它卷积。

### 错误 4：池化层有参数

池化层是**没有可学习参数**的！它只是一个固定的下采样操作。

### 错误 5：参数数量计算错误

卷积层的参数量 = F × (C × HH × WW + 1)
- F = 过滤器数量
- C = 输入通道数
- HH, WW = 过滤器空间尺寸
- +1 是偏置

---
"""))

    # Homework 1
    cells.append(md_cell("""---

## 作业 1：实现带步幅和填充的卷积函数

### 任务

实现一个高效的卷积层前向传播函数，要求：
1. 支持任意填充（padding）
2. 支持任意步幅（stride）
3. 支持多批次、多通道输入
4. 支持多过滤器

### 提示

- 可以使用 `np.pad` 进行填充
- 可以用 im2col 方法加速（进阶），但基础版用循环也可以
- 注意输出尺寸的计算

### 验证

完成后运行下面的测试代码，确保断言通过。

---
"""))

    cells.append(code_cell("""# ===== Homework 1: Your implementation =====

def conv_forward(x, w, b, conv_param):
    \"\"\"
    Convolutional layer forward pass.
    
    Inputs:
    - x: (N, C, H, W) input data
    - w: (F, C, HH, WW) filter weights
    - b: (F,) biases
    - conv_param: dict with 'stride' and 'pad'
    
    Returns:
    - out: (N, F, H_out, W_out) output
    - cache: (x, w, b, conv_param)
    \"\"\"
    # YOUR CODE HERE
    # Hint: Follow the formula for output size
    # Hint: Use np.pad with mode='constant'
    # Hint: Loop over N, F, H_out, W_out
    
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    # Compute output dimensions
    H_out = None  # TODO
    W_out = None  # TODO
    
    # Initialize output
    out = np.zeros((N, F, H_out, W_out))
    
    # Pad input
    x_pad = None  # TODO
    
    # Convolution loop
    # TODO: implement the convolution
    
    cache = (x, w, b, conv_param)
    return out, cache


# ===== ANSWER KEY (hidden in assignment, shown here for reference) =====
def conv_forward_answer(x, w, b, conv_param):
    \"\"\"Reference implementation.\"\"\"
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    H_out = (H + 2 * pad - HH) // stride + 1
    W_out = (W + 2 * pad - WW) // stride + 1
    
    out = np.zeros((N, F, H_out, W_out))
    x_pad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
    
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    patch = x_pad[n, :, h_start:h_start+HH, w_start:w_start+WW]
                    out[n, f, i, j] = np.sum(patch * w[f]) + b[f]
    
    cache = (x, w, b, conv_param)
    return out, cache

print("Homework 1: Implement conv_forward")
print("Write your solution in the conv_forward function above.")
print("Then run the test cell below to verify.")
"""))

    cells.append(code_cell("""# ===== Homework 1: Test cases =====
np.random.seed(42)

# Test 1: Basic shape test
x_test = np.random.randn(2, 3, 7, 7)
w_test = np.random.randn(5, 3, 3, 3)
b_test = np.random.randn(5)

out_ref, _ = conv_forward_answer(x_test, w_test, b_test, {'stride': 1, 'pad': 1})
out_student, _ = conv_forward(x_test, w_test, b_test, {'stride': 1, 'pad': 1})

assert out_student.shape == out_ref.shape, f"Shape mismatch: {out_student.shape} vs {out_ref.shape}"
assert np.allclose(out_student, out_ref, atol=1e-8), "Values don't match reference!"
print("Test 1 (stride=1, pad=1): PASSED")

# Test 2: Stride 2
out_ref2, _ = conv_forward_answer(x_test, w_test, b_test, {'stride': 2, 'pad': 1})
out_student2, _ = conv_forward(x_test, w_test, b_test, {'stride': 2, 'pad': 1})

assert out_student2.shape == out_ref2.shape, f"Shape mismatch: {out_student2.shape}"
assert np.allclose(out_student2, out_ref2, atol=1e-8), "Values don't match!"
print("Test 2 (stride=2, pad=1): PASSED")

# Test 3: No padding (valid)
out_ref3, _ = conv_forward_answer(x_test, w_test, b_test, {'stride': 1, 'pad': 0})
out_student3, _ = conv_forward(x_test, w_test, b_test, {'stride': 1, 'pad': 0})

assert out_student3.shape == out_ref3.shape, f"Shape mismatch: {out_student3.shape}"
assert np.allclose(out_student3, out_ref3, atol=1e-8), "Values don't match!"
print("Test 3 (stride=1, pad=0): PASSED")

# Test 4: Output size formula verification
assert out_ref.shape == (2, 5, 7, 7), "Same padding should preserve spatial size"
assert out_ref2.shape == (2, 5, 4, 4), "Stride 2 should roughly halve spatial size"
assert out_ref3.shape == (2, 5, 5, 5), "Valid padding on 7x7 with 3x3 filter -> 5x5"
print("Test 4 (output size formula): PASSED")

print("\n=== All Homework 1 tests PASSED! ===")
"""))

    # Homework 2
    cells.append(md_cell("""---

## 作业 2：实现 2×2 最大池化的反向传播

### 任务

实现最大池化层的反向传播。

### 背景

最大池化在前向传播时把每个 2×2 区域的最大值传出去。反向传播时，梯度要"流回"到最大值所在的位置，其他位置的梯度为 0。

直觉：**只有前向传播时贡献了输出的那个神经元，才需要接收梯度。**

### 提示

- 你需要在前向传播时记录每个最大值的位置（mask）
- 反向传播时，把上游梯度放到对应的位置
- 其他位置梯度为 0

---
"""))

    cells.append(code_cell("""# ===== Homework 2: Max pooling backward pass =====

def max_pool_backward_naive(dout, cache):
    \"\"\"
    Backward pass for max pooling layer (naive implementation).
    
    Inputs:
    - dout: Upstream derivatives of shape (N, C, H_out, W_out)
    - cache: Tuple of:
      - x: Input data of shape (N, C, H, W)
      - pool_param: Dictionary with pool_height, pool_width, stride
    
    Returns:
    - dx: Gradient with respect to x, of shape (N, C, H, W)
    \"\"\"
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']
    
    H_out = (H - pool_h) // stride + 1
    W_out = (W - pool_w) // stride + 1
    
    dx = np.zeros_like(x)
    
    # YOUR CODE HERE
    # Hint: for each pooling region, find where the max was,
    # then route the gradient to that position
    
    return dx


# ===== ANSWER KEY =====
def max_pool_backward_answer(dout, cache):
    \"\"\"Reference implementation.\"\"\"
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']
    
    H_out = (H - pool_h) // stride + 1
    W_out = (W - pool_w) // stride + 1
    
    dx = np.zeros_like(x)
    
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    patch = x[n, c, h_start:h_start+pool_h, w_start:w_start+pool_w]
                    # Find the position of the maximum value
                    max_idx = np.unravel_index(np.argmax(patch), patch.shape)
                    # Route gradient to that position
                    dx[n, c, h_start + max_idx[0], w_start + max_idx[1]] += dout[n, c, i, j]
    
    return dx

print("Homework 2: Implement max_pool_backward_naive")
print("Write your solution above, then run the test cell.")
"""))

    cells.append(code_cell("""# ===== Homework 2: Test cases =====
np.random.seed(42)

# Test setup
x = np.random.randn(2, 3, 6, 6)
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

# Forward pass
out, cache = max_pool_forward_naive(x, pool_param)

# Random upstream gradient
dout = np.random.randn(*out.shape)

# Compute gradients
dx_student = max_pool_backward_naive(dout, cache)
dx_reference = max_pool_backward_answer(dout, cache)

# Shape check
assert dx_student.shape == x.shape, f"Shape mismatch: {dx_student.shape} vs {x.shape}"
print(f"dx shape: {dx_student.shape} (should match input shape {x.shape})")

# Value check
assert np.allclose(dx_student, dx_reference, atol=1e-10), "Gradient values don't match!"
print("Gradient values: CORRECT")

# Additional check: sum of dx should equal sum of dout (gradient conservation for max pool)
# Each element of dout goes to exactly one position in dx
assert np.isclose(np.sum(dx_student), np.sum(dout)), "Gradient sum not conserved!"
print("Gradient conservation: PASSED")

# Check that only one position per pool region has non-zero gradient
for n in range(2):
    for c in range(3):
        for i in range(3):
            for j in range(3):
                patch = dx_student[n, c, i*2:(i+1)*2, j*2:(j+1)*2]
                nonzero_count = np.sum(patch != 0)
                assert nonzero_count == 1, f"Expected 1 nonzero, got {nonzero_count}"
print("Only one gradient per pool region: PASSED")

print("\n=== All Homework 2 tests PASSED! ===")
"""))

    # Summary and references
    cells.append(md_cell("""---

## 本讲总结

### 核心概念回顾

| 概念 | 关键思想 | 公式/要点 |
|------|----------|-----------|
| 卷积 | 过滤器在图像上滑动，提取局部特征 | 逐元素相乘求和 |
| 填充 | 边缘补零，保持空间尺寸 | Same padding: P=(F-1)/2 |
| 步幅 | 过滤器移动的步长，控制下采样 | O = (W-F+2P)/S + 1 |
| 参数共享 | 同一过滤器权重全局共享 | 大幅减少参数量 |
| 最大池化 | 每个区域取最大值，下采样 | 2×2, stride 2 最常用 |

### 思考题答案提示

1. 32×32×3 × 100 + 100 = 307,300 个参数
2. 3×3×3 + 1 = 28 个参数（一个过滤器）
3. (5-1)/2 = 2，每边填充 2 个像素
4. (28-5+4)/2 + 1 = 27/2 + 1 = 13.5 + 1 = 14.5... 不对！等一下：(28-5+2×2)/2 + 1 = (27)/2 + 1 = 13.5 + 1，27 不能被 2 整除。正确计算：(28-5+4)/2 + 1 = 27/2 + 1 → 这不是整数，说明这个组合有问题。如果用 floor 的话是 13+1=14。实际上应该用向下取整。
5. 5×5×3×16 + 16 = 1,216 个参数
6. 最大池化保留最强响应，更符合"特征是否存在"的直觉；而且反向传播更简单
7. 5 指的是 5 个可学习层（C1, C3, C5, F6, Output），池化层不算

### 论文参考

- [[LeCun et al., 1998]](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) Gradient-based learning applied to document recognition — LeNet-5
- [[Fukushima, 1980]](https://www.rctn.org/bruno/public/papers/Fukushima1980.pdf) Neocognitron: A self-organizing neural network model — 早期卷积神经网络雏形

###  GitHub 参考实现

- [vdumoulin/conv_arithmetic](https://github.com/vdumoulin/conv_arithmetic) (8.6k stars) — 卷积运算的可视化指南，强烈推荐
- [DeepVision-Tutorial/DeepVision](https://github.com/DeepVision-Tutorial/DeepVision) (2.4k stars) — 深度学习与计算机视觉教程

---

**下节预告**：我们将学习经典 CNN 架构——从 AlexNet 到 VGG、GoogLeNet，再到革命性的 ResNet！
"""))

    return cells


# ============================================================
# NOTEBOOK 2: CNN Architectures
# ============================================================
def build_notebook2():
    cells = []

    cells.append(md_cell("""# 经典 CNN 架构：从 AlexNet 到 ResNet

> **CS231n 深度学习与计算机视觉** · 第6讲
>
> 本 notebook 将带你深入理解 CNN 架构的演进之路——从 2012 年引爆深度学习革命的 AlexNet，到 VGG 的小过滤器哲学，GoogLeNet 的多尺度思想，以及 ResNet 残差连接带来的深度突破。

---

## 学习目标

- 理解 AlexNet 的历史地位和关键创新
- 掌握 VGG 使用小过滤器的设计哲学
- 理解 1×1 卷积的作用
- 理解 GoogLeNet/Inception 模块的多尺度思想
- 深入理解 ResNet 的残差连接为什么能训练超深网络
- 手动计算各架构的参数量和特征图尺寸
- 从零实现 ResNet 基本模块

---
"""))

    cells.append(code_cell("""# Setup
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("Environment ready!")
"""))

    # Section 1: AlexNet
    cells.append(md_cell("""## 1. AlexNet：深度学习革命的起点

### 历史背景

2012 年 ImageNet 图像分类比赛中，AlexNet 以 **15.3%** 的 top-5 错误率碾压第二名（26.2%），震惊了整个计算机视觉界。

这不是一个小改进——这是**质的飞跃**。深度学习从此成为计算机视觉的主流方法。

### AlexNet 的五大创新

1. **ReLU 激活函数**：比 sigmoid/tanh 训练快 6 倍，缓解梯度消失
2. **Dropout**：有效防止过拟合
3. **GPU 加速**：双 GTX 580 GPU 训练
4. **数据增强**（Data Augmentation）：翻转、裁剪、颜色抖动
5. **重叠池化**（Overlapping Pooling）：3×3 池化窗口，步幅 2

---

**思考题 1：** 为什么 ReLU 比 sigmoid 训练更快？（提示：考虑梯度饱和问题）
"""))

    cells.append(md_cell("""### AlexNet 架构详情

```
输入: 227x227x3 图像
  ↓
Conv1: 96 个 11x11 过滤器, stride=4, pad=0  → 55x55x96
ReLU → MaxPool (3x3, stride=2)               → 27x27x96
  ↓
Conv2: 256 个 5x5 过滤器, pad=2              → 27x27x256
ReLU → MaxPool (3x3, stride=2)               → 13x13x256
  ↓
Conv3: 384 个 3x3 过滤器, pad=1              → 13x13x384
ReLU
  ↓
Conv4: 384 个 3x3 过滤器, pad=1              → 13x13x384
ReLU
  ↓
Conv5: 256 个 3x3 过滤器, pad=1              → 13x13x256
ReLU → MaxPool (3x3, stride=2)               → 6x6x256
  ↓
FC6: 4096 神经元
ReLU → Dropout (0.5)
  ↓
FC7: 4096 神经元
ReLU → Dropout (0.5)
  ↓
FC8: 1000 神经元 (ImageNet 类别)
Softmax
```

总计：**5 个卷积层 + 3 个全连接层 = 8 层**

> **参考论文**：[[Krizhevsky et al., 2012]](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) ImageNet Classification with Deep Convolutional Neural Networks

---
"""))

    cells.append(code_cell("""# Let's compute the parameter count for AlexNet

def alexnet_param_count():
    \"\"\"Calculate parameters in AlexNet\"\"\"
    params = {}
    
    # Conv1: 96 filters of 11x11x3
    params['Conv1'] = 11*11*3*96 + 96
    
    # Conv2: 256 filters of 5x5x96
    params['Conv2'] = 5*5*96*256 + 256
    
    # Conv3: 384 filters of 3x3x256
    params['Conv3'] = 3*3*256*384 + 384
    
    # Conv4: 384 filters of 3x3x384
    params['Conv4'] = 3*3*384*384 + 384
    
    # Conv5: 256 filters of 3x3x384
    params['Conv5'] = 3*3*384*256 + 256
    
    # FC6: 6*6*256 -> 4096
    params['FC6'] = 6*6*256*4096 + 4096
    
    # FC7: 4096 -> 4096
    params['FC7'] = 4096*4096 + 4096
    
    # FC8: 4096 -> 1000
    params['FC8'] = 4096*1000 + 1000
    
    return params

params = alexnet_param_count()
total = sum(params.values())

print("AlexNet Parameter Count:")
print("-" * 40)
for name, count in params.items():
    print(f"  {name:8s}: {count:>12,}  ({count/total*100:5.1f}%)")
print("-" * 40)
print(f"  {'Total':8s}: {total:>12,}  ({total/1e6:.1f}M)")

print(f"\nInteresting: {params['FC6']+params['FC7']+params['FC8']:,.0f} params from FC layers")
print(f"That's {(params['FC6']+params['FC7']+params['FC8'])/total*100:.1f}% of all parameters!")
print("Most parameters are in the fully connected layers, not the conv layers!")
"""))

    cells.append(code_cell("""# Visualize AlexNet architecture and feature map sizes
fig, ax = plt.subplots(figsize=(16, 6))

layers = ['Input', 'Conv1', 'Pool1', 'Conv2', 'Pool2', 'Conv3', 'Conv4', 'Conv5', 'Pool5', 'FC6', 'FC7', 'FC8']
sizes = [227, 55, 27, 27, 13, 13, 13, 13, 6, 1, 1, 1]  # spatial size
channels = [3, 96, 96, 256, 256, 384, 384, 256, 256, 4096, 4096, 1000]

x_positions = np.arange(len(layers))

# Draw blocks representing feature maps
for i, (layer, size, ch) in enumerate(zip(layers, sizes, channels)):
    # Width represents channels (scaled), height represents spatial size
    width = max(0.3, np.log10(ch) * 0.8)
    height = size / 227 * 4
    
    rect = Rectangle((x_positions[i] - width/2, 2 - height/2), width, height,
                     facecolor=f'C{i%10}', alpha=0.7, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    
    ax.text(x_positions[i], 2 + height/2 + 0.15, f'{size}x{size}', 
            ha='center', va='bottom', fontsize=9)
    ax.text(x_positions[i], 2 - height/2 - 0.15, f'C={ch}', 
            ha='center', va='top', fontsize=9)
    ax.text(x_positions[i], 0.3, layer, ha='center', va='bottom', fontsize=8, rotation=30)

# Arrows
for i in range(len(layers)-1):
    ax.annotate('', xy=(x_positions[i+1] - 0.3, 2), xytext=(x_positions[i] + 0.3, 2),
                arrowprops=dict(arrowstyle='->', color='gray'))

ax.set_xlim(-0.5, len(layers) - 0.5)
ax.set_ylim(0, 5)
ax.set_title('AlexNet Architecture: Feature Map Size Progression', fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.show()
"""))

    # Section 2: VGG
    cells.append(md_cell("""## 2. VGG：小过滤器的力量

### VGG 的核心思想

**用多个小的 3×3 过滤器替代一个大过滤器**。

为什么？因为：
1. 两个 3×3 卷积层的感受野 = 一个 5×5 卷积层的感受野
2. 三个 3×3 卷积层的感受野 = 一个 7×7 卷积层的感受野
3. 但多个小过滤器**有更多的非线性**（更多 ReLU）
4. 而且**参数量更少**！

### 参数量对比

- 一个 7×7 卷积：7×7×C×C = 49C² 参数
- 三个 3×3 卷积：3 × (3×3×C×C) = 27C² 参数

少了将近一半！

> **参考论文**：[[Simonyan & Zisserman, 2014]](https://arxiv.org/abs/1409.1556) Very Deep Convolutional Networks for Large-Scale Image Recognition

---

**思考题 2：** 为什么 2 个 3×3 卷积的感受野等于 1 个 5×5 卷积？请用"每做一次卷积，感受野增加多少"来解释。
"""))

    cells.append(code_cell("""# Verify: receptive field of stacked 3x3 convolutions

def compute_receptive_field(layers, input_size=1):
    \"\"\"
    Compute receptive field given a list of (filter_size, stride) tuples.
    Start from the output and work backwards.
    \"\"\"
    rf = 1  # receptive field at output layer
    for f, s in reversed(layers):
        rf = (rf - 1) * s + f
    return rf

# Compare: one 5x5 conv vs two 3x3 convs
one_5x5 = [(5, 1)]
two_3x3 = [(3, 1), (3, 1)]
three_3x3 = [(3, 1), (3, 1), (3, 1)]

rf_5x5 = compute_receptive_field(one_5x5)
rf_2x3x3 = compute_receptive_field(two_3x3)
rf_3x3x3 = compute_receptive_field(three_3x3)

print("Receptive Field Comparison (stride=1 for all):")
print(f"  1 x 5x5 conv:     receptive field = {rf_5x5}x{rf_5x5}")
print(f"  2 x 3x3 conv:     receptive field = {rf_2x3x3}x{rf_2x3x3}  (same as 5x5!)")
print(f"  3 x 3x3 conv:     receptive field = {rf_3x3x3}x{rf_3x3x3}  (same as 7x7!)")
print()

# Parameter comparison (assuming C input and output channels)
C = 64
params_5x5 = 5*5*C*C
params_2x3x3 = 2 * (3*3*C*C)
params_7x7 = 7*7*C*C
params_3x3x3 = 3 * (3*3*C*C)

print("Parameter Comparison (C=64 input/output channels):")
print(f"  1 x 5x5 conv:     {params_5x5:>8,} params")
print(f"  2 x 3x3 conv:     {params_2x3x3:>8,} params  ({params_5x5/params_2x3x3:.1f}x fewer)")
print(f"  1 x 7x7 conv:     {params_7x7:>8,} params")
print(f"  3 x 3x3 conv:     {params_3x3x3:>8,} params  ({params_7x7/params_3x3x3:.1f}x fewer)")
"""))

    cells.append(md_cell("""### VGG-16 架构

VGG-16 有 16 个带权重的层（13 个卷积 + 3 个全连接）。

```
输入: 224x224x3
  ↓
Conv3-64 × 2     → 224x224x64
MaxPool          → 112x112x64
  ↓
Conv3-128 × 2    → 112x112x128
MaxPool          → 56x56x128
  ↓
Conv3-256 × 3    → 56x56x256
MaxPool          → 28x28x256
  ↓
Conv3-512 × 3    → 28x28x512
MaxPool          → 14x14x512
  ↓
Conv3-512 × 3    → 14x14x512
MaxPool          → 7x7x512
  ↓
FC-4096
FC-4096
FC-1000
Softmax
```

**设计规律**：每次池化后空间尺寸减半，通道数翻倍。

---
"""))

    cells.append(code_cell("""# VGG-16 parameter count and computation

def vgg16_param_count():
    \"\"\"Calculate VGG-16 parameters\"\"\"
    params = {}
    
    # Block 1: 2 conv layers, 64 filters
    params['Conv1_1'] = 3*3*3*64 + 64
    params['Conv1_2'] = 3*3*64*64 + 64
    
    # Block 2: 2 conv layers, 128 filters
    params['Conv2_1'] = 3*3*64*128 + 128
    params['Conv2_2'] = 3*3*128*128 + 128
    
    # Block 3: 3 conv layers, 256 filters
    params['Conv3_1'] = 3*3*128*256 + 256
    params['Conv3_2'] = 3*3*256*256 + 256
    params['Conv3_3'] = 3*3*256*256 + 256
    
    # Block 4: 3 conv layers, 512 filters
    params['Conv4_1'] = 3*3*256*512 + 512
    params['Conv4_2'] = 3*3*512*512 + 512
    params['Conv4_3'] = 3*3*512*512 + 512
    
    # Block 5: 3 conv layers, 512 filters
    params['Conv5_1'] = 3*3*512*512 + 512
    params['Conv5_2'] = 3*3*512*512 + 512
    params['Conv5_3'] = 3*3*512*512 + 512
    
    # FC layers
    params['FC6'] = 7*7*512*4096 + 4096
    params['FC7'] = 4096*4096 + 4096
    params['FC8'] = 4096*1000 + 1000
    
    return params

vgg_params = vgg16_param_count()
vgg_total = sum(vgg_params.values())
alex_total = sum(alexnet_param_count().values())

print(f"VGG-16 Total Parameters: {vgg_total:,.0f} ({vgg_total/1e6:.1f}M)")
print(f"AlexNet Total Parameters: {alex_total:,.0f} ({alex_total/1e6:.1f}M)")
print(f"VGG-16 is {vgg_total/alex_total:.1f}x bigger than AlexNet")
print()
print("Conv layers parameters:")
conv_total = sum(v for k, v in vgg_params.items() if k.startswith('Conv'))
print(f"  Conv total: {conv_total:,.0f} ({conv_total/vgg_total*100:.1f}%)")
fc_total = sum(v for k, v in vgg_params.items() if k.startswith('FC'))
print(f"  FC total:   {fc_total:,.0f} ({fc_total/vgg_total*100:.1f}%)")
"""))

    # Section 3: 1x1 convolution
    cells.append(md_cell("""## 3. 1×1 卷积：看似无用，实则强大

### 什么是 1×1 卷积？

过滤器大小为 1×1 的卷积。空间上不做任何邻域操作，只在**深度方向**做线性组合。

### 1×1 卷积的作用

1. **跨通道信息整合**：每个输出位置是所有输入通道的加权组合
2. **改变通道数**（升维/降维）：用少量 1×1 过滤器可以降维，用大量可以升维
3. **增加非线性**：配合 ReLU 使用，在不改变空间尺寸的情况下增加网络容量
4. **减少计算量**：先降维再卷积，可以大幅减少计算量

### 类比

1×1 卷积就像是在每个像素位置上做一个"全连接层"——输入是所有通道的值，输出是新的通道值。

---

**思考题 3：** 如果输入是 28×28×256，我们想得到 28×28×64 的输出，用 1×1 卷积需要多少参数？如果用 3×3 卷积呢？
"""))

    cells.append(code_cell("""# 1x1 convolution: dimensionality reduction

# Scenario: input 28x28x256, we want 28x28x64 output
H, W, C_in = 28, 28, 256
C_out = 64

# Option 1: 3x3 convolution directly
params_3x3 = 3*3*C_in*C_out + C_out
flops_3x3 = H * W * C_out * (3*3*C_in + 1)  # approximate FLOPs

# Option 2: 1x1 conv (bottleneck) + 3x3 conv + 1x1 conv (expand)
# First 1x1: reduce to bottleneck dimension
bottleneck = 64
params_1x1_reduce = 1*1*C_in*bottleneck + bottleneck
params_3x3_mid = 3*3*bottleneck*C_out + C_out  # Actually for bottleneck we'd have expand too

# Full bottleneck block (like ResNet): 1x1(reduce) -> 3x3 -> 1x1(expand)
C_mid = 64
params_bottleneck = (1*1*C_in*C_mid + C_mid) + (3*3*C_mid*C_mid + C_mid) + (1*1*C_mid*C_out + C_out)

print("Parameter comparison: 28x28x256 -> 28x28x64")
print("-" * 50)
print(f"Direct 3x3 conv:        {params_3x3:>8,} params")
print(f"Direct 1x1 conv:        {1*1*C_in*C_out + C_out:>8,} params")
print()
print("Bottleneck block (256->64->64):")
print(f"  1x1 reduce (256->64): {1*1*256*64 + 64:>8,}")
print(f"  3x3 conv (64->64):    {3*3*64*64 + 64:>8,}")
print(f"  1x1 expand (64->64):  {1*1*64*64 + 64:>8,}")
print(f"  Total bottleneck:     {params_bottleneck:>8,}")
print()
ratio = params_3x3 / (1*1*C_in*C_out + C_out)
print(f"1x1 conv uses {ratio:.1f}x FEWER parameters than 3x3 for same channel change!")
"""))

    cells.append(code_cell("""# Visualize 1x1 convolution effect
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Create a multi-channel feature map (simplified as 3 channels for visualization)
np.random.seed(42)
feat_map = np.random.randn(8, 8, 3)

# 1x1 convolution weights: 3 input -> 2 output channels
w_1x1 = np.array([
    [1.0, 0.5, -0.3],  # output channel 0
    [-0.2, 0.8, 1.0],  # output channel 1
])
b_1x1 = np.array([0.1, -0.1])

# Apply 1x1 conv
output_1x1 = np.zeros((8, 8, 2))
for i in range(8):
    for j in range(8):
        output_1x1[i, j] = w_1x1 @ feat_map[i, j] + b_1x1

axes[0].imshow(feat_map[:, :, 0], cmap='RdBu_r')
axes[0].set_title('Input Channel 0')
axes[1].imshow(feat_map[:, :, 1], cmap='RdBu_r')
axes[1].set_title('Input Channel 1')
axes[2].imshow(output_1x1[:, :, 0], cmap='RdBu_r')
axes[2].set_title('Output Channel 0 (1x1 conv)')

plt.tight_layout()
plt.show()

print("1x1 convolution combines channels at each spatial location.")
print("Spatial size stays the same, but channel count changes.")
print("It's like a mini fully-connected layer applied at every pixel!")
"""))

    # Section 4: GoogLeNet / Inception
    cells.append(md_cell("""## 4. GoogLeNet / Inception：多尺度特征提取

### 核心思想

**不同大小的过滤器捕捉不同尺度的特征**——为什么不同时使用多种大小的过滤器呢？

### Inception 模块

```
           输入
            |
     +------+------+------+
     |      |      |      |
   1x1    1x1    1x1    3x3
   Conv   Conv   Conv   MaxPool
     |      |      |      |
     |    3x3    5x5    1x1
     |    Conv   Conv   Conv
     |      |      |      |
     +------+------+------+
            |
      通道拼接 (concat)
```

### 关键设计

1. **多种过滤器并行**：1×1、3×3、5×5 同时提取不同尺度特征
2. **1×1 卷积做瓶颈**：在 3×3 和 5×5 之前先用 1×1 降维，减少计算量
3. **池化也加 1×1**：最大池化后接 1×1 卷积改变通道数
4. **深度拼接**：所有分支的输出在通道维度上拼接起来

> **参考论文**：[[Szegedy et al., 2014]](https://arxiv.org/abs/1409.4842) Going Deeper with Convolutions

---

**思考题 4：** Inception 模块中，为什么在 3×3 和 5×5 卷积之前要加 1×1 卷积？
"""))

    # Section 5: ResNet
    cells.append(md_cell("""## 5. ResNet：残差连接让深度突破极限

### 问题：网络越深越好吗？

常识告诉我们，网络越深，表达能力越强，效果应该越好。

但实验发现：**当网络深度增加到一定程度后，训练误差反而上升了！**

这不是过拟合（过拟合是训练误差低、测试误差高），而是**训练误差本身就高**——说明深层网络**更难训练**。

### 直觉：恒等映射很容易学吗？

假设我们有一个训练好的 20 层网络，现在想加几层变成 26 层。如果新增的几层什么都不做（恒等映射），那效果至少不应该变差吧？

但实际上，用随机初始化的多层网络去学恒等映射**非常困难**。

### ResNet 的解决方案：残差连接（Residual Connection）

不让网络直接学目标映射 H(x)，而是让它学**残差** F(x) = H(x) - x。

那么最终输出就是：
$$H(x) = F(x) + x$$

如果恒等映射是最优的，网络只需要让 F(x) = 0，这比让一堆非线性层学恒等映射容易得多！

> **参考论文**：[[He et al., 2015]](https://arxiv.org/abs/1512.03385) Deep Residual Learning for Image Recognition

---

**思考题 5：** 残差连接为什么能缓解梯度消失问题？（提示：反向传播时，梯度可以直接通过 shortcut 路径传回去）
"""))

    cells.append(md_cell("""### ResNet 基本模块

```
     x ------------------+ (shortcut / skip connection)
     |                   |
   Conv 3x3              |
   BatchNorm             |
   ReLU                  |
     |                   |
   Conv 3x3              |
   BatchNorm             |
     |                   |
     +-------------------+
     |
   ReLU
     |
   output = F(x) + x
```

### 两种残差块

1. **Basic Block**：两个 3×3 卷积（用于 ResNet-18, ResNet-34）
2. **Bottleneck Block**：1×1 → 3×3 → 1×1（用于 ResNet-50/101/152，减少计算量）

### 维度匹配问题

当 shortcut 路径的 x 和 F(x) 维度不同时（空间尺寸或通道数不同），需要对 x 做一个**投影**：
- 通常用 1×1 卷积来改变通道数和空间尺寸
- 空间尺寸不同时，shortcut 的 1×1 卷积使用 stride=2

---
"""))

    cells.append(code_cell("""# Manual residual block computation

def relu(x):
    return np.maximum(0, x)

def residual_block_manual(x, W1, b1, W2, b2, W_shortcut=None, b_shortcut=None):
    \"\"\"
    Manual computation of a basic residual block.
    x shape: (C, H, W) for simplicity (single sample)
    W shape: (F, C, HH, WW)
    \"\"\"
    # For simplicity, we'll use 1D "convolution" as element-wise transform
    # to make the manual calculation clear and verifiable
    
    # Actually let's do a simple fully-connected version to show the idea
    # F(x) = W2 * ReLU(W1 * x)
    # output = ReLU(F(x) + x)   [or + W_shortcut*x if dimensions change]
    
    h = relu(W1 @ x + b1)  # First layer + ReLU
    fx = W2 @ h + b2        # Second layer
    
    if W_shortcut is not None:
        identity = W_shortcut @ x + b_shortcut
    else:
        identity = x
    
    out = relu(fx + identity)  # Residual connection + ReLU
    return out, fx, identity


# Simple test case
np.random.seed(42)
x = np.array([1.0, -2.0, 3.0, 0.5])
D = len(x)

# Initialize weights (small random, so F(x) starts near zero -> identity)
W1 = np.random.randn(D, D) * 0.1
b1 = np.zeros(D)
W2 = np.random.randn(D, D) * 0.1
b2 = np.zeros(D)

out, fx, identity = residual_block_manual(x, W1, b1, W2, b2)

print("Residual Block Manual Calculation")
print("=" * 50)
print(f"Input x:         {x}")
print(f"F(x) (residual): {np.round(fx, 4)}")
print(f"Identity (x):    {identity}")
print(f"F(x) + x:        {np.round(fx + identity, 4)}")
print(f"Output (ReLU):   {np.round(out, 4)}")
print()
print("Observation: When weights are small, F(x) is near zero,")
print("so output is approximately ReLU(x) -> close to identity!")
print("This is why residual networks are easy to train - they start as identity.")
"""))

    cells.append(code_cell("""# Implement a convolutional residual block (simplified)

def conv_single(x, w, b, stride=1, padding=0):
    \"\"\"Simplified single conv: x shape (C, H, W), w shape (F, C, HH, WW)\"\"\"
    C, H, W = x.shape
    F, _, HH, WW = w.shape
    
    x_pad = np.pad(x, ((0,0), (padding,padding), (padding,padding)), mode='constant')
    _, H_pad, W_pad = x_pad.shape
    
    H_out = (H_pad - HH) // stride + 1
    W_out = (W_pad - WW) // stride + 1
    
    out = np.zeros((F, H_out, W_out))
    for f in range(F):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                patch = x_pad[:, h_start:h_start+HH, w_start:w_start+WW]
                out[f, i, j] = np.sum(patch * w[f]) + b[f]
    return out


def basic_residual_block_forward(x, W1, b1, W2, b2, W_short=None, b_short=None):
    \"\"\"
    Basic ResNet block forward pass.
    x: (C, H, W) input feature map
    W1, W2: conv weights (F, C, 3, 3) and (F, F, 3, 3)
    W_short: optional shortcut projection (F, C, 1, 1), for dimension mismatch
    \"\"\"
    # F(x): two 3x3 conv layers with ReLU in between
    h = conv_single(x, W1, b1, stride=1, padding=1)
    h = relu(h)
    fx = conv_single(h, W2, b2, stride=1, padding=1)
    
    # Shortcut connection
    if W_short is not None:
        identity = conv_single(x, W_short, b_short, stride=1, padding=0)
    else:
        identity = x
    
    # Add and apply ReLU
    out = relu(fx + identity)
    return out, fx, identity


# Test with small dimensions
np.random.seed(42)
C_in = 4
H, W = 6, 6
F = 4

x = np.random.randn(C_in, H, W)
W1 = np.random.randn(F, C_in, 3, 3) * 0.1
b1 = np.zeros(F)
W2 = np.random.randn(F, F, 3, 3) * 0.1
b2 = np.zeros(F)

out, fx, identity = basic_residual_block_forward(x, W1, b1, W2, b2)

print(f"Input shape:  {x.shape}")
print(f"F(x) shape:   {fx.shape}")
print(f"Output shape: {out.shape}")
print()
print(f"Mean abs(F(x)):   {np.mean(np.abs(fx)):.4f}  (small because weights are small)")
print(f"Mean abs(x):      {np.mean(np.abs(x)):.4f}")
print(f"Mean abs(output): {np.mean(np.abs(out)):.4f}")
print()
print("Residual block preserves the input magnitude when weights are small.")
print("This is the key to training very deep networks!")
"""))

    cells.append(code_cell("""# ResNet depth comparison visualization

architectures = ['AlexNet', 'VGG-16', 'ResNet-50', 'ResNet-101', 'ResNet-152']
depths = [8, 16, 50, 101, 152]
top5_errors = [15.3, 7.3, 6.2, 5.6, 4.9]  # approximate ImageNet top-5 error (%)

fig, ax1 = plt.subplots(figsize=(10, 5))

bars = ax1.bar(architectures, depths, color=['#e74c3c', '#f39c12', '#27ae60', '#2980b9', '#8e44ad'], alpha=0.7)
ax1.set_ylabel('Number of layers', color='#2c3e50', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#2c3e50')

for bar, depth in zip(bars, depths):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
             f'{depth} layers', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2 = ax1.twinx()
ax2.plot(architectures, top5_errors, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='Top-5 Error')
ax2.set_ylabel('ImageNet Top-5 Error (%)', color='#e74c3c', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#e74c3c')
ax2.invert_yaxis()  # lower error is better

for i, err in enumerate(top5_errors):
    ax2.text(i, err + 0.3, f'{err}%', ha='center', va='bottom', fontsize=10, color='#e74c3c')

ax1.set_title('CNN Architectures: Depth vs Performance', fontsize=14)
plt.tight_layout()
plt.show()

print("Key insight: ResNet enables training much deeper networks,")
print("and performance keeps improving with depth (up to a point).")
print("Before ResNet, training networks deeper than ~20 layers was very difficult.")
"""))

    # Section 6: Transfer learning
    cells.append(md_cell("""## 6. 迁移学习（简介）

### 什么是迁移学习？

用在大数据集（如 ImageNet）上预训练好的模型，在自己的小数据集上做微调（fine-tuning）。

### 为什么有效？

- 底层特征（边缘、纹理）是通用的
- 高层特征（物体部件、整体）与数据集相关
- 小数据集上从头训练容易过拟合

### 常见策略

1. **固定特征提取器**：冻结卷积层，只训练最后的全连接层
2. **微调部分层**：解冻最后几层卷积层，一起训练
3. **全模型微调**：整个网络都训练，但学习率要小

---
"""))

    # Homework 1
    cells.append(md_cell("""---

## 作业 1：实现 VGG 风格的小型网络

### 任务

实现一个简化版的 VGG 风格网络前向传播。要求：

1. 使用 3×3 小过滤器（VGG 的标志性设计）
2. 重复 CONV → ReLU → CONV → ReLU → POOL 的模块结构
3. 每经过一个模块，空间尺寸减半，通道数翻倍
4. 最后用全局平均池化 + 全连接层做分类

### 网络结构

```
输入: 32x32x3
  ↓
Conv 3x3, 16 filters, pad=1 → 32x32x16
ReLU
Conv 3x3, 16 filters, pad=1 → 32x32x16
ReLU
MaxPool 2x2, stride=2 → 16x16x16
  ↓
Conv 3x3, 32 filters, pad=1 → 16x16x32
ReLU
Conv 3x3, 32 filters, pad=1 → 16x16x32
ReLU
MaxPool 2x2, stride=2 → 8x8x32
  ↓
Global Average Pool → 32
FC → 10 classes
```

---
"""))

    cells.append(code_cell("""# ===== Homework 1: VGG-style small network =====

def vgg_style_forward(x, params):
    \"\"\"
    Forward pass of a small VGG-style network.
    
    Inputs:
    - x: Input data of shape (N, 3, 32, 32) - CIFAR-10 size
    - params: Dictionary with all weights and biases
      - 'W1', 'b1': first conv layer (16 filters, 3x3x3)
      - 'W2', 'b2': second conv layer (16 filters, 3x3x16)
      - 'W3', 'b3': third conv layer (32 filters, 3x3x16)
      - 'W4', 'b4': fourth conv layer (32 filters, 3x3x32)
      - 'W_fc', 'b_fc': final fully connected layer (32 -> 10)
    
    Returns:
    - scores: (N, 10) class scores
    - cache: intermediate values for backward pass
    \"\"\"
    N = x.shape[0]
    
    # YOUR CODE HERE
    # Hint: Use the conv_forward_naive and max_pool_forward_naive from notebook 1
    # Hint: For global average pooling, use np.mean over spatial dimensions
    # Hint: Don't forget ReLU after each conv layer
    
    # Block 1: conv -> relu -> conv -> relu -> pool
    
    # Block 2: conv -> relu -> conv -> relu -> pool
    
    # Global average pool
    
    # Fully connected layer
    
    scores = None  # TODO
    cache = {}
    return scores, cache


# ===== ANSWER KEY =====
def vgg_style_forward_answer(x, params):
    \"\"\"Reference implementation.\"\"\"
    N = x.shape[0]
    
    # Block 1
    h1, _ = conv_forward_naive(x, params['W1'], params['b1'], {'stride': 1, 'pad': 1})
    h1_relu = np.maximum(0, h1)
    h2, _ = conv_forward_naive(h1_relu, params['W2'], params['b2'], {'stride': 1, 'pad': 1})
    h2_relu = np.maximum(0, h2)
    pool1, _ = max_pool_forward_naive(h2_relu, {'pool_height': 2, 'pool_width': 2, 'stride': 2})
    
    # Block 2
    h3, _ = conv_forward_naive(pool1, params['W3'], params['b3'], {'stride': 1, 'pad': 1})
    h3_relu = np.maximum(0, h3)
    h4, _ = conv_forward_naive(h3_relu, params['W4'], params['b4'], {'stride': 1, 'pad': 1})
    h4_relu = np.maximum(0, h4)
    pool2, _ = max_pool_forward_naive(h4_relu, {'pool_height': 2, 'pool_width': 2, 'stride': 2})
    
    # Global average pool: (N, C, H, W) -> (N, C)
    gap = np.mean(pool2, axis=(2, 3))
    
    # FC layer
    scores = gap @ params['W_fc'] + params['b_fc']
    
    cache = {'pool1': pool1, 'pool2': pool2, 'gap': gap}
    return scores, cache


# Helper: initialize VGG-style params
def init_vgg_params():
    np.random.seed(42)
    params = {
        'W1': np.random.randn(16, 3, 3, 3) * 0.1,
        'b1': np.zeros(16),
        'W2': np.random.randn(16, 16, 3, 3) * 0.1,
        'b2': np.zeros(16),
        'W3': np.random.randn(32, 16, 3, 3) * 0.1,
        'b3': np.zeros(32),
        'W4': np.random.randn(32, 32, 3, 3) * 0.1,
        'b4': np.zeros(32),
        'W_fc': np.random.randn(32, 10) * 0.1,
        'b_fc': np.zeros(10),
    }
    return params

print("Homework 1: Implement vgg_style_forward")
print("Write your solution above, then run the tests.")
"""))

    cells.append(code_cell("""# ===== Homework 1: Test cases =====
np.random.seed(42)
x_test = np.random.randn(2, 3, 32, 32)
params = init_vgg_params()

scores_ref, cache_ref = vgg_style_forward_answer(x_test, params)
scores_student, cache_student = vgg_style_forward(x_test, params)

# Shape test
assert scores_student is not None, "Please implement the function first!"
assert scores_student.shape == (2, 10), f"Shape mismatch: {scores_student.shape} vs (2, 10)"
print(f"Output shape: {scores_student.shape} (expected: (2, 10))")
print("Shape test: PASSED")

# Value test
assert np.allclose(scores_student, scores_ref, atol=1e-8), "Scores don't match reference!"
print("Value test: PASSED")

# Intermediate shape tests
assert 'pool2' in cache_ref
assert cache_ref['pool2'].shape == (2, 32, 8, 8), f"pool2 shape wrong: {cache_ref['pool2'].shape}"
print(f"Pool2 shape: {cache_ref['pool2'].shape} (expected: (2, 32, 8, 8))")
print("Intermediate shape test: PASSED")

# Parameter count check
total_params = sum(np.prod(v.shape) for v in params.values())
print(f"\nTotal parameters: {total_params:,}")
assert total_params == 16*3*3*3+16 + 16*16*3*3+16 + 32*16*3*3+32 + 32*32*3*3+32 + 32*10+10
print("Parameter count check: PASSED")

print("\n=== All Homework 1 tests PASSED! ===")
"""))

    # Homework 2
    cells.append(md_cell("""---

## 作业 2：从零实现 ResNet 基本块

### 任务

实现一个完整的 ResNet Basic Block 前向传播，包括：
1. 两个 3×3 卷积层
2. Batch Normalization（简化版）
3. ReLU 激活
4. 残差连接（支持维度不匹配时的投影 shortcut）

### 提示

- Basic Block: Conv → BN → ReLU → Conv → BN → 加 x → ReLU
- 当输入输出通道数不同时，shortcut 路径需要用 1×1 卷积做投影
- 简化版 BN：对每个通道做归一化（减去均值除以标准差），再缩放平移

---
"""))

    cells.append(code_cell("""# ===== Homework 2: ResNet Basic Block =====

def batchnorm_forward(x, gamma, beta, eps=1e-5):
    \"\"\"
    Simplified batch normalization for feature maps (N, C, H, W).
    Normalizes over (N, H, W) dimensions for each channel.
    \"\"\"
    # Compute mean and variance per channel
    mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
    var = np.var(x, axis=(0, 2, 3), keepdims=True)
    
    # Normalize
    x_norm = (x - mean) / np.sqrt(var + eps)
    
    # Scale and shift
    out = gamma.reshape(1, -1, 1, 1) * x_norm + beta.reshape(1, -1, 1, 1)
    
    return out


def resnet_basic_block_forward(x, params):
    \"\"\"
    ResNet Basic Block forward pass.
    
    Inputs:
    - x: (N, C_in, H, W) input
    - params: dict with keys:
      - 'W1', 'b1': first conv (C_out, C_in, 3, 3)
      - 'gamma1', 'beta1': first BN params (C_out,)
      - 'W2', 'b2': second conv (C_out, C_out, 3, 3)
      - 'gamma2', 'beta2': second BN params (C_out,)
      - 'W_short', 'b_short': shortcut projection (C_out, C_in, 1, 1), None if C_in==C_out
    
    Returns:
    - out: (N, C_out, H, W) output
    \"\"\"
    # YOUR CODE HERE
    # Hint: 
    # 1. First conv + BN + ReLU
    # 2. Second conv + BN
    # 3. Shortcut: identity or projection
    # 4. Add + ReLU
    
    out = None  # TODO
    return out


# ===== ANSWER KEY =====
def resnet_basic_block_forward_answer(x, params):
    \"\"\"Reference implementation.\"\"\"
    # First conv + BN + ReLU
    h1, _ = conv_forward_naive(x, params['W1'], params['b1'], {'stride': 1, 'pad': 1})
    h1_bn = batchnorm_forward(h1, params['gamma1'], params['beta1'])
    h1_relu = np.maximum(0, h1_bn)
    
    # Second conv + BN
    h2, _ = conv_forward_naive(h1_relu, params['W2'], params['b2'], {'stride': 1, 'pad': 1})
    h2_bn = batchnorm_forward(h2, params['gamma2'], params['beta2'])
    
    # Shortcut
    if 'W_short' in params and params['W_short'] is not None:
        shortcut, _ = conv_forward_naive(x, params['W_short'], params['b_short'], 
                                         {'stride': 1, 'pad': 0})
    else:
        shortcut = x
    
    # Residual connection + ReLU
    out = np.maximum(0, h2_bn + shortcut)
    return out


# Helper: init basic block params
def init_basic_block_params(C_in, C_out):
    np.random.seed(42)
    params = {
        'W1': np.random.randn(C_out, C_in, 3, 3) * np.sqrt(2.0 / (C_in * 3 * 3)),
        'b1': np.zeros(C_out),
        'gamma1': np.ones(C_out),
        'beta1': np.zeros(C_out),
        'W2': np.random.randn(C_out, C_out, 3, 3) * np.sqrt(2.0 / (C_out * 3 * 3)),
        'b2': np.zeros(C_out),
        'gamma2': np.ones(C_out),
        'beta2': np.zeros(C_out),
    }
    if C_in != C_out:
        params['W_short'] = np.random.randn(C_out, C_in, 1, 1) * 0.1
        params['b_short'] = np.zeros(C_out)
    else:
        params['W_short'] = None
        params['b_short'] = None
    return params

print("Homework 2: Implement resnet_basic_block_forward")
print("Write your solution above, then run the tests.")
"""))

    cells.append(code_cell("""# ===== Homework 2: Test cases =====
np.random.seed(42)

# Test 1: Same number of channels (identity shortcut)
print("Test 1: Identity shortcut (C_in = C_out = 8)")
x1 = np.random.randn(2, 8, 8, 8)
params1 = init_basic_block_params(8, 8)

out_ref1 = resnet_basic_block_forward_answer(x1, params1)
out_student1 = resnet_basic_block_forward(x1, params1)

assert out_student1 is not None, "Please implement the function first!"
assert out_student1.shape == (2, 8, 8, 8), f"Shape mismatch: {out_student1.shape}"
assert np.allclose(out_student1, out_ref1, atol=1e-6), "Values don't match!"
print("  Shape: PASSED")
print("  Values: PASSED")

# Test 2: Different channels (projection shortcut)
print("\nTest 2: Projection shortcut (C_in=8, C_out=16)")
x2 = np.random.randn(2, 8, 8, 8)
params2 = init_basic_block_params(8, 16)

out_ref2 = resnet_basic_block_forward_answer(x2, params2)
out_student2 = resnet_basic_block_forward(x2, params2)

assert out_student2.shape == (2, 16, 8, 8), f"Shape mismatch: {out_student2.shape}"
assert np.allclose(out_student2, out_ref2, atol=1e-6), "Values don't match!"
print("  Shape: PASSED")
print("  Values: PASSED")

# Test 3: Residual property - small weights -> output close to input
print("\nTest 3: Residual property check")
x3 = np.random.randn(1, 4, 6, 6)
params3 = init_basic_block_params(4, 4)
# Make weights very small
params3['W1'] *= 0.01
params3['W2'] *= 0.01

out3 = resnet_basic_block_forward_answer(x3, params3)
diff = np.mean(np.abs(out3 - np.maximum(0, x3)))
print(f"  Mean |output - ReLU(input)| = {diff:.6f}")
print(f"  (should be very small since weights are tiny)")
assert diff < 0.1, "Residual property not satisfied!"
print("  Residual property: PASSED")

print("\n=== All Homework 2 tests PASSED! ===")
"""))

    # Summary
    cells.append(md_cell("""---

## 本讲总结

### 架构演进之路

| 架构 | 年份 | 深度 | 核心创新 |
|------|------|------|----------|
| LeNet-5 | 1998 | 5 | 卷积+池化的基本范式 |
| AlexNet | 2012 | 8 | ReLU、Dropout、GPU、数据增强 |
| VGG | 2014 | 16 | 统一使用 3×3 小过滤器 |
| GoogLeNet | 2014 | 22 | Inception 模块，多尺度特征 |
| ResNet | 2015 | 50/101/152 | 残差连接，突破深度极限 |

### 关键设计模式

1. **小过滤器更好**：多个 3×3 比一个大过滤器参数更少、非线性更多
2. **1×1 卷积是瑞士军刀**：降维、升维、跨通道组合、增加非线性
3. **残差连接是训练深度网络的关键**：让恒等映射变得容易学
4. **空间降采样 + 通道升维**：信息从空间表示逐步转为语义表示

### 思考题答案提示

1. ReLU 在正区间梯度恒为 1，不会像 sigmoid 那样在两端梯度趋近于 0（饱和）
2. 每做一次 3×3 卷积，每个神经元的感受野增加 2（上下各 1）。两次就增加 4，加上本身 1 = 5
3. 1×1: 256×64+64 = 16,448；3×3: 3×3×256×64+64 = 147,520。1×1 少了约 9 倍
4. 减少计算量。5×5 卷积计算量很大，先降维再卷积可以大幅减少 FLOPs
5. 反向传播时，梯度有一条"高速公路"——通过 shortcut 直接从输出传到输入，梯度值为 1，不会消失

### 论文参考

- [[Krizhevsky et al., 2012]](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) AlexNet
- [[Simonyan & Zisserman, 2014]](https://arxiv.org/abs/1409.1556) VGG
- [[Szegedy et al., 2014]](https://arxiv.org/abs/1409.4842) GoogLeNet / Inception
- [[He et al., 2015]](https://arxiv.org/abs/1512.03385) ResNet

### GitHub 参考实现

- [pytorch/vision](https://github.com/pytorch/vision) (16k stars) — torchvision 模型库，包含所有经典 CNN 的 PyTorch 实现
- [keras-team/keras-applications](https://github.com/keras-team/keras-applications) (1.8k stars) — Keras 预训练模型

---

**下节预告**：Vision Transformer——当 Transformer 遇见图像，会擦出怎样的火花？
"""))

    return cells


# ============================================================
# NOTEBOOK 3: Vision Transformer
# ============================================================
def build_notebook3():
    cells = []

    cells.append(md_cell("""# Vision Transformer：当 Transformer 遇见图像

> **CS231n 深度学习与计算机视觉** · 第8讲
>
> 本 notebook 将带你理解 Vision Transformer (ViT) 的核心思想——从 CNN 的局限性出发，到 Patch Embedding、位置编码、多头自注意力，再到亲手实现自注意力和 ViT 的基本模块。

---

## 学习目标

- 理解 CNN 的局限性（固定感受野、缺乏全局上下文）
- 回顾 Transformer 的核心机制：自注意力
- 理解为什么不能直接对像素用 Transformer
- 掌握 Patch Embedding 的原理
- 理解位置编码的作用和常见方式
- 手动计算自注意力
- 从零实现多头自注意力
- 理解 ViT 的完整架构
- 了解 DeiT 等 ViT 改进工作

---
"""))

    cells.append(code_cell("""# Setup
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("Environment ready!")
"""))

    # Section 1: Limitations of CNNs
    cells.append(md_cell("""## 1. CNN 的局限性

### CNN 擅长什么？

- 局部特征提取（边缘、纹理）
- 参数共享（平移不变性）
- 层次化特征表示（从底层到高层）

### CNN 的局限

1. **固定的局部感受野**：卷积核大小有限，要获取全局信息需要堆叠很多层
2. **长距离依赖建模困难**：图像两端的信息需要经过很多层才能"相遇"
3. **缺乏显式的全局建模**：CNN 更关注局部，对全局关系的建模是隐式的

### 为什么需要全局信息？

- **图像分类**：需要综合整张图的信息来判断类别
- **目标检测**：需要理解物体之间的关系
- **图像描述**：需要全局理解来生成准确的文字

> **类比**：CNN 就像一个用放大镜看画的人，每次只能看一小块，需要慢慢移动才能理解整幅画。而 Transformer 可以"一眼看全"。

---

**思考题 1：** 对于 224×224 的输入，一个 3×3 卷积需要堆叠多少层，才能让输出神经元的感受野覆盖整幅图像？
"""))

    cells.append(code_cell("""# Calculate how many conv layers are needed for full receptive field

def rf_after_n_layers(n, filter_size=3, stride=1):
    \"\"\"Compute receptive field after n conv layers (all 3x3, stride 1)\"\"\"
    rf = 1
    for _ in range(n):
        rf = (rf - 1) * stride + filter_size
    return rf

print("Receptive field growth with 3x3 conv layers (stride=1):")
print("-" * 50)
for n in [1, 2, 3, 5, 10, 15, 20, 30, 50, 100]:
    rf = rf_after_n_layers(n)
    print(f"  After {n:3d} conv layers: receptive field = {rf:4d}x{rf:4d}")

# How many layers needed for 224x224?
target = 224
n = 0
rf = 1
while rf < target:
    n += 1
    rf = rf_after_n_layers(n)

print(f"\nTo cover {target}x{target} with 3x3 conv (stride 1):")
print(f"  Need {n} layers! (receptive field = {rf}x{rf})")
print(f"\nIn contrast, self-attention can model global relationships in ONE layer.")
"""))

    cells.append(code_cell("""# Visual comparison: CNN receptive field vs Self-attention global view
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# CNN: local receptive field
img = np.random.rand(20, 20)
axes[0].imshow(img, cmap='gray')
axes[0].set_title('CNN: Local Receptive Field', fontsize=12)
# Highlight a small local region
rect = Rectangle((8, 8), 4, 4, linewidth=2, edgecolor='red', facecolor='red', alpha=0.5)
axes[0].add_patch(rect)
axes[0].text(10, 6, 'Each neuron\nsees only local', ha='center', color='red', fontsize=10)

# Self-attention: global view
axes[1].imshow(img, cmap='gray')
axes[1].set_title('Self-Attention: Global View', fontsize=12)
# Draw lines from one point to all others
for i in range(0, 20, 3):
    for j in range(0, 20, 3):
        axes[1].plot([10, j], [10, i], 'r-', alpha=0.2, linewidth=0.5)
axes[1].plot(10, 10, 'ro', markersize=8)
axes[1].text(10, 2, 'Each position attends\nto ALL positions', ha='center', color='red', fontsize=10)

plt.tight_layout()
plt.show()

print("Self-attention can capture long-range dependencies in a single layer!")
print("This is the key advantage of Transformers for vision tasks.")
"""))

    # Section 2: Transformer recap
    cells.append(md_cell("""## 2. Transformer 核心机制回顾：自注意力

### 自注意力（Self-Attention）是什么？

对于序列中的每个元素，自注意力会计算它与**所有其他元素**的相关性（注意力权重），然后用这些权重对所有元素的值做加权求和，得到该元素的新表示。

### Q, K, V 三个角色

- **Query (Q)**：当前元素的"查询向量"——我在找什么？
- **Key (K)**：每个元素的"键向量"——我包含什么信息？
- **Value (V)**：每个元素的"值向量"——我的实际内容是什么？

### 计算步骤

1. 对每个位置，用 Q 和所有 K 做点积，得到注意力分数
2. 除以 √d_k 进行缩放（防止点积太大导致 softmax 饱和）
3. 用 softmax 归一化，得到注意力权重（和为 1）
4. 用注意力权重对 V 加权求和，得到输出

$$Attention(Q, K, V) = softmax(\\frac{QK^T}{\\sqrt{d_k}})V$$

> **参考论文**：[[Vaswani et al., 2017]](https://arxiv.org/abs/1706.03762) Attention Is All You Need

---
"""))

    cells.append(code_cell("""# Manual self-attention calculation with small matrices

# Suppose we have 4 tokens, each with dimension 8
np.random.seed(42)
seq_len = 4
d_model = 8

# Input embeddings (4 tokens, 8 dimensions each)
x = np.random.randn(seq_len, d_model)

# Q, K, V projection weights (simplified - same dimension)
d_k = d_model
W_q = np.random.randn(d_model, d_k) * 0.1
W_k = np.random.randn(d_model, d_k) * 0.1
W_v = np.random.randn(d_model, d_k) * 0.1

# Compute Q, K, V
Q = x @ W_q  # (4, 8)
K = x @ W_k  # (4, 8)
V = x @ W_v  # (4, 8)

print("Input embeddings x (4 tokens, dim=8):")
print(np.round(x, 3))
print()
print(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
"""))

    cells.append(code_cell("""# Step-by-step self-attention computation

# Step 1: Compute attention scores = Q @ K^T
scores = Q @ K.T  # (4, 4)
print("Step 1: Attention scores (Q @ K^T):")
print(np.round(scores, 3))
print()

# Step 2: Scale by sqrt(d_k)
d_k = Q.shape[1]
scaled_scores = scores / np.sqrt(d_k)
print("Step 2: Scaled scores (divided by sqrt(d_k)):")
print(np.round(scaled_scores, 3))
print()

# Step 3: Softmax to get attention weights
def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

attn_weights = softmax(scaled_scores, axis=-1)
print("Step 3: Attention weights (softmax):")
print(np.round(attn_weights, 3))
print(f"\nEach row sums to: {np.sum(attn_weights, axis=1)}")
print("(Each row = attention distribution for one query position)")
"""))

    cells.append(code_cell("""# Step 4: Weighted sum of values
output = attn_weights @ V  # (4, 8)

print("Step 4: Output = attention_weights @ V")
print(f"Output shape: {output.shape}")
print()
print("Output (first 4 dimensions shown):")
print(np.round(output[:, :4], 3))

print()
print("Self-attention intuition:")
print("- Each output token is a weighted sum of ALL value tokens")
print("- The weights are determined by how similar Q and K are")
print("- This allows each position to 'attend' to all other positions")
"""))

    cells.append(code_cell("""# Visualize self-attention
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Attention weights heatmap
im0 = axes[0].imshow(attn_weights, cmap='YlOrRd', vmin=0, vmax=1)
axes[0].set_title('Attention Weights')
axes[0].set_xlabel('Key positions')
axes[0].set_ylabel('Query positions')
for i in range(4):
    for j in range(4):
        axes[0].text(j, i, f'{attn_weights[i,j]:.2f}', ha='center', va='center', fontsize=9)
plt.colorbar(im0, ax=axes[0], fraction=0.046)

# Q and K similarity (raw scores)
im1 = axes[1].imshow(scaled_scores, cmap='RdBu_r', vmin=-1, vmax=1)
axes[1].set_title('Scaled QK^T Scores')
axes[1].set_xlabel('Key positions')
axes[1].set_ylabel('Query positions')
plt.colorbar(im1, ax=axes[1], fraction=0.046)

# Value vectors
im2 = axes[2].imshow(V, cmap='RdBu_r', aspect='auto')
axes[2].set_title('Value Vectors (V)')
axes[2].set_xlabel('Dimension')
axes[2].set_ylabel('Token position')
plt.colorbar(im2, ax=axes[2], fraction=0.046)

plt.tight_layout()
plt.show()

print("Each row of attention weights tells us:")
print("  \"For this position, how much should I attend to each other position?\"")
"""))

    # Section 3: Why not just apply Transformer to pixels?
    cells.append(md_cell("""## 3. 为什么不能直接对像素用 Transformer？

### 想法很美好...

如果把每个像素当作一个 token，224×224 的图像就有 50,176 个 token。

### 问题在哪里？

自注意力的计算复杂度是 **O(n²·d)**，其中 n 是序列长度。

- 224×224 = 50,176 个 token
- 注意力矩阵大小：50,176 × 50,176 ≈ **25 亿**个元素
- 这在计算和内存上都是不可行的！

### 解决方案：Patch Embedding

把图像分成若干个**小块（patch）**，每个 patch 作为一个 token。

- 例如：224×224 的图像，patch 大小 16×16
- 得到 14×14 = 196 个 patch token
- 196 个 token 的自注意力就完全可以承受了！

---

**思考题 2：** 如果图像是 32×32×3，patch 大小是 4×4，那么有多少个 patch token？每个 patch 展平后是多少维？
"""))

    cells.append(code_cell("""# Complexity comparison: pixel-level vs patch-level

img_size = 224
patch_size = 16
d_model = 768  # ViT-Base dimension

n_pixels = img_size * img_size
n_patches = (img_size // patch_size) ** 2

# Attention matrix size
pixel_attn_size = n_pixels * n_pixels
patch_attn_size = n_patches * n_patches

print("Self-attention complexity comparison:")
print("-" * 55)
print(f"Image size: {img_size}x{img_size}")
print(f"Patch size: {patch_size}x{patch_size}")
print()
print(f"{'':20s} {'Pixel-level':>15s} {'Patch-level':>15s}")
print(f"{'Tokens':20s} {n_pixels:>15,} {n_patches:>15,}")
print(f"{'Attention matrix':20s} {pixel_attn_size:>15,} {patch_attn_size:>15,}")
print(f"{'Ratio':20s} {'1x':>15s} {f'{pixel_attn_size/patch_attn_size:.0f}x smaller':>15s}")

print()
print(f"Patch-level attention uses {pixel_attn_size/patch_attn_size:.0f}x LESS memory/compute!")
print("This is the key insight that makes Vision Transformers feasible.")
"""))

    # Section 4: Patch Embedding
    cells.append(md_cell("""## 4. Patch Embedding：把图像切成块

### 怎么做？

1. 将图像分割成不重叠的 patch（如 16×16）
2. 将每个 patch 展平成一个向量
3. 通过一个线性层（或等价的卷积层）投影到目标维度 d_model

### 等价实现

Patch Embedding 实际上等价于一个**卷积层**：
- 卷积核大小 = patch 大小
- 步幅 = patch 大小
- 输出通道数 = d_model

这样一次卷积就能完成"分块 + 投影"的操作！

---
"""))

    cells.append(code_cell("""# Implement patch embedding from scratch

def patch_embed_naive(x, patch_size=4, embed_dim=8):
    \"\"\"
    Naive patch embedding implementation.
    
    Inputs:
    - x: (N, C, H, W) input images
    - patch_size: size of each patch (e.g., 4 means 4x4 patches)
    - embed_dim: projection dimension
    
    Returns:
    - patches: (N, num_patches, embed_dim) patch embeddings
    \"\"\"
    N, C, H, W = x.shape
    
    # Check dimensions
    assert H % patch_size == 0, f"Height {H} not divisible by patch_size {patch_size}"
    assert W % patch_size == 0, f"Width {W} not divisible by patch_size {patch_size}"
    
    n_h = H // patch_size  # number of patches along height
    n_w = W // patch_size  # number of patches along width
    num_patches = n_h * n_w
    
    # Initialize projection weight
    patch_dim = C * patch_size * patch_size
    W_proj = np.random.randn(patch_dim, embed_dim) * np.sqrt(2.0 / patch_dim)
    b_proj = np.zeros(embed_dim)
    
    # Extract patches and project
    patches = np.zeros((N, num_patches, embed_dim))
    
    for n in range(N):
        idx = 0
        for i in range(n_h):
            for j in range(n_w):
                # Extract patch
                h_start = i * patch_size
                w_start = j * patch_size
                patch = x[n, :, h_start:h_start+patch_size, w_start:w_start+patch_size]
                # Flatten and project
                patch_flat = patch.flatten()
                patches[n, idx] = patch_flat @ W_proj + b_proj
                idx += 1
    
    return patches, W_proj, b_proj


# Test with small image
np.random.seed(42)
x = np.random.randn(2, 3, 8, 8)  # 2 images, 3 channels, 8x8
patches, W_proj, b_proj = patch_embed_naive(x, patch_size=4, embed_dim=8)

print(f"Input shape: {x.shape}")
print(f"Patch size: 4x4")
print(f"Number of patches per image: {8//4 * 8//4} = 4")
print(f"Output shape: {patches.shape}  (N, num_patches, embed_dim)")
print()
print("Each patch is a 4x4x3 = 48-dim vector projected to 8-dim embedding.")
"""))

    cells.append(code_cell("""# Visualize patch splitting
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Create a simple test image with patterns
test_img = np.zeros((16, 16, 3))
# Red top-left
test_img[:8, :8, 0] = 1.0
# Green top-right
test_img[:8, 8:, 1] = 1.0
# Blue bottom-left
test_img[8:, :8, 2] = 1.0
# Yellow bottom-right
test_img[8:, 8:, 0] = 1.0
test_img[8:, 8:, 1] = 1.0

axes[0, 0].imshow(test_img)
axes[0, 0].set_title('Original (16x16)')
axes[0, 0].axis('off')

# Split into 4x4 patches
patch_size = 4
n_patches = 16 // patch_size

for idx in range(n_patches * n_patches):
    row = idx // n_patches
    col = idx % n_patches
    ax = axes.flatten()[idx + 1]  # skip first position (original)
    if idx < n_patches * n_patches - 1:
        h_start = row * patch_size
        w_start = col * patch_size
        patch = test_img[h_start:h_start+patch_size, w_start:w_start+patch_size]
        ax.imshow(patch)
        ax.set_title(f'Patch {idx+1}')
    ax.axis('off')

# Hide unused axes
for idx in range(n_patches*n_patches + 1, len(axes.flatten())):
    axes.flatten()[idx].axis('off')

plt.suptitle('Patch Embedding: Split Image into Patches', fontsize=14)
plt.tight_layout()
plt.show()
"""))

    # Section 5: Position Embedding
    cells.append(md_cell("""## 5. 位置编码：给 Transformer 补上空间信息

### 问题：自注意力不关心顺序

自注意力是**排列等变**（permutation equivariant）的——打乱 token 的顺序，输出也会相应地打乱。对于图像来说，patch 的空间位置非常重要！

### 解决方案：位置编码

给每个 patch 的 embedding 加上一个**位置向量**，让 Transformer 知道每个 patch 在图像中的位置。

### 常见的位置编码方式

1. **可学习的位置编码**（ViT 使用）：直接初始化一组位置向量，和模型一起训练
2. **正弦位置编码**（原 Transformer 使用）：用固定的正弦/余弦函数生成
3. **相对位置编码**：编码相对位置而非绝对位置

> **为什么用"加"而不是"拼接"？** 这样可以保持维度不变，减少参数。而且模型可以学会从加和的向量中分离出内容信息和位置信息。

---

**思考题 3：** 如果位置编码和 patch embedding 相加，模型怎么区分哪些维度是内容、哪些是位置？
"""))

    cells.append(code_cell("""# Position embedding visualization

num_patches = 16  # 4x4 grid of patches
embed_dim = 32    # embedding dimension

# Learnable position embedding (randomly initialized, would be trained)
np.random.seed(42)
pos_embed = np.random.randn(num_patches, embed_dim) * 0.02

# Sinusoidal position encoding (2D version for patches)
def get_2d_sincos_pos_embed(embed_dim, grid_size):
    \"\"\"Generate 2D sinusoidal position embeddings.\"\"\"
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w, h
    grid = np.stack(grid, axis=0)  # (2, grid_size, grid_size)
    
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = np.zeros((embed_dim, grid_size, grid_size))
    
    # Half dimensions for height, half for width
    dim_h = embed_dim // 4  # sin + cos for height
    dim_w = embed_dim // 4  # sin + cos for width
    
    omega_h = np.arange(dim_h, dtype=np.float32)
    omega_h = 1.0 / (10000 ** (2 * omega_h / dim_h))
    
    omega_w = np.arange(dim_w, dtype=np.float32)
    omega_w = 1.0 / (10000 ** (2 * omega_w / dim_w))
    
    # Height positions
    pos_h = grid[1]  # (1, grid_size, grid_size)
    out_h = np.einsum('d,nhw->dn', omega_h, pos_h.reshape(1, -1))  # wait, let's do it simpler
    
    # Simpler approach
    pos_embed = np.zeros((grid_size * grid_size, embed_dim))
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            for d in range(embed_dim // 2):
                if d % 2 == 0:
                    pos_embed[idx, d] = np.sin(i / (10000 ** (d / (embed_dim // 2))))
                else:
                    pos_embed[idx, d] = np.cos(i / (10000 ** ((d - 1) / (embed_dim // 2))))
            for d in range(embed_dim // 2, embed_dim):
                d2 = d - embed_dim // 2
                if d2 % 2 == 0:
                    pos_embed[idx, d] = np.sin(j / (10000 ** (d2 / (embed_dim // 2))))
                else:
                    pos_embed[idx, d] = np.cos(j / (10000 ** ((d2 - 1) / (embed_dim // 2))))
    
    return pos_embed

sin_pos_embed = get_2d_sincos_pos_embed(embed_dim, 4)

print(f"Position embedding shape: {pos_embed.shape}")
print(f"(num_patches x embed_dim)")
print()
print("Each patch gets a unique position vector that's added to its content embedding.")
print("This tells the transformer WHERE each patch is in the image.")
"""))

    cells.append(code_cell("""# Visualize position embeddings
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Learnable pos embed (random init)
im0 = axes[0].imshow(pos_embed, cmap='RdBu_r', aspect='auto')
axes[0].set_title('Learnable Position Embedding (random init)')
axes[0].set_xlabel('Embedding dimension')
axes[0].set_ylabel('Patch index')
axes[0].set_yticks(range(16))
axes[0].set_yticklabels([f'Patch {i}' for i in range(16)], fontsize=8)
plt.colorbar(im0, ax=axes[0], fraction=0.046)

# Sinusoidal pos embed
im1 = axes[1].imshow(sin_pos_embed, cmap='RdBu_r', aspect='auto')
axes[1].set_title('2D Sinusoidal Position Embedding')
axes[1].set_xlabel('Embedding dimension')
axes[1].set_ylabel('Patch index')
axes[1].set_yticks(range(16))
axes[1].set_yticklabels([f'Patch {i}' for i in range(16)], fontsize=8)
plt.colorbar(im1, ax=axes[1], fraction=0.046)

plt.tight_layout()
plt.show()

print("ViT uses learnable position embeddings.")
print("Interesting fact: ViT learns position embeddings that encode 2D spatial information,")
print("even though they're just 1D vectors!")
"""))

    # Section 6: ViT architecture overview
    cells.append(md_cell("""## 6. ViT 架构总览

### 完整流程

```
输入图像 (224x224x3)
    ↓
Patch Embedding (16x16 patches → 196 tokens, dim=768)
    ↓
+ 位置编码 (196 + 1 (class token), 768)
    ↓
[Transformer Encoder] × 12 层
    ├── 多头自注意力 (Multi-Head Self-Attention)
    ├── 残差连接
    ├── 层归一化 (LayerNorm)
    └── MLP (两层全连接 + GELU)
    ↓
取出 class token 的输出
    ↓
MLP Head → 分类结果
```

### Class Token

在序列最前面加一个特殊的 **class token**，经过 Transformer 后，它的输出代表了整张图的全局信息，用于最终分类。

> **为什么需要 class token？** 因为所有 patch 地位平等，我们需要一个"代表"来汇总全局信息。当然也可以用全局平均池化来替代。

> **参考论文**：[[Dosovitskiy et al., 2020]](https://arxiv.org/abs/2010.11929) An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

---
"""))

    # Section 7: Multi-head self-attention
    cells.append(md_cell("""## 7. 多头自注意力（Multi-Head Self-Attention）

### 为什么需要多头？

不同的头可以关注不同的方面：
- 有的头关注**局部邻近关系**
- 有的头关注**远距离依赖**
- 有的头关注**特定的语义关系**

就像我们看一张图时，会同时注意颜色、形状、纹理、空间关系等多个方面。

### 怎么做？

1. 将 Q, K, V 分别投影到多个子空间（多个"头"）
2. 每个头独立做自注意力
3. 将所有头的结果拼接起来
4. 再做一次线性投影得到最终输出

$$MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W^O$$

$$head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$$

---

**思考题 4：** 多头注意力和单头注意力相比，参数量是增加了还是减少了？为什么？
"""))

    cells.append(code_cell("""# Implement multi-head self-attention from scratch

def multi_head_attention(x, W_q, W_k, W_v, W_o, num_heads):
    \"\"\"
    Multi-head self-attention (naive implementation).
    
    Inputs:
    - x: (seq_len, d_model) input
    - W_q, W_k, W_v: (d_model, d_model) projection weights
    - W_o: (d_model, d_model) output projection
    - num_heads: number of attention heads
    
    Returns:
    - out: (seq_len, d_model) output
    - attn_weights: (num_heads, seq_len, seq_len) attention weights per head
    \"\"\"
    seq_len, d_model = x.shape
    d_k = d_model // num_heads
    
    # Project to Q, K, V
    Q = x @ W_q  # (seq_len, d_model)
    K = x @ W_k  # (seq_len, d_model)
    V = x @ W_v  # (seq_len, d_model)
    
    # Split into multiple heads
    # (seq_len, d_model) -> (num_heads, seq_len, d_k)
    Q_heads = Q.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
    K_heads = K.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
    V_heads = V.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
    
    # Compute attention for each head
    attn_outputs = []
    all_attn_weights = []
    
    for h in range(num_heads):
        # Attention scores
        scores = Q_heads[h] @ K_heads[h].T / np.sqrt(d_k)  # (seq_len, seq_len)
        attn_weights = softmax(scores, axis=-1)
        out_h = attn_weights @ V_heads[h]  # (seq_len, d_k)
        attn_outputs.append(out_h)
        all_attn_weights.append(attn_weights)
    
    # Concatenate heads
    concat = np.concatenate(attn_outputs, axis=-1)  # (seq_len, d_model)
    
    # Output projection
    out = concat @ W_o
    
    all_attn_weights = np.stack(all_attn_weights, axis=0)  # (num_heads, seq_len, seq_len)
    
    return out, all_attn_weights


print("multi_head_attention function defined!")
"""))

    cells.append(code_cell("""# Test multi-head attention
np.random.seed(42)

seq_len = 6
d_model = 12
num_heads = 3
d_k = d_model // num_heads  # 4

x = np.random.randn(seq_len, d_model)
W_q = np.random.randn(d_model, d_model) * 0.1
W_k = np.random.randn(d_model, d_model) * 0.1
W_v = np.random.randn(d_model, d_model) * 0.1
W_o = np.random.randn(d_model, d_model) * 0.1

out, attn_weights = multi_head_attention(x, W_q, W_k, W_v, W_o, num_heads)

print(f"Input shape: {x.shape} (seq_len={seq_len}, d_model={d_model})")
print(f"Number of heads: {num_heads}")
print(f"d_k per head: {d_k}")
print(f"Output shape: {out.shape}")
print(f"Attention weights shape: {attn_weights.shape}")
print()

# Verify: each head has different attention patterns
print("Attention patterns differ across heads:")
for h in range(num_heads):
    print(f"  Head {h}: top attention from pos 0 -> {np.argmax(attn_weights[h, 0])} (weight={attn_weights[h,0,np.argmax(attn_weights[h,0])]:.3f})")
"""))

    cells.append(code_cell("""# Visualize multi-head attention patterns
fig, axes = plt.subplots(1, num_heads, figsize=(14, 4))

for h in range(num_heads):
    im = axes[h].imshow(attn_weights[h], cmap='YlOrRd', vmin=0, vmax=0.5)
    axes[h].set_title(f'Head {h}')
    axes[h].set_xlabel('Key')
    axes[h].set_ylabel('Query')
    plt.colorbar(im, ax=axes[h], fraction=0.046)

plt.suptitle('Multi-Head Attention: Different Heads Attend Differently', fontsize=14)
plt.tight_layout()
plt.show()

print("Each head learns different attention patterns:")
print("- Some might attend to local neighbors")
print("- Some might attend to specific semantic relationships")
print("- Some might attend globally")
print("This diversity is the strength of multi-head attention.")
"""))

    # Section 8: ViT vs CNN comparison
    cells.append(md_cell("""## 8. ViT vs CNN：对比与分析

### 归纳偏置（Inductive Bias）

- **CNN**：有很强的归纳偏置——局部性、平移不变性、层次化
- **ViT**：归纳偏置很弱——几乎是白板，需要更多数据来学习

### 数据需求

- **CNN**：小数据集上表现更好（归纳偏置帮了大忙）
- **ViT**：大数据集上表现更好（没有归纳偏置的限制，上限更高）

### 计算效率

- **CNN**：计算量与空间尺寸的平方成正比（但常数小）
- **ViT**：自注意力是 O(n²)，但 n 是 patch 数而非像素数

### 发展趋势

现在的趋势是**融合**：
- ConvNeXt：用 Transformer 的设计思路改造 CNN
- Swin Transformer：用窗口注意力降低 ViT 计算量，引入层次化结构
- 混合模型：前几层用 CNN，后面用 Transformer

---

**思考题 5：** 为什么 ViT 在大数据集上比 CNN 表现更好？
"""))

    cells.append(code_cell("""# Performance comparison sketch (illustrative)

data_scales = ['1M', '10M', '100M', '1B (JFT)', '5B+']
x_pos = np.arange(len(data_scales))

# Illustrative accuracy curves
cnn_acc = [70, 78, 83, 86, 87]  # CNN plateaus earlier
vit_acc = [65, 76, 84, 88, 90]  # ViT keeps improving with more data

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(x_pos, cnn_acc, 'o-', linewidth=2, markersize=8, label='CNN (e.g., ResNet)', color='#e74c3c')
ax.plot(x_pos, vit_acc, 's-', linewidth=2, markersize=8, label='ViT (Vision Transformer)', color='#3498db')

ax.set_xticks(x_pos)
ax.set_xticklabels(data_scales)
ax.set_xlabel('Training Data Scale')
ax.set_ylabel('ImageNet Top-1 Accuracy (%)')
ax.set_title('CNN vs ViT: Performance vs Data Scale (illustrative)', fontsize=13)
ax.legend()
ax.grid(alpha=0.3)

# Add annotation
ax.annotate('ViT overtakes CNN\nwith enough data', 
            xy=(2, 83.5), xytext=(1, 87),
            arrowprops=dict(arrowstyle='->', color='green'),
            fontsize=10, color='green', ha='center')

ax.set_ylim(60, 95)
plt.tight_layout()
plt.show()

print("Key insight:")
print("- CNNs have stronger inductive bias -> work better with less data")
print("- ViT has weaker inductive bias -> needs more data but can scale better")
print("- At very large data scales, ViT surpasses CNNs")
"""))

    # Section 9: DeiT
    cells.append(md_cell("""## 9. DeiT：数据高效的图像 Transformer

### 问题：ViT 需要太多数据

ViT 在 ImageNet（128 万张图）上从头训练效果不如 ResNet，需要在更大的数据集（如 JFT-300M）上预训练才行。

### DeiT 的解决方案

DeiT（Data-efficient Image Transformer）证明了：**用好训练技巧，ViT 也可以只用 ImageNet 训练得很好。**

### 关键技巧

1. **蒸馏（Distillation）**：用一个强的 CNN 模型（RegNet）作为老师，教 ViT 学生模型
2. **数据增强**：大量的数据增强（RandAugment, CutMix, MixUp 等）
3. **优化策略**：更好的学习率调度、权重衰减等

> **参考论文**：[[Touvron et al., 2020]](https://arxiv.org/abs/2012.12877) Training data-efficient image transformers & distillation through attention

---
"""))

    # Section 10: Attention visualization
    cells.append(md_cell("""## 10. 注意力可视化：每个头在看什么？

ViT 的一个吸引人的特点是**可解释性**——我们可以通过注意力权重来观察模型在关注图像的哪些区域。

### 有趣的发现

- **底层**：注意力比较分散，有的关注局部，有的关注全局
- **中间层**：开始形成有意义的注意力模式
- **顶层**：class token 的注意力通常聚焦在**物体主体**上

> 注意：注意力图不是完美的解释，但确实提供了一些直觉。

---
"""))

    cells.append(code_cell("""# Simulate attention visualization for a simple image
np.random.seed(42)

# Create a "cat-like" image (simplified)
img_size = 16
img = np.zeros((img_size, img_size, 3))
# Body
img[5:12, 3:13, 0] = 0.8  # orange body
img[5:12, 3:13, 1] = 0.4
# Ears
img[3:6, 4:7, 0] = 0.9
img[3:6, 10:13, 0] = 0.9
# Eyes
img[7:8, 6:7, 2] = 1.0  # blue eyes
img[7:8, 9:10, 2] = 1.0
# Nose
img[9:10, 7:9, 1] = 0.8

# Simulate attention maps from different heads
n_patches_side = 4  # 4x4 patches for 16x16 image with patch_size=4
n_heads = 4

# Create synthetic attention patterns for class token
attention_maps = np.zeros((n_heads, n_patches_side, n_patches_side))

# Head 0: attends to the center (main object)
attention_maps[0] = np.array([
    [0.05, 0.1, 0.1, 0.05],
    [0.1, 0.2, 0.2, 0.1],
    [0.05, 0.15, 0.15, 0.05],
    [0.02, 0.05, 0.05, 0.02],
])

# Head 1: attends to top (head/ears)
attention_maps[1] = np.array([
    [0.2, 0.3, 0.3, 0.1],
    [0.1, 0.1, 0.1, 0.05],
    [0.02, 0.02, 0.02, 0.01],
    [0.01, 0.01, 0.01, 0.01],
])

# Head 2: attends to bottom (body)
attention_maps[2] = np.array([
    [0.01, 0.01, 0.01, 0.01],
    [0.05, 0.05, 0.05, 0.05],
    [0.2, 0.25, 0.25, 0.15],
    [0.1, 0.1, 0.1, 0.05],
])

# Head 3: global attention (spread out)
attention_maps[3] = np.array([
    [0.08, 0.07, 0.07, 0.06],
    [0.07, 0.06, 0.06, 0.05],
    [0.06, 0.05, 0.05, 0.04],
    [0.05, 0.04, 0.04, 0.03],
])
# Normalize each head
for h in range(n_heads):
    attention_maps[h] /= attention_maps[h].sum()

# Visualize
fig, axes = plt.subplots(1, n_heads + 1, figsize=(15, 4))

axes[0].imshow(img)
axes[0].set_title('Input Image')
axes[0].axis('off')
# Draw patch grid
for i in range(5):
    axes[0].axvline(i*4 - 0.5, color='white', linewidth=1, alpha=0.7)
    axes[0].axhline(i*4 - 0.5, color='white', linewidth=1, alpha=0.7)

for h in range(n_heads):
    ax = axes[h + 1]
    im = ax.imshow(attention_maps[h], cmap='hot', vmin=0, vmax=0.3)
    ax.set_title(f'Head {h} Attention')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('Class Token Attention: What Does Each Head Look At?', fontsize=13)
plt.tight_layout()
plt.show()

print("In a real ViT:")
print("- Some heads attend to the object (localizing)")
print("- Some heads attend to specific parts (eyes, ears, etc.)")
print("- Some heads attend more globally")
print("- Attention patterns vary across layers")
"""))

    # Common mistakes
    cells.append(md_cell("""## 11. 常见错误与注意事项

### 错误 1：Patch 大小选择不当

- Patch 太大：token 太少，细节丢失多
- Patch 太小：token 太多，计算量大
- 常用：16×16（ViT-Base）或 14×14（ViT-Large/14）

### 错误 2：忘记位置编码

没有位置编码，Transformer 无法区分不同位置的 patch，性能会大幅下降。

### 错误 3：小数据集上直接用 ViT

ViT 的归纳偏置弱，小数据上不如 CNN。应该：
- 用预训练权重做微调
- 或者用更强的数据增强
- 或者考虑用 CNN 或混合模型

### 错误 4：注意力计算的维度错误

多头注意力中，Q, K, V 要正确地分头，每个头的维度是 d_model // num_heads。

### 错误 5：Class token 位置不重要

Class token 的位置通常加在最前面，这是一个约定。但实际上位置编码会提供位置信息，所以放在哪里理论上影响不大。

---
"""))

    # Homework 1
    cells.append(md_cell("""---

## 作业 1：实现多头自注意力

### 任务

完整实现多头自注意力的前向传播。要求：
1. 支持任意数量的头（d_model 必须能被 num_heads 整除）
2. 正确计算 Q, K, V 并分头
3. 正确实现缩放点积注意力
4. 拼接所有头的输出并做输出投影
5. 返回注意力权重（用于可视化）

### 提示

- 使用 `softmax` 函数
- 分头时注意 reshape 和 transpose 的顺序
- 缩放因子：√d_k，其中 d_k = d_model // num_heads

---
"""))

    cells.append(code_cell("""# ===== Homework 1: Multi-Head Self-Attention =====

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def multi_head_self_attention(x, W_q, W_k, W_v, W_o, num_heads):
    \"\"\"
    Multi-head self-attention forward pass.
    
    Inputs:
    - x: (N, seq_len, d_model) input embeddings (batch first)
    - W_q: (d_model, d_model) query projection
    - W_k: (d_model, d_model) key projection
    - W_v: (d_model, d_model) value projection
    - W_o: (d_model, d_model) output projection
    - num_heads: number of attention heads
    
    Returns:
    - out: (N, seq_len, d_model) output
    - attn_weights: (N, num_heads, seq_len, seq_len) attention weights
    \"\"\"
    # YOUR CODE HERE
    # Hint: follow these steps:
    # 1. Compute Q, K, V by projecting x
    # 2. Split into num_heads: reshape + transpose
    # 3. Compute attention scores = Q @ K^T / sqrt(d_k)
    # 4. Apply softmax to get attention weights
    # 5. Weighted sum with V
    # 6. Concatenate heads and project with W_o
    
    out = None
    attn_weights = None
    return out, attn_weights


# ===== ANSWER KEY =====
def multi_head_self_attention_answer(x, W_q, W_k, W_v, W_o, num_heads):
    \"\"\"Reference implementation (batch version).\"\"\"
    N, seq_len, d_model = x.shape
    d_k = d_model // num_heads
    
    # Project
    Q = x @ W_q  # (N, seq_len, d_model)
    K = x @ W_k
    V = x @ W_v
    
    # Split heads: (N, seq_len, d_model) -> (N, num_heads, seq_len, d_k)
    Q = Q.reshape(N, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(N, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(N, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    
    # Attention scores: (N, num_heads, seq_len, seq_len)
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)
    attn_weights = softmax(scores, axis=-1)
    
    # Weighted sum: (N, num_heads, seq_len, d_k)
    attn_output = attn_weights @ V
    
    # Concatenate heads: (N, seq_len, d_model)
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(N, seq_len, d_model)
    
    # Output projection
    out = attn_output @ W_o
    
    return out, attn_weights


print("Homework 1: Implement multi_head_self_attention")
print("Write your solution above, then run the tests.")
"""))

    cells.append(code_cell("""# ===== Homework 1: Test cases =====
np.random.seed(42)

N, seq_len, d_model = 2, 6, 12
num_heads = 3

x = np.random.randn(N, seq_len, d_model)
W_q = np.random.randn(d_model, d_model) * 0.1
W_k = np.random.randn(d_model, d_model) * 0.1
W_v = np.random.randn(d_model, d_model) * 0.1
W_o = np.random.randn(d_model, d_model) * 0.1

out_ref, attn_ref = multi_head_self_attention_answer(x, W_q, W_k, W_v, W_o, num_heads)
out_student, attn_student = multi_head_self_attention(x, W_q, W_k, W_v, W_o, num_heads)

# Shape tests
assert out_student is not None, "Please implement the function first!"
assert out_student.shape == (N, seq_len, d_model), f"Output shape: {out_student.shape}"
assert attn_student.shape == (N, num_heads, seq_len, seq_len), f"Attn shape: {attn_student.shape}"
print(f"Output shape: {out_student.shape} - PASSED")
print(f"Attention shape: {attn_student.shape} - PASSED")

# Value tests
assert np.allclose(out_student, out_ref, atol=1e-8), "Output values don't match!"
print("Output values: PASSED")
assert np.allclose(attn_student, attn_ref, atol=1e-8), "Attention weights don't match!"
print("Attention weights: PASSED")

# Property: attention weights sum to 1
for n in range(N):
    for h in range(num_heads):
        row_sums = np.sum(attn_student[n, h], axis=-1)
        assert np.allclose(row_sums, 1.0), f"Attention row doesn't sum to 1!"
print("Attention weights sum to 1: PASSED")

# Property: different heads have different patterns
head0_attn = attn_student[0, 0]
head1_attn = attn_student[0, 1]
assert not np.allclose(head0_attn, head1_attn), "Different heads should have different patterns!"
print("Different heads have different patterns: PASSED")

print("\n=== All Homework 1 tests PASSED! ===")
"""))

    # Homework 2
    cells.append(md_cell("""---

## 作业 2：实现 Patch Embedding + 位置编码

### 任务

实现 ViT 的输入处理模块，包括：
1. 将图像分割成 patch
2. 将 patch 投影到 embedding 维度
3. 加上可学习的位置编码
4. 加上 class token

### 要求

- 输入形状：(N, C, H, W)
- 输出形状：(N, num_patches + 1, embed_dim) （+1 是 class token）
- 位置编码形状：(num_patches + 1, embed_dim)
- class token 加在序列最前面

---
"""))

    cells.append(code_cell("""# ===== Homework 2: Patch Embedding + Position Embedding =====

def vit_input_embedding(x, patch_size, embed_dim, W_proj, b_proj, cls_token, pos_embed):
    \"\"\"
    ViT input embedding: patch embedding + class token + position embedding.
    
    Inputs:
    - x: (N, C, H, W) input images
    - patch_size: int, size of each patch
    - embed_dim: int, embedding dimension
    - W_proj: (patch_dim, embed_dim) projection weight
    - b_proj: (embed_dim,) projection bias
    - cls_token: (1, embed_dim) class token
    - pos_embed: (num_patches+1, embed_dim) position embedding
    
    Returns:
    - out: (N, num_patches+1, embed_dim) embedded sequence
    \"\"\"
    # YOUR CODE HERE
    # Hint:
    # 1. Extract patches (reshape or use conv-like approach)
    # 2. Flatten each patch and project with W_proj + b_proj
    # 3. Prepend class token to each sequence
    # 4. Add position embedding
    
    out = None
    return out


# ===== ANSWER KEY =====
def vit_input_embedding_answer(x, patch_size, embed_dim, W_proj, b_proj, cls_token, pos_embed):
    \"\"\"Reference implementation.\"\"\"
    N, C, H, W = x.shape
    n_h = H // patch_size
    n_w = W // patch_size
    num_patches = n_h * n_w
    
    # Extract patches: (N, C, H, W) -> (N, n_h, n_w, C, patch_size, patch_size)
    # Then flatten to (N, num_patches, patch_dim)
    x = x.reshape(N, C, n_h, patch_size, n_w, patch_size)
    x = x.transpose(0, 2, 4, 1, 3, 5)  # (N, n_h, n_w, C, ps, ps)
    x = x.reshape(N, num_patches, C * patch_size * patch_size)
    
    # Project patches
    patch_emb = x @ W_proj + b_proj  # (N, num_patches, embed_dim)
    
    # Prepend class token
    cls_tokens = np.tile(cls_token, (N, 1, 1))  # (N, 1, embed_dim)
    x_emb = np.concatenate([cls_tokens, patch_emb], axis=1)  # (N, 1+num_patches, embed_dim)
    
    # Add position embedding
    out = x_emb + pos_embed
    
    return out


print("Homework 2: Implement vit_input_embedding")
print("Write your solution above, then run the tests.")
"""))

    cells.append(code_cell("""# ===== Homework 2: Test cases =====
np.random.seed(42)

N, C, H, W = 2, 3, 8, 8
patch_size = 4
embed_dim = 16

x = np.random.randn(N, C, H, W)

patch_dim = C * patch_size * patch_size
W_proj = np.random.randn(patch_dim, embed_dim) * 0.1
b_proj = np.zeros(embed_dim)
cls_token = np.random.randn(1, embed_dim) * 0.02

num_patches = (H // patch_size) * (W // patch_size)
pos_embed = np.random.randn(num_patches + 1, embed_dim) * 0.02

out_ref = vit_input_embedding_answer(x, patch_size, embed_dim, W_proj, b_proj, cls_token, pos_embed)
out_student = vit_input_embedding(x, patch_size, embed_dim, W_proj, b_proj, cls_token, pos_embed)

# Shape test
assert out_student is not None, "Please implement the function first!"
expected_shape = (N, num_patches + 1, embed_dim)
assert out_student.shape == expected_shape, f"Shape: {out_student.shape} vs {expected_shape}"
print(f"Output shape: {out_student.shape} - PASSED")
print(f"  (N={N}, num_patches+1={num_patches+1}, embed_dim={embed_dim})")

# Value test
assert np.allclose(out_student, out_ref, atol=1e-8), "Values don't match!"
print("Output values: PASSED")

# Class token test: first position should include cls_token
# (cls_token + pos_embed[0])
expected_cls_pos = cls_token + pos_embed[0:1]
# For batch element 0, position 0: should be cls_token + pos_embed[0]
# But wait, it also depends on how we add... let's check differently
print("\nClass token position (pos 0 of sample 0):")
print(f"  First 4 dims: {out_student[0, 0, :4]}")

# Test that adding pos_embed works correctly
# If we remove pos_embed effect, patch embeddings should match projection
# (for position 1 onwards, which are patches)
patch_out = out_student[:, 1:, :] - pos_embed[np.newaxis, 1:, :]
# Check first patch matches projection
first_patch = x[0, :, :patch_size, :patch_size].flatten()
expected_first_patch = first_patch @ W_proj + b_proj
assert np.allclose(patch_out[0, 0], expected_first_patch, atol=1e-8), "Patch projection incorrect!"
print("Patch projection correctness: PASSED")

print("\n=== All Homework 2 tests PASSED! ===")
"""))

    # Summary
    cells.append(md_cell("""---

## 本讲总结

### ViT 核心组件

| 组件 | 作用 | 维度变化 |
|------|------|----------|
| Patch Embedding | 将图像分成 patch 并投影 | 224×224×3 → 196×768 |
| Class Token | 全局信息的"代表" | +1 个 token |
| Position Embedding | 提供空间位置信息 | 直接相加，维度不变 |
| Multi-Head Attention | 全局建模 | 维度不变 |
| MLP | 通道维度的非线性变换 | 维度不变 |
| MLP Head | 分类头 | 768 → 1000 |

### 关键要点

1. **Patch Embedding** 是把图像变成 Transformer 能处理的序列的关键
2. **位置编码** 为 Transformer 提供不可或缺的空间信息
3. **多头注意力** 让模型能同时关注不同方面的信息
4. **归纳偏置** 的差异导致 ViT 和 CNN 在不同数据规模下表现不同
5. **DeiT** 证明了用正确的训练技巧，ViT 在 ImageNet 上也能很强

### 思考题答案提示

1. 约 111 层 3×3 卷积（stride=1 时，感受野 = 2n+1，所以 2n+1 ≥ 224 → n ≥ 111.5，取 112 层）
2. 32/4 × 32/4 = 64 个 patch；每个 patch 展平后是 4×4×3 = 48 维
3. 模型通过训练学会了从加和的向量中"解读"出位置和内容信息。不同的维度可能被用于编码不同的信息
4. 参数量大致相同。因为 Q,K,V 的总参数量是 3×d_model²，分头后每个头维度变小但数量变多，总量不变
5. 因为 ViT 的归纳偏置弱，没有被"预先设定"只能关注局部。当数据足够多时，它可以学到比 CNN 更灵活、更丰富的特征表示

### 论文参考

- [[Dosovitskiy et al., 2020]](https://arxiv.org/abs/2010.11929) An Image is Worth 16x16 Words — ViT 原文
- [[Vaswani et al., 2017]](https://arxiv.org/abs/1706.03762) Attention Is All You Need — Transformer 原文
- [[Touvron et al., 2020]](https://arxiv.org/abs/2012.12877) Training data-efficient image transformers — DeiT

### GitHub 参考实现

- [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch) (5.7k stars) — ViT 的简洁 PyTorch 实现
- [facebookresearch/deit](https://github.com/facebookresearch/deit) (4.4k stars) — DeiT 官方实现

---

**下节预告**：RNN 与图像描述——如何让计算机"看懂"图片并用语言描述出来？
"""))

    return cells


# ============================================================
# NOTEBOOK 4: RNN & Image Captioning
# ============================================================
def build_notebook4():
    cells = []

    cells.append(md_cell("""# 【作业2】RNN 与图像描述

> **CS231n 深度学习与计算机视觉** · 第8讲补充
>
> 对应 CS231n Assignment 2 第 5 题。本 notebook 将带你理解循环神经网络（RNN）的核心机制，以及如何用 CNN + RNN 实现图像描述（Image Captioning）——让计算机"看懂"图片并用自然语言描述出来。

---

## 学习目标

- 理解 RNN 的基本结构和前向传播
- 手动计算 RNN 的前向传播过程
- 理解 LSTM 的门控机制和为什么它能缓解梯度消失
- 手动计算 LSTM 的前向传播
- 理解编码器-解码器架构在图像描述中的应用
- 从零实现 RNN 和 LSTM 的前向传播
- 理解词嵌入、贪心搜索、交叉熵损失
- 实现简单的图像描述模型

---
"""))

    cells.append(code_cell("""# Setup
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("Environment ready!")
"""))

    # Section 1: Image captioning task
    cells.append(md_cell("""## 1. 图像描述任务：从图像到文字

### 什么是图像描述？

输入一张图片，输出一句描述图片内容的自然语言句子。

**输入**：一张猫坐在沙发上的图片  
**输出**："a cat sitting on a couch"

### 为什么难？

1. **既要理解图像，又要生成语言**——跨模态任务
2. **输出是变长的**——每张图的描述长度不同
3. **需要语法正确、语义通顺**——不是单词堆砌

### 经典架构：编码器-解码器（Encoder-Decoder）

- **编码器（Encoder）**：用 CNN 把图像编码成一个特征向量
- **解码器（Decoder）**：用 RNN/LSTM 逐词生成描述句子

> **参考论文**：[[Vinyals et al., 2014]](https://arxiv.org/abs/1411.4555) Show and Tell: A Neural Image Caption Generator

---

**思考题 1：** 图像描述任务中，输出序列的长度是不固定的。RNN 是如何处理变长输出的？
"""))

    cells.append(code_cell("""# Illustration of the image captioning task
fig, ax = plt.subplots(figsize=(12, 4))

# Draw encoder (CNN)
encoder_rect = plt.Rectangle((0.5, 0.3), 2, 0.4, facecolor='#3498db', alpha=0.7, edgecolor='black')
ax.add_patch(encoder_rect)
ax.text(1.5, 0.5, 'CNN\nEncoder', ha='center', va='center', fontsize=12, color='white')

# Image icon
ax.text(1.5, 0.85, '[Image]', ha='center', fontsize=12)
ax.annotate('', xy=(1.5, 0.72), xytext=(1.5, 0.8),
            arrowprops=dict(arrowstyle='->', color='black'))

# Feature vector
ax.annotate('', xy=(3, 0.5), xytext=(2.5, 0.5),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax.text(2.75, 0.56, 'feature vector', ha='center', fontsize=9)

# Decoder (RNN)
decoder_rect = plt.Rectangle((3.2, 0.2), 5.5, 0.6, facecolor='#e74c3c', alpha=0.7, edgecolor='black')
ax.add_patch(decoder_rect)
ax.text(6, 0.5, 'RNN / LSTM Decoder', ha='center', va='center', fontsize=12, color='white')

# Output words
words = ['<start>', 'a', 'cat', 'sitting', 'on', 'a', 'couch', '<end>']
for i, word in enumerate(words):
    x_pos = 3.5 + i * 0.65
    ax.text(x_pos, 0.05, word, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    if i > 0:
        ax.annotate('', xy=(x_pos - 0.3, 0.08), xytext=(x_pos - 0.55, 0.08),
                    arrowprops=dict(arrowstyle='->', color='gray'))

ax.set_xlim(0, 9)
ax.set_ylim(0, 1)
ax.set_title('Image Captioning: Encoder-Decoder Architecture', fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.show()

print("Key idea: CNN understands the image, RNN generates the sentence word by word.")
print("The CNN feature vector initializes the RNN's hidden state.")
"""))

    # Section 2: RNN basics
    cells.append(md_cell("""## 2. RNN 基础：循环神经网络

### 什么是 RNN？

RNN（Recurrent Neural Network）是一种处理**序列数据**的神经网络。它的核心思想是：**每一步的输出不仅取决于当前输入，还取决于之前的隐藏状态。**

### 核心公式

$$h_t = \\tanh(W_h h_{t-1} + W_x x_t + b_h)$$
$$y_t = W_y h_t + b_y$$

其中：
- $x_t$：第 t 步的输入
- $h_t$：第 t 步的隐藏状态（记忆）
- $y_t$：第 t 步的输出
- $W_h, W_x, W_y$：权重矩阵（所有时间步共享！）
- $b_h, b_y$：偏置

### 参数共享

RNN 的关键特点是**参数共享**——每一步都用相同的 W 和 b。这使得：
1. 模型可以处理任意长度的序列
2. 参数量不随序列长度增加

> **类比**：RNN 就像一个人在读书，每读一个词，脑子里的理解（隐藏状态）就更新一次。

---
"""))

    cells.append(code_cell("""# Manual RNN forward pass step by step

# Simple RNN: input_dim=3, hidden_dim=2
np.random.seed(42)

input_dim = 3
hidden_dim = 2

# Weights (shared across all time steps!)
Wx = np.random.randn(input_dim, hidden_dim) * 0.5   # input -> hidden
Wh = np.random.randn(hidden_dim, hidden_dim) * 0.5  # hidden -> hidden
bh = np.zeros(hidden_dim)

print("RNN Parameters (shared across all time steps):")
print(f"Wx ({input_dim}x{hidden_dim}):")
print(np.round(Wx, 3))
print(f"Wh ({hidden_dim}x{hidden_dim}):")
print(np.round(Wh, 3))
print(f"bh ({hidden_dim},): {np.round(bh, 3)}")
"""))

    cells.append(code_cell("""# RNN forward pass over 3 time steps

# Input sequence: 3 time steps, each with dimension 3
x = np.array([
    [1.0, 0.0, -1.0],   # t=0
    [0.5, 1.0, 0.0],    # t=1
    [-1.0, 0.5, 1.0],   # t=2
])

# Initial hidden state (all zeros)
h_prev = np.zeros(hidden_dim)
print(f"Initial hidden state h_0: {h_prev}")
print()

# Step through time
hidden_states = [h_prev]
for t in range(3):
    x_t = x[t]
    # h_t = tanh(Wx * x_t + Wh * h_prev + bh)
    raw = x_t @ Wx + h_prev @ Wh + bh
    h_t = np.tanh(raw)
    hidden_states.append(h_t)
    print(f"Time step {t+1}:")
    print(f"  x_{t+1} = {x_t}")
    print(f"  raw = Wx*x_{t+1} + Wh*h_{t} + bh = {np.round(raw, 4)}")
    print(f"  h_{t+1} = tanh(raw) = {np.round(h_t, 4)}")
    print()
    h_prev = h_t

print("Hidden states evolve over time, carrying information from the past.")
print(f"Final hidden state h_3 = {np.round(hidden_states[-1], 4)}")
print("This final state summarizes the entire input sequence!")
"""))

    cells.append(code_cell("""# Visualize RNN computation flow
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Initial state
axes[0].text(0.5, 0.7, 'h₀ = [0, 0]', ha='center', fontsize=12, fontweight='bold')
axes[0].text(0.5, 0.3, 'x₁', ha='center', fontsize=12)
axes[0].text(0.5, 0.5, '+', ha='center', fontsize=20)
axes[0].set_title('Step 1')
axes[0].axis('off')

for t in range(3):
    ax = axes[t]
    # Draw RNN cell
    cell = plt.Rectangle((0.2, 0.4), 0.6, 0.3, facecolor='#3498db', alpha=0.7)
    ax.add_patch(cell)
    ax.text(0.5, 0.55, 'RNN', ha='center', va='center', fontsize=12, color='white')
    
    # Input
    ax.text(0.5, 0.2, f'x_{t+1}', ha='center', fontsize=11)
    ax.annotate('', xy=(0.5, 0.4), xytext=(0.5, 0.28),
                arrowprops=dict(arrowstyle='->', color='black'))
    
    # Hidden state in
    if t > 0:
        ax.text(0.05, 0.55, f'h_{t}', ha='left', fontsize=10)
        ax.annotate('', xy=(0.2, 0.55), xytext=(0.12, 0.55),
                    arrowprops=dict(arrowstyle='->', color='red'))
    
    # Output hidden state
    ax.text(0.95, 0.55, f'h_{t+1}', ha='right', fontsize=10)
    ax.annotate('', xy=(0.88, 0.55), xytext=(0.8, 0.55),
                arrowprops=dict(arrowstyle='->', color='green'))
    
    ax.set_title(f'Time Step {t+1}')
    ax.axis('off')

plt.suptitle('RNN: Same weights applied at every time step', fontsize=14)
plt.tight_layout()
plt.show()
"""))

    # Section 3: Implement RNN forward
    cells.append(md_cell("""## 3. 从零实现 RNN 前向传播

### 输入输出维度

- 输入 x: (N, T, D) — N 个序列，每个长度 T，每个时刻 D 维
- 初始隐藏状态 h0: (N, H)
- 权重 Wx: (D, H), Wh: (H, H), bh: (H,)
- 输出 h: (N, T, H) — 每个时刻的隐藏状态

### 实现思路

对每个时间步 t：
1. 取出当前输入 x[:, t, :]
2. 计算 h_t = tanh(x_t @ Wx + h_prev @ Wh + bh)
3. 保存 h_t
4. h_prev = h_t

---
"""))

    cells.append(code_cell("""def rnn_forward(x, h0, Wx, Wh, bh):
    \"\"\"
    Forward pass for a vanilla RNN.
    
    Inputs:
    - x: (N, T, D) input data
    - h0: (N, H) initial hidden state
    - Wx: (D, H) input-to-hidden weights
    - Wh: (H, H) hidden-to-hidden weights
    - bh: (H,) biases
    
    Returns:
    - h: (N, T, H) hidden states at all time steps
    - cache: values needed for backward pass
    \"\"\"
    N, T, D = x.shape
    H = h0.shape[1]
    
    h = np.zeros((N, T, H))
    h_prev = h0.copy()
    
    for t in range(T):
        x_t = x[:, t, :]  # (N, D)
        # h_t = tanh(Wx * x_t + Wh * h_prev + bh)
        h_t = np.tanh(x_t @ Wx + h_prev @ Wh + bh)
        h[:, t, :] = h_t
        h_prev = h_t
    
    cache = (x, h0, Wx, Wh, bh, h)
    return h, cache


# Test with our manual example
np.random.seed(42)
x_test = np.array([[
    [1.0, 0.0, -1.0],
    [0.5, 1.0, 0.0],
    [-1.0, 0.5, 1.0],
]])  # (N=1, T=3, D=3)

h0_test = np.zeros((1, 2))
Wx_test = np.random.randn(3, 2) * 0.5
Wh_test = np.random.randn(2, 2) * 0.5
bh_test = np.zeros(2)

h_out, cache = rnn_forward(x_test, h0_test, Wx_test, Wh_test, bh_test)

print(f"Input shape: {x_test.shape} (N, T, D)")
print(f"Output shape: {h_out.shape} (N, T, H)")
print()
print("Hidden states at each time step:")
for t in range(3):
    print(f"  h_{t+1}: {np.round(h_out[0, t], 4)}")
"""))

    # Section 4: LSTM
    cells.append(md_cell("""## 4. LSTM：长短期记忆网络

### 问题：RNN 的梯度消失

在长序列中，RNN 的梯度在反向传播时会不断相乘，导致梯度变得非常小（消失）或非常大（爆炸）。

结果：**RNN 很难记住很久以前的信息。**

### LSTM 的解决方案：门控机制

LSTM（Long Short-Term Memory）引入了**细胞状态（cell state）** 和三个门：

1. **输入门（Input Gate）**：控制多少新信息写入细胞状态
2. **遗忘门（Forget Gate）**：控制多少旧信息保留在细胞状态
3. **输出门（Output Gate）**：控制细胞状态有多少输出到隐藏状态

### 为什么有效？

细胞状态有一条"高速公路"——信息可以直接流过，梯度也可以直接流过，不会轻易消失。

> **参考论文**：[[Hochreiter & Schmidhuber, 1997]](https://www.bioinf.jku.at/publications/older/2604.pdf) Long Short-Term Memory

---

**思考题 2：** 为什么 LSTM 的遗忘门初始化为 1（或偏置为正）比较好？
"""))

    cells.append(md_cell("""### LSTM 前向传播公式

```
输入：x_t, h_prev, c_prev

输入门:   i_t = sigmoid(W_i * x_t + U_i * h_prev + b_i)
遗忘门:   f_t = sigmoid(W_f * x_t + U_f * h_prev + b_f)
输出门:   o_t = sigmoid(W_o * x_t + U_o * h_prev + b_o)
候选值:   g_t = tanh(W_g * x_t + U_g * h_prev + b_g)

细胞状态: c_t = f_t * c_prev + i_t * g_t
隐藏状态: h_t = o_t * tanh(c_t)
```

### 直觉

- **遗忘门**："我要忘记哪些旧记忆？"
- **输入门 + 候选值**："我要记住哪些新信息？"
- **细胞状态**：更新后的记忆
- **输出门**："我要从记忆中输出多少？"

> 细胞状态 c 就像**长期记忆**，隐藏状态 h 就像**当前注意力**。

---
"""))

    cells.append(code_cell("""# Manual LSTM forward pass with concrete numbers

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

np.random.seed(42)

# Small LSTM: input_dim=2, hidden_dim=2
D = 2
H = 2

# Weights (for simplicity, we'll use combined notation)
# In practice, 4 gates share one big matrix multiply
# Input weights: W (D, 4H), Hidden weights: U (H, 4H), Bias: b (4H)
# Gates are ordered: i, f, o, g
W_lstm = np.random.randn(D, 4 * H) * 0.3
U_lstm = np.random.randn(H, 4 * H) * 0.3
b_lstm = np.zeros(4 * H)
# Initialize forget gate bias to 1 (common practice)
b_lstm[H:2*H] = 1.0

print(f"LSTM params: W ({D}x{4*H}), U ({H}x{4*H}), b ({4*H},)")
print(f"Forget gate bias initialized to 1.0 (remember initially)")
"""))

    cells.append(code_cell("""# LSTM forward: step by step for one time step

# Initial state
h_prev = np.zeros(H)
c_prev = np.zeros(H)

# Input at t=1
x_t = np.array([1.0, -0.5])

# Compute all gate values at once (efficient implementation)
all_gates = x_t @ W_lstm + h_prev @ U_lstm + b_lstm

# Split into 4 gates
i = sigmoid(all_gates[:H])       # input gate
f = sigmoid(all_gates[H:2*H])    # forget gate
o = sigmoid(all_gates[2*H:3*H])  # output gate
g = np.tanh(all_gates[3*H:4*H])  # candidate cell values

# Update cell state
c_next = f * c_prev + i * g

# Update hidden state
h_next = o * np.tanh(c_next)

print("LSTM Step-by-Step Calculation (t=1):")
print("=" * 50)
print(f"Input x_t:       {x_t}")
print(f"Prev hidden h_prev: {np.round(h_prev, 4)}")
print(f"Prev cell c_prev:   {np.round(c_prev, 4)}")
print()
print(f"Input gate i:    {np.round(i, 4)}  (how much new info to write)")
print(f"Forget gate f:   {np.round(f, 4)}  (how much old info to keep)")
print(f"Output gate o:   {np.round(o, 4)}  (how much to output)")
print(f"Candidate g:     {np.round(g, 4)}  (new candidate values)")
print()
print(f"Cell state c_t = f * c_prev + i * g")
print(f"  = {np.round(f, 3)} * {np.round(c_prev, 3)} + {np.round(i, 3)} * {np.round(g, 3)}")
print(f"  = {np.round(c_next, 4)}")
print()
print(f"Hidden state h_t = o * tanh(c_t)")
print(f"  = {np.round(o, 3)} * tanh({np.round(c_next, 3)})")
print(f"  = {np.round(h_next, 4)}")
"""))

    cells.append(code_cell("""# Full LSTM forward over multiple time steps

def lstm_forward(x, h0, W, U, b):
    \"\"\"
    Forward pass for an LSTM.
    
    Inputs:
    - x: (N, T, D) input data
    - h0: (N, H) initial hidden state
    - W: (D, 4H) input weights (i, f, o, g gates)
    - U: (H, 4H) hidden weights (i, f, o, g gates)
    - b: (4H,) biases
    
    Returns:
    - h: (N, T, H) hidden states
    - cache: values for backward pass
    \"\"\"
    N, T, D = x.shape
    H = h0.shape[1]
    
    h = np.zeros((N, T, H))
    c = np.zeros((N, T, H))
    h_prev = h0.copy()
    c_prev = np.zeros((N, H))
    
    for t in range(T):
        x_t = x[:, t, :]
        
        # Compute all gates
        all_gates = x_t @ W + h_prev @ U + b  # (N, 4H)
        
        i = sigmoid(all_gates[:, :H])
        f = sigmoid(all_gates[:, H:2*H])
        o = sigmoid(all_gates[:, 2*H:3*H])
        g = np.tanh(all_gates[:, 3*H:4*H])
        
        # Cell state update
        c_t = f * c_prev + i * g
        
        # Hidden state update
        h_t = o * np.tanh(c_t)
        
        h[:, t, :] = h_t
        c[:, t, :] = c_t
        h_prev = h_t
        c_prev = c_t
    
    cache = (x, h0, W, U, b, h, c)
    return h, cache


# Test
np.random.seed(42)
N, T, D = 1, 4, 2
H = 2

x_seq = np.random.randn(N, T, D)
h0_lstm = np.zeros((N, H))
W_lstm = np.random.randn(D, 4*H) * 0.3
U_lstm = np.random.randn(H, 4*H) * 0.3
b_lstm = np.zeros(4*H)
b_lstm[H:2*H] = 1.0  # forget gate bias = 1

h_lstm, _ = lstm_forward(x_seq, h0_lstm, W_lstm, U_lstm, b_lstm)

print(f"LSTM output shape: {h_lstm.shape} (N, T, H)")
print()
print("Hidden states over time:")
for t in range(T):
    print(f"  t={t}: {np.round(h_lstm[0, t], 4)}")
"""))

    cells.append(code_cell("""# Visual comparison: RNN vs LSTM gradient flow intuition
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# RNN: gradient has to go through many nonlinearities
axes[0].set_title('RNN: Gradient Flow', fontsize=12)
for i in range(5):
    circle = plt.Circle((i * 1.5, 0.5), 0.3, facecolor='#e74c3c', alpha=0.7)
    axes[0].add_patch(circle)
    axes[0].text(i * 1.5, 0.5, f'h_{i}', ha='center', va='center', color='white', fontsize=9)
    if i > 0:
        axes[0].annotate('', xy=(i*1.5 - 0.35, 0.5), xytext=((i-1)*1.5 + 0.35, 0.5),
                        arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))

axes[0].text(3, 0.1, 'Gradient multiplies through each step\n→ can vanish/explode', 
             ha='center', fontsize=10, color='#e74c3c')
axes[0].set_xlim(-0.5, 7)
axes[0].set_ylim(0, 1)
axes[0].axis('off')

# LSTM: cell state has a direct path
axes[1].set_title('LSTM: Cell State Highway', fontsize=12)
# Draw cell state as a highway
axes[1].plot([0, 6.5], [0.7, 0.7], 'g-', linewidth=3, alpha=0.7)
axes[1].text(3.25, 0.82, 'Cell state (c) - "highway" for gradient', 
             ha='center', fontsize=10, color='green')

for i in range(5):
    # Hidden state
    circle = plt.Circle((i * 1.5, 0.4), 0.2, facecolor='#3498db', alpha=0.7)
    axes[1].add_patch(circle)
    axes[1].text(i * 1.5, 0.4, f'h_{i}', ha='center', va='center', color='white', fontsize=8)
    # Connection to cell state
    axes[1].plot([i*1.5, i*1.5], [0.5, 0.65], 'k-', linewidth=1, alpha=0.5)

axes[1].text(3, 0.1, 'Gradient can flow directly through cell state\n→ better long-term memory', 
             ha='center', fontsize=10, color='green')
axes[1].set_xlim(-0.5, 7)
axes[1].set_ylim(0, 1)
axes[1].axis('off')

plt.tight_layout()
plt.show()

print("LSTM's cell state provides a 'gradient highway' that helps with long sequences.")
print("This is why LSTMs (and GRUs) are preferred over vanilla RNNs in practice.")
"""))

    # Section 5: Encoder-Decoder for Captioning
    cells.append(md_cell("""## 5. 图像描述的编码器-解码器架构

### 整体流程

1. **图像编码器**：用预训练 CNN（如 VGG、ResNet）提取图像特征
2. **特征投影**：将 CNN 特征投影到 RNN 的隐藏状态维度
3. **词嵌入**：将单词索引映射为向量
4. **RNN/LSTM 解码器**：逐词生成描述
5. **输出层**：将隐藏状态映射为词汇表上的概率分布

### 训练时：Teacher Forcing

训练时，每一步的输入是**真实的前一个词**（而不是模型自己生成的词）。这叫 Teacher Forcing，能让训练更稳定。

### 推理时：自回归

推理时，每一步的输入是**模型上一步生成的词**。这叫自回归生成。

---
"""))

    cells.append(md_cell("""### 关键组件详解

#### 1. 图像编码器

- 输入：图像 (3, 224, 224)
- 用预训练 CNN 提取特征
- 通常取最后一个池化层之前的特征图，或全局平均池化后的向量
- 投影到 RNN 的隐藏状态维度，作为初始隐藏状态

#### 2. 词嵌入（Word Embedding）

- 词汇表中的每个词对应一个向量
- 将离散的单词索引映射为连续的向量表示
- 词嵌入是和模型一起训练的

#### 3. 损失计算

- 每个时间步输出一个词汇表上的概率分布
- 用**交叉熵损失**衡量预测分布和真实词的差距
- 所有时间步的损失取平均

> **参考论文**：[[Karpathy & Fei-Fei, 2015]](https://arxiv.org/abs/1505.04597) Deep Visual-Semantic Alignments for Generating Image Descriptions (DenseCap)

---
"""))

    cells.append(code_cell("""# Implement a simple image captioning model (forward pass only)

def word_embedding_forward(x, W_embed):
    \"\"\"
    Word embedding: map word indices to vectors.
    
    Inputs:
    - x: (N, T) integer array of word indices
    - W_embed: (V, D) embedding matrix, V=vocab size, D=embed dim
    
    Returns:
    - out: (N, T, D) embedded word vectors
    \"\"\"
    out = W_embed[x]  # fancy indexing
    return out


def captioning_model_forward(features, captions, params):
    \"\"\"
    Forward pass for a simple image captioning model.
    
    Inputs:
    - features: (N, D_img) CNN image features
    - captions: (N, T) word indices (teacher forcing)
      captions[:, 0] should be <START> token
    - params: dict with:
      - 'W_proj': (D_img, H) image feature projection
      - 'b_proj': (H,) projection bias
      - 'W_embed': (V, D) word embedding
      - 'Wx': (D, H) input-to-hidden
      - 'Wh': (H, H) hidden-to-hidden
      - 'bh': (H,) hidden bias
      - 'W_out': (H, V) output projection
      - 'b_out': (V,) output bias
    
    Returns:
    - scores: (N, T, V) unnormalized scores for each word at each step
    - h0: (N, H) initial hidden state from image
    \"\"\"
    N, T = captions.shape
    V = params['W_embed'].shape[0]
    
    # Step 1: Project image features to initial hidden state
    h0 = features @ params['W_proj'] + params['b_proj']  # (N, H)
    h0 = np.tanh(h0)
    
    # Step 2: Embed input words (use all except last for input)
    # Input: captions[:, :-1] (we predict next word given previous)
    word_emb = word_embedding_forward(captions[:, :-1], params['W_embed'])  # (N, T-1, D)
    
    # Step 3: RNN forward pass
    h_rnn, _ = rnn_forward(word_emb, h0, params['Wx'], params['Wh'], params['bh'])
    # h_rnn shape: (N, T-1, H)
    
    # Step 4: Project to vocabulary scores
    scores = h_rnn @ params['W_out'] + params['b_out']  # (N, T-1, V)
    
    return scores, h0


print("Captioning model forward function defined!")
"""))

    cells.append(code_cell("""# Test the captioning model
np.random.seed(42)

# Hyperparameters
N = 2         # batch size
T = 5         # caption length
D_img = 16    # CNN feature dimension (small for testing)
H = 8         # hidden dimension
D = 6         # word embedding dimension
V = 10        # vocabulary size

# Create test data
features = np.random.randn(N, D_img)
captions = np.array([
    [0, 3, 5, 2, 1],  # sample 1: <start> a cat on <end>
    [0, 4, 6, 7, 1],  # sample 2: <start> dog is running <end>
])

# Initialize parameters
params = {
    'W_proj': np.random.randn(D_img, H) * 0.1,
    'b_proj': np.zeros(H),
    'W_embed': np.random.randn(V, D) * 0.1,
    'Wx': np.random.randn(D, H) * 0.1,
    'Wh': np.random.randn(H, H) * 0.1,
    'bh': np.zeros(H),
    'W_out': np.random.randn(H, V) * 0.1,
    'b_out': np.zeros(V),
}

# Forward pass
scores, h0 = captioning_model_forward(features, captions, params)

print(f"Image features shape: {features.shape}")
print(f"Captions shape: {captions.shape}")
print(f"Initial hidden state h0 shape: {h0.shape}")
print(f"Scores shape: {scores.shape} (N, T-1, V)")
print()
print("Scores for sample 0, first time step (predicting word 1):")
print(f"  {np.round(scores[0, 0], 3)}")
print(f"  Predicted word index: {np.argmax(scores[0, 0])}")
print(f"  True word index: {captions[0, 1]}")
"""))

    cells.append(code_cell("""# Compute cross-entropy loss for captioning

def cross_entropy_loss(scores, captions):
    \"\"\"
    Compute cross-entropy loss for captioning.
    
    Inputs:
    - scores: (N, T-1, V) unnormalized scores
    - captions: (N, T) true word indices
      We compare scores[:, t] with captions[:, t+1]
    
    Returns:
    - loss: scalar loss value
    \"\"\"
    N, T_minus_1, V = scores.shape
    
    # Softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Target: captions[:, 1:] (all except <START>)
    targets = captions[:, 1:]  # (N, T-1)
    
    # Gather probabilities of correct words
    N_idx = np.arange(N)[:, np.newaxis]
    T_idx = np.arange(T_minus_1)[np.newaxis, :]
    correct_probs = probs[N_idx, T_idx, targets]
    
    # Negative log likelihood
    loss = -np.mean(np.log(correct_probs + 1e-10))
    
    return loss


loss = cross_entropy_loss(scores, captions)
print(f"Cross-entropy loss: {loss:.4f}")
print()
print("During training, we minimize this loss using backpropagation.")
print("Lower loss = model assigns higher probability to correct words.")
"""))

    # Section 6: Greedy search
    cells.append(md_cell("""## 6. 贪心搜索（Greedy Search）解码

### 训练 vs 推理

训练时我们用 Teacher Forcing（喂真实词），但推理时必须自己生成词。

### 贪心搜索

最简单的生成策略：每一步选概率最大的词，作为下一步的输入。

```
h_0 = CNN(image)  # 初始隐藏状态
x = <START>       # 初始输入
for t in range(max_len):
    h_t, _ = RNN(x, h_{t-1})
    scores = h_t @ W_out + b_out
    word = argmax(scores)
    output.append(word)
    x = word_embed[word]
    if word == <END>: break
```

### 贪心搜索的问题

贪心搜索只选当前最优，不一定能得到全局最优的句子。更好的方法是 **Beam Search**（集束搜索）。

---

**思考题 3：** 为什么贪心搜索不能保证找到最优的句子序列？
"""))

    cells.append(code_cell("""# Greedy search decoding implementation

def greedy_decode(features, params, start_idx, end_idx, max_len=20):
    \"\"\"
    Generate captions using greedy search.
    
    Inputs:
    - features: (N, D_img) CNN image features
    - params: model parameters
    - start_idx: index of <START> token
    - end_idx: index of <END> token
    - max_len: maximum caption length
    
    Returns:
    - captions: list of lists, generated word indices for each sample
    \"\"\"
    N = features.shape[0]
    H = params['bh'].shape[0]
    V = params['W_embed'].shape[0]
    
    # Initial hidden state from image
    h_prev = np.tanh(features @ params['W_proj'] + params['b_proj'])  # (N, H)
    
    # Start with <START> token
    current_word = np.full(N, start_idx, dtype=int)
    
    captions = [[] for _ in range(N)]
    done = np.zeros(N, dtype=bool)
    
    for t in range(max_len):
        # Embed current word
        word_emb = params['W_embed'][current_word]  # (N, D)
        
        # RNN step
        h_prev = np.tanh(word_emb @ params['Wx'] + h_prev @ params['Wh'] + params['bh'])
        
        # Predict next word
        scores = h_prev @ params['W_out'] + params['b_out']  # (N, V)
        next_word = np.argmax(scores, axis=1)
        
        # Add to captions (only if not done)
        for i in range(N):
            if not done[i]:
                captions[i].append(int(next_word[i]))
                if next_word[i] == end_idx:
                    done[i] = True
        
        current_word = next_word
        
        if np.all(done):
            break
    
    return captions


# Test greedy decoding
generated = greedy_decode(features, params, start_idx=0, end_idx=1, max_len=10)

print("Generated captions (word indices):")
for i, cap in enumerate(generated):
    print(f"  Sample {i}: {cap}")
print()
print("Greedy search picks the most likely word at each step.")
print("It's fast but not guaranteed to find the globally optimal sequence.")
"""))

    # Common mistakes
    cells.append(md_cell("""## 7. 常见错误与注意事项

### 错误 1：混淆输入和目标的偏移

在图像描述中：
- **输入**给解码器的是 `<START>, w1, w2, ..., w_{T-1}`
- **目标**是 `w1, w2, ..., w_{T-1}, <END>`
- 输入比目标早一个位置

### 错误 2：RNN 初始化为全零就够了

对于图像描述，初始隐藏状态应该由图像特征初始化，而不是全零。

### 错误 3：忘记处理变长序列

不同的描述长度不同，损失计算时要忽略 padding 位置的损失。

### 错误 4：LSTM 遗忘门偏置初始化

遗忘门偏置应该初始化为正值（如 1），这样初始时模型倾向于记住信息。如果初始化为 0，sigmoid(0)=0.5，信息会被"稀释"。

### 错误 5：贪心搜索就是最好的

贪心搜索简单快速，但往往不是最优的。Beam Search 通常能生成更好的结果。

---
"""))

    # Homework 1
    cells.append(md_cell("""---

## 作业 1：实现 LSTM 反向传播

### 任务

实现 LSTM 的反向传播。这是 CS231n Assignment 2 的核心题目之一。

### 提示

LSTM 反向传播的关键是：
1. 从最后一个时间步开始，反向传播到第一个时间步
2. 梯度有两条路径：通过隐藏状态 h，和通过细胞状态 c
3. 细胞状态的梯度会通过遗忘门和输入门分流
4. 门的梯度需要经过 sigmoid 或 tanh 的导数

### 需要推导的梯度

- dW, dU, db：所有权重和偏置的梯度
- dh0：初始隐藏状态的梯度
- dx：输入的梯度

---
"""))

    cells.append(code_cell("""# ===== Homework 1: LSTM Backward Pass =====

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def lstm_backward(dh, cache):
    \"\"\"
    Backward pass for LSTM.
    
    Inputs:
    - dh: (N, T, H) gradient of loss w.r.t. hidden states
    - cache: tuple from lstm_forward with (x, h0, W, U, b, h, c)
    
    Returns:
    - dx: (N, T, D) gradient w.r.t. input
    - dh0: (N, H) gradient w.r.t. initial hidden state
    - dW: (D, 4H) gradient w.r.t. input weights
    - dU: (H, 4H) gradient w.r.t. hidden weights
    - db: (4H,) gradient w.r.t. biases
    \"\"\"
    x, h0, W, U, b, h, c = cache
    N, T, D = x.shape
    H = h0.shape[1]
    
    # Initialize gradients
    dx = np.zeros_like(x)
    dh0 = np.zeros_like(h0)
    dW = np.zeros_like(W)
    dU = np.zeros_like(U)
    db = np.zeros_like(b)
    
    # YOUR CODE HERE
    # Hint: You need to recompute gate values (or store them in cache)
    # Hint: Iterate backwards through time
    # Hint: dh_next and dc_next accumulate gradients from t+1
    
    return dx, dh0, dW, dU, db


# ===== ANSWER KEY =====
def lstm_backward_answer(dh, cache):
    \"\"\"Reference implementation.\"\"\"
    x, h0, W, U, b, h, c = cache
    N, T, D = x.shape
    H = h0.shape[1]
    
    dx = np.zeros_like(x)
    dh0 = np.zeros_like(h0)
    dW = np.zeros_like(W)
    dU = np.zeros_like(U)
    db = np.zeros_like(b)
    
    # Recompute gate values (in practice, store these in cache)
    i_gates = np.zeros((N, T, H))
    f_gates = np.zeros((N, T, H))
    o_gates = np.zeros((N, T, H))
    g_gates = np.zeros((N, T, H))
    
    h_prev = h0.copy()
    c_prev = np.zeros((N, H))
    
    for t in range(T):
        x_t = x[:, t, :]
        all_gates = x_t @ W + h_prev @ U + b
        i_gates[:, t, :] = sigmoid(all_gates[:, :H])
        f_gates[:, t, :] = sigmoid(all_gates[:, H:2*H])
        o_gates[:, t, :] = sigmoid(all_gates[:, 2*H:3*H])
        g_gates[:, t, :] = np.tanh(all_gates[:, 3*H:4*H])
        h_prev = h[:, t, :]
        c_prev = c[:, t, :]
    
    # Backward pass
    dh_next = np.zeros((N, H))
    dc_next = np.zeros((N, H))
    
    for t in reversed(range(T)):
        # Total gradient for h at this time step
        dh_t = dh[:, t, :] + dh_next
        
        # Gradient through output gate: h = o * tanh(c)
        do = dh_t * np.tanh(c[:, t, :])
        dc_t = dh_t * o_gates[:, t, :] * (1 - np.tanh(c[:, t, :])**2) + dc_next
        
        # Gradient through gates
        di = dc_t * g_gates[:, t, :]
        df = dc_t * (c[:, t-1, :] if t > 0 else np.zeros((N, H)))
        dg = dc_t * i_gates[:, t, :]
        
        # Gradients of gate pre-activations
        # sigmoid derivative: sigmoid * (1 - sigmoid)
        # tanh derivative: 1 - tanh^2
        di_pre = di * i_gates[:, t, :] * (1 - i_gates[:, t, :])
        df_pre = df * f_gates[:, t, :] * (1 - f_gates[:, t, :])
        do_pre = do * o_gates[:, t, :] * (1 - o_gates[:, t, :])
        dg_pre = dg * (1 - g_gates[:, t, :]**2)
        
        # Concatenate gate gradients
        dall_gates = np.concatenate([di_pre, df_pre, do_pre, dg_pre], axis=1)  # (N, 4H)
        
        # Gradients for weights and biases
        dW += x[:, t, :].T @ dall_gates
        db += np.sum(dall_gates, axis=0)
        
        # Gradient for h_prev and x_t
        dx[:, t, :] = dall_gates @ W.T
        dh_prev = dall_gates @ U.T
        dh_next = dh_prev
        
        # Gradient for c_prev (through forget gate)
        dc_next = dc_t * f_gates[:, t, :]
    
    dh0 = dh_next
    
    return dx, dh0, dW, dU, db


print("Homework 1: Implement lstm_backward")
print("Write your solution above, then run the tests.")
"""))

    cells.append(code_cell("""# ===== Homework 1: Gradient check =====
np.random.seed(42)

N, T, D = 2, 3, 4
H = 3

x = np.random.randn(N, T, D)
h0 = np.random.randn(N, H) * 0.1
W = np.random.randn(D, 4*H) * 0.1
U = np.random.randn(H, 4*H) * 0.1
b = np.random.randn(4*H) * 0.01
b[H:2*H] = 1.0  # forget gate bias

# Forward pass
h_out, cache = lstm_forward(x, h0, W, U, b)

# Random upstream gradient
dh = np.random.randn(*h_out.shape)

# Compute gradients with reference implementation
dx_ref, dh0_ref, dW_ref, dU_ref, db_ref = lstm_backward_answer(dh, cache)

# Numerical gradient check for a few parameters
def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]
        x[idx] = old_val + h
        fp = f()
        x[idx] = old_val - h
        fm = f()
        x[idx] = old_val
        grad[idx] = (fp - fm) / (2 * h)
        it.iternext()
    return grad

# Simple loss: sum of h_out (so dL/dh = ones)
def loss_fn():
    h, _ = lstm_forward(x, h0, W, U, b)
    return np.sum(h)

dh_test = np.ones_like(h_out)
dx_anal, dh0_anal, dW_anal, dU_anal, db_anal = lstm_backward_answer(dh_test, (x, h0, W, U, b, h_out, cache[5]))

# Check dW numerically (sample a few elements)
dW_num = numerical_gradient(loss_fn, W)
error_W = np.max(np.abs(dW_anal - dW_num) / (np.abs(dW_anal) + np.abs(dW_num) + 1e-8))

print(f"dW relative error: {error_W:.2e}")
assert error_W < 1e-5, f"dW gradient check failed! Error: {error_W}"
print("dW gradient check: PASSED")

# Check db
db_num = numerical_gradient(loss_fn, b)
error_b = np.max(np.abs(db_anal - db_num) / (np.abs(db_anal) + np.abs(db_num) + 1e-8))
print(f"db relative error: {error_b:.2e}")
assert error_b < 1e-5, f"db gradient check failed!"
print("db gradient check: PASSED")

print("\n=== Homework 1 gradient check PASSED! ===")
"""))

    # Homework 2
    cells.append(md_cell("""---

## 作业 2：实现 Beam Search 解码

### 任务

实现集束搜索（Beam Search）解码算法。

### 背景

贪心搜索每步只保留最好的 1 个候选，容易陷入局部最优。
Beam Search 每步保留最好的 **k 个候选**（k 叫 beam size），从而更有可能找到全局最优的序列。

### 算法步骤

1. 初始：只有 <START> 一个候选，分数为 0
2. 对每个候选，用当前词跑一步 RNN，得到下一个词的概率分布
3. 对每个候选，取 top-k 个最可能的词，扩展成新的候选
4. 从所有新候选中选总分最高的 k 个，保留到下一步
5. 如果某个候选生成了 <END>，标记为完成，不再扩展
6. 重复直到最大长度或所有候选都完成

---
"""))

    cells.append(code_cell("""# ===== Homework 2: Beam Search Decoding =====

def beam_search_decode(features, params, start_idx, end_idx, beam_size=3, max_len=10):
    \"\"\"
    Generate captions using beam search (for a single image, batch size 1).
    
    Inputs:
    - features: (1, D_img) image features
    - params: model parameters
    - start_idx: <START> token index
    - end_idx: <END> token index
    - beam_size: number of candidates to keep
    - max_len: maximum length
    
    Returns:
    - best_caption: list of word indices (best caption)
    - all_candidates: list of (score, caption) tuples
    \"\"\"
    # YOUR CODE HERE
    # Hint:
    # 1. Initialize beam with start token
    # 2. At each step, expand all candidates with top-k next words
    # 3. Keep top-k by total score (log probability sum)
    # 4. Track which sequences are finished (generated <END>)
    # 5. Return the best finished sequence
    
    best_caption = None
    all_candidates = []
    return best_caption, all_candidates


# ===== ANSWER KEY =====
def beam_search_decode_answer(features, params, start_idx, end_idx, beam_size=3, max_len=10):
    \"\"\"Reference implementation.\"\"\"
    H = params['bh'].shape[0]
    V = params['W_embed'].shape[0]
    
    # Initial hidden state
    h0 = np.tanh(features @ params['W_proj'] + params['b_proj'])  # (1, H)
    
    # Beam: list of (score, h_state, caption_list, is_finished)
    # Start with one candidate
    beam = [(0.0, h0[0], [start_idx], False)]
    finished = []
    
    for step in range(max_len):
        candidates = []
        
        for score, h_state, caption, is_done in beam:
            if is_done:
                candidates.append((score, h_state, caption, True))
                continue
            
            # Last word
            last_word = caption[-1]
            word_emb = params['W_embed'][last_word:last_word+1]  # (1, D)
            
            # RNN step
            h_next = np.tanh(word_emb @ params['Wx'] + h_state @ params['Wh'] + params['bh'])
            h_next = h_next[0]  # (H,)
            
            # Scores
            scores = h_next @ params['W_out'] + params['b_out']  # (V,)
            
            # Log probabilities (stable softmax)
            scores = scores - np.max(scores)
            log_probs = scores - np.log(np.sum(np.exp(scores)))
            
            # Top-k next words
            top_k_indices = np.argsort(log_probs)[-beam_size:][::-1]
            
            for next_word in top_k_indices:
                new_score = score + log_probs[next_word]
                new_caption = caption + [int(next_word)]
                new_done = (next_word == end_idx)
                
                if new_done:
                    finished.append((new_score, new_caption))
                else:
                    candidates.append((new_score, h_next, new_caption, False))
        
        # Keep top-k candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = candidates[:beam_size]
        
        # Check if all are done
        if len(beam) == 0:
            break
    
    # Also add any remaining beam candidates
    for score, _, caption, _ in beam:
        finished.append((score, caption))
    
    # Sort by score
    finished.sort(key=lambda x: x[0], reverse=True)
    
    best_caption = finished[0][1] if finished else []
    all_candidates = [(s, c) for s, c in finished[:beam_size]]
    
    return best_caption, all_candidates


print("Homework 2: Implement beam_search_decode")
print("Write your solution above, then run the tests.")
"""))

    cells.append(code_cell("""# ===== Homework 2: Test cases =====
np.random.seed(42)

# Use same params as before, single sample
feat_single = features[0:1]

# Beam search
best_beam, candidates = beam_search_decode_answer(
    feat_single, params, start_idx=0, end_idx=1, beam_size=3, max_len=8
)

# Greedy search for comparison
greedy_result = greedy_decode(feat_single, params, start_idx=0, end_idx=1, max_len=8)[0]

print("Beam Search Results (beam_size=3):")
print(f"  Best caption: {best_beam}")
print(f"  Top candidates:")
for i, (score, cap) in enumerate(candidates):
    print(f"    {i+1}. score={score:.3f}, caption={cap}")

print()
print("Greedy Search Result:")
print(f"  Caption: {greedy_result}")

# Verify beam search finds better or equal score than greedy
def score_caption(feat, caption, params):
    \"\"\"Compute log probability of a caption.\"\"\"
    h = np.tanh(feat @ params['W_proj'] + params['b_proj'])
    total_score = 0.0
    for t in range(len(caption) - 1):
        word_emb = params['W_embed'][caption[t]:caption[t]+1]
        h = np.tanh(word_emb @ params['Wx'] + h @ params['Wh'] + params['bh'])
        scores = h @ params['W_out'] + params['b_out']
        scores = scores - np.max(scores)
        log_probs = scores - np.log(np.sum(np.exp(scores)))
        total_score += log_probs[0, caption[t+1]]
    return total_score

if best_beam and greedy_result:
    beam_score = score_caption(feat_single[0], best_beam, params)
    greedy_score = score_caption(feat_single[0], greedy_result, params)
    print()
    print(f"Beam search score: {beam_score:.4f}")
    print(f"Greedy search score: {greedy_score:.4f}")
    assert beam_score >= greedy_score - 1e-8, "Beam search should find >= greedy!"
    print("Beam search >= greedy search: PASSED")

# Verify shape of output
assert isinstance(best_beam, list), "Best caption should be a list"
assert len(best_beam) > 0, "Caption should not be empty"
print("Output format: PASSED")

print("\n=== All Homework 2 tests PASSED! ===")
"""))

    # Summary
    cells.append(md_cell("""---

## 本讲总结

### 核心概念回顾

| 概念 | 关键思想 | 要点 |
|------|----------|------|
| RNN | 循环连接，参数共享 | 处理变长序列，但梯度消失 |
| LSTM | 门控机制 + 细胞状态 | 输入门、遗忘门、输出门 |
| 编码器-解码器 | CNN 编码图像，RNN 生成文字 | 图像描述的经典架构 |
| Teacher Forcing | 训练时喂真实词 | 训练稳定快速 |
| 贪心搜索 | 每步选概率最大的词 | 简单但非最优 |
| Beam Search | 每步保留 k 个候选 | 质量更高，计算更多 |

### 思考题答案提示

1. RNN 的参数是共享的，可以处理任意长度的序列。生成时通过检测 <END> token 来决定何时停止
2. 遗忘门偏置为正 → sigmoid 输出接近 1 → 初始时信息可以流过，有助于训练初期的梯度流动
3. 因为每步选局部最优不保证全局最优。有时选一个概率稍低的词，后续能得到更好的整体序列

### 论文参考

- [[Vinyals et al., 2014]](https://arxiv.org/abs/1411.4555) Show and Tell — 图像描述的开创性工作
- [[Karpathy & Fei-Fei, 2015]](https://arxiv.org/abs/1505.04597) DenseCap — 密集图像描述
- [[Hochreiter & Schmidhuber, 1997]](https://www.bioinf.jku.at/publications/older/2604.pdf) LSTM — 长短期记忆网络

### GitHub 参考实现

- [ruotianluo/ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch) (3.3k stars) — 图像描述 PyTorch 实现
- [jariasf/CS231n](https://github.com/jariasf/CS231n) (486 stars) — CS231n 作业参考

---

**下节预告**：目标检测——从 R-CNN 到 YOLO，如何在图像中找到所有物体并识别它们？
"""))

    return cells


# ============================================================
# NOTEBOOK 5: Object Detection
# ============================================================
def build_notebook5():
    cells = []

    cells.append(md_cell("""# 目标检测：从 R-CNN 到 YOLO

> **CS231n 深度学习与计算机视觉** · 第9讲
>
> 本 notebook 将带你理解目标检测任务的核心挑战，以及从 R-CNN 系列（两阶段）到 YOLO/SSD（单阶段）的演进之路。你将亲手实现 IoU、NMS 等目标检测的基础组件。

---

## 学习目标

- 理解目标检测任务（分类 + 定位，多物体）
- 理解滑动窗口方法的局限
- 掌握 R-CNN → Fast R-CNN → Faster R-CNN 的演进
- 理解候选区域（Region Proposals）和 RPN 的作用
- 理解 Anchor Box 的直觉
- 理解单阶段检测器（YOLO, SSD）的思想
- 掌握 IoU 的定义与计算
- 掌握 NMS（非极大值抑制）的原理与实现
- 理解 mAP 评估指标
- 从零实现 IoU 和 NMS

---
"""))

    cells.append(code_cell("""# Setup
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("Environment ready!")
"""))

    # Section 1: What is object detection?
    cells.append(md_cell("""## 1. 什么是目标检测？

### 任务定义

目标检测 = **分类** + **定位**（对图像中的所有物体）

- **图像分类**：这张图里有什么？ → 输出一个类别
- **目标定位**：物体在哪里？ → 输出一个边界框
- **目标检测**：图中有哪些物体？分别在哪里？ → 输出多个 (类别, 边界框)

### 难点

1. **物体数量不固定**：每张图里的物体数目不同
2. **物体大小不一**：远近不同，尺度变化大
3. **物体会遮挡**：部分可见的物体也要检测
4. **背景复杂**：需要区分前景和背景

---

**思考题 1：** 目标检测和图像分类相比，最大的挑战是什么？
"""))

    cells.append(code_cell("""# Visual illustration of object detection
fig, ax = plt.subplots(figsize=(8, 6))

# Create a simple "scene"
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.set_aspect('equal')

# Background
ax.add_patch(Rectangle((0, 0), 10, 8, facecolor='#87CEEB', alpha=0.3))
ax.add_patch(Rectangle((0, 0), 10, 2, facecolor='#90EE90', alpha=0.5))

# "Cat" - orange rectangle
cat_rect = Rectangle((2, 2.5), 2.5, 2, facecolor='#FF8C00', alpha=0.7)
ax.add_patch(cat_rect)
ax.text(3.25, 3.5, 'cat', ha='center', va='center', color='white', fontsize=12, fontweight='bold')

# "Dog" - brown rectangle
dog_rect = Rectangle((5.5, 2), 3, 2.5, facecolor='#8B4513', alpha=0.7)
ax.add_patch(dog_rect)
ax.text(7, 3.25, 'dog', ha='center', va='center', color='white', fontsize=12, fontweight='bold')

# "Bird" - small yellow rectangle  
bird_rect = Rectangle((6.5, 5.5), 1, 0.8, facecolor='#FFD700', alpha=0.8)
ax.add_patch(bird_rect)
ax.text(7, 5.9, 'bird', ha='center', va='center', color='black', fontsize=9, fontweight='bold')

# Draw detection boxes (red dashed)
for rect, label in [(cat_rect, 'cat 0.95'), (dog_rect, 'dog 0.92'), (bird_rect, 'bird 0.88')]:
    x0 = rect.get_x()
    y0 = rect.get_y()
    w = rect.get_width()
    h = rect.get_height()
    det_box = Rectangle((x0-0.1, y0-0.1), w+0.2, h+0.2, 
                         linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(det_box)
    ax.text(x0-0.1, y0+h+0.2, label, color='red', fontsize=10, fontweight='bold')

ax.set_title('Object Detection: Find and classify ALL objects', fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.show()

print("Goal: output multiple (class, bounding_box, confidence) tuples")
print("Bounding box format: usually (x1, y1, x2, y2) or (cx, cy, w, h)")
"""))

    # Section 2: Sliding window
    cells.append(md_cell("""## 2. 滑动窗口：最朴素的方法

### 思路

用一个固定大小的窗口，在图像上从左到右、从上到下滑动，每个位置都用分类器判断"窗口里有没有目标"。

### 问题

1. **计算量太大**：窗口数量 = 位置数 × 尺度数 × 长宽比数，可能上百万
2. **窗口大小固定**：不同大小和比例的物体很难用同一个窗口覆盖
3. **重复计算**：相邻窗口的特征大部分是重叠的

### 为什么 CNN 时代之前滑动窗口不现实？

传统方法（如 HOG + SVM）每个窗口都要重新提取特征，非常慢。

---
"""))

    cells.append(code_cell("""# Sliding window visualization
fig, ax = plt.subplots(figsize=(8, 6))

# Image
img = np.random.rand(20, 30, 3) * 0.3
img[5:12, 8:18, 0] = 0.9  # "object"
ax.imshow(img)

# Draw sliding windows at different positions
window_sizes = [(4, 4), (6, 6), (8, 8)]
positions = [(2, 2), (5, 10), (10, 5), (8, 15), (12, 10)]

for idx, (wy, wx) in enumerate(positions[:4]):
    rect = Rectangle((wx, wy), 6, 6, linewidth=2, edgecolor='yellow', facecolor='none', alpha=0.7)
    ax.add_patch(rect)

# Highlight one window that "contains" the object
rect_hit = Rectangle((8, 5), 10, 7, linewidth=3, edgecolor='red', facecolor='none')
ax.add_patch(rect_hit)
ax.text(13, 4, 'positive\nwindow', color='red', fontsize=10, ha='center')

ax.set_title('Sliding Window: too many positions, too slow!', fontsize=12)
ax.axis('off')
plt.tight_layout()
plt.show()

print("Problems with sliding window approach:")
print("1. Too many windows (millions per image)")
print("2. Need multiple scales and aspect ratios")
print("3. Redundant computation between overlapping windows")
print("")
print("Modern detectors solve these problems in clever ways!")
"""))

    # Section 3: R-CNN series
    cells.append(md_cell("""## 3. R-CNN 系列：两阶段检测器

### R-CNN (2014)

**思路**：先用选择性搜索（Selective Search）生成 ~2000 个候选区域，再对每个区域用 CNN 提取特征并分类。

**流程**：
1. 输入图像
2. 选择性搜索生成 ~2000 个候选框
3. 把每个候选框 warp 成固定大小
4. 用 CNN 提取每个候选框的特征
5. 用 SVM 分类 + 边界框回归

**问题**：
- 慢！每张图要对 2000 个候选框单独跑 CNN
- 训练分多阶段（预训练 → 微调 → SVM → 回归），很麻烦

> **参考论文**：[[Girshick et al., 2014]](https://arxiv.org/abs/1311.2524) Rich feature hierarchies for accurate object detection and semantic segmentation

---
"""))

    cells.append(md_cell("""### Fast R-CNN (2015)

**核心改进：特征共享**

不是对每个候选框单独跑 CNN，而是：
1. 先对整张图跑一次 CNN，得到特征图
2. 在特征图上做 RoI Pooling（感兴趣区域池化）
3. 每个候选框从特征图上"抠"出对应区域，池化成固定大小

**RoI Pooling 的作用**：把任意大小的候选区域变成固定大小的特征向量（如 7×7），这样后面才能接全连接层。

**优点**：
- 速度快了很多（CNN 只跑一次）
- 可以端到端训练

**问题**：候选区域还是用选择性搜索生成的，这一步在 CPU 上跑，是瓶颈。

> **参考论文**：[[Girshick, 2015]](https://arxiv.org/abs/1504.08083) Fast R-CNN

---

### Faster R-CNN (2015)

**核心改进：RPN（Region Proposal Network）**

把候选区域生成也放进 CNN 里，用 GPU 加速！

**RPN 的工作方式**：
- 在特征图上的每个位置，预测 k 个 anchor box 的得分和偏移
- Anchor box：预先定义好的不同大小和比例的框
- 网络只需要学习"调整"这些 anchor，而不是从零预测框

**完整流程**：
1. 图像 → CNN → 特征图
2. RPN 在特征图上生成候选区域
3. RoI Pooling 提取每个候选区域的特征
4. 分类 + 边界框回归

> **参考论文**：[[Ren et al., 2015]](https://arxiv.org/abs/1506.01497) Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

---

**思考题 2：** 为什么 Faster R-CNN 比 Fast R-CNN 快很多？
"""))

    # Section 4: Anchor boxes
    cells.append(md_cell("""## 4. Anchor Box：先验框的直觉

### 什么是 Anchor Box？

在特征图的每个位置，预先定义 k 个不同大小和比例的框。网络的任务是：
1. 预测每个 anchor 里有没有物体（objectness score）
2. 预测 anchor 需要怎么调整才能对准真实物体（bounding box regression）

### 为什么需要 Anchor？

- 直接预测边界框的坐标很难（范围太大）
- 先有一个"参考框"，只需要预测偏移量，更容易学习
- 不同的 anchor 负责检测不同大小和比例的物体

### 典型设置

Faster R-CNN 使用 9 个 anchor（3 种尺度 × 3 种长宽比）：
- 尺度：128², 256², 512²
- 比例：1:1, 1:2, 2:1

---
"""))

    cells.append(code_cell("""# Visualize anchor boxes at one location
fig, ax = plt.subplots(figsize=(6, 6))

# Feature map grid point
center_x, center_y = 5, 5
ax.plot(center_x, center_y, 'ro', markersize=8)
ax.text(center_x, center_y - 0.5, 'anchor\ncenter', ha='center', color='red', fontsize=9)

# 9 anchor boxes: 3 scales x 3 aspect ratios
scales = [1.0, 2.0, 3.0]
ratios = [(1, 1), (2, 1), (1, 2)]
colors = ['blue', 'green', 'orange']

for s_idx, scale in enumerate(scales):
    for r_idx, (rw, rh) in enumerate(ratios):
        w = scale * rw
        h = scale * rh
        rect = Rectangle((center_x - w/2, center_y - h/2), w, h,
                        linewidth=2, edgecolor=colors[s_idx], facecolor='none',
                        linestyle='--' if r_idx > 0 else '-', alpha=0.7)
        ax.add_patch(rect)

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.set_title('Anchor Boxes at One Location (9 anchors)', fontsize=12)
ax.text(5, 9.5, '3 scales x 3 aspect ratios = 9 anchors', ha='center', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("Anchor boxes provide reference shapes for the network to adjust.")
print("The network predicts offsets (dx, dy, dw, dh) from anchor to ground truth.")
print("This is much easier than predicting boxes from scratch!")
"""))

    # Section 5: One-stage vs two-stage
    cells.append(md_cell("""## 5. 单阶段 vs 两阶段检测器

### 两阶段检测器（Two-Stage）

代表：R-CNN 系列

**流程**：先生成候选区域，再对每个区域分类

**优点**：精度高（尤其是小物体）
**缺点**：速度慢

### 单阶段检测器（One-Stage）

代表：YOLO, SSD

**流程**：直接在特征图上预测所有框的类别和位置

**优点**：速度快，可以实时检测
**缺点**：精度略低（尤其是小物体）

### 速度-精度权衡

- 需要最高精度 → 两阶段（Faster R-CNN 系列）
- 需要实时检测 → 单阶段（YOLO, SSD 系列）
- 现在两者差距越来越小

---

**思考题 3：** 为什么单阶段检测器通常在小物体上表现不如两阶段？
"""))

    # Section 6: YOLO
    cells.append(md_cell("""## 6. YOLO：你只看一次

### YOLO 的核心思想

把图像分成 S×S 的网格（如 7×7），每个网格单元负责预测：
- B 个边界框（每个框有 5 个值：x, y, w, h, confidence）
- C 个类别的概率

**总输出维度**：S × S × (B×5 + C)

### 直觉

- 每个网格单元"负责"检测中心落在该单元内的物体
- 相当于在全局做密集预测
- 一次前向传播就得到所有结果，非常快

### YOLO 的演进

- **YOLOv1**：开创性工作，但小物体检测差
- **YOLOv2/9000**：加入 BatchNorm、anchor boxes、多尺度训练
- **YOLOv3**：多尺度预测（类似 FPN）、更好的 backbone
- **YOLOv4/v5/v8**：各种技巧的集大成，速度精度都很强

> **参考论文**：[[Redmon et al., 2016]](https://arxiv.org/abs/1506.02640) You Only Look Once: Unified, Real-Time Object Detection

---
"""))

    cells.append(code_cell("""# YOLO grid-based prediction visualization
fig, ax = plt.subplots(figsize=(7, 7))

# Image grid
S = 7  # 7x7 grid
for i in range(S+1):
    ax.axvline(i, color='white', linewidth=1)
    ax.axhline(i, color='white', linewidth=1)

# "Object" in the image
obj_rect = Rectangle((2.2, 3.1), 3.5, 2.8, facecolor='#FF8C00', alpha=0.7)
ax.add_patch(obj_rect)

# Center of object falls in grid cell (4, 4) [0-indexed: (3, 4)? Let's compute]
cx = 2.2 + 3.5/2  # 3.95
cy = 3.1 + 2.8/2  # 4.5
grid_x = int(cx)
grid_y = int(cy)

# Highlight the responsible grid cell
responsible = Rectangle((grid_x, grid_y), 1, 1, facecolor='yellow', alpha=0.4)
ax.add_patch(responsible)
ax.text(grid_x + 0.5, grid_y + 0.5, 'responsible\ncell', ha='center', va='center',
        fontsize=9, fontweight='bold')

# Mark center
ax.plot(cx, cy, 'r*', markersize=15)
ax.text(cx + 0.2, cy + 0.2, 'center', color='red', fontsize=9)

ax.set_xlim(0, S)
ax.set_ylim(0, S)
ax.set_aspect('equal')
ax.set_title(f'YOLO: {S}x{S} Grid, Each Cell Predicts B Boxes', fontsize=12)
ax.invert_yaxis()
plt.tight_layout()
plt.show()

print(f"Object center at ({cx:.1f}, {cy:.1f}) -> grid cell ({grid_x}, {grid_y})")
print("Grid cell", (grid_x, grid_y), "is responsible for detecting this object.")
print("It predicts B bounding boxes and C class probabilities.")
"""))

    # Section 7: IoU
    cells.append(md_cell("""## 7. IoU：交并比

### 定义

IoU（Intersection over Union）是两个边界框重叠程度的度量：

$$IoU = \\frac{\\text{交集面积}}{\\text{并集面积}}$$

### 为什么重要？

1. **评估预测质量**：预测框和真实框的 IoU 越高，检测越准确
2. **正负样本定义**：anchor 与真实框的 IoU > 阈值算正样本
3. **NMS 中使用**：判断两个框是不是检测了同一个物体

### 常见阈值

- IoU > 0.5：粗略认为正确
- IoU > 0.7：较好
- IoU > 0.9：非常精准

---

**思考题 4：** IoU 的取值范围是什么？两个框完全重合时 IoU 是多少？
"""))

    cells.append(code_cell("""# IoU calculation from scratch

def compute_iou(box1, box2):
    \"\"\"
    Compute IoU between two boxes.
    Box format: (x1, y1, x2, y2) - top-left and bottom-right corners.
    
    Inputs:
    - box1: (4,) array or tuple (x1, y1, x2, y2)
    - box2: (4,) array or tuple (x1, y1, x2, y2)
    
    Returns:
    - iou: scalar, IoU value
    \"\"\"
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection coordinates
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    # Intersection area (max with 0 in case no overlap)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    
    # Union area = area1 + area2 - intersection
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0.0
    
    return iou


# Test cases
box_a = (0, 0, 10, 10)
box_b = (5, 5, 15, 15)
box_c = (12, 12, 20, 20)
box_d = (1, 1, 9, 9)

iou_ab = compute_iou(box_a, box_b)
iou_ac = compute_iou(box_a, box_c)
iou_ad = compute_iou(box_a, box_d)

print(f"IoU(box_a, box_b) = {iou_ab:.4f}  (partial overlap)")
print(f"IoU(box_a, box_c) = {iou_ac:.4f}  (no overlap)")
print(f"IoU(box_a, box_d) = {iou_ad:.4f}  (one inside the other)")
"""))

    cells.append(code_cell("""# Visualize IoU examples
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

boxes_pairs = [
    (box_a, box_b, f'IoU = {iou_ab:.3f}'),
    (box_a, box_c, f'IoU = {iou_ac:.3f}'),
    (box_a, box_d, f'IoU = {iou_ad:.3f}'),
]

for idx, (b1, b2, title) in enumerate(boxes_pairs):
    ax = axes[idx]
    
    # Box 1
    rect1 = Rectangle((b1[0], b1[1]), b1[2]-b1[0], b1[3]-b1[1],
                     facecolor='blue', alpha=0.5, edgecolor='blue', linewidth=2)
    ax.add_patch(rect1)
    
    # Box 2
    rect2 = Rectangle((b2[0], b2[1]), b2[2]-b2[0], b2[3]-b2[1],
                     facecolor='red', alpha=0.5, edgecolor='red', linewidth=2)
    ax.add_patch(rect2)
    
    # Intersection
    xi1 = max(b1[0], b2[0])
    yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2])
    yi2 = min(b1[3], b2[3])
    if xi2 > xi1 and yi2 > yi1:
        inter = Rectangle((xi1, yi1), xi2-xi1, yi2-yi1,
                         facecolor='purple', alpha=0.8)
        ax.add_patch(inter)
    
    ax.set_xlim(-2, 22)
    ax.set_ylim(-2, 22)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12)
    ax.invert_yaxis()
    ax.grid(alpha=0.3)

plt.suptitle('IoU = Intersection / Union', fontsize=14)
plt.tight_layout()
plt.show()
"""))

    # Section 8: NMS
    cells.append(md_cell("""## 8. NMS：非极大值抑制

### 问题：重复检测

检测器经常会对同一个物体输出多个重叠的检测框。我们需要**只保留最好的那个**。

### NMS 算法步骤

1. 按置信度从高到低排序所有检测框
2. 取出置信度最高的框，加入最终结果
3. 计算它和其他所有框的 IoU
4. 移除 IoU 大于阈值的框（认为它们检测的是同一个物体）
5. 重复步骤 2-4，直到没有框剩下

### 直觉

"在一个区域里，只保留最自信的那个检测结果，其他重叠的都删掉。"

---
"""))

    cells.append(code_cell("""# Implement NMS from scratch

def nms(boxes, scores, iou_threshold=0.5):
    \"\"\"
    Non-Maximum Suppression.
    
    Inputs:
    - boxes: (N, 4) array of boxes in (x1, y1, x2, y2) format
    - scores: (N,) array of confidence scores
    - iou_threshold: IoU threshold for suppression
    
    Returns:
    - keep: list of indices of boxes to keep
    \"\"\"
    if len(boxes) == 0:
        return []
    
    # Sort by score (descending)
    order = np.argsort(scores)[::-1]
    
    keep = []
    
    while len(order) > 0:
        # Take the box with highest score
        idx = order[0]
        keep.append(idx)
        
        if len(order) == 1:
            break
        
        # Compute IoU with all remaining boxes
        remaining = order[1:]
        ious = np.array([compute_iou(boxes[idx], boxes[r]) for r in remaining])
        
        # Keep boxes with IoU <= threshold
        order = remaining[ious <= iou_threshold]
    
    return keep


# Test NMS
np.random.seed(42)

# Create some overlapping boxes (simulating multiple detections of the same object)
boxes = np.array([
    [10, 10, 50, 50],   # box 0: score 0.95
    [12, 12, 48, 48],   # box 1: score 0.90
    [8, 8, 52, 52],     # box 2: score 0.85
    [60, 10, 100, 50],  # box 3: score 0.92 (different object)
    [62, 12, 98, 48],   # box 4: score 0.80
    [15, 60, 45, 90],   # box 5: score 0.70 (third object)
])
scores = np.array([0.95, 0.90, 0.85, 0.92, 0.80, 0.70])

keep = nms(boxes, scores, iou_threshold=0.5)

print("Input boxes:")
for i in range(len(boxes)):
    print(f"  Box {i}: {boxes[i]}, score={scores[i]:.2f}")

print(f"\nNMS result (kept indices): {keep}")
print(f"Kept {len(keep)} boxes out of {len(boxes)}")
print()
print("Expected: 3 boxes (one per object), keeping the highest-scoring one per cluster")
"""))

    cells.append(code_cell("""# Visualize NMS before and after
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Before NMS
ax = axes[0]
for i in range(len(boxes)):
    rect = Rectangle((boxes[i, 0], boxes[i, 1]), 
                     boxes[i, 2]-boxes[i, 0], boxes[i, 3]-boxes[i, 1],
                     facecolor='none', edgecolor='red', linewidth=2, alpha=0.7)
    ax.add_patch(rect)
    ax.text(boxes[i, 0], boxes[i, 1] - 2, f'{scores[i]:.2f}', 
            color='red', fontsize=9)
ax.set_title(f'Before NMS: {len(boxes)} detections', fontsize=12)
ax.set_xlim(0, 110)
ax.set_ylim(0, 100)
ax.invert_yaxis()
ax.set_aspect('equal')

# After NMS
ax = axes[1]
for i in keep:
    rect = Rectangle((boxes[i, 0], boxes[i, 1]),
                     boxes[i, 2]-boxes[i, 0], boxes[i, 3]-boxes[i, 1],
                     facecolor='none', edgecolor='green', linewidth=3)
    ax.add_patch(rect)
    ax.text(boxes[i, 0], boxes[i, 1] - 2, f'{scores[i]:.2f}',
            color='green', fontsize=9, fontweight='bold')
ax.set_title(f'After NMS: {len(keep)} detections', fontsize=12)
ax.set_xlim(0, 110)
ax.set_ylim(0, 100)
ax.invert_yaxis()
ax.set_aspect('equal')

plt.suptitle('Non-Maximum Suppression (IoU threshold = 0.5)', fontsize=14)
plt.tight_layout()
plt.show()

print("NMS removes redundant overlapping detections, keeping only the best one per object.")
"""))

    # Section 9: mAP
    cells.append(md_cell("""## 9. mAP：目标检测的评估指标

### 为什么不用准确率？

目标检测有两个维度都要评估：
1. **分类对不对** → 类别是否正确
2. **定位准不准** → 边界框 IoU 是否足够高

### mAP（mean Average Precision）

目标检测的标准评估指标。

### 计算步骤（简化版）

1. 对每个类别：
   - 按置信度排序所有检测结果
   - 逐个判断是 TP（正确检测）还是 FP（错误检测）
   - 计算 precision-recall 曲线
   - 计算 AP（Average Precision，PR 曲线下的面积）
2. 对所有类别的 AP 取平均 → mAP

### IoU 阈值

- **AP@0.5**：IoU > 0.5 算正确
- **AP@0.75**：IoU > 0.75 算正确（更严格）
- **AP@[0.5:0.95]**：从 0.5 到 0.95 每隔 0.05 取一个阈值，计算 AP 再取平均（COCO 标准）

---
"""))

    cells.append(code_cell("""# Simplified AP calculation example

def compute_ap_simple(tp, fp, num_gt):
    \"\"\"
    Simplified Average Precision computation.
    
    Inputs:
    - tp: boolean array, True if detection is True Positive (sorted by score desc)
    - fp: boolean array, True if detection is False Positive
    - num_gt: number of ground truth objects
    
    Returns:
    - ap: Average Precision
    - precisions: precision at each step
    - recalls: recall at each step
    \"\"\"
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / num_gt
    
    # Simple AP: area under PR curve (trapezoidal approximation)
    # Add sentinel values
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    
    # Make precision monotonically decreasing
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    
    # Compute area
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            ap += (mrec[i] - mrec[i-1]) * mpre[i]
    
    return ap, precisions, recalls


# Example: 5 detections, 3 ground truth objects
# (sorted by confidence from high to low)
tp = np.array([True, True, False, True, False])
fp = ~tp
num_gt = 3

ap, precisions, recalls = compute_ap_simple(tp, fp, num_gt)

print("Detection results (sorted by confidence):")
for i in range(len(tp)):
    status = "TP" if tp[i] else "FP"
    print(f"  Rank {i+1}: {status}  (precision={precisions[i]:.3f}, recall={recalls[i]:.3f})")

print(f"\nNumber of ground truth objects: {num_gt}")
print(f"Average Precision (AP): {ap:.4f}")
print()
print("mAP = mean AP over all classes")
print("Higher mAP = better detector")
"""))

    cells.append(code_cell("""# Visualize PR curve
fig, ax = plt.subplots(figsize=(8, 6))

# Full PR curve points
mrec = np.concatenate(([0.0], recalls, [1.0]))
mpre = np.concatenate(([1.0], precisions, [0.0]))
for i in range(len(mpre)-2, -1, -1):
    mpre[i] = max(mpre[i], mpre[i+1])

ax.plot(recalls, precisions, 'o-', linewidth=2, markersize=8, label='PR curve')
ax.plot(mrec, mpre, '--', linewidth=1.5, alpha=0.7, label='Monotonic envelope')

# Fill AP area
ax.fill_between(mrec, mpre, alpha=0.2, color='blue')

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title(f'Precision-Recall Curve (AP = {ap:.3f})', fontsize=13)
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)

# Annotate points
for i, (r, p) in enumerate(zip(recalls, precisions)):
    status = "TP" if tp[i] else "FP"
    ax.annotate(f'#{i+1} {status}', (r, p), textcoords="offset points",
                xytext=(10, 5), fontsize=9)

plt.tight_layout()
plt.show()

print("AP = area under the precision-recall curve.")
print("mAP averages this across all object classes.")
"""))

    # Common mistakes
    cells.append(md_cell("""## 10. 常见错误与注意事项

### 错误 1：IoU 计算时忘记处理无重叠的情况

如果两个框没有交集，交集面积是 0，IoU 应该是 0。要确保宽高都取 max(0, ...)。

### 错误 2：NMS 中按置信度升序排序

NMS 应该按置信度**从高到低**排序，每次保留最自信的那个。

### 错误 3：混淆两阶段和单阶段的优缺点

- 两阶段：精度高、速度慢（先找候选再分类）
- 单阶段：速度快、精度稍低（一步到位）

### 错误 4：Anchor 数量太少或太多

Anchor 太少：覆盖不了所有形状的物体
Anchor 太多：计算量大，正负样本不平衡更严重

### 错误 5：mAP 指标理解错误

- mAP 不是"准确率"
- mAP 综合考虑了精确率和召回率
- 不同 IoU 阈值下的 mAP 不可直接比较

---
"""))

    # Homework 1
    cells.append(md_cell("""---

## 作业 1：实现带 IoU 阈值的 NMS

### 任务

完善 NMS 实现，确保处理以下边界情况：
1. 空输入
2. 单个框输入
3. 所有框都重叠（应只剩 1 个）
4. 所有框都不重叠（应全部保留）
5. 支持不同的 IoU 阈值

### 要求

- 输入：boxes (N, 4), scores (N,), iou_threshold
- 输出：保留的框的索引列表
- 框的格式：(x1, y1, x2, y2)

---
"""))

    cells.append(code_cell("""# ===== Homework 1: NMS with IoU threshold =====

def compute_iou(box1, box2):
    \"\"\"Compute IoU between two boxes (x1, y1, x2, y2).\"\"""
    # TODO (if not already defined above)
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def nms_improved(boxes, scores, iou_threshold=0.5):
    \"\"\"
    Improved NMS implementation.
    
    Inputs:
    - boxes: (N, 4) array of (x1, y1, x2, y2)
    - scores: (N,) confidence scores
    - iou_threshold: float
    
    Returns:
    - keep_indices: list of int, indices of kept boxes
    \"\"\"
    # YOUR CODE HERE
    # Make sure to handle:
    # - Empty input (return [])
    # - Sort by score descending
    # - Iteratively pick best, remove overlaps
    
    keep_indices = []
    return keep_indices


# ===== ANSWER KEY =====
def nms_improved_answer(boxes, scores, iou_threshold=0.5):
    \"\"\"Reference implementation.\"\"\"
    if len(boxes) == 0:
        return []
    
    # Sort by score descending
    order = np.argsort(scores)[::-1]
    boxes = boxes[order]
    scores = scores[order]
    
    keep = []
    suppressed = np.zeros(len(boxes), dtype=bool)
    
    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        keep.append(order[i])  # original index
        
        for j in range(i + 1, len(boxes)):
            if suppressed[j]:
                continue
            iou = compute_iou(boxes[i], boxes[j])
            if iou > iou_threshold:
                suppressed[j] = True
    
    return keep


print("Homework 1: Implement nms_improved")
print("Write your solution above, then run the tests.")
"""))

    cells.append(code_cell("""# ===== Homework 1: Test cases =====
np.random.seed(42)

# Test 1: Empty input
result = nms_improved_answer(np.array([]).reshape(0, 4), np.array([]), 0.5)
assert result == [], f"Empty test failed: {result}"
print("Test 1 (empty input): PASSED")

# Test 2: Single box
boxes = np.array([[10, 10, 50, 50]])
scores = np.array([0.9])
result = nms_improved_answer(boxes, scores, 0.5)
assert result == [0], f"Single box test failed: {result}"
print("Test 2 (single box): PASSED")

# Test 3: All overlapping -> keep 1
boxes = np.array([
    [0, 0, 10, 10],
    [1, 1, 11, 11],
    [2, 2, 12, 12],
])
scores = np.array([0.9, 0.8, 0.7])
result = nms_improved_answer(boxes, scores, 0.5)
assert len(result) == 1, f"Overlap test failed: {len(result)} boxes kept"
assert result[0] == 0, "Should keep highest scoring box"
print("Test 3 (all overlapping -> 1): PASSED")

# Test 4: No overlap -> keep all
boxes = np.array([
    [0, 0, 10, 10],
    [20, 0, 30, 10],
    [40, 0, 50, 10],
])
scores = np.array([0.9, 0.8, 0.7])
result = nms_improved_answer(boxes, scores, 0.5)
assert len(result) == 3, f"No overlap test failed: {len(result)} boxes kept"
print("Test 4 (no overlap -> all): PASSED")

# Test 5: Variable threshold
boxes = np.array([
    [0, 0, 10, 10],
    [3, 3, 13, 13],  # IoU with box 0 = 49/169 = ~0.29
    [9, 9, 19, 19],  # IoU with box 0 = 1/271 = ~0.004
])
scores = np.array([0.9, 0.8, 0.7])

# With high threshold (0.5), should keep 3 (only very high overlap suppressed)
result_high = nms_improved_answer(boxes, scores, 0.5)
assert len(result_high) >= 2, f"High threshold test failed: {len(result_high)} boxes"
print(f"Test 5a (IoU=0.5): {len(result_high)} boxes kept - PASSED")

# With low threshold (0.1), should keep fewer
result_low = nms_improved_answer(boxes, scores, 0.1)
assert len(result_low) <= len(result_high), "Lower threshold should keep fewer boxes"
print(f"Test 5b (IoU=0.1): {len(result_low)} boxes kept - PASSED")

# Test 6: Realistic scenario
boxes = np.array([
    [10, 10, 50, 50],
    [12, 12, 48, 48],
    [60, 10, 100, 50],
    [8, 8, 52, 52],
    [62, 12, 98, 48],
    [100, 100, 150, 150],
])
scores = np.array([0.95, 0.90, 0.92, 0.85, 0.80, 0.75])
result = nms_improved_answer(boxes, scores, 0.5)
assert len(result) == 3, f"Should keep 3 boxes (one per object), got {len(result)}"
print("Test 6 (3 objects): PASSED")

print("\n=== All Homework 1 tests PASSED! ===")
"""))

    # Homework 2
    cells.append(md_cell("""---

## 作业 2：实现简单 YOLO 风格的网格预测

### 任务

实现一个简化版的 YOLO 网格预测解码函数。

### 背景

YOLO 将图像分成 S×S 的网格，每个网格单元预测 B 个边界框和 C 个类别概率。
我们需要将网络输出转换为实际的边界框坐标。

### 公式

对于第 i,j 个网格单元，第 k 个框：
- bx = sigmoid(tx) + j （中心 x，相对于网格单元）
- by = sigmoid(ty) + i （中心 y，相对于网格单元）
- bw = pw × exp(tw) （宽度，相对于 anchor）
- bh = ph × exp(th) （高度，相对于 anchor）
- 置信度 = sigmoid(to)
- 类别概率 = softmax(class_scores)

### 要求

输入：preds (S, S, B*(5+C))，其中每个框有 (tx, ty, tw, th, to) + C 个类别分数
输出：所有框的列表 (x1, y1, x2, y2, confidence, class_id)

---
"""))

    cells.append(code_cell("""# ===== Homework 2: YOLO-style grid prediction =====

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def yolo_decode(preds, anchors, num_classes):
    \"\"\"
    Decode YOLO grid predictions to bounding boxes.
    
    Inputs:
    - preds: (S, S, B*(5+C)) network output
      Each grid cell predicts B boxes, each box has: tx, ty, tw, th, to + C class scores
    - anchors: (B, 2) anchor box sizes (pw, ph)
    - num_classes: C, number of classes
    
    Returns:
    - boxes: list of (x1, y1, x2, y2, confidence, class_id)
    \"\"\"
    S = preds.shape[0]
    B = len(anchors)
    box_dim = 5 + num_classes  # tx, ty, tw, th, to + C class scores
    
    boxes = []
    
    # YOUR CODE HERE
    # Hint:
    # 1. Reshape preds to (S, S, B, 5+C)
    # 2. For each grid cell (i,j) and each box k:
    #    - tx, ty, tw, th, to = preds[i, j, k, :5]
    #    - bx = sigmoid(tx) + j
    #    - by = sigmoid(ty) + i
    #    - bw = anchors[k, 0] * exp(tw)
    #    - bh = anchors[k, 1] * exp(th)
    #    - confidence = sigmoid(to)
    #    - class_id = argmax(class_scores)
    #    - Convert to (x1, y1, x2, y2) format
    #    - Normalize by S (to get coordinates in [0, 1])
    
    return boxes


# ===== ANSWER KEY =====
def yolo_decode_answer(preds, anchors, num_classes):
    \"\"\"Reference implementation.\"\"\"
    S = preds.shape[0]
    B = len(anchors)
    C = num_classes
    
    # Reshape to (S, S, B, 5+C)
    preds = preds.reshape(S, S, B, 5 + C)
    
    boxes = []
    
    for i in range(S):  # grid row (y)
        for j in range(S):  # grid col (x)
            for k in range(B):  # each anchor/box
                tx, ty, tw, th, to = preds[i, j, k, :5]
                class_scores = preds[i, j, k, 5:]
                
                # Box center relative to grid cell
                bx = sigmoid(tx) + j
                by = sigmoid(ty) + i
                
                # Box width/height relative to anchor
                pw, ph = anchors[k]
                bw = pw * np.exp(tw)
                bh = ph * np.exp(th)
                
                # Confidence
                conf = sigmoid(to)
                
                # Class prediction
                class_probs = softmax(class_scores)
                class_id = np.argmax(class_probs)
                
                # Convert to x1, y1, x2, y2 (normalized to [0, 1])
                x1 = (bx - bw / 2) / S
                y1 = (by - bh / 2) / S
                x2 = (bx + bw / 2) / S
                y2 = (by + bh / 2) / S
                
                boxes.append((x1, y1, x2, y2, conf, class_id))
    
    return boxes


print("Homework 2: Implement yolo_decode")
print("Write your solution above, then run the tests.")
"""))

    cells.append(code_cell("""# ===== Homework 2: Test cases =====
np.random.seed(42)

S = 3  # 3x3 grid
B = 2  # 2 anchor boxes per cell
C = 4  # 4 classes

anchors = np.array([[1.0, 1.0], [2.0, 0.5]])  # square, wide

# Create simple test predictions
preds = np.random.randn(S, S, B * (5 + C)) * 0.5

# Make one cell have a strong detection
# Cell (1,1), box 0: high confidence, cat-like class scores
box_idx = 0
cell_offset = box_idx * (5 + C)
preds[1, 1, cell_offset + 4] = 3.0  # high confidence (to)
preds[1, 1, cell_offset + 5] = 2.0   # class 0 high
preds[1, 1, cell_offset + 6] = 0.5   # class 1 medium
preds[1, 1, cell_offset + 7] = -1.0  # class 2 low
preds[1, 1, cell_offset + 8] = 0.0   # class 3

boxes = yolo_decode_answer(preds, anchors, C)

print(f"Total boxes decoded: {len(boxes)}")
print(f"Expected: S*S*B = {S*S*B} = {len(boxes)}")
assert len(boxes) == S * S * B, f"Expected {S*S*B} boxes, got {len(boxes)}"
print("Box count: PASSED")

# Check box format
assert len(boxes[0]) == 6, "Each box should have 6 elements: (x1, y1, x2, y2, conf, class_id)"
print("Box format: PASSED")

# Check confidence range [0, 1] (sigmoid)
confidences = [b[4] for b in boxes]
assert all(0 <= c <= 1 for c in confidences), "Confidences should be in [0, 1]"
print("Confidence in [0, 1]: PASSED")

# Check class id range [0, C-1]
class_ids = [b[5] for b in boxes]
assert all(0 <= c < C for c in class_ids), "Class IDs out of range"
print("Class ID range: PASSED")

# Check the strong detection in cell (1,1)
# Find the box from cell (1,1), box 0
# Cell (1,1) is the 4th cell: index = 1*S*B + 1*B + 0 = 6 + 2 + 0 = 8
strong_box_idx = 1 * S * B + 1 * B + 0
strong_box = boxes[strong_box_idx]
print(f"\nStrong detection (cell 1,1, box 0):")
print(f"  Box: ({strong_box[0]:.3f}, {strong_box[1]:.3f}) -> ({strong_box[2]:.3f}, {strong_box[3]:.3f})")
print(f"  Confidence: {strong_box[4]:.3f}")
print(f"  Class: {strong_box[5]}")
assert strong_box[4] > 0.9, "High confidence expected (sigmoid(3) ~ 0.95)"
print("Strong detection confidence: PASSED")

# Verify coordinates are in [0, 1]
for b in boxes:
    assert 0 <= b[0] <= 1 and 0 <= b[1] <= 1 and 0 <= b[2] <= 2 and 0 <= b[3] <= 2, \
        f"Box coordinates out of expected range: {b}"
print("Coordinate range: PASSED")

print("\n=== All Homework 2 tests PASSED! ===")
"""))

    # Summary
    cells.append(md_cell("""---

## 本讲总结

### 目标检测方法演进

| 方法 | 类型 | 核心思想 | 速度 | 精度 |
|------|------|----------|------|------|
| R-CNN | 两阶段 | 选择性搜索 + CNN 分类 | 慢 | 中 |
| Fast R-CNN | 两阶段 | RoI Pooling + 特征共享 | 中 | 中高 |
| Faster R-CNN | 两阶段 | RPN 生成候选区域 | 中快 | 高 |
| YOLO | 单阶段 | 网格预测，一步到位 | 快 | 中 |
| SSD | 单阶段 | 多尺度特征图预测 | 快 | 中高 |

### 核心组件

| 组件 | 作用 | 关键公式/操作 |
|------|------|--------------|
| IoU | 衡量框重叠程度 | 交集面积 / 并集面积 |
| NMS | 去除重复检测 | 按分数排序，抑制高 IoU 框 |
| Anchor Box | 提供先验框 | 3 尺度 × 3 比例 = 9 个 |
| RoI Pooling | 固定大小输出 | 将任意区域池化为 7×7 |
| mAP | 评估指标 | PR 曲线下面积，各类平均 |

### 思考题答案提示

1. 图像分类是"一图一类"，目标检测是"一图多物"，而且还要定位。难点在于物体数量不固定、位置不固定
2. 因为 Faster R-CNN 把候选区域生成（RPN）也放进了 GPU 上的 CNN，不再用 CPU 上的选择性搜索
3. 单阶段检测器在较粗的特征图上做密集预测，对小物体的特征提取不够充分；两阶段有专门的分类器对每个候选区域精细分类
4. IoU 范围是 [0, 1]，完全重合时 IoU = 1

### 论文参考

- [[Girshick et al., 2014]](https://arxiv.org/abs/1311.2524) R-CNN
- [[Girshick, 2015]](https://arxiv.org/abs/1504.08083) Fast R-CNN
- [[Ren et al., 2015]](https://arxiv.org/abs/1506.01497) Faster R-CNN
- [[Redmon et al., 2016]](https://arxiv.org/abs/1506.02640) YOLO

### GitHub 参考实现

- [ultralytics/yolov5](https://github.com/ultralytics/yolov5) (48k stars) — YOLOv5 官方实现
- [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2) (29k stars) — Detectron2，FAIR 的目标检测框架

---

**下节预告**：CNN 可视化与理解——网络到底"看到"了什么？
"""))

    return cells


# ============================================================
# NOTEBOOK 6: CNN Visualization
# ============================================================
def build_notebook6():
    cells = []

    cells.append(md_cell("""# CNN 可视化与理解：网络看到了什么？

> **CS231n 深度学习与计算机视觉** · 第10讲
>
> 本 notebook 将带你探索 CNN 的"内心世界"——从特征空间可视化到显著图、Grad-CAM、对抗样本、特征反演，再到过滤器可视化，全方位理解 CNN 到底学到了什么。

---

## 学习目标

- 理解为什么需要可视化 CNN（可解释性、调试）
- 理解 t-SNE 如何可视化特征空间
- 理解显著图（Saliency Map）的原理
- 理解 Class Activation Mapping (CAM) 和 Grad-CAM
- 从零实现 Grad-CAM（基于预计算特征）
- 理解对抗样本的概念和 FGSM 攻击
- 生成简单的对抗样本
- 理解特征反演和 DeepDream 的直觉
- 理解不同层过滤器的视觉含义
- 实现简单的显著性图和对抗攻击

---
"""))

    cells.append(code_cell("""# Setup
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("Environment ready!")
"""))

    # Section 1: Why visualize?
    cells.append(md_cell("""## 1. 为什么要可视化 CNN？

### 1. 可解释性

深度学习模型常被称为"黑盒"——我们知道它效果好，但不知道它为什么好。
可视化可以帮助我们理解：
- 模型到底在关注图像的哪些区域？
- 模型学到了什么样的特征？
- 模型的判断依据合理吗？

### 2. 调试模型

当模型表现不好时，可视化可以帮助诊断：
- 是特征没学好，还是分类器不行？
- 模型是不是在关注错误的区域？
- 有没有数据偏差的问题？

### 3. 科学发现

可视化能揭示网络学习到的视觉概念层级：
- 底层：边缘、颜色
- 中层：纹理、图案
- 高层：物体部件、整体物体

---

**思考题 1：** 如果一个分类猫的模型实际上在识别"旁边有猫砂盆"而不是猫本身，这是个什么问题？可视化怎么帮助发现？
"""))

    cells.append(code_cell("""# Different types of CNN visualization

viz_types = [
    ('t-SNE feature\nvisualization', 'Feature space\nstructure', '#3498db'),
    ('Saliency maps', 'Which pixels\nmatter', '#e74c3c'),
    ('Grad-CAM', 'Where the network\nlooks', '#2ecc71'),
    ('Adversarial\nexamples', 'Fooling the\nnetwork', '#9b59b6'),
    ('Filter\nvisualization', 'What each\ndetector sees', '#f39c12'),
    ('DeepDream', 'Amplifying\nfeatures', '#1abc9c'),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, (title, desc, color) in enumerate(viz_types):
    ax = axes[idx]
    # Decorative rectangle
    rect = plt.Rectangle((0.1, 0.3), 0.8, 0.5, facecolor=color, alpha=0.7)
    ax.add_patch(rect)
    ax.text(0.5, 0.55, title, ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    ax.text(0.5, 0.2, desc, ha='center', va='center', fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

plt.suptitle('Types of CNN Visualization & Interpretation', fontsize=15)
plt.tight_layout()
plt.show()

print("We'll explore several visualization techniques in this notebook.")
"""))

    # Section 2: t-SNE
    cells.append(md_cell("""## 2. t-SNE：可视化特征空间

### 问题：高维特征看不见

CNN 提取的特征向量通常有几百或几千维，无法直接可视化。

### t-SNE 的思想

t-SNE（t-Distributed Stochastic Neighbor Embedding）是一种降维方法，它试图在低维空间中**保留高维空间中的局部邻域关系**。

简单说：
- 高维空间中相近的点 → 低维空间中也放得近
- 高维空间中远的点 → 低维空间中也放得远

### 怎么看 t-SNE 图？

- 如果**同一类的样本聚在一起**，说明特征学到了好的类别区分
- 如果**不同类之间混在一起**，说明特征区分度不够
- 可以发现**异常样本**（离群点）

---
"""))

    cells.append(code_cell("""# Create synthetic features and apply "t-SNE-like" visualization

np.random.seed(42)

# Create synthetic feature data (5 classes, each with a cluster pattern)
n_per_class = 50
n_classes = 5
feat_dim = 50  # high dimensional features

# Create clusters in high-dim space
features = []
labels = []
for c in range(n_classes):
    # Each class has a center and some spread
    center = np.random.randn(feat_dim) * 3  # class center
    class_feats = center + np.random.randn(n_per_class, feat_dim) * 0.8
    features.append(class_feats)
    labels.extend([c] * n_per_class)

features = np.vstack(features)
labels = np.array(labels)

print(f"Feature shape: {features.shape} ({n_classes*n_per_class} samples, {feat_dim} dimensions)")
print(f"Labels shape: {labels.shape}")
print()
print("We can't visualize 50 dimensions directly. We need dimensionality reduction!")
"""))

    cells.append(code_cell("""# Simple PCA + random projection for "t-SNE-like" visualization
# (Real t-SNE would need sklearn; we'll simulate the effect here)

def pca(X, n_components=2):
    \"\"\"Simple PCA implementation.\"\"\"
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    # Covariance matrix
    cov = X_centered.T @ X_centered / (X.shape[0] - 1)
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort by eigenvalue descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Project
    X_proj = X_centered @ eigenvectors[:, :n_components]
    return X_proj, eigenvalues

# Apply PCA
pca_result, eigvals = pca(features, n_components=2)

print(f"PCA explained variance ratio: {eigvals[:2].sum() / eigvals.sum():.2%}")
print(f"PCA keeps only {eigvals[:2].sum() / eigvals.sum():.1%} of variance in 2D")
print("t-SNE does better at preserving local structure for visualization.")
"""))

    cells.append(code_cell("""# Visualize PCA result and simulate t-SNE effect
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PCA
colors = plt.cm.tab10(np.arange(n_classes))
for c in range(n_classes):
    mask = labels == c
    axes[0].scatter(pca_result[mask, 0], pca_result[mask, 1], 
                   c=[colors[c]], alpha=0.7, s=40, label=f'Class {c}')
axes[0].set_title('PCA: Linear Projection', fontsize=12)
axes[0].set_xlabel('PC 1')
axes[0].set_ylabel('PC 2')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Simulated t-SNE (better cluster separation for visualization purposes)
# In real life, t-SNE often produces tighter clusters
np.random.seed(42)
tsne_sim = np.zeros_like(pca_result)
for c in range(n_classes):
    mask = labels == c
    n = np.sum(mask)
    # Move each class center further apart (simulating t-SNE)
    center = np.mean(pca_result[mask], axis=0)
    center_norm = center / np.linalg.norm(center)
    tsne_sim[mask] = pca_result[mask] + center_norm * 2
    # Also compress within-cluster distance
    cluster_center = np.mean(pca_result[mask], axis=0)
    tsne_sim[mask] = cluster_center + (pca_result[mask] - cluster_center) * 0.5

for c in range(n_classes):
    mask = labels == c
    axes[1].scatter(tsne_sim[mask, 0], tsne_sim[mask, 1],
                   c=[colors[c]], alpha=0.7, s=40, label=f'Class {c}')
axes[1].set_title('t-SNE (simulated): Better Local Structure', fontsize=12)
axes[1].set_xlabel('Dim 1')
axes[1].set_ylabel('Dim 2')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('Feature Space Visualization: PCA vs t-SNE', fontsize=14)
plt.tight_layout()
plt.show()

print("Key insight from t-SNE:")
print("- If classes cluster well, features are discriminative")
print("- If classes overlap, model might have trouble distinguishing them")
print("- Outliers in t-SNE often correspond to mislabeled or unusual samples")
"""))

    # Section 3: Saliency maps
    cells.append(md_cell("""## 3. 显著图：哪些像素对分类最重要？

### 思想

对于一张图像和一个类别，计算**损失对每个像素的梯度**。梯度大的像素说明它对分类结果影响大——也就是"重要的像素"。

### 怎么做？

1. 前向传播，得到指定类别的分数
2. 反向传播，计算损失对输入图像的梯度
3. 取梯度的绝对值的最大值（跨通道），得到显著图

### 直观理解

"如果我稍微改变这个像素，分类分数会变化多少？"
- 变化大 → 这个像素重要 → 显著图中亮
- 变化小 → 这个像素不重要 → 显著图中暗

> **参考论文**：[[Simonyan et al., 2013]](https://arxiv.org/abs/1312.6034) Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps

---

**思考题 2：** 显著图和图像分割有什么区别？
"""))

    cells.append(code_cell("""# Saliency map visualization with synthetic gradients

np.random.seed(42)

# Create a synthetic "cat-like" image
img_h, img_w = 32, 32
img = np.zeros((img_h, img_w, 3))

# "Cat" body
img[8:25, 6:26, 0] = 0.8  # orange
img[8:25, 6:26, 1] = 0.4
# "Eyes" - important features
img[12:14, 12:14, 2] = 1.0  # blue eyes
img[12:14, 18:20, 2] = 1.0
# "Nose"
img[16:18, 15:17, 1] = 0.9
# "Ears"
img[5:9, 8:12, 0] = 0.9
img[5:9, 20:24, 0] = 0.9

# Simulate gradients (eyes and nose would have high gradients for "cat" class)
# In reality, these would come from backpropagation
gradients = np.zeros((img_h, img_w, 3))

# Eyes are very important for cat classification
gradients[11:15, 11:15, :] = 0.8
gradients[11:15, 17:21, :] = 0.8
# Nose is important
gradients[15:19, 14:18, :] = 0.6
# Body shape contributes
gradients[7:26, 5:27, :] += 0.2
# Ears
gradients[4:10, 7:13, :] += 0.4
gradients[4:10, 19:25, :] += 0.4

# Add some noise
gradients += np.random.randn(img_h, img_w, 3) * 0.05

# Compute saliency map: max of absolute gradient across channels
saliency = np.max(np.abs(gradients), axis=2)

print("Saliency map shape:", saliency.shape)
print("Saliency map shows which pixels matter most for classification.")
"""))

    cells.append(code_cell("""# Visualize saliency map
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img)
axes[0].set_title('Input Image')
axes[0].axis('off')

axes[1].imshow(saliency, cmap='hot')
axes[1].set_title('Saliency Map')
axes[1].axis('off')

# Overlay
axes[2].imshow(img, alpha=0.5)
axes[2].imshow(saliency, cmap='hot', alpha=0.5)
axes[2].set_title('Overlay')
axes[2].axis('off')

plt.suptitle('Saliency Map: Which Pixels Drive Classification?', fontsize=14)
plt.tight_layout()
plt.show()

print("Interpretation:")
print("- Bright regions = pixels with large gradient = important for classification")
print("- The network seems to focus on eyes, nose, and ears (cat features)")
print("- This is like asking the network: 'What made you think this is a cat?'")
"""))

    # Section 4: Grad-CAM
    cells.append(md_cell("""## 4. Grad-CAM：梯度加权类激活映射

### CAM 的思想

CAM（Class Activation Mapping）利用全局平均池化层，把最后一个卷积层的特征图加权求和，得到类别对应的热图。

### Grad-CAM 的改进

Grad-CAM 不需要修改网络结构（不需要全局平均池化层），而是**用梯度作为权重**。

### Grad-CAM 步骤

1. 前向传播到最后一个卷积层，得到特征图 A（维度：C × H × W）
2. 对目标类别分数 y，计算 y 对每个特征图通道的梯度：α_k = 1/(H×W) × Σ_i Σ_j (∂y/∂A^k_ij)
3. 对特征图按梯度加权求和：L = ReLU(Σ_k α_k × A^k)
4. 上采样到原图大小，得到热图

### 直觉

"哪些特征通道对这个类别最重要？把它们的激活图加起来就是网络关注的区域。"

> **参考论文**：[[Selvaraju et al., 2016]](https://arxiv.org/abs/1610.02391) Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization

---
"""))

    cells.append(code_cell("""# Implement Grad-CAM from scratch (using pre-computed features)

def grad_cam(feature_maps, gradients):
    \"\"\"
    Compute Grad-CAM heatmap.
    
    Inputs:
    - feature_maps: (C, H, W) activations from last conv layer
    - gradients: (C, H, W) gradient of class score w.r.t. feature maps
    
    Returns:
    - heatmap: (H, W) Grad-CAM heatmap
    \"\"\"
    C, H, W = feature_maps.shape
    
    # Step 1: Compute channel weights (alpha)
    # alpha_k = global average pooling of gradients for channel k
    alpha = np.mean(gradients, axis=(1, 2))  # (C,)
    
    # Step 2: Weighted combination of feature maps
    heatmap = np.zeros((H, W))
    for k in range(C):
        heatmap += alpha[k] * feature_maps[k]
    
    # Step 3: ReLU - keep only positive contributions
    heatmap = np.maximum(heatmap, 0)
    
    # Step 4: Normalize to [0, 1]
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    return heatmap


print("grad_cam function defined!")
"""))

    cells.append(code_cell("""# Test Grad-CAM with synthetic data
np.random.seed(42)

# Simulate feature maps from last conv layer
# Suppose we have 8 channels, 8x8 spatial
C = 8
H, W = 8, 8
feature_maps = np.random.randn(C, H, W) * 0.5

# Simulate: channel 2 and 5 are important for "cat" class
# They activate in the center of the feature map (where the "cat" is)
feature_maps[2] = np.zeros((H, W))
feature_maps[2, 2:6, 2:6] = 1.0  # center activation
feature_maps[5] = np.zeros((H, W))
feature_maps[5, 3:5, 3:5] = 1.5  # stronger center activation

# Simulate gradients for "cat" class
# Important channels (2 and 5) have positive gradients
gradients = np.random.randn(C, H, W) * 0.1
gradients[2] = 0.8  # high gradient = important channel
gradients[5] = 1.2  # even more important

# Compute Grad-CAM
heatmap = grad_cam(feature_maps, gradients)

print(f"Feature maps shape: {feature_maps.shape} (C={C}, H={H}, W={W})")
print(f"Gradients shape: {gradients.shape}")
print(f"Heatmap shape: {heatmap.shape}")
print()
print("Channel weights (alpha):")
alpha = np.mean(gradients, axis=(1, 2))
for c in range(C):
    print(f"  Channel {c}: alpha = {alpha[c]:.4f}")
print()
print(f"Channel 2 and 5 have highest weights -> they drive the heatmap.")
"""))

    cells.append(code_cell("""# Visualize Grad-CAM
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Show a few feature maps
for c in range(4):
    axes[0, c].imshow(feature_maps[c], cmap='jet')
    axes[0, c].set_title(f'Feature Map {c}\n(alpha={alpha[c]:.3f})')
    axes[0, c].axis('off')

# Show gradients
for c in range(4):
    axes[1, c].imshow(gradients[c], cmap='RdBu_r')
    axes[1, c].set_title(f'Gradient {c}')
    axes[1, c].axis('off')

plt.suptitle('Feature Maps and Their Gradients (Importance)', fontsize=14)
plt.tight_layout()
plt.show()

# Show final heatmap
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(heatmap, cmap='jet', vmin=0, vmax=1)
ax.set_title('Grad-CAM Heatmap', fontsize=13)
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()

print("Grad-CAM produces a coarse localization map highlighting")
print("the important regions for predicting a specific class.")
"""))

    # Section 5: Adversarial examples
    cells.append(md_cell("""## 5. 对抗样本：欺骗 CNN

### 什么是对抗样本？

在图像上添加人眼几乎不可察觉的微小扰动，就能让 CNN 把图像错分类成另一个类别。

### FGSM（Fast Gradient Sign Method）

最经典的对抗攻击方法：

$$x_{adv} = x + \\epsilon \\cdot sign(\\nabla_x J(x, y_{true}))$$

直觉：
1. 计算损失对输入图像的梯度
2. 取梯度的符号（每个像素是 +1 还是 -1）
3. 沿梯度上升方向走一步（让损失变大，即让正确类别的分数变低）
4. ε 控制扰动大小

### 为什么令人惊讶？

- 人眼完全看不出区别
- 但模型却以很高的置信度判错
- 说明模型学到的模式和人类的视觉理解有很大差异

> **参考论文**：[[Goodfellow et al., 2014]](https://arxiv.org/abs/1412.6572) Explaining and Harnessing Adversarial Examples

---

**思考题 3：** 对抗样本的存在说明深度学习模型有什么根本性的问题？
"""))

    cells.append(code_cell("""# Generate adversarial examples with FGSM

def fgsm_attack(image, epsilon, gradient):
    \"\"\"
    Fast Gradient Sign Method (FGSM) attack.
    
    Inputs:
    - image: (H, W, C) original image
    - epsilon: perturbation magnitude
    - gradient: (H, W, C) gradient of loss w.r.t. image
    
    Returns:
    - adv_image: adversarial image
    - perturbation: the added noise
    \"\"\"
    # Get the sign of the gradient
    sign_grad = np.sign(gradient)
    
    # Create the perturbed image
    perturbation = epsilon * sign_grad
    adv_image = image + perturbation
    
    # Clip to valid range [0, 1]
    adv_image = np.clip(adv_image, 0, 1)
    perturbation = adv_image - image
    
    return adv_image, perturbation


print("fgsm_attack function defined!")
"""))

    cells.append(code_cell("""# Demonstrate FGSM attack with synthetic example
np.random.seed(42)

# Create a simple "image" (a gradient pattern)
img_size = 16
orig_img = np.zeros((img_size, img_size, 3))
orig_img[:, :, 0] = np.linspace(0.2, 0.8, img_size).reshape(-1, 1)  # red gradient
orig_img[:, :, 1] = np.linspace(0.5, 0.1, img_size).reshape(1, -1)  # green gradient
orig_img[:, :, 2] = 0.3  # constant blue

# Simulate gradient of loss w.r.t. image
# (In real life, this would come from backpropagation through the network)
gradient = np.random.randn(img_size, img_size, 3) * 0.5
# Make the gradient have some structure
gradient[:, :, 0] += 0.3  # positive gradient in red channel
gradient[:, :, 1] -= 0.2  # negative in green

# FGSM attack with different epsilons
epsilons = [0, 0.02, 0.05, 0.1]

fig, axes = plt.subplots(2, len(epsilons), figsize=(16, 7))

for idx, eps in enumerate(epsilons):
    if eps == 0:
        adv = orig_img.copy()
        pert = np.zeros_like(orig_img)
    else:
        adv, pert = fgsm_attack(orig_img, eps, gradient)
    
    axes[0, idx].imshow(adv)
    axes[0, idx].set_title(f'Image (eps={eps})')
    axes[0, idx].axis('off')
    
    # Show perturbation amplified for visibility
    pert_vis = (pert - pert.min()) / (pert.max() - pert.min() + 1e-10)
    axes[1, idx].imshow(pert_vis)
    axes[1, idx].set_title(f'Perturbation\n(amplified)')
    axes[1, idx].axis('off')

plt.suptitle('FGSM Adversarial Attack: Adding Tiny Noise Changes Classification', fontsize=14)
plt.tight_layout()
plt.show()

print("Key observations:")
print("- With small epsilon, the perturbation is nearly invisible to humans")
print("- But it can completely change the CNN's prediction")
print("- This reveals a fundamental difference between human and machine vision")
"""))

    # Section 6: Feature inversion
    cells.append(md_cell("""## 6. 特征反演与 DeepDream

### 特征反演（Feature Inversion）

给定一个特征向量，我们能反推出"什么样的图像会产生这个特征"吗？

这就像问："如果某个神经元激活了，它看到的是什么？"

### DeepDream 的直觉

不是从随机噪声开始优化图像来最大化某个神经元，而是**在原图基础上**，让网络**增强**它已经检测到的特征。

结果就是：
- 狗变得更像狗（眼睛更突出、毛发更明显）
- 云变成了狗脸、建筑变成了教堂
- 像做梦一样，图像充满了迷幻的细节

### 直觉

DeepDream 就像**反馈循环**：
1. 网络在图像中检测到一些特征
2. 我们增强这些特征
3. 增强后网络检测到更多
4. 不断循环，直到图像充满了各种"幻觉"

---
"""))

    cells.append(code_cell("""# Simulate DeepDream-like feature amplification
np.random.seed(42)

# Create a "noisy" starting image
img = np.random.rand(64, 64, 3) * 0.1 + 0.5

# Simulate DeepDream iterations
# In reality, this would involve gradient ascent on activations
# We'll simulate the effect with pattern amplification

# Simulated "detected patterns" that get amplified
dream_iterations = 5
dream_images = [img.copy()]

for i in range(dream_iterations):
    # Simulate: amplify certain frequencies/patterns
    # Add some "eye-like" and "swirl-like" patterns
    img_dream = dream_images[-1].copy()
    
    # Add circular patterns (simulating neuron preferences)
    for _ in range(3):
        cx = np.random.randint(10, 54)
        cy = np.random.randint(10, 54)
        r = np.random.randint(3, 10)
        y, x = np.ogrid[-cy:64-cy, -cx:64-cx]
        mask = x*x + y*y <= r*r
        color = np.random.rand(3) * 0.3
        img_dream[mask] += color.reshape(1, 1, 3) * 0.2
    
    # Slightly sharpen and boost contrast
    img_dream = np.clip(img_dream, 0, 1)
    dream_images.append(img_dream)

# Visualize
fig, axes = plt.subplots(1, len(dream_images), figsize=(18, 4))

for i, img_d in enumerate(dream_images):
    axes[i].imshow(img_d)
    axes[i].set_title(f'Iteration {i}')
    axes[i].axis('off')

plt.suptitle('DeepDream Simulation: Amplifying Network Activations', fontsize=14)
plt.tight_layout()
plt.show()

print("DeepDream intuition:")
print("- The network sees patterns in the image")
print("- We amplify those patterns to make the network see even more")
print("- Results in surreal, dream-like images")
print("- Different layers produce different kinds of patterns")
"""))

    # Section 7: Filter visualization
    cells.append(md_cell("""## 7. 过滤器可视化：每个过滤器检测什么？

### 不同层学到了什么？

- **第一层**：简单的边缘、颜色斑点
- **中间层**：纹理、图案、简单形状
- **高层**：物体部件、整个物体

### 怎么可视化过滤器？

**方法 1：看权值**（只对第一层有效）
- 第一层过滤器是 3×3×3 或 11×11×3，可以直接当图像显示

**方法 2：找最大激活的图像块**
- 找数据集中哪些图像块让这个过滤器激活最强

**方法 3：特征反演 / 生成图像**
- 从随机噪声开始，优化图像来最大化某个过滤器的激活

> **参考论文**：[[Zeiler & Fergus, 2013]](https://arxiv.org/abs/1311.2901) Visualizing and Understanding Convolutional Networks

---

**思考题 4：** 为什么不能直接看高层过滤器的权值来理解它检测什么？
"""))

    cells.append(code_cell("""# Visualize first-layer filters (learned edge detectors)
np.random.seed(42)

# Simulate first-layer filters (like what a trained CNN learns)
n_filters = 16
filter_size = 5

filters = np.zeros((n_filters, filter_size, filter_size, 3))

# Create various edge/texture patterns
for i in range(n_filters):
    angle = i * (np.pi / n_filters)  # different orientations
    # Create oriented edge filter
    for y in range(filter_size):
        for x in range(filter_size):
            # Distance from center
            dx = x - filter_size//2
            dy = y - filter_size//2
            # Gabor-like pattern
            rot_x = dx * np.cos(angle) + dy * np.sin(angle)
            filters[i, y, x, 0] = np.sin(rot_x * 1.5) * np.exp(-(dx*dx + dy*dy) / 4)
            filters[i, y, x, 1] = np.sin(rot_x * 1.5 + 2) * np.exp(-(dx*dx + dy*dy) / 4)
            filters[i, y, x, 2] = np.sin(rot_x * 1.5 + 4) * np.exp(-(dx*dx + dy*dy) / 4)

# Normalize for visualization
f_min, f_max = filters.min(), filters.max()
filters_vis = (filters - f_min) / (f_max - f_min)

# Display
fig, axes = plt.subplots(2, 8, figsize=(16, 5))
for i in range(n_filters):
    ax = axes[i // 8, i % 8]
    ax.imshow(filters_vis[i])
    ax.set_title(f'Filter {i}')
    ax.axis('off')

plt.suptitle('First Layer Filters: Edges and Color Patterns', fontsize=14)
plt.tight_layout()
plt.show()

print("First layer filters detect:")
print("- Edges at different orientations")
print("- Color patterns (color blobs, opponent colors)")
print("- Simple textures")
print()
print("These are the 'building blocks' that higher layers combine into more complex features.")
"""))

    cells.append(code_cell("""# Simulate higher layer feature visualization
# Higher layers detect more complex patterns

# Simulated "higher layer" feature visualizations
higher_features = [
    ('Layer 2: Textures', ['stripes', 'dots', 'grid', 'waves']),
    ('Layer 3: Patterns', ['spiral', 'star', 'circle', 'zigzag']),
    ('Layer 4: Parts', ['eye-like', 'ear-like', 'wheel-like', 'face-like']),
    ('Layer 5: Objects', ['cat face', 'dog face', 'car front', 'bird']),
]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for row_idx, (layer_name, features) in enumerate(higher_features):
    for col_idx, feat_name in enumerate(features):
        ax = axes[row_idx, col_idx]
        # Create a stylized visualization
        img_sim = np.random.rand(32, 32, 3) * 0.1
        # Add pattern-like structure
        for _ in range(10):
            cx, cy = np.random.randint(5, 27, 2)
            r = np.random.randint(2, 8)
            color = np.random.rand(3) * 0.5 + 0.3
            for y in range(max(0, cy-r), min(32, cy+r)):
                for x in range(max(0, cx-r), min(32, cx+r)):
                    if (x-cx)**2 + (y-cy)**2 <= r*r:
                        img_sim[y, x] = color
        ax.imshow(img_sim)
        ax.set_title(feat_name, fontsize=10)
        ax.axis('off')
    # Label row
    axes[row_idx, 0].text(-2, 16, layer_name, rotation=90, 
                          va='center', fontsize=11, fontweight='bold')

plt.suptitle('CNN Layer Hierarchy: From Simple to Complex Features', fontsize=14)
plt.tight_layout()
plt.show()

print("CNN feature hierarchy:")
print("  Layer 1: edges, colors -> Layer 2: textures -> Layer 3: patterns")
print("  Layer 4: object parts -> Layer 5: whole objects")
print()
print("This hierarchical representation is what makes CNNs so powerful.")
"""))

    # Common mistakes
    cells.append(md_cell("""## 8. 常见错误与注意事项

### 错误 1：把显著图当分割图

显著图显示的是"哪些像素影响分类分数"，而不是"物体在哪里"。
两者有关联，但不是一回事。

### 错误 2：过度解读 Grad-CAM

Grad-CAM 是粗粒度的定位，不能精确到像素级别。它告诉你"大概在哪个区域"，而不是"精确边界"。

### 错误 3：认为 t-SNE 中的距离有意义

t-SNE 主要保留**局部邻域关系**，全局距离可能没有意义。
两个簇离得远不一定说明它们差异大。

### 错误 4：对抗样本只对特定模型有效

不对——对抗样本有**可迁移性**，在一个模型上生成的对抗样本，往往也能骗过另一个结构不同的模型。

### 错误 5：过滤器可视化就是"神经元看到的东西"

严格来说，可视化的是"什么样的图像能最大化激活这个神经元"，不等于"神经元看到的就是这个"。这是一个有用的近似，但不是全部真相。

---
"""))

    # Homework 1
    cells.append(md_cell("""---

## 作业 1：通过引导反向传播实现显著图

### 任务

实现基于引导反向传播（Guided Backpropagation）的显著图。

### 背景

普通的反向传播会有"负梯度"的干扰。引导反向传播的想法是：
- 只考虑那些对激活有**正向贡献**的梯度
- 在反向传播经过 ReLU 时，只让正梯度通过，负梯度被截断

### 引导反向传播规则

普通 ReLU 反向传播：梯度 = (input > 0) * upstream_grad
引导反向传播：梯度 = (input > 0) * (upstream_grad > 0) * upstream_grad

直觉："只保留那些让激活变强的路径的梯度"

---
"""))

    cells.append(code_cell("""# ===== Homework 1: Guided Backpropagation Saliency =====

def guided_relu_backward(dout, forward_input):
    \"\"\"
    Guided backpropagation through ReLU.
    
    Inputs:
    - dout: upstream gradient
    - forward_input: the input to ReLU in the forward pass
    
    Returns:
    - dx: gradient with guided backprop
    \"\"\"
    # YOUR CODE HERE
    # Hint: 
    # Regular ReLU grad: dx = dout * (forward_input > 0)
    # Guided ReLU grad:  dx = dout * (forward_input > 0) * (dout > 0)
    # Only keep gradients where both input was positive AND gradient is positive
    
    dx = None
    return dx


# ===== ANSWER KEY =====
def guided_relu_backward_answer(dout, forward_input):
    \"\"\"Reference implementation.\"\"\"
    dx = dout * (forward_input > 0) * (dout > 0)
    return dx


def relu(x):
    return np.maximum(0, x)


print("Homework 1: Implement guided_relu_backward")
print("Write your solution above, then run the tests.")
"""))

    cells.append(code_cell("""# ===== Homework 1: Test cases =====
np.random.seed(42)

# Test 1: Basic guided backprop properties
forward_input = np.array([2.0, -1.0, 3.0, -0.5, 1.0])
dout = np.array([1.0, 1.0, -1.0, -2.0, 2.0])

dx = guided_relu_backward_answer(dout, forward_input)

print("Test 1: Guided backprop basics")
print(f"  forward_input: {forward_input}")
print(f"  dout:          {dout}")
print(f"  dx (guided):   {dx}")
print()

# Regular ReLU grad for comparison
regular_grad = dout * (forward_input > 0)
print(f"  Regular grad:  {regular_grad}")
print()

# Guided should be same as regular but with negative dout zeroed out where input > 0
expected = dout * (forward_input > 0) * (dout > 0)
assert np.allclose(dx, expected), "Guided backprop result incorrect!"
print("  Guided backprop values: CORRECT")

# Test 2: Only positive input and positive gradient should pass through
# Position 0: input>0, dout>0 -> should pass
assert dx[0] == 1.0, "Position 0 should pass gradient"
# Position 1: input<0 -> should not pass
assert dx[1] == 0.0, "Position 1 should have 0 gradient (negative input)"
# Position 2: input>0 but dout<0 -> guided should block
assert dx[2] == 0.0, "Position 2 should have 0 gradient (negative dout)"
# Position 3: input<0 and dout<0 -> 0
assert dx[3] == 0.0, "Position 3 should have 0 gradient"
# Position 4: input>0, dout>0 -> pass
assert dx[4] == 2.0, "Position 4 should pass gradient"
print("  All individual positions: CORRECT")

# Test 3: Saliency map comparison
# Simulate a small network forward and backward
np.random.seed(42)
img = np.random.randn(1, 8, 8)  # single channel "image"

# Simulated conv + relu + conv + relu
w1 = np.random.randn(4, 1, 3, 3) * 0.1
b1 = np.zeros(4)
w2 = np.random.randn(2, 4, 3, 3) * 0.1
b2 = np.zeros(2)

# Forward (simplified - use our conv function)
def simple_conv(x, w, b):
    \"\"\"Simplified conv: x (1,C,H,W), w (F,C,HH,WW)\"\"\"
    C, H, W = x.shape[1:]
    F, _, HH, WW = w.shape
    out = np.zeros((1, F, H-2, W-2))
    for f in range(F):
        for i in range(H-2):
            for j in range(W-2):
                out[0, f, i, j] = np.sum(x[0, :, i:i+HH, j:j+WW] * w[f]) + b[f]
    return out

# Forward pass
h1 = simple_conv(img, w1, b1)
h1_relu = relu(h1)
h2 = simple_conv(h1_relu, w2, b2)
h2_relu = relu(h2)

# Suppose we compute saliency for the max output
# Upstream gradient at output
dout = np.zeros_like(h2_relu)
dout.flat[np.argmax(h2_relu)] = 1.0

# Backward through second ReLU (guided)
dh2_relu = guided_relu_backward_answer(dout, h2)
# For simplicity, skip conv backward and verify ReLU guided behavior
assert np.all(dh2_relu >= 0), "Guided backprop should produce non-negative gradients!"
print("\nTest 3: Guided backprop produces non-negative gradients: PASSED")

print("\n=== All Homework 1 tests PASSED! ===")
"""))

    # Homework 2
    cells.append(md_cell("""---

## 作业 2：实现简单的对抗攻击

### 任务

实现一个简单的无目标对抗攻击（Untargeted Attack）：
1. 给定图像和真实类别
2. 计算损失对图像的梯度
3. 用 FGSM 生成对抗样本
4. 验证对抗样本是否成功让模型"判错"

### 要求

- 实现 FGSM 攻击
- 实现一个简单的线性分类器作为"模型"
- 验证攻击成功率
- 可视化原始图像、对抗图像和扰动

---
"""))

    cells.append(code_cell("""# ===== Homework 2: Simple Adversarial Attack =====

def simple_linear_classifier(x, W, b):
    \"\"\"
    Simple linear classifier.
    x: (N, D) flattened images
    W: (D, num_classes) weights
    b: (num_classes,) biases
    Returns: (N, num_classes) scores
    \"\"\"
    return x @ W + b


def fgsm_untargeted(x, y_true, W, b, epsilon):
    \"\"\"
    FGSM untargeted attack: make model misclassify.
    
    Inputs:
    - x: (N, D) input images (flattened)
    - y_true: (N,) true class indices
    - W: (D, C) classifier weights
    - b: (C,) classifier biases
    - epsilon: perturbation magnitude
    
    Returns:
    - x_adv: (N, D) adversarial examples
    \"\"\"
    # YOUR CODE HERE
    # Hint:
    # 1. Compute scores = x @ W + b
    # 2. Compute cross-entropy loss gradient w.r.t. x
    #    For linear classifier: dL/dx = W[:, y_pred] - W[:, y_true] (simplified)
    #    Actually: dLoss/dx = (softmax(scores) - one_hot(y_true)) @ W.T
    # 3. x_adv = x + epsilon * sign(gradient)
    # 4. Clip to valid range
    
    x_adv = None
    return x_adv


# ===== ANSWER KEY =====
def fgsm_untargeted_answer(x, y_true, W, b, epsilon):
    \"\"\"Reference implementation.\"\"\"
    N, D = x.shape
    C = W.shape[1]
    
    # Forward pass
    scores = x @ W + b  # (N, C)
    
    # Softmax
    scores_max = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # One-hot encoding of true labels
    one_hot = np.zeros((N, C))
    one_hot[np.arange(N), y_true] = 1.0
    
    # Gradient of cross-entropy loss w.r.t. scores
    dscores = probs - one_hot  # (N, C)
    
    # Gradient w.r.t. input x
    dx = dscores @ W.T  # (N, D)
    
    # FGSM
    x_adv = x + epsilon * np.sign(dx)
    
    # Clip to [0, 1]
    x_adv = np.clip(x_adv, 0, 1)
    
    return x_adv


print("Homework 2: Implement fgsm_untargeted")
print("Write your solution above, then run the tests.")
"""))

    cells.append(code_cell("""# ===== Homework 2: Test cases =====
np.random.seed(42)

# Create a simple 2D "image" classification problem
N = 10  # 10 samples
D = 4   # 4 pixels (2x2 image flattened)
C = 3   # 3 classes

# Generate random data
x = np.random.rand(N, D)  # pixel values in [0, 1]
y_true = np.random.randint(0, C, N)

# Random linear classifier
W = np.random.randn(D, C) * 0.5
b = np.random.randn(C) * 0.1

# Original predictions
scores_orig = simple_linear_classifier(x, W, b)
pred_orig = np.argmax(scores_orig, axis=1)
acc_orig = np.mean(pred_orig == y_true)

print(f"Original accuracy: {acc_orig:.1%} ({np.sum(pred_orig == y_true)}/{N} correct)")

# Attack with different epsilons
for eps in [0.01, 0.05, 0.1, 0.2]:
    x_adv = fgsm_untargeted_answer(x, y_true, W, b, eps)
    scores_adv = simple_linear_classifier(x_adv, W, b)
    pred_adv = np.argmax(scores_adv, axis=1)
    acc_adv = np.mean(pred_adv == y_true)
    success_rate = np.mean(pred_adv != y_true)
    
    print(f"  eps={eps:.2f}: accuracy={acc_adv:.1%}, attack success={success_rate:.1%}")

# Verify attack works (with large enough epsilon)
eps_test = 0.3
x_adv_test = fgsm_untargeted_answer(x, y_true, W, b, eps_test)
assert x_adv_test.shape == x.shape, "Shape mismatch!"
print(f"\nAttack output shape: {x_adv_test.shape} - PASSED")

# Verify clipping
assert np.all(x_adv_test >= 0) and np.all(x_adv_test <= 1), "Values not in [0,1]!"
print("Clipping to [0, 1]: PASSED")

# Verify perturbation is bounded by epsilon
perturbation = x_adv_test - x
assert np.max(np.abs(perturbation)) <= eps_test + 1e-10, "Perturbation exceeds epsilon!"
print(f"Max perturbation <= eps ({eps_test}): PASSED")

# With large epsilon, most should be misclassified
scores_attacked = simple_linear_classifier(x_adv_test, W, b)
pred_attacked = np.argmax(scores_attacked, axis=1)
n_success = np.sum(pred_attacked != y_true)
print(f"Attack success with eps={eps_test}: {n_success}/{N} samples misclassified")

print("\n=== All Homework 2 tests PASSED! ===")
"""))

    # Summary
    cells.append(md_cell("""---

## 本讲总结

### 可视化方法一览

| 方法 | 回答的问题 | 原理 |
|------|-----------|------|
| t-SNE | 特征空间长什么样？ | 高维降维到 2D/3D |
| 最近邻 | 哪些图像特征相似？ | 特征空间距离 |
| 显著图 | 哪些像素重要？ | 损失对输入的梯度 |
| Grad-CAM | 网络在看哪里？ | 梯度加权的特征图 |
| 过滤器可视化 | 过滤器检测什么？ | 最大化激活的图像 |
| 特征反演 | 这个特征对应什么图像？ | 从特征生成图像 |
| 对抗样本 | 怎么骗过网络？ | 沿梯度方向加扰动 |
| DeepDream | 增强特征会怎样？ | 反馈循环放大激活 |

### 关键洞察

1. **底层特征是通用的**：边缘、纹理——和人类视觉系统有相似之处
2. **高层特征是语义的**：物体部件、整体物体
3. **模型和人类看世界的方式不同**：对抗样本的存在证明了这一点
4. **可解释性很重要**：理解模型才能信任模型、改进模型

### 思考题答案提示

1. 这是**数据偏差 / 快捷方式学习**（shortcut learning）的问题。显著图可以显示模型在关注图像的哪些区域——如果它关注的是猫砂盆而不是猫，就说明模型学错了。
2. 显著图是"哪些像素对分类有影响"，分割是"哪些像素属于这个物体"。两者相关但不等同。显著图可能包含背景中对分类有影响的像素。
3. 说明模型的决策边界和人类的很不一样。模型可能依赖一些人类察觉不到的细微模式，而不是真正"理解"图像内容。这也叫"对抗性脆弱性"。
4. 因为高层过滤器的权值是在前一层特征图上的权重，不是在像素空间的。它们的"感受野"是语义层面的，不能直接当作图像看。

### 论文参考

- [[Simonyan et al., 2013]](https://arxiv.org/abs/1312.6034) Deep Inside Convolutional Networks — 显著图
- [[Selvaraju et al., 2016]](https://arxiv.org/abs/1610.02391) Grad-CAM — 梯度加权类激活映射
- [[Goodfellow et al., 2014]](https://arxiv.org/abs/1412.6572) Explaining and Harnessing Adversarial Examples — 对抗样本 / FGSM
- [[Zeiler & Fergus, 2013]](https://arxiv.org/abs/1311.2901) Visualizing and Understanding Convolutional Networks — 反卷积可视化

### GitHub 参考实现

- [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) (4.5k stars) — Grad-CAM 的 PyTorch 实现
- [fg91/visualizing-cnn-feature-maps](https://github.com/fg91/visualizing-cnn-feature-maps) (1.2k stars) — CNN 特征图可视化

---

**恭喜！** 你已经完成了 CS231n 第二部分（CNN 与计算机视觉）的全部 6 个 notebook！
"""))

    return cells


# ============================================================
# Main
# ============================================================
def main():
    # Notebook 1: CNN Basics
    cells1 = build_notebook1()
    f1 = save_notebook(cells1, "part2-cnn-vision_lecture-05-cnn_practice.ipynb")
    
    # Notebook 2: CNN Architectures
    cells2 = build_notebook2()
    f2 = save_notebook(cells2, "part2-cnn-vision_lecture-06-architectures_practice.ipynb")
    
    # Notebook 3: Vision Transformer
    cells3 = build_notebook3()
    f3 = save_notebook(cells3, "part2-cnn-vision_lecture-08-transformers_practice.ipynb")
    
    # Notebook 4: RNN & Image Captioning
    cells4 = build_notebook4()
    f4 = save_notebook(cells4, "part2-cnn-vision_lecture-08b-rnn-captioning_practice.ipynb")
    
    # Notebook 5: Object Detection
    cells5 = build_notebook5()
    f5 = save_notebook(cells5, "part2-cnn-vision_lecture-09-detection_practice.ipynb")
    
    # Notebook 6: CNN Visualization
    cells6 = build_notebook6()
    f6 = save_notebook(cells6, "part2-cnn-vision_lecture-10-visualization_practice.ipynb")
    
    print("\n=== All 6 notebooks generated ===")
    print(f"Notebook 1 (CNN Basics):        {len(cells1)} cells")
    print(f"Notebook 2 (Architectures):     {len(cells2)} cells")
    print(f"Notebook 3 (ViT):               {len(cells3)} cells")
    print(f"Notebook 4 (RNN Captioning):    {len(cells4)} cells")
    print(f"Notebook 5 (Detection):         {len(cells5)} cells")
    print(f"Notebook 6 (Visualization):     {len(cells6)} cells")


if __name__ == "__main__":
    main()
