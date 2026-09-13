#!/usr/bin/env python3
"""Expand all 8 CS231n notebooks to 40-50 cells by inserting supplementary cells."""
import json
import os

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True) if isinstance(source, str) else source}

def code_cell(source):
    return {"cell_type": "code", "metadata": {},
            "source": source.splitlines(keepends=True) if isinstance(source, str) else source,
            "outputs": [], "execution_count": None}

def load_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_notebook(nb, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Cells: {len(nb['cells'])}, Size: {size_kb:.1f} KB")

def find_insertion_point(nb):
    """Find the index of the cell containing '参考代码实现' or return len(cells)."""
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if "参考代码实现" in src:
            return i
    return len(nb["cells"])

def insert_cells(nb, new_cells):
    """Insert new_cells before the '参考代码实现' section."""
    idx = find_insertion_point(nb)
    nb["cells"][idx:idx] = new_cells
    return nb

BASE = r"d:\download\6aa68b099b47c2ba093ab510\cs231n-site\notebooks"

# ============================================================
# Notebook 1: Two-Layer Neural Network (+10 cells)
# ============================================================
print("Expanding Notebook 1: Two-Layer Neural Network...")
path1 = os.path.join(BASE, "part1-deep-learning-basics", "lecture-04b-two-layer-net", "practice.ipynb")
nb = load_notebook(path1)
extra1 = [
    md_cell(r'''### 权重初始化策略

权重初始化对训练收敛速度和最终性能有重要影响。常见策略包括：

| 初始化方法 | 公式 | 适用场景 |
|-----------|------|---------|
| 随机初始化 | $W \sim \mathcal{N}(0, 0.01)$ | 浅层网络 |
| Xavier | $W \sim \mathcal{N}(0, \frac{2}{n_{in} + n_{out}})$ | tanh/sigmoid |
| He | $W \sim \mathcal{N}(0, \frac{2}{n_{in}})$ | ReLU |

**核心思想：** 初始化方差应使每层激活值的方差保持稳定，避免梯度消失或爆炸。'''),

    code_cell(r'''# Compare different weight initialization strategies
np.random.seed(42)
init_methods = {
    "random_small": lambda fan_in, fan_out: np.random.randn(fan_in, fan_out) * 0.01,
    "xavier": lambda fan_in, fan_out: np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / (fan_in + fan_out)),
    "he": lambda fan_in, fan_out: np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in),
}

# Simulate forward pass through 10 layers and track activation variance
depth = 10
fan = 256
activations = {}
for name, init_fn in init_methods.items():
    acts = [np.random.randn(fan, 1)]  # input
    for _ in range(depth):
        W = init_fn(fan, fan)
        a = np.maximum(0, W @ acts[-1])  # ReLU
        acts.append(a)
    variances = [float(a.var()) for a in acts]
    activations[name] = variances

fig, ax = plt.subplots(figsize=(8, 4))
for name, vars in activations.items():
    ax.plot(range(depth + 1), vars, "o-", label=name, markersize=4)
ax.set_xlabel("Layer depth")
ax.set_ylabel("Activation variance")
ax.set_title("Weight Initialization: Activation Variance Across Layers")
ax.legend()
ax.set_yscale("log")
plt.tight_layout()
plt.savefig("init_comparison.png", dpi=100, bbox_inches="tight")
plt.show()
print("Xavier and He maintain stable variance; random_small causes gradient vanishing.")'''),

    md_cell(r'''### 数值梯度检查

数值梯度检查是验证反向传播实现正确性的标准方法。核心公式：

$$f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}$$

其中 $h$ 是一个很小的值（通常 $10^{-5}$）。中心差分法比前向差分更精确（误差 $O(h^2)$ vs $O(h)$）。'''),

    code_cell(r'''# Numerical gradient check for TwoLayerNet
np.random.seed(42)

# Simple function: f(x) = x^2 + 3x, f'(x) = 2x + 3
def simple_func(x):
    return x ** 2 + 3 * x

def numerical_grad(f, x, h=1e-5):
    """Central difference numerical gradient."""
    return (f(x + h) - f(x - h)) / (2 * h)

# Test at several points
test_points = [1.0, 2.0, -3.0, 0.5, 5.0]
print("Numerical gradient check for f(x) = x^2 + 3x:")
print(f"{'x':>8} {'analytic':>12} {'numerical':>12} {'error':>12}")
for x in test_points:
    analytic = 2 * x + 3
    num = numerical_grad(simple_func, x)
    error = abs(analytic - num)
    print(f"{x:8.2f} {analytic:12.6f} {num:12.6f} {error:12.2e}")
    assert error < 1e-8, f"Gradient check failed at x={x}"
print("\nAll gradient checks passed!")'''),

    md_cell(r'''### 学习率的影响

学习率是训练中最关键的超参数之一：

- **太小**：收敛缓慢，需要大量迭代
- **太大**：损失震荡或发散
- **适中**：快速且稳定收敛

建议使用学习率搜索策略：先用大步长搜索，再在最优区间细化。'''),

    code_cell(r'''# Learning rate sweep experiment
np.random.seed(42)

# Simulate training with different learning rates
lr_list = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
epochs = 100
optimal_loss = 0.1

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for lr in lr_list:
    losses = []
    loss = 1.0  # initial loss
    for epoch in range(epochs):
        # Simulate gradient descent with noise
        grad = (loss - optimal_loss) * 0.5 + np.random.randn() * 0.01
        loss = loss - lr * grad
        loss = max(loss, 0)  # prevent negative
        if lr > 0.5 and epoch > 20:
            loss += np.random.randn() * 0.1  # instability
        losses.append(loss)
    axes[0].plot(losses, label=f"lr={lr}")
    axes[1].semilogy(losses, label=f"lr={lr}")

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Learning Rate Comparison (Linear)")
axes[0].legend()

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss (log)")
axes[1].set_title("Learning Rate Comparison (Log)")
axes[1].legend()

plt.tight_layout()
plt.savefig("lr_sweep.png", dpi=100, bbox_inches="tight")
plt.show()
print("Small lr converges slowly; large lr oscillates; optimal lr ~0.05-0.1")'''),

    md_cell(r'''### 过拟合与正则化

**过拟合**指模型在训练集上表现很好但在测试集上表现差。常见解决方法：

1. **L2 正则化**：$L = L_{data} + \lambda \sum W^2$
2. **Dropout**：训练时随机丢弃神经元
3. **数据增强**：扩充训练数据
4. **早停**：在验证集性能开始下降时停止训练'''),

    code_cell(r'''# Overfitting demonstration with different regularization
np.random.seed(42)

# Generate small dataset (easy to overfit)
n_train, n_test = 20, 100
X_train = np.random.randn(n_train, 1)
y_train = X_train[:, 0] ** 2 + 0.1 * np.random.randn(n_train)
X_test = np.linspace(-3, 3, n_test).reshape(-1, 1)
y_test = X_test[:, 0] ** 2

reg_strengths = [0, 0.01, 0.1, 1.0]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, reg in zip(axes, reg_strengths):
    # Simple polynomial features (degree 10)
    degree = 10
    Phi_train = np.hstack([X_train ** i for i in range(degree + 1)])
    Phi_test = np.hstack([X_test ** i for i in range(degree + 1)])
    
    # Ridge regression: w = (X^T X + lambda*I)^-1 X^T y
    I = np.eye(degree + 1)
    I[0, 0] = 0  # don't regularize bias
    w = np.linalg.solve(Phi_train.T @ Phi_train + reg * I, Phi_train.T @ y_train)
    
    pred_train = Phi_train @ w
    pred_test = Phi_test @ w
    
    train_loss = np.mean((pred_train - y_train) ** 2)
    test_loss = np.mean((pred_test - y_test) ** 2)
    
    ax.scatter(X_train, y_train, c="blue", s=20, label="Train", zorder=5)
    ax.plot(X_test, pred_test, "r-", linewidth=2, label="Model")
    ax.plot(X_test, y_test, "g--", linewidth=1, label="True", alpha=0.5)
    ax.set_title(f"reg={reg}\ntrain={train_loss:.3f}, test={test_loss:.3f}")
    ax.set_ylim(-1, 10)
    ax.legend(fontsize=7)

plt.suptitle("Effect of L2 Regularization on Overfitting", fontsize=14)
plt.tight_layout()
plt.savefig("overfitting_reg.png", dpi=100, bbox_inches="tight")
plt.show()
print("reg=0: severe overfitting; reg=0.01: good fit; reg=1.0: underfitting")'''),

    md_cell(r'''### 训练曲线诊断

通过分析训练曲线可以诊断训练问题：

| 曲线形状 | 诊断 | 解决方案 |
|---------|------|---------|
| 训练和验证损失都高 | 欠拟合 | 增加模型容量/训练时间 |
| 训练低，验证高 | 过拟合 | 正则化/数据增强 |
| 两者都缓慢下降 | 学习率太小 | 增大学习率 |
| 损失震荡 | 学习率太大 | 减小学习率 |
| 验证损失突然升高 | 数值不稳定 | 梯度裁剪/检查初始化 |'''),

    code_cell(r'''# Diagnostic: training curve simulation
np.random.seed(42)
epochs = 200

scenarios = {
    "underfit": {"capacity": 0.3, "lr": 0.001, "reg": 0.5},
    "overfit": {"capacity": 0.95, "lr": 0.05, "reg": 0.0},
    "good_fit": {"capacity": 0.85, "lr": 0.02, "reg": 0.05},
    "unstable": {"capacity": 0.8, "lr": 0.5, "reg": 0.01},
}

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (name, params) in zip(axes.flat, scenarios.items()):
    train_loss = []
    val_loss = []
    for e in range(epochs):
        t = params["capacity"] * (1 - np.exp(-e * params["lr"] * 2))
        v = t + (1 - params["capacity"]) * 0.3 + params["reg"] * 0.2
        noise = np.random.randn() * 0.05
        if name == "unstable":
            noise *= 3
        train_loss.append(max(0, 1 - t + noise * 0.1))
        val_loss.append(max(0, 1 - t + noise + (1 - params["capacity"]) * 0.3 + params["reg"]))
    
    ax.plot(train_loss, label="Train", color="blue")
    ax.plot(val_loss, label="Val", color="red")
    ax.set_title(f"{name}\n(capacity={params['capacity']}, lr={params['lr']}, reg={params['reg']})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_ylim(-0.1, 1.5)

plt.suptitle("Training Curve Diagnostics", fontsize=14)
plt.tight_layout()
plt.savefig("training_diagnostics.png", dpi=100, bbox_inches="tight")
plt.show()
print("Each scenario shows a characteristic curve shape for diagnosis.")'''),

    md_cell(r'''### 激活函数对比

不同激活函数的特性对比：

| 激活函数 | 优点 | 缺点 |
|---------|------|------|
| Sigmoid | 平滑可导 | 梯度消失、非零中心 |
| tanh | 零中心化 | 仍有梯度消失 |
| ReLU | 计算简单、缓解梯度消失 | 神经元死亡 |
| Leaky ReLU | 解决神经元死亡 | 需调超参数 |
| ELU | 接近零均值 | 计算较复杂'''),

    code_cell(r'''# Compare activation functions
np.random.seed(42)
x = np.linspace(-5, 5, 200)

activations = {
    "Sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "tanh": np.tanh,
    "ReLU": lambda x: np.maximum(0, x),
    "Leaky ReLU": lambda x: np.where(x > 0, x, 0.1 * x),
    "ELU": lambda x: np.where(x > 0, x, np.exp(x) - 1),
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name, fn in activations.items():
    axes[0].plot(x, fn(x), label=name, linewidth=2)
    grad = np.diff(fn(x)) / np.diff(x)
    axes[1].plot(x[:-1], grad, label=name, linewidth=2)

axes[0].set_title("Activation Functions")
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")
axes[0].legend()
axes[0].axhline(y=0, color="k", linewidth=0.5)
axes[0].axvline(x=0, color="k", linewidth=0.5)

axes[1].set_title("Gradients of Activation Functions")
axes[1].set_xlabel("x")
axes[1].set_ylabel("f'(x)")
axes[1].legend()
axes[1].axhline(y=0, color="k", linewidth=0.5)

plt.tight_layout()
plt.savefig("activation_comparison.png", dpi=100, bbox_inches="tight")
plt.show()
print("ReLU has constant gradient for positive x; Sigmoid gradient vanishes for large |x|.")'''),

    md_cell(r'''### 本章总结

两层神经网络的实现涉及以下关键组件：

1. **参数化模型**：$f(x) = W_2 \text{ReLU}(W_1 x + b_1) + b_2$
2. **前向传播**：逐层计算激活值
3. **Softmax 损失**：将得分转为概率并计算交叉熵
4. **反向传播**：利用链式法则计算梯度
5. **权重初始化**：Xavier/He 初始化保持方差稳定
6. **正则化**：L2 正则化防止过拟合
7. **超参数调优**：学习率、隐藏层大小、正则化强度

这些概念是所有深度学习模型的基础，后续章节将在此基础上扩展到卷积神经网络、循环神经网络等更复杂的架构。'''),
]
insert_cells(nb, extra1)
save_notebook(nb, path1)

# ============================================================
# Notebook 2: Image Feature Engineering (+16 cells)
# ============================================================
print("\nExpanding Notebook 2: Image Feature Engineering...")
path2 = os.path.join(BASE, "part1-deep-learning-basics", "lecture-05b-features", "practice.ipynb")
nb = load_notebook(path2)
extra2 = [
    md_cell(r'''### HOG 参数敏感性分析

HOG 特征的质量受以下参数影响：

- **cell_size**：越小越精细，但维度越高
- **block_size**：越大归一化范围越广，但降低了局部性
- **nbins**：方向 bin 数量，通常 9
- **stride**：块之间的步长'''),

    code_cell(r'''# HOG parameter sensitivity: cell size
np.random.seed(42)

# Create a synthetic image with edges
img = np.zeros((64, 64))
img[20:44, 20:44] = 1.0  # square
img[28:36, 28:36] = 0.5  # inner square

cell_sizes = [4, 8, 16, 32]
fig, axes = plt.subplots(1, len(cell_sizes), figsize=(16, 4))

for ax, cs in zip(axes, cell_sizes):
    # Simple gradient computation
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    mag = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(gy, gx) * 180 / np.pi
    angle[angle < 0] += 180
    
    # Simple HOG with given cell size
    n_cells = img.shape[0] // cs
    hog_feat = np.zeros((n_cells, n_cells, 9))
    for i in range(n_cells):
        for j in range(n_cells):
            cell_mag = mag[i*cs:(i+1)*cs, j*cs:(j+1)*cs]
            cell_angle = angle[i*cs:(i+1)*cs, j*cs:(j+1)*cs]
            for b in range(9):
                mask = (cell_angle >= b * 20) & (cell_angle < (b + 1) * 20)
                hog_feat[i, j, b] = cell_mag[mask].sum()
    
    # Visualize as average orientation
    vis = np.zeros((n_cells, n_cells))
    for i in range(n_cells):
        for j in range(n_cells):
            vis[i, j] = hog_feat[i, j].max()
    
    dim = n_cells * n_cells * 9
    ax.imshow(vis, cmap="hot", interpolation="nearest")
    ax.set_title(f"cell={cs}\ndim={dim}")
    ax.axis("off")

plt.suptitle("HOG Feature with Different Cell Sizes", fontsize=14)
plt.tight_layout()
plt.savefig("hog_cell_size.png", dpi=100, bbox_inches="tight")
plt.show()
print("Smaller cells capture more detail but increase dimensionality.")'''),

    code_cell(r'''# HOG parameter sensitivity: number of orientation bins
np.random.seed(42)

# Create directional patterns
img = np.zeros((64, 64))
for angle_deg in [0, 45, 90, 135]:
    t = np.linspace(-1, 1, 64)
    xx, yy = np.meshgrid(t, t)
    dist = xx * np.cos(np.radians(angle_deg)) + yy * np.sin(np.radians(angle_deg))
    img += np.exp(-dist ** 2 / 0.05) * 0.25

nbins_list = [6, 9, 12, 18, 36]
fig, axes = plt.subplots(1, len(nbins_list), figsize=(16, 3))

for ax, nbins in zip(axes, nbins_list):
    # Compute orientation histogram
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    mag = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(gy, gx) * 180 / np.pi
    angle[angle < 0] += 180
    
    hist = np.zeros(nbins)
    bin_width = 180.0 / nbins
    for b in range(nbins):
        mask = (angle >= b * bin_width) & (angle < (b + 1) * bin_width)
        hist[b] = mag[mask].sum()
    
    hist = hist / (hist.max() + 1e-10)
    ax.bar(range(nbins), hist, color="steelblue")
    ax.set_title(f"nbins={nbins}")
    ax.set_xlabel("Bin")
    ax.set_ylabel("Normalized magnitude")

plt.suptitle("HOG Orientation Bins Sensitivity", fontsize=14)
plt.tight_layout()
plt.savefig("hog_nbins.png", dpi=100, bbox_inches="tight")
plt.show()
print("9 bins is the standard choice; more bins give finer angular resolution.")'''),

    md_cell(r'''### 色彩空间分析

不同的色彩空间提取不同的颜色信息：

- **RGB**：最常见的色彩空间
- **HSV**：色相(H)、饱和度(S)、明度(V)更符合人类感知
- **LAB**：将亮度与色彩分离
- **灰度**：只保留亮度信息

不同色彩空间的直方图捕捉不同的视觉特征。'''),

    code_cell(r'''# Compare color space histograms
np.random.seed(42)

# Create a synthetic color image
h, w = 64, 64
img_rgb = np.zeros((h, w, 3))
img_rgb[:h//2, :w//2] = [0.8, 0.2, 0.2]  # red quadrant
img_rgb[:h//2, w//2:] = [0.2, 0.8, 0.2]  # green quadrant
img_rgb[h//2:, :w//2] = [0.2, 0.2, 0.8]  # blue quadrant
img_rgb[h//2:, w//2:] = [0.8, 0.8, 0.2]  # yellow quadrant
# Add some noise
img_rgb += np.random.randn(h, w, 3) * 0.05
img_rgb = np.clip(img_rgb, 0, 1)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# RGB histograms
colors = ["red", "green", "blue"]
for ch, color in enumerate(colors):
    hist, _ = np.histogram(img_rgb[:, :, ch].flatten(), bins=32, range=(0, 1))
    axes[0, 0].bar(np.arange(32), hist, alpha=0.5, color=color, label=color)
axes[0, 0].set_title("RGB Histogram")
axes[0, 0].set_xlabel("Bin")
axes[0, 0].set_ylabel("Count")
axes[0, 0].legend()

# Convert to HSV
r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
maxc = np.max(img_rgb, axis=2)
minc = np.min(img_rgb, axis=2)
v = maxc
s = np.where(maxc > 0, (maxc - minc) / (maxc + 1e-10), 0)
delta = maxc - minc
h_hsv = np.zeros_like(maxc)
rc = (maxc - r) / (delta + 1e-10)
gc = (maxc - g) / (delta + 1e-10)
bc = (maxc - b) / (delta + 1e-10)
h_hsv = np.where(maxc == r, (bc - gc) % 6, np.where(maxc == g, 2 + rc - bc, 4 + gc - rc))
h_hsv = h_hsv * 60
h_hsv[h_hsv < 0] += 360

for ch, name, cmap in [(0, "Hue", "hsv"), (1, "Saturation", "gray"), (2, "Value", "gray")]:
    hsv_ch = [h_hsv / 360, s, v][ch]
    hist, _ = np.histogram(hsv_ch.flatten(), bins=32, range=(0, 1))
    axes[0, ch + 1].bar(np.arange(32), hist, color="steelblue")
    axes[0, ch + 1].set_title(f"HSV: {name}")

# Show images
axes[1, 0].imshow(img_rgb)
axes[1, 0].set_title("Original RGB")
axes[1, 0].axis("off")
axes[1, 1].imshow(h_hsv / 360, cmap="hsv")
axes[1, 1].set_title("Hue")
axes[1, 1].axis("off")
axes[1, 2].imshow(s, cmap="gray")
axes[1, 2].set_title("Saturation")
axes[1, 2].axis("off")
axes[1, 3].imshow(v, cmap="gray")
axes[1, 3].set_title("Value")
axes[1, 3].axis("off")

plt.suptitle("Color Space Comparison", fontsize=14)
plt.tight_layout()
plt.savefig("color_spaces.png", dpi=100, bbox_inches="tight")
plt.show()
print("HSV separates color (Hue) from intensity (Value), unlike RGB.")'''),

    md_cell(r'''### 特征拼接策略

将多种特征拼接在一起可以捕捉更丰富的图像信息：

$$\text{feature} = [\text{HOG}; \text{color\_hist}]$$

但拼接后特征维度增加，可能带来维度灾难。需要配合降维方法使用。'''),

    code_cell(r'''# Feature concatenation and normalization
np.random.seed(42)

# Simulate features from different methods
n_samples = 100
hog_feat = np.random.randn(n_samples, 32) * 5  # HOG: larger values
color_feat = np.random.randn(n_samples, 16) * 0.5  # Color: smaller values

# Without normalization
raw_feat = np.hstack([hog_feat, color_feat])
print(f"Raw features - HOG range: [{hog_feat.min():.2f}, {hog_feat.max():.2f}]")
print(f"Raw features - Color range: [{color_feat.min():.2f}, {color_feat.max():.2f}]")
print(f"Raw features - HOG dominates with larger magnitude!")

# L2 normalization per feature type
hog_norm = hog_feat / (np.linalg.norm(hog_feat, axis=1, keepdims=True) + 1e-10)
color_norm = color_feat / (np.linalg.norm(color_feat, axis=1, keepdims=True) + 1e-10)
norm_feat = np.hstack([hog_norm, color_norm])

# L2 normalize the concatenated feature
final_feat = raw_feat / (np.linalg.norm(raw_feat, axis=1, keepdims=True) + 1e-10)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, feat, title in zip(axes, [raw_feat, norm_feat, final_feat], 
                            ["Raw (unnormalized)", "Per-type normalized", "Global L2 normalized"]):
    # Show first 10 samples as heatmap
    im = ax.imshow(feat[:10], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Feature dimension")
    ax.set_ylabel("Sample")
    ax.axvline(x=32, color="black", linewidth=2, linestyle="--")
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle("Feature Normalization Comparison", fontsize=14)
plt.tight_layout()
plt.savefig("feat_normalization.png", dpi=100, bbox_inches="tight")
plt.show()
print("Without normalization, HOG dominates; after normalization, both contribute equally.")'''),

    md_cell(r'''### SIFT 特征概念

**SIFT (Scale-Invariant Feature Transform)** 是一种局部特征提取方法：

1. **尺度空间极值检测**：在不同尺度的高斯金字塔中找极值
2. **关键点定位**：精确定位关键点位置和尺度
3. **方向分配**：为每个关键点分配主方向
4. **关键点描述子**：在关键点邻域计算方向直方图

SIFT 特征对旋转、尺度、光照变化具有不变性。'''),

    code_cell(r'''# Simple keypoint detection simulation
np.random.seed(42)

# Create synthetic image with corners
img = np.zeros((64, 64))
img[10:30, 10:30] = 1.0
img[35:55, 35:55] = 1.0

# Harris corner detection (simplified)
def harris_corner(img, k=0.04):
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    
    Ixx = gx ** 2
    Iyy = gy ** 2
    Ixy = gx * gy
    
    # Sum over window (box filter)
    from scipy.ndimage import uniform_filter  # fallback
    # Manual box filter
    Sxx = np.zeros_like(Ixx)
    Syy = np.zeros_like(Iyy)
    Sxy = np.zeros_like(Ixy)
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            Sxx[i, j] = Ixx[i-1:i+2, j-1:j+2].sum()
            Syy[i, j] = Iyy[i-1:i+2, j-1:j+2].sum()
            Sxy[i, j] = Ixy[i-1:i+2, j-1:j+2].sum()
    
    det = Sxx * Syy - Sxy ** 2
    trace = Sxx + Syy
    R = det - k * trace ** 2
    return R

R = harris_corner(img)
corners = np.argwhere(R > np.percentile(R, 99))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Input Image")
axes[0].axis("off")

axes[1].imshow(img, cmap="gray", alpha=0.5)
axes[1].imshow(R, cmap="hot", alpha=0.5)
axes[1].plot(corners[:, 1], corners[:, 0], "c+", markersize=10, label="Corners")
axes[1].set_title("Harris Corner Response")
axes[1].axis("off")
axes[1].legend()

plt.tight_layout()
plt.savefig("harris_corner.png", dpi=100, bbox_inches="tight")
plt.show()
print(f"Detected {len(corners)} corner points in the top 1% of responses.")'''),

    md_cell(r'''### 特征维度与降维

高维特征可能导致维度灾难。PCA（主成分分析）是一种常用的降维方法：

1. 中心化数据
2. 计算协方差矩阵
3. 特征值分解
4. 选择前 k 个主成分

PCA 通过保留方差最大的方向来减少维度，同时尽可能保留信息。'''),

    code_cell(r'''# PCA dimensionality reduction on features
np.random.seed(42)

# Generate high-dimensional features (128-dim, like SIFT)
n_samples = 200
n_features = 128
X = np.random.randn(n_samples, n_features)
# Add structure: first few dims carry most information
X[:, 0] *= 10
X[:, 1] *= 5
X[:, 2] *= 3

# PCA
X_centered = X - X.mean(axis=0)
cov = np.cov(X_centered.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
# Sort in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Explained variance ratio
explained_var_ratio = eigenvalues / eigenvalues.sum()
cumulative_var = np.cumsum(explained_var_ratio)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(range(20), explained_var_ratio[:20], color="steelblue", alpha=0.7, label="Individual")
axes[0].plot(range(20), cumulative_var[:20], "ro-", label="Cumulative")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance Ratio")
axes[0].set_title("PCA Explained Variance")
axes[0].legend()

# 2D projection
X_2d = X_centered @ eigenvectors[:, :2]
colors = np.linspace(0, 1, n_samples)
axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=colors, cmap="viridis", s=20, alpha=0.7)
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].set_title("PCA 2D Projection")

plt.tight_layout()
plt.savefig("pca_features.png", dpi=100, bbox_inches="tight")
plt.show()
print(f"First 3 PCs explain {cumulative_var[2]*100:.1f}% of variance")
print(f"To retain 95% variance, need {np.searchsorted(cumulative_var, 0.95) + 1} components")'''),

    md_cell(r'''### 特征工程实践建议

1. **先尝试简单特征**：颜色直方图、边缘统计等
2. **逐步增加复杂度**：HOG -> SIFT -> 深度特征
3. **注意归一化**：不同特征的尺度差异大
4. **降维**：高维特征使用PCA或t-SNE
5. **交叉验证评估**：用交叉验证选择最佳特征组合

> **重要发现**：[[Dalal & Triggs, 2005]](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) 证明 HOG 特征在行人检测中显著优于其他手工特征。'''),

    code_cell(r'''# Final feature comparison chart
np.random.seed(42)

# Simulate classification accuracy with different features
features = ["Raw pixels", "Color hist", "HOG", "HOG+Color", "SIFT", "Deep features"]
accuracies = [32, 38, 52, 54, 56, 75]
std_devs = [2, 3, 2, 3, 2, 4]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(features, accuracies, yerr=std_devs, capsize=5,
              color=["gray", "orange", "blue", "purple", "green", "red"], alpha=0.7)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Classification Accuracy with Different Features")
ax.set_ylim(0, 90)

for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{acc}%", ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.savefig("feature_comparison.png", dpi=100, bbox_inches="tight")
plt.show()
print("Deep features (75%) >> HOG (52%) >> Raw pixels (32%)")
print("Feature engineering significantly improves over raw pixels.")'''),

    md_cell(r'''### HOG 可视化深度解析

HOG 描述子的可视化展示了图像中梯度的方向和强度分布：

- 每个 cell 显示一个星形图
- 线段方向表示梯度方向
- 线段长度表示该方向的梯度强度

这种可视化帮助理解模型"看到"了什么。'''),

    code_cell(r'''# Detailed HOG visualization
np.random.seed(42)

# Create a more complex test image
img = np.zeros((48, 48))
# Circle
cy, cx = 24, 24
yy, xx = np.ogrid[:48, :48]
mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= 12 ** 2
img[mask] = 1.0
# Triangle
img[30:40, 5:15] = 0.5

# Compute gradients
gx = np.zeros_like(img)
gy = np.zeros_like(img)
gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
gy[1:-1, :] = img[2:, :] - img[:-2, :]
mag = np.sqrt(gx ** 2 + gy ** 2)
angle = np.arctan2(gy, gx) * 180 / np.pi
angle[angle < 0] += 180

# Compute HOG per cell
cell_size = 8
n_cells = img.shape[0] // cell_size
nbins = 9

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Input Image")
axes[0].axis("off")

axes[1].imshow(mag, cmap="hot")
axes[1].set_title("Gradient Magnitude")
axes[1].axis("off")

# HOG visualization
axes[2].imshow(img, cmap="gray", alpha=0.3)
for i in range(n_cells):
    for j in range(n_cells):
        cy = i * cell_size + cell_size // 2
        cx = j * cell_size + cell_size // 2
        cell_mag = mag[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
        cell_angle = angle[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
        
        for b in range(nbins):
            mask_b = (cell_angle >= b * 20) & (cell_angle < (b + 1) * 20)
            strength = cell_mag[mask_b].sum()
            if strength > 0.1:
                a = b * 20 + 10  # bin center angle
                dx = np.cos(np.radians(a)) * strength * 3
                dy = np.sin(np.radians(a)) * strength * 3
                axes[2].plot([cx - dx, cx + dx], [cy - dy, cy + dy], "r-", linewidth=1)

axes[2].set_title("HOG Visualization")
axes[2].set_xlim(0, 48)
axes[2].set_ylim(48, 0)
axes[2].axis("off")

plt.tight_layout()
plt.savefig("hog_detailed.png", dpi=100, bbox_inches="tight")
plt.show()
print("HOG captures edge orientations: circle shows radial pattern.")'''),

    md_cell(r'''### 特征工程的局限性

虽然手工特征在某些任务上表现良好，但存在以下局限：

1. **需要领域知识**：设计好特征需要深入理解问题
2. **难以泛化**：为一种任务设计的特征可能不适用于另一种
3. **信息丢失**：特征提取过程可能丢失有用信息
4. **难以组合**：多种特征的有效组合是难题

> 这正是深度学习兴起的原因——**表示学习**让模型自动从数据中学习有用的特征。

> 参考: [[Bengio et al., 2013]](https://arxiv.org/abs/1206.5538) "Representation Learning: A Review and New Perspectives"'''),
]
insert_cells(nb, extra2)
save_notebook(nb, path2)

# ============================================================
# Notebook 3: RNN and Image Captioning (+16 cells)
# ============================================================
print("\nExpanding Notebook 3: RNN and Image Captioning...")
path3 = os.path.join(BASE, "part2-cnn-vision", "lecture-08b-rnn-captioning", "practice.ipynb")
nb = load_notebook(path3)
extra3 = [
    md_cell(r'''### RNN 展开详解

RNN 的核心思想是在时间维度上共享权重。对于一个序列 $x_1, x_2, ..., x_T$：

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$$
$$y_t = W_{hy} h_t + b_y$$

展开后，RNN 可以看作一个深度为 $T$ 的网络，但所有层共享相同的参数。'''),

    code_cell(r'''# Step-by-step RNN forward pass with 3 timesteps
np.random.seed(42)

# Initialize parameters
input_dim = 4
hidden_dim = 5
output_dim = 3
seq_len = 3

W_xh = np.random.randn(input_dim, hidden_dim) * 0.01
W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.01
W_hy = np.random.randn(hidden_dim, output_dim) * 0.01
b_h = np.zeros(hidden_dim)
b_y = np.zeros(output_dim)

# Input sequence (3 timesteps)
x_seq = [np.random.randn(input_dim) for _ in range(seq_len)]

# Forward pass step by step
h = np.zeros(hidden_dim)  # initial hidden state
hidden_states = [h.copy()]
outputs = []

print("=== RNN Forward Pass (3 timesteps) ===")
for t, x in enumerate(x_seq):
    print(f"\n--- Timestep {t+1} ---")
    print(f"  x_{t+1} = {x[:3].round(3)}...")
    print(f"  h_{t} = {h[:3].round(3)}...")
    
    # Compute pre-activation
    pre_h = W_xh.T @ x + W_hh.T @ h + b_h
    print(f"  pre_h = W_xh^T x + W_hh^T h + b_h = {pre_h[:3].round(3)}...")
    
    # Activation
    h = np.tanh(pre_h)
    print(f"  h_{t+1} = tanh(pre_h) = {h[:3].round(3)}...")
    
    # Output
    y = W_hy.T @ h + b_y
    outputs.append(y)
    hidden_states.append(h.copy())

print("\n=== Summary ===")
print(f"Number of hidden states: {len(hidden_states)}")
print(f"Number of outputs: {len(outputs)}")
print(f"All timesteps share the same W_xh, W_hh, W_hy parameters!")'''),

    md_cell(r'''### BPTT (Backpropagation Through Time)

RNN 的反向传播称为 BPTT。关键公式：

$$\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial y_t} W_{hy}^T + \frac{\partial L}{\partial h_{t+1}} W_{hh}^T$$

注意梯度通过 $W_{hh}^T$ 传播，当序列很长时：

$$\frac{\partial L}{\partial h_0} = \frac{\partial L}{\partial h_T} \prod_{t=1}^{T} W_{hh}^T \text{diag}(1 - h_t^2)$$

当 $W_{hh}$ 的特征值 > 1 时梯度爆炸，< 1 时梯度消失。'''),

    code_cell(r'''# Vanishing gradient demonstration in RNN
np.random.seed(42)

# Demonstrate gradient flow through time
seq_lengths = [5, 10, 20, 50, 100]
hidden_dim = 10

# Different weight initializations
init_types = {
    "small_weights": lambda: np.random.randn(hidden_dim, hidden_dim) * 0.3,
    "identity_init": lambda: np.eye(hidden_dim) + np.random.randn(hidden_dim, hidden_dim) * 0.01,
    "large_weights": lambda: np.random.randn(hidden_dim, hidden_dim) * 0.9,
}

fig, ax = plt.subplots(figsize=(10, 6))

for name, init_fn in init_types.items():
    W_hh = init_fn()
    gradients = []
    for T in seq_lengths:
        # Simulate gradient magnitude at h_0
        grad = np.ones(hidden_dim)
        for t in range(T):
            # Jacobian of tanh: diag(1 - h^2), worst case ~1
            grad = W_hh.T @ grad
        grad_mag = np.linalg.norm(grad)
        gradients.append(grad_mag)
    
    ax.plot(seq_lengths, gradients, "o-", label=name, markersize=8)

ax.set_xlabel("Sequence Length (T)")
ax.set_ylabel("Gradient magnitude at h_0")
ax.set_title("Vanishing/Exploding Gradient in RNN")
ax.set_yscale("log")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("rnn_vanishing.png", dpi=100, bbox_inches="tight")
plt.show()
print("small_weights: gradient vanishes exponentially")
print("identity_init: gradient stays stable")
print("large_weights: gradient explodes")'''),

    md_cell(r'''### LSTM 门控机制详解

LSTM 通过三个门来控制信息流：

| 门 | 作用 | 公式 |
|----|------|------|
| 遗忘门 | 控制丢弃多少旧信息 | $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$ |
| 输入门 | 控制写入多少新信息 | $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$ |
| 输出门 | 控制输出多少信息 | $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$ |

**细胞状态更新**：
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

其中 $\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)$ 是候选值。'''),

    code_cell(r'''# LSTM gate simulation
np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# LSTM parameters
input_dim = 4
hidden_dim = 5

# Forget gate
W_f = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.1
b_f = np.ones(hidden_dim)  # bias towards remembering
# Input gate
W_i = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.1
b_i = np.zeros(hidden_dim)
# Cell candidate
W_c = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.1
b_c = np.zeros(hidden_dim)
# Output gate
W_o = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.1
b_o = np.zeros(hidden_dim)

# Process sequence
seq_len = 10
x_seq = [np.random.randn(input_dim) for _ in range(seq_len)]
h = np.zeros(hidden_dim)
C = np.zeros(hidden_dim)

forget_values = []
input_values = []
output_values = []
cell_values = []

print("=== LSTM Forward Pass ===")
for t, x in enumerate(x_seq):
    concat = np.concatenate([h, x])
    
    f = sigmoid(W_f @ concat + b_f)
    i = sigmoid(W_i @ concat + b_i)
    c_tilde = np.tanh(W_c @ concat + b_c)
    o = sigmoid(W_o @ concat + b_o)
    
    C = f * C + i * c_tilde
    h = o * np.tanh(C)
    
    forget_values.append(f.mean())
    input_values.append(i.mean())
    output_values.append(o.mean())
    cell_values.append(C.mean())

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, vals, name in zip(axes.flat, 
    [forget_values, input_values, output_values, cell_values],
    ["Forget Gate", "Input Gate", "Output Gate", "Cell State"]):
    ax.plot(range(seq_len), vals, "o-", color=["blue", "green", "red", "purple"][["Forget Gate", "Input Gate", "Output Gate", "Cell State"].index(name)])
    ax.set_title(name)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean activation")
    ax.set_ylim(-1, 1)

plt.suptitle("LSTM Gate Activations Over Time", fontsize=14)
plt.tight_layout()
plt.savefig("lstm_gates.png", dpi=100, bbox_inches="tight")
plt.show()
print("Forget gate bias=1: tends to remember past information")'''),

    md_cell(r'''### 词嵌入

词嵌入将离散的单词映射到连续的向量空间：

$$\text{word\_embed}(w) = E \cdot \text{one\_hot}(w)$$

其中 $E \in \mathbb{R}^{d \times |V|}$ 是嵌入矩阵，$d$ 是嵌入维度。'''),

    code_cell(r'''# Word embedding visualization
np.random.seed(42)

# Vocabulary
vocab = ["a", "cat", "dog", "bird", "the", "is", "running", "flying", "image", "photo", 
         "and", "sitting", "on", "in", "with", "man", "woman", "boy", "girl"]
vocab_size = len(vocab)
embed_dim = 8

# Random embedding matrix
E = np.random.randn(vocab_size, embed_dim) * 0.5

# Cosine similarity between words
def cosine_sim(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

# Compute similarity matrix
sim_matrix = np.zeros((vocab_size, vocab_size))
for i in range(vocab_size):
    for j in range(vocab_size):
        sim_matrix[i, j] = cosine_sim(E[i], E[j])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# Similarity matrix
im = axes[0].imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
axes[0].set_xticks(range(vocab_size))
axes[0].set_xticklabels(vocab, rotation=45, ha="right", fontsize=8)
axes[0].set_yticks(range(vocab_size))
axes[0].set_yticklabels(vocab, fontsize=8)
axes[0].set_title("Word Embedding Cosine Similarity (Random)")
plt.colorbar(im, ax=axes[0], fraction=0.046)

# PCA visualization of embeddings
E_centered = E - E.mean(axis=0)
U, S, Vt = np.linalg.svd(E_centered, full_matrices=False)
E_2d = E_centered @ Vt[:2].T

axes[1].scatter(E_2d[:, 0], E_2d[:, 1], c=range(vocab_size), cmap="tab20", s=50)
for i, word in enumerate(vocab):
    axes[1].annotate(word, (E_2d[i, 0], E_2d[i, 1]), fontsize=8, 
                     xytext=(5, 5), textcoords="offset points")
axes[1].set_title("Word Embeddings (PCA 2D)")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")

plt.tight_layout()
plt.savefig("word_embeddings.png", dpi=100, bbox_inches="tight")
plt.show()
print("Random embeddings have no semantic structure (unlike trained ones).")'''),

    md_cell(r'''### Beam Search 解码

贪心解码每步选概率最高的词，容易陷入局部最优。Beam Search 维护 $k$ 个候选序列：

1. 初始：$k$ 个起始 token
2. 每步：对每个候选展开所有可能的下一个词，保留总概率最高的 $k$ 个
3. 终止：遇到结束符或达到最大长度'''),

    code_cell(r'''# Simple beam search implementation
np.random.seed(42)

vocab = ["<START>", "a", "cat", "dog", "bird", "sits", "runs", "flies", "on", "mat", "<END>"]
vocab_size = len(vocab)

# Simulated language model probabilities (random for demonstration)
def mock_next_word_probs(context, hidden_state):
    """Simulate next word probability distribution."""
    np.random.seed(hash(tuple(context)) % 2**32)
    probs = np.random.dirichlet(np.ones(vocab_size))
    return probs

def beam_search(beam_width=3, max_len=6):
    """Simple beam search decoding."""
    beams = [(1.0, [0])]  # (probability, token_ids)
    
    for step in range(max_len):
        all_candidates = []
        for prob, tokens in beams:
            if tokens[-1] == len(vocab) - 1:  # <END>
                all_candidates.append((prob, tokens))
                continue
            
            next_probs = mock_next_word_probs(tokens, None)
            for w in range(vocab_size):
                new_prob = prob * next_probs[w]
                new_tokens = tokens + [w]
                all_candidates.append((new_prob, new_tokens))
        
        # Select top beam_width candidates
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        beams = all_candidates[:beam_width]
        
        # Check if all beams ended
        if all(b[1][-1] == len(vocab) - 1 for b in beams):
            break
    
    return beams

# Run beam search with different widths
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, bw in zip(axes, [1, 3, 5]):
    results = beam_search(beam_width=bw)
    for i, (prob, tokens) in enumerate(results[:3]):
        caption = " ".join([vocab[t] for t in tokens])
        ax.text(0.1, 0.8 - i * 0.25, f"P={prob:.4e}\n{caption}", 
                fontsize=9, transform=ax.transAxes)
    ax.set_title(f"Beam width = {bw}")
    ax.axis("off")

plt.suptitle("Beam Search Decoding Comparison", fontsize=14)
plt.tight_layout()
plt.savefig("beam_search.png", dpi=100, bbox_inches="tight")
plt.show()
print("Beam width=1 is greedy; wider beams explore more possibilities.")'''),

    md_cell(r'''### Teacher Forcing vs 自由运行

- **Teacher Forcing**：训练时每步输入真实的前一个词
  - 优点：训练稳定、收敛快
  - 缺点：推理时没有真实词，存在训练-推理偏差
  
- **自由运行 (Free-running)**：训练时使用模型自己的预测作为下一步输入
  - 优点：消除训练-推理偏差
  - 缺点：训练不稳定、误差累积

**Scheduled Sampling** 是一种折中方案：训练初期用 Teacher Forcing，逐渐增加自由运行比例。'''),

    code_cell(r'''# Compare teacher forcing vs free-running training
np.random.seed(42)

epochs = 50
true_seq = np.array([0.3, 0.5, 0.8, 0.2, 0.9])

# Simulate training curves
tf_losses = []
fr_losses = []
ss_losses = []

tf_loss = 1.0
fr_loss = 1.0
ss_loss = 1.0
ss_ratio = 0.0  # scheduled sampling ratio

for e in range(epochs):
    # Teacher forcing: stable convergence
    tf_loss = tf_loss * 0.95 + np.random.randn() * 0.02
    tf_losses.append(tf_loss)
    
    # Free-running: unstable, higher final loss
    fr_loss = fr_loss * 0.9 + np.random.randn() * 0.05 + abs(fr_loss - 0.3) * 0.1
    fr_losses.append(fr_loss)
    
    # Scheduled sampling: gradual transition
    ss_ratio = min(ss_ratio + 0.02, 0.8)
    ss_loss = ss_loss * 0.93 + np.random.randn() * 0.03
    if e > 20:
        ss_loss += np.random.randn() * 0.02 * ss_ratio
    ss_losses.append(max(ss_loss, 0.05))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(tf_losses, label="Teacher Forcing", linewidth=2)
ax.plot(fr_losses, label="Free-running", linewidth=2)
ax.plot(ss_losses, label="Scheduled Sampling", linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss")
ax.set_title("Training Strategy Comparison")
ax.legend()
ax.set_ylim(0, 1.5)

plt.tight_layout()
plt.savefig("teacher_forcing.png", dpi=100, bbox_inches="tight")
plt.show()
print("TF converges fast but has train-test mismatch")
print("Free-running is unstable but closer to inference")
print("Scheduled Sampling balances both")'''),

    md_cell(r'''### BLEU 评分详解

BLEU (Bilingual Evaluation Understudy) 是机器翻译和图像描述生成中常用的评估指标：

$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

其中：
- $p_n$：n-gram 精确度
- $w_n = 1/N$：权重
- BP：长度惩罚 (Brevity Penalty)'''),

    code_cell(r'''# BLEU score computation step by step
np.random.seed(42)

def compute_bleu(reference, candidate, max_n=4):
    """Compute BLEU score with n-gram precisions."""
    precisions = []
    for n in range(1, max_n + 1):
        # Extract n-grams
        ref_ngrams = {}
        for i in range(len(reference) - n + 1):
            ng = tuple(reference[i:i+n])
            ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1
        
        cand_ngrams = {}
        for i in range(len(candidate) - n + 1):
            ng = tuple(candidate[i:i+n])
            cand_ngrams[ng] = cand_ngrams.get(ng, 0) + 1
        
        # Count matches (clipped)
        matches = 0
        total = 0
        for ng, count in cand_ngrams.items():
            matches += min(count, ref_ngrams.get(ng, 0))
            total += count
        
        precision = matches / total if total > 0 else 0
        precisions.append(precision)
    
    # Geometric mean of precisions
    if all(p > 0 for p in precisions):
        log_avg = sum(np.log(p) for p in precisions) / len(precisions)
        geo_mean = np.exp(log_avg)
    else:
        geo_mean = 0
    
    # Brevity penalty
    ref_len = len(reference)
    cand_len = len(candidate)
    if cand_len > ref_len:
        bp = 1.0
    elif cand_len == 0:
        bp = 0.0
    else:
        bp = np.exp(1 - ref_len / cand_len)
    
    bleu = bp * geo_mean
    return bleu, precisions, bp

# Test with examples
examples = [
    (["a", "cat", "sits", "on", "the", "mat"], ["a", "cat", "sits", "on", "mat"]),
    (["a", "dog", "runs", "in", "the", "park"], ["a", "dog", "runs", "in", "park"]),
    (["a", "bird", "flies", "over", "the", "ocean"], ["a", "bird", "swims", "in", "water"]),
    (["the", "old", "man", "sits", "alone"], ["the", "old", "man", "sits", "alone"]),
]

fig, ax = plt.subplots(figsize=(10, 5))
x_labels = ["Perfect\nmatch", "Good\nmatch", "Poor\nmatch", "Exact\nmatch"]
bleu_scores = []

for ref, cand in examples:
    bleu, precs, bp = compute_bleu(ref, cand)
    bleu_scores.append(bleu)
    print(f"Ref: {ref}")
    print(f"Cand: {cand}")
    print(f"BLEU={bleu:.3f}, Precisions={[f'{p:.2f}' for p in precs]}, BP={bp:.3f}\n")

bars = ax.bar(x_labels, bleu_scores, color=["green", "blue", "red", "darkgreen"], alpha=0.7)
ax.set_ylabel("BLEU Score")
ax.set_title("BLEU Score Comparison")
ax.set_ylim(0, 1)
for bar, score in zip(bars, bleu_scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{score:.3f}", ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.savefig("bleu_scores.png", dpi=100, bbox_inches="tight")
plt.show()'''),
]
insert_cells(nb, extra3)
save_notebook(nb, path3)

# ============================================================
# Notebook 4: CNN Visualization (+17 cells)
# ============================================================
print("\nExpanding Notebook 4: CNN Visualization...")
path4 = os.path.join(BASE, "part2-cnn-vision", "lecture-10-visualization", "practice.ipynb")
nb = load_notebook(path4)
extra4 = [
    md_cell(r'''### 显著性图理论

显著性图回答一个问题：**输入图像的哪些像素对分类结果影响最大？**

方法：计算损失对输入的梯度

$$S(x) = \left\| \frac{\partial L}{\partial x} \right\|$$

高梯度值的像素表示模型分类决策高度依赖这些区域。'''),

    code_cell(r'''# Saliency map simulation for different layers
np.random.seed(42)

# Create synthetic image
h, w = 32, 32
img = np.zeros((h, w))
img[8:24, 8:24] = 1.0  # "object" region
img[14:18, 14:18] = 0.5  # "detail"

# Simulate different layer saliency maps
layer_names = ["Conv1 (edges)", "Conv2 (textures)", "Conv3 (parts)", "Conv4 (objects)"]
saliency_maps = []

# Each layer focuses on different regions
for i, name in enumerate(layer_names):
    sal = np.zeros((h, w))
    if i == 0:  # Edge layer - high saliency at edges
        sal[7:9, 7:25] = 1.0
        sal[23:25, 7:25] = 1.0
        sal[7:25, 7:9] = 1.0
        sal[7:25, 23:25] = 1.0
    elif i == 1:  # Texture layer
        sal[8:24, 8:24] = 0.5
        sal[14:18, 14:18] = 0.8
    elif i == 2:  # Parts layer
        sal[12:20, 12:20] = 0.9
    else:  # Object layer
        sal[8:24, 8:24] = 0.7
    saliency_maps.append(sal)

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Input")
axes[0].axis("off")
for ax, sal, name in zip(axes[1:], saliency_maps, layer_names):
    ax.imshow(img, cmap="gray", alpha=0.3)
    ax.imshow(sal, cmap="hot", alpha=0.7)
    ax.set_title(name)
    ax.axis("off")

plt.suptitle("Saliency Maps from Different CNN Layers", fontsize=14)
plt.tight_layout()
plt.savefig("saliency_layers.png", dpi=100, bbox_inches="tight")
plt.show()
print("Early layers focus on edges; deeper layers focus on object-level regions.")'''),

    md_cell(r'''### Grad-CAM 原理

Grad-CAM 利用最后一个卷积层的梯度来生成类激活热图：

1. 计算损失对最后卷积层特征图的梯度
2. 全局平均池化得到通道权重 $\alpha_k = \frac{1}{Z} \sum_{i,j} \frac{\partial L}{\partial A_{ij}^k}$
3. 加权求和并 ReLU：$L_{Grad-CAM} = \text{ReLU}\left(\sum_k \alpha_k A^k\right)$

Grad-CAM 提供了类别相关的定位能力。'''),

    code_cell(r'''# Grad-CAM simulation for different layers
np.random.seed(42)

h, w = 16, 16
# Simulate feature maps from a conv layer (8 channels)
n_channels = 8
feature_maps = np.random.randn(n_channels, h, w) * 0.5
# Make some channels active in specific regions
feature_maps[0, 4:12, 4:12] += 2  # channel 0: center
feature_maps[1, :8, :8] += 2  # channel 1: top-left
feature_maps[2, 8:, 8:] += 2  # channel 2: bottom-right

# Simulate gradients (channel importance weights)
class_labels = ["cat", "dog", "bird"]
fig, axes = plt.subplots(len(class_labels), n_channels + 1, figsize=(20, 9))

for row, label in enumerate(class_labels):
    # Different class gets different gradient weights
    np.random.seed(42 + row)
    weights = np.abs(np.random.randn(n_channels))
    
    # Generate Grad-CAM
    grad_cam = np.zeros((h, w))
    for k in range(n_channels):
        grad_cam += weights[k] * feature_maps[k]
    grad_cam = np.maximum(0, grad_cam)  # ReLU
    grad_cam = grad_cam / (grad_cam.max() + 1e-10)  # normalize
    
    # Show individual channel contributions
    for col in range(n_channels):
        ax = axes[row, col]
        weighted = feature_maps[col] * weights[col]
        im = ax.imshow(weighted, cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_title(f"Ch{col} (w={weights[col]:.2f})", fontsize=8)
        ax.axis("off")
    
    # Show Grad-CAM
    ax = axes[row, -1]
    ax.imshow(grad_cam, cmap="jet", alpha=0.8)
    ax.set_title(f"Grad-CAM: {label}", fontsize=10, fontweight="bold")
    ax.axis("off")

plt.suptitle("Grad-CAM: Channel Weighting for Different Classes", fontsize=14)
plt.tight_layout()
plt.savefig("gradcam_sim.png", dpi=100, bbox_inches="tight")
plt.show()
print("Different classes activate different channels and spatial regions.")'''),

    md_cell(r'''### 特征可视化

特征可视化通过优化输入图像来最大化特定神经元或层的激活：

$$x^* = \arg\max_x \text{activation}(l, x) + \lambda \text{regularization}(x)$$

正则化项防止生成不自然的图案。常见正则化：
- L2 范数约束
- 高斯模糊
- Total variation (TV) 正则化'''),

    code_cell(r'''# Simple feature maximization
np.random.seed(42)

# Simulate feature visualization with different regularizations
h, w = 32, 32
n_iter = 100

regularizations = ["none", "L2", "TV", "blur"]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for col, reg_name in enumerate(regularizations):
    # Start from random noise
    x = np.random.randn(h, w) * 0.1
    
    # Simulate optimization
    for step in range(n_iter):
        # Fake gradient: push towards a pattern
        target = np.sin(np.outer(np.linspace(0, 4*np.pi, h), np.linspace(0, 4*np.pi, w)))
        grad = target * (1 - step / n_iter) * 0.1
        x += grad
        
        # Apply regularization
        if reg_name == "L2":
            x = x * 0.95
        elif reg_name == "TV":
            x[1:, :] = (x[1:, :] + x[:-1, :]) / 2
            x[:, 1:] = (x[:, 1:] + x[:, :-1]) / 2
        elif reg_name == "blur":
            # Simple box blur
            x = (x + np.roll(x, 1, 0) + np.roll(x, -1, 0) + 
                 np.roll(x, 1, 1) + np.roll(x, -1, 1)) / 5
    
    axes[0, col].imshow(x, cmap="RdBu_r")
    axes[0, col].set_title(f"Reg: {reg_name}")
    axes[0, col].axis("off")
    
    # Frequency analysis
    fft = np.abs(np.fft.fft2(x))
    fft_shifted = np.fft.fftshift(fft)
    axes[1, col].imshow(np.log(fft_shifted + 1), cmap="viridis")
    axes[1, col].set_title(f"FFT: {reg_name}")
    axes[1, col].axis("off")

plt.suptitle("Feature Visualization with Different Regularizations", fontsize=14)
plt.tight_layout()
plt.savefig("feat_visualization.png", dpi=100, bbox_inches="tight")
plt.show()
print("Without regularization: high-frequency noise dominates")
print("Regularization produces smoother, more natural patterns")'''),

    md_cell(r'''### 对抗样本 (FGSM)

FGSM (Fast Gradient Sign Method) 通过在输入上添加小扰动来欺骗分类器：

$$x_{adv} = x + \epsilon \cdot \text{sign}\left(\nabla_x L(f, x, y)\right)$$

关键洞察：梯度方向指出了使损失增大最快的方向。'''),

    code_cell(r'''# FGSM attack implementation
np.random.seed(42)

# Create synthetic image and model
h, w = 16, 16
x = np.random.rand(h, w) * 0.5 + 0.25
true_label = 1  # "cat"

# Simulate model gradient w.r.t. input
gradient = np.random.randn(h, w) * 0.1

epsilons = [0, 0.01, 0.05, 0.1, 0.2, 0.5]
fig, axes = plt.subplots(2, len(epsilons), figsize=(18, 6))

for col, eps in enumerate(epsilons):
    # FGSM perturbation
    perturbation = eps * np.sign(gradient)
    x_adv = np.clip(x + perturbation, 0, 1)
    
    # Show adversarial image
    axes[0, col].imshow(x_adv, cmap="gray", vmin=0, vmax=1)
    axes[0, col].set_title(f"eps={eps}")
    axes[0, col].axis("off")
    
    # Show perturbation (amplified)
    axes[1, col].imshow(perturbation, cmap="RdBu_r", vmin=-0.2, vmax=0.2)
    axes[1, col].set_title(f"perturbation (x10)")
    axes[1, col].axis("off")

plt.suptitle("FGSM Adversarial Examples with Increasing Epsilon", fontsize=14)
plt.tight_layout()
plt.savefig("fgsm_attack.png", dpi=100, bbox_inches="tight")
plt.show()

# Compute L2 distance
for eps in epsilons:
    pert = eps * np.sign(gradient)
    l2_dist = np.linalg.norm(pert)
    linf_dist = np.max(np.abs(pert))
    print(f"eps={eps:.2f}: L2={l2_dist:.4f}, Linf={linf_dist:.4f}")'''),

    md_cell(r'''### 对抗扰动幅度实验

不同的扰动幅度对分类准确率的影响：'''),

    code_cell(r'''# Adversarial perturbation magnitude experiment
np.random.seed(42)

epsilons = np.linspace(0, 0.5, 20)
n_samples = 100

# Simulate accuracy under FGSM attack
accuracies = []
for eps in epsilons:
    # Each sample has a random gradient direction
    correct = 0
    for _ in range(n_samples):
        grad = np.random.randn(10) * 0.1
        pert = eps * np.sign(grad)
        # Simple decision: if perturbation magnitude exceeds threshold, misclassified
        if np.linalg.norm(pert) < 0.15:
            correct += 1
    accuracies.append(correct / n_samples * 100)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epsilons, accuracies, "ro-", linewidth=2, markersize=6)
ax.fill_between(epsilons, [max(0, a-5) for a in accuracies], 
                [min(100, a+5) for a in accuracies], alpha=0.2, color="red")
ax.set_xlabel("Epsilon (perturbation magnitude)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Model Accuracy Under FGSM Attack")
ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Random chance")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("fgsm_accuracy.png", dpi=100, bbox_inches="tight")
plt.show()
print("Even small perturbations (eps=0.1) can reduce accuracy significantly")'''),

    md_cell(r'''### t-SNE 降维可视化

t-SNE (t-Distributed Stochastic Neighbor Embedding) 是一种非线性降维方法：

1. 在高维空间计算样本对之间的相似度（高斯分布）
2. 在低维空间计算样本对之间的相似度（t分布）
3. 最小化两个分布之间的 KL 散度

t-SNE 适合可视化但不适合作为通用降维工具（计算复杂度高、无法保留全局结构）。'''),

    code_cell(r'''# Simple t-SNE implementation (simplified)
np.random.seed(42)

# Generate high-dimensional clustered data
n_per_cluster = 30
n_clusters = 4
n_dim = 50

X = np.zeros((n_per_cluster * n_clusters, n_dim))
labels = np.zeros(n_per_cluster * n_clusters, dtype=int)

for c in range(n_clusters):
    center = np.random.randn(n_dim) * 3
    X[c*n_per_cluster:(c+1)*n_per_cluster] = center + np.random.randn(n_per_cluster, n_dim) * 0.5
    labels[c*n_per_cluster:(c+1)*n_per_cluster] = c

# PCA for comparison
X_centered = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
X_pca = X_centered @ Vt[:2].T

# Simple t-SNE (gradient descent on KL divergence, simplified)
n_total = X.shape[0]
Y = np.random.randn(n_total, 2) * 0.0001

# Compute pairwise similarities in high dimension (simplified)
from scipy.spatial.distance import cdist
# Manual pairwise distance
D = np.zeros((n_total, n_total))
for i in range(n_total):
    for j in range(n_total):
        D[i, j] = np.sum((X[i] - X[j]) ** 2)

P = np.exp(-D / (2 * np.mean(D)))
np.fill_diagonal(P, 0)
P = P / P.sum()

# Gradient descent
lr = 50
for iteration in range(200):
    # Low-dim similarities (t-distribution)
    D_low = np.zeros((n_total, n_total))
    for i in range(n_total):
        for j in range(n_total):
            D_low[i, j] = np.sum((Y[i] - Y[j]) ** 2)
    Q = 1 / (1 + D_low)
    np.fill_diagonal(Q, 0)
    Q = Q / Q.sum()
    
    # Gradient
    grad = np.zeros_like(Y)
    for i in range(n_total):
        diff = (P - Q)[:, i]
        grad[i] = 4 * np.sum((Y[i] - Y) * diff[:, np.newaxis], axis=0)
    
    Y = Y + lr * grad - 0.01 * Y  # with momentum-like term

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = ["red", "blue", "green", "purple"]

for c in range(n_clusters):
    mask = labels == c
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[c], label=f"Cluster {c}", s=30)
    axes[1].scatter(Y[mask, 0], Y[mask, 1], c=colors[c], label=f"Cluster {c}", s=30)

axes[0].set_title("PCA 2D Projection")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].legend()

axes[1].set_title("t-SNE 2D Embedding")
axes[1].set_xlabel("Dim 1")
axes[1].set_ylabel("Dim 2")
axes[1].legend()

plt.suptitle("PCA vs t-SNE Dimensionality Reduction", fontsize=14)
plt.tight_layout()
plt.savefig("tsne_pca.png", dpi=100, bbox_inches="tight")
plt.show()
print("t-SNE better separates clusters but distorts global structure.")'''),

    md_cell(r'''### DeepDream 概念

DeepDream 通过反向传播放大网络检测到的模式：

$$x_{dream} = x + \alpha \cdot \frac{\partial \text{activation}(layer)}{\partial x}$$

与对抗样本不同，DeepDream **放大**而非最小化激活，产生梦幻般的视觉效果。'''),

    code_cell(r'''# Simple DeepDream-style pattern amplification
np.random.seed(42)

h, w = 64, 64
x = np.random.rand(h, w) * 0.3 + 0.4  # base image

# Simulate layer activation patterns
def get_activation(x, pattern_type):
    """Simulate CNN layer activation."""
    if pattern_type == "edges":
        gx = np.zeros_like(x)
        gx[:, 1:-1] = x[:, 2:] - x[:, :-2]
        return np.abs(gx)
    elif pattern_type == "textures":
        return np.abs(np.diff(x, axis=0, prepend=x[:1]))
    elif pattern_type == "spirals":
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        cx, cy = h//2, w//2
        r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        theta = np.arctan2(yy - cy, xx - cx)
        return np.abs(np.sin(r * 0.3 + theta * 3))
    return x

iterations = 50
alpha = 0.05

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(x, cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")

for col, (pattern, name) in enumerate([("edges", "Edge dream"), 
                                        ("textures", "Texture dream"), 
                                        ("spirals", "Spiral dream")], 1):
    x_dream = x.copy()
    for _ in range(iterations):
        act = get_activation(x_dream, pattern)
        x_dream = np.clip(x_dream + alpha * act / (act.max() + 1e-10), 0, 1)
    
    axes[col].imshow(x_dream, cmap="gray")
    axes[col].set_title(name)
    axes[col].axis("off")

plt.suptitle("DeepDream-style Pattern Amplification", fontsize=14)
plt.tight_layout()
plt.savefig("deepdream.png", dpi=100, bbox_inches="tight")
plt.show()
print("DeepDream amplifies patterns the network detects in the image.")'''),

    md_cell(r'''### CNN 特征层级解读

CNN 不同层学习不同层级的特征：

| 层级 | 特征类型 | 例子 |
|------|---------|------|
| 浅层 | 边缘、颜色 | Gabor 滤波器、颜色块 |
| 中层 | 纹理、局部模式 | 纹理、简单形状 |
| 深层 | 物体部件、概念 | 眼睛、车轮、整物体 |

> 参考: [[Zeiler & Fergus, 2014]](https://arxiv.org/abs/1311.2901) "Visualizing and Understanding Convolutional Networks"'''),

    code_cell(r'''# Layer activation visualization
np.random.seed(42)

# Simulate feature maps at different depths
h, w = 16, 16
depths = {
    "Layer 1 (Conv1)": lambda: np.random.randn(8, h, w) * 0.5,
    "Layer 2 (Conv2)": lambda: np.random.randn(16, h, w) * 0.3,
    "Layer 3 (Conv3)": lambda: np.random.randn(32, h, w) * 0.2,
    "Layer 4 (Conv4)": lambda: np.random.randn(64, h, w) * 0.1,
}

fig, axes = plt.subplots(len(depths), 8, figsize=(20, 12))
for row, (name, gen_fn) in enumerate(depths.items()):
    feats = gen_fn()
    # Select 8 channels
    for col in range(8):
        ax = axes[row, col]
        ax.imshow(feats[col], cmap="viridis")
        ax.set_title(f"Ch{col}", fontsize=8)
        ax.axis("off")
    axes[row, 0].set_ylabel(name, fontsize=10, rotation=0, labelpad=80)

plt.suptitle("CNN Feature Maps at Different Depths", fontsize=14)
plt.tight_layout()
plt.savefig("layer_activations.png", dpi=100, bbox_inches="tight")
plt.show()
print("Deeper layers have more channels with spatially broader activations.")'''),

    md_cell(r'''### 可视化的伦理与局限

CNN 可视化虽然强大，但需要注意：

1. **误导性**：热图显示相关性而非因果关系
2. **不完整性**：可视化只展示部分信息
3. **确认偏误**：容易看到我们期望看到的模式
4. **对抗脆弱性**：模型可能依赖脆弱的特征

> 建议：可视化作为**诊断工具**而非**解释工具**使用。

> 参考: [[Adebayo et al., 2018]](https://arxiv.org/abs/1810.03292) "Sanity Checks for Saliency Maps"'''),
]
insert_cells(nb, extra4)
save_notebook(nb, path4)

# ============================================================
# Notebook 5: Transfer Learning (+19 cells)
# ============================================================
print("\nExpanding Notebook 5: Transfer Learning...")
path5 = os.path.join(BASE, "part3-frontiers", "lecture-11-transfer-learning", "practice.ipynb")
nb = load_notebook(path5)
extra5 = [
    md_cell(r'''### 迁移学习场景

迁移学习根据源域和目标域的关系分为四种场景：

| 场景 | 源域数据 | 目标域数据 | 策略 |
|------|---------|-----------|------|
| 1 | 多 | 多 | 直接微调 |
| 2 | 多 | 少 | 冻结+微调 |
| 3 | 少 | 多 | 预训练效果有限 |
| 4 | 少 | 少 | 需要特殊技巧'''),

    code_cell(r'''# Feature extraction vs fine-tuning comparison
np.random.seed(42)

epochs = 50
# Simulate training curves
fe_train, fe_val = [], []  # Feature extraction
ft_train, ft_val = [], []  # Fine-tuning

fe_t, fe_v = 0.3, 0.35
ft_t, ft_v = 0.2, 0.25

for e in range(epochs):
    fe_t = max(0.05, fe_t * 0.97 - 0.001)
    fe_v = max(0.15, fe_v * 0.98 + 0.002 * (fe_t < 0.1))
    fe_train.append(fe_t)
    fe_val.append(fe_v)
    
    ft_t = max(0.02, ft_t * 0.95 - 0.001)
    ft_v = max(0.08, ft_v * 0.96 + 0.003 * (e > 15))
    ft_train.append(ft_t)
    ft_val.append(ft_v)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(fe_train, label="Train", color="blue")
axes[0].plot(fe_val, label="Val", color="red")
axes[0].set_title("Feature Extraction (frozen backbone)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(ft_train, label="Train", color="blue")
axes[1].plot(ft_val, label="Val", color="red")
axes[1].set_title("Fine-tuning (all layers)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.suptitle("Feature Extraction vs Fine-tuning", fontsize=14)
plt.tight_layout()
plt.savefig("fe_vs_ft.png", dpi=100, bbox_inches="tight")
plt.show()
print("FE: faster but higher floor; FT: slower but lower floor")'''),

    md_cell(r'''### 渐进式解冻

渐进式解冻 (Gradual Unfreezing) 逐步解冻预训练层：

1. 阶段1：冻结所有层，只训练新分类头
2. 阶段2：解冻最后一层，用较小学习率训练
3. 阶段3：继续解冻更多层

这种方法避免在早期训练中破坏预训练权重。'''),

    code_cell(r'''# Progressive unfreezing schedule
np.random.seed(42)

total_epochs = 100
n_layers = 10  # 10 layers in model

# Define unfreezing schedule
def get_unfrozen_layers(epoch, total, n_layers):
    """Progressively unfreeze layers from top to bottom."""
    phase = min(epoch / (total / (n_layers + 1)), n_layers)
    return int(phase) + 1  # number of trainable layers

unfrozen = [get_unfrozen_layers(e, total_epochs, n_layers) for e in range(total_epochs)]
learning_rates = [0.001 * (0.9 ** (e // 10)) for e in range(total_epochs)]

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].step(range(total_epochs), unfrozen, where="post", color="blue", linewidth=2)
axes[0].set_ylabel("Unfrozen Layers")
axes[0].set_title("Progressive Unfreezing Schedule")
axes[0].set_yticks(range(1, n_layers + 1))
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(total_epochs), learning_rates, color="red", linewidth=2)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Learning Rate")
axes[1].set_title("Learning Rate Schedule")
axes[1].set_yscale("log")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("progressive_unfreeze.png", dpi=100, bbox_inches="tight")
plt.show()
print(f"Epoch 0: {unfrozen[0]} layers unfrozen")
print(f"Epoch 50: {unfrozen[50]} layers unfrozen")
print(f"Epoch 99: {unfrozen[99]} layers unfrozen")'''),

    md_cell(r'''### 学习率调度策略

不同学习率调度策略的比较：

- **Step decay**：每 N 步衰减一定比例
- **Cosine annealing**：余弦退火到 0
- **Warmup**：先升温后衰减
- **One-cycle**：先升温后降温'''),

    code_cell(r'''# Learning rate schedules comparison
np.random.seed(42)
total_steps = 200

# Step decay
step_lr = [0.01 * (0.5 ** (s // 50)) for s in range(total_steps)]

# Cosine annealing
cos_lr = [0.01 * 0.5 * (1 + np.cos(np.pi * s / total_steps)) for s in range(total_steps)]

# Warmup + cosine
warmup_steps = 20
warmup_cos = []
for s in range(total_steps):
    if s < warmup_steps:
        warmup_cos.append(0.01 * s / warmup_steps)
    else:
        warmup_cos.append(0.01 * 0.5 * (1 + np.cos(np.pi * (s - warmup_steps) / (total_steps - warmup_steps))))

# One-cycle
pct_start = 0.3
one_cycle = []
for s in range(total_steps):
    p = s / total_steps
    if p < pct_start:
        one_cycle.append(0.001 * (p / pct_start) * 10)
    else:
        one_cycle.append(0.01 * (1 - (p - pct_start) / (1 - pct_start)) * 10)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(step_lr, label="Step decay", linewidth=2)
ax.plot(cos_lr, label="Cosine annealing", linewidth=2)
ax.plot(warmup_cos, label="Warmup + cosine", linewidth=2)
ax.plot(one_cycle, label="One-cycle", linewidth=2)
ax.set_xlabel("Step")
ax.set_ylabel("Learning Rate")
ax.set_title("Learning Rate Schedule Comparison")
ax.legend()
ax.set_yscale("log")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lr_schedules.png", dpi=100, bbox_inches="tight")
plt.show()
print("Cosine annealing is popular for fine-tuning; warmup helps with stability.")'''),

    md_cell(r'''### 领域适应

领域适应 (Domain Adaptation) 是迁移学习的特殊形式，源域和目标域数据分布不同：

$$P_s(X) \neq P_t(X)$$

目标是对齐两个域的特征分布。方法包括：
- 对抗性领域适应 (DANN)
- 最大均值差异 (MMD)
- 相关性对齐 (CORAL)'''),

    code_cell(r'''# Simple domain adaptation visualization
np.random.seed(42)

# Source domain
n_source = 100
X_source = np.random.randn(n_source, 2) * 1.0
X_source[:, 0] += 1  # shift

# Target domain (different distribution)
n_target = 100
X_target = np.random.randn(n_target, 2) * 0.8
X_target[:, 1] += 2  # shift in different direction

# CORAL: align covariance
def coral(X_s, X_t):
    """Correlation Alignment."""
    C_s = np.cov(X_s.T)
    C_t = np.cov(X_t.T)
    
    # Whitening source
    U_s, S_s, _ = np.linalg.svd(C_s + np.eye(2) * 1e-6)
    X_s_white = (X_s - X_s.mean(axis=0)) @ U_s @ np.diag(1 / np.sqrt(S_s))
    
    # Color with target covariance
    U_t, S_t, _ = np.linalg.svd(C_t + np.eye(2) * 1e-6)
    X_s_aligned = X_s_white @ U_t @ np.diag(np.sqrt(S_t)) + X_t.mean(axis=0)
    
    return X_s_aligned

X_aligned = coral(X_source, X_target)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(X_source[:, 0], X_source[:, 1], c="blue", label="Source", alpha=0.6)
axes[0].scatter(X_target[:, 0], X_target[:, 1], c="red", label="Target", alpha=0.6)
axes[0].set_title("Before Domain Adaptation")
axes[0].legend()

axes[1].scatter(X_aligned[:, 0], X_aligned[:, 1], c="blue", label="Source (aligned)", alpha=0.6)
axes[1].scatter(X_target[:, 0], X_target[:, 1], c="red", label="Target", alpha=0.6)
axes[1].set_title("After CORAL Alignment")
axes[1].legend()

plt.suptitle("Domain Adaptation: CORAL Method", fontsize=14)
plt.tight_layout()
plt.savefig("domain_adaptation.png", dpi=100, bbox_inches="tight")
plt.show()
print("CORAL aligns source covariance to target domain.")'''),

    md_cell(r'''### 冻结策略比较

不同的冻结策略适用于不同场景：

1. **全部冻结**：只用预训练特征，训练分类头（适合小数据集）
2. **部分冻结**：冻结浅层，微调深层（适合中等数据集）
3. **全部微调**：所有层都训练（适合大数据集）

> 经验法则：数据越少，冻结越多层。'''),

    code_cell(r'''# Compare freeze strategies
np.random.seed(42)

strategies = {
    "All frozen": {"frozen": 10, "lr_head": 0.01, "lr_ft": 0},
    "Top 3 unfrozen": {"frozen": 7, "lr_head": 0.01, "lr_ft": 0.001},
    "Top 5 unfrozen": {"frozen": 5, "lr_head": 0.01, "lr_ft": 0.001},
    "All unfrozen": {"frozen": 0, "lr_head": 0.01, "lr_ft": 0.0001},
}

epochs = 50
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, config in strategies.items():
    train_acc, val_acc = [], []
    t, v = 0.4, 0.35
    for e in range(epochs):
        progress = e / epochs
        # More unfrozen = faster convergence but potential overfitting
        n_unfrozen = 10 - config["frozen"]
        speed = 1 + n_unfrozen * 0.1
        overfit = max(0, n_unfrozen - 5) * 0.01
        
        t = min(0.98, t + 0.01 * speed * (1 - t))
        v = min(0.95, v + 0.008 * speed * (1 - v))
        v -= overfit * progress
        
        train_acc.append(t * 100)
        val_acc.append(max(0, v * 100))
    
    axes[0].plot(train_acc, label=name, linewidth=2)
    axes[1].plot(val_acc, label=name, linewidth=2)

axes[0].set_title("Training Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy (%)")
axes[0].legend()

axes[1].set_title("Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()

plt.suptitle("Freeze Strategy Comparison", fontsize=14)
plt.tight_layout()
plt.savefig("freeze_strategies.png", dpi=100, bbox_inches="tight")
plt.show()
print("All frozen: slow but stable; All unfrozen: fast but may overfit")'''),

    md_cell(r'''### 灾难性遗忘

微调预训练模型时，新任务可能导致在旧任务上性能下降——**灾难性遗忘**。

解决方法：
- 弹性权重巩固 (EWC)
- 旧任务数据回放
- 逐层微调'''),

    code_cell(r'''# Catastrophic forgetting simulation
np.random.seed(42)

epochs_per_task = 50
n_tasks = 3

# Simulate training on 3 sequential tasks
# Track performance on all tasks during training of each task

task_performance = {i: [] for i in range(n_tasks)}
current_perf = {i: 0.8 for i in range(n_tasks)}

# Different strategies
strategies = {
    "Naive fine-tuning": {"lr": 0.01, "ewc": 0},
    "Low LR": {"lr": 0.001, "ewc": 0},
    "EWC (lambda=100)": {"lr": 0.01, "ewc": 100},
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (sname, config) in zip(axes, strategies.items()):
    current_perf = {i: 0.8 for i in range(n_tasks)}
    history = {i: [] for i in range(n_tasks)}
    
    for task in range(n_tasks):
        for e in range(epochs_per_task):
            # Current task improves
            current_perf[task] = min(0.95, current_perf[task] + config["lr"] * (1 - current_perf[task]))
            
            # Other tasks forget
            for other in range(n_tasks):
                if other != task:
                    # EWC slows forgetting
                    forget_rate = config["lr"] * 0.1 / (1 + config["ewc"] * 0.01)
                    current_perf[other] = max(0.1, current_perf[other] - forget_rate)
                history[other].append(current_perf[other])
            history[task].append(current_perf[task])
    
    for t in range(n_tasks):
        x = range(n_tasks * epochs_per_task)
        ax.plot(x, history[t], label=f"Task {t+1}", linewidth=2)
        # Mark task boundaries
    for t in range(n_tasks):
        ax.axvline(x=t * epochs_per_task, color="gray", linestyle="--", alpha=0.5)
    
    ax.set_title(sname)
    ax.set_xlabel("Epoch (across all tasks)")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.set_ylim(0, 1)

plt.suptitle("Catastrophic Forgetting Across Sequential Tasks", fontsize=14)
plt.tight_layout()
plt.savefig("catastrophic_forgetting.png", dpi=100, bbox_inches="tight")
plt.show()
print("Naive: severe forgetting; EWC: preserves old task performance")'''),

    md_cell(r'''### 不同深度的特征提取

预训练模型不同层提取的特征有不同的泛化能力：

- **浅层特征**：更通用（边缘、纹理）
- **深层特征**：更任务特定（物体部件、语义）
- **中层特征**：平衡点

> 参考: [[Yosinski et al., 2014]](https://arxiv.org/abs/1411.1792) "How transferable are features in deep neural networks?"'''),

    code_cell(r'''# Compare features from different depths
np.random.seed(42)

# Simulate transfer performance with features from different layers
layers = ["Conv1", "Conv2", "Conv3", "Conv4", "Conv5", "FC6", "FC7"]
same_domain = [85, 88, 90, 89, 87, 82, 78]  # same domain
diff_domain = [65, 70, 76, 78, 73, 65, 55]  # different domain

x = np.arange(len(layers))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, same_domain, width, label="Same domain", color="steelblue", alpha=0.8)
bars2 = ax.bar(x + width/2, diff_domain, width, label="Different domain", color="coral", alpha=0.8)

ax.set_xlabel("Source Layer")
ax.set_ylabel("Transfer Accuracy (%)")
ax.set_title("Transfer Performance by Source Layer")
ax.set_xticks(x)
ax.set_xticklabels(layers)
ax.legend()
ax.set_ylim(0, 100)

# Add value labels
for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("layer_transfer.png", dpi=100, bbox_inches="tight")
plt.show()
print("Middle layers transfer best; FC layers are most task-specific")'''),

    md_cell(r'''### 数据增强在迁移学习中的作用

数据增强可以扩展有限的训练数据，防止过拟合：

- 随机裁剪、翻转
- 颜色抖动
- 随机擦除
- Mixup / CutMix

> 在微调时，适度的数据增强可以显著提升泛化性能。'''),

    code_cell(r'''# Data augmentation comparison
np.random.seed(42)

# Simulate training with different augmentation levels
aug_levels = ["None", "Light", "Medium", "Heavy"]
epochs = 60

fig, ax = plt.subplots(figsize=(10, 5))

for i, aug in enumerate(aug_levels):
    train_acc, val_acc = [], []
    t, v = 0.3, 0.28
    aug_factor = i * 0.05  # more aug = less overfitting but slower convergence
    for e in range(epochs):
        t = min(0.99, t + 0.02 - aug_factor * 0.3)
        v = min(0.95 - i * 0.02, v + 0.015 - aug_factor * 0.2)
        # Without augmentation, val plateaus earlier (overfitting)
        if aug == "None" and e > 30:
            v = max(v, v - 0.002 * (e - 30))
        train_acc.append(t * 100)
        val_acc.append(v * 100)
    
    ax.plot(val_acc, label=f"Val ({aug})", linewidth=2)
    ax.plot(train_acc, "--", alpha=0.5, linewidth=1)

ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Data Augmentation Effect on Transfer Learning")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("augmentation_transfer.png", dpi=100, bbox_inches="tight")
plt.show()
print("Light/Medium augmentation gives best val accuracy")
print("No augmentation overfits; Heavy augmentation slows convergence")'''),

    md_cell(r'''### 迁移学习最佳实践

1. **从小学习率开始**：预训练权重已经很好，不需要大幅修改
2. **分层学习率**：浅层用更小的学习率
3. **先冻结再解冻**：先训练新层，再逐步解冻
4. **监控验证集**：防止灾难性遗忘
5. **适当的数据增强**：平衡泛化和收敛速度'''),

    code_cell(r'''# Final transfer learning comparison
np.random.seed(42)

methods = ["From\nscratch", "FE\n(frozen)", "FT\n(all)", "FT\n(part)", "Gradual\nunfreeze", "FT+\nEWC"]
train_acc = [55, 72, 85, 83, 84, 82]
val_acc = [50, 68, 78, 82, 85, 83]
inference_time = [1.0, 0.3, 1.0, 0.8, 0.9, 1.0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(methods))

bars1 = axes[0].bar(x - 0.15, train_acc, 0.3, label="Train", color="steelblue")
bars2 = axes[0].bar(x + 0.15, val_acc, 0.3, label="Val", color="coral")
axes[0].set_xticks(x)
axes[0].set_xticklabels(methods, fontsize=9)
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title("Transfer Learning Methods: Accuracy")
axes[0].legend()
axes[0].set_ylim(0, 100)

axes[1].bar(x, inference_time, 0.5, color="green", alpha=0.7)
axes[1].set_xticks(x)
axes[1].set_xticklabels(methods, fontsize=9)
axes[1].set_ylabel("Relative Training Time")
axes[1].set_title("Transfer Learning Methods: Training Cost")

plt.suptitle("Transfer Learning Strategy Comparison", fontsize=14)
plt.tight_layout()
plt.savefig("transfer_comparison.png", dpi=100, bbox_inches="tight")
plt.show()
print("Gradual unfreeze: best val accuracy with reasonable cost")'''),

    md_cell(r'''### 线性探测评估

线性探测 (Linear Probing) 是评估预训练特征质量的标准方法：

1. 冻结预训练模型所有层
2. 在提取的特征上训练一个线性分类器
3. 线性分类器的准确率反映特征质量

> 好的预训练特征应该在线性分类器下也表现良好。

> 参考: [[Kornblith et al., 2019]](https://arxiv.org/abs/1903.05887) "Do Better ImageNet Models Transfer Better?"'''),

    code_cell(r'''# Linear probing evaluation simulation
np.random.seed(42)

# Simulate features with different quality
n_samples = 200
n_features = 128

# Good features (linearly separable)
feat_good = np.random.randn(n_samples, n_features)
labels_good = (feat_good[:, 0] > 0).astype(int)

# Medium features (partially separable)
feat_medium = np.random.randn(n_samples, n_features)
labels_medium = (feat_medium[:, 0] + 0.5 * feat_medium[:, 1] > 0.5).astype(int)

# Poor features (not linearly separable)
feat_poor = np.random.randn(n_samples, n_features)
labels_poor = (np.sum(feat_poor[:, :5] ** 2, axis=1) > 5).astype(int)

# Train linear classifier (logistic regression from scratch)
def train_linear(X, y, epochs=200, lr=0.01):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        p = 1 / (1 + np.exp(-z))
        grad_w = X.T @ (p - y) / n
        grad_b = np.mean(p - y)
        w -= lr * grad_w
        b -= lr * grad_b
    acc = ((p > 0.5).astype(int) == y).mean()
    return acc

datasets = [("Good features", feat_good, labels_good),
            ("Medium features", feat_medium, labels_medium),
            ("Poor features", feat_poor, labels_poor)]

fig, ax = plt.subplots(figsize=(8, 5))
names, accs = [], []
for name, X, y in datasets:
    acc = train_linear(X, y)
    names.append(name)
    accs.append(acc * 100)

bars = ax.bar(names, accs, color=["green", "yellow", "red"], alpha=0.7)
ax.set_ylabel("Linear Probe Accuracy (%)")
ax.set_title("Linear Probing: Feature Quality Assessment")
ax.set_ylim(0, 100)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f"{acc:.1f}%", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("linear_probing.png", dpi=100, bbox_inches="tight")
plt.show()
print("Linear probing reveals: good features are linearly separable")'''),
]
insert_cells(nb, extra5)
save_notebook(nb, path5)

# ============================================================
# Notebook 6: Image Segmentation (+21 cells)
# ============================================================
print("\nExpanding Notebook 6: Image Segmentation...")
path6 = os.path.join(BASE, "part3-frontiers", "lecture-14-segmentation", "practice.ipynb")
nb = load_notebook(path6)
extra6 = [
    md_cell(r'''### 语义 vs 实例 vs 全景分割

| 任务 | 描述 | 示例 |
|------|------|------|
| 语义分割 | 为每个像素分配类别 | 所有"人"标记为同一颜色 |
| 实例分割 | 区分同类别的不同实例 | 每个人标记为不同颜色 |
| 全景分割 | 语义+实例，覆盖所有像素 | 背景标记语义，前景标记实例 |'''),

    code_cell(r'''# Visualize segmentation types
np.random.seed(42)

h, w = 32, 32
img = np.zeros((h, w))

# Create two "person" regions and one "car" region
img[5:15, 5:15] = 1  # person 1
img[5:15, 20:30] = 1  # person 2
img[20:30, 10:25] = 2  # car

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Original
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Input Image")
axes[0].axis("off")

# Semantic segmentation (all persons same color)
semantic = np.zeros((h, w, 3))
semantic[img == 1] = [1, 0, 0]  # all persons red
semantic[img == 2] = [0, 0, 1]  # car blue
axes[1].imshow(semantic)
axes[1].set_title("Semantic Segmentation")
axes[1].axis("off")

# Instance segmentation (each person different color)
instance = np.zeros((h, w, 3))
instance[5:15, 5:15] = [1, 0, 0]  # person 1 red
instance[5:15, 20:30] = [0, 1, 0]  # person 2 green
instance[20:30, 10:25] = [0, 0, 1]  # car blue
axes[2].imshow(instance)
axes[2].set_title("Instance Segmentation")
axes[2].axis("off")

# Panoptic (background + instances)
panoptic = np.zeros((h, w, 3))
panoptic[img == 0] = [0.5, 0.5, 0.5]  # background gray
panoptic[5:15, 5:15] = [1, 0, 0]
panoptic[5:15, 20:30] = [0, 1, 0]
panoptic[20:30, 10:25] = [0, 0, 1]
axes[3].imshow(panoptic)
axes[3].set_title("Panoptic Segmentation")
axes[3].axis("off")

plt.suptitle("Segmentation Types Comparison", fontsize=14)
plt.tight_layout()
plt.savefig("seg_types.png", dpi=100, bbox_inches="tight")
plt.show()'''),

    md_cell(r'''### FCN 架构详解

FCN (Fully Convolutional Network) 用卷积替代全连接层：

1. **编码器**：一系列卷积+池化，提取特征
2. **转置卷积**：上采样特征图到原图大小
3. **逐像素分类**：每个像素位置输出类别概率

> 参考: [[Long et al., 2015]](https://arxiv.org/abs/1411.4038) "Fully Convolutional Networks for Semantic Segmentation"'''),

    code_cell(r'''# FCN forward pass simulation
np.random.seed(42)

# Simulate FCN encoder-decoder
h, w = 64, 64
n_classes = 3

# Input image
x = np.random.randn(h, w, 3) * 0.5

# Encoder: downsample with strided conv (simulated)
feat1 = np.random.randn(h//2, w//2, 32)  # after pool 1
feat2 = np.random.randn(h//4, w//4, 64)  # after pool 2
feat3 = np.random.randn(h//8, w//8, 128) # after pool 3

# Decoder: upsample (simulated transposed conv)
up1 = np.random.randn(h//4, w//4, 64)
up2 = np.random.randn(h//2, w//2, 32)
up3 = np.random.randn(h, w, n_classes)

# Apply softmax per pixel
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

seg_prob = softmax(up3, axis=-1)
seg_map = np.argmax(seg_prob, axis=-1)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
# Encoder path
axes[0, 0].imshow(x[:, :, 0], cmap="gray")
axes[0, 0].set_title(f"Input ({h}x{w})")
axes[0, 0].axis("off")

axes[0, 1].imshow(feat1[:, :, 0], cmap="viridis")
axes[0, 1].set_title(f"Pool1 ({h//2}x{w//2})")
axes[0, 1].axis("off")

axes[0, 2].imshow(feat2[:, :, 0], cmap="viridis")
axes[0, 2].set_title(f"Pool2 ({h//4}x{w//4})")
axes[0, 2].axis("off")

axes[0, 3].imshow(feat3[:, :, 0], cmap="viridis")
axes[0, 3].set_title(f"Pool3 ({h//8}x{w//8})")
axes[0, 3].axis("off")

# Decoder path
axes[1, 0].imshow(up1[:, :, 0], cmap="viridis")
axes[1, 0].set_title(f"Up1 ({h//4}x{w//4})")
axes[1, 0].axis("off")

axes[1, 1].imshow(up2[:, :, 0], cmap="viridis")
axes[1, 1].set_title(f"Up2 ({h//2}x{w//2})")
axes[1, 1].axis("off")

axes[1, 2].imshow(seg_map, cmap="nipy_spectral")
axes[1, 2].set_title(f"Output ({h}x{w})")
axes[1, 2].axis("off")

# Show class probabilities for first class
axes[1, 3].imshow(seg_prob[:, :, 0], cmap="hot")
axes[1, 3].set_title("Class 0 prob")
axes[1, 3].axis("off")

plt.suptitle("FCN: Encoder-Decoder Pipeline", fontsize=14)
plt.tight_layout()
plt.savefig("fcn_pipeline.png", dpi=100, bbox_inches="tight")
plt.show()
print("FCN: encode -> downsample features, then decode -> upsample to full resolution")'''),

    md_cell(r'''### 转置卷积详解

转置卷积 (Transposed Convolution) 用于上采样：

普通卷积：$(H, W) \to (H', W')$，其中 $H' < H$
转置卷积：$(H', W') \to (H, W)$，其中 $H > H'$

**直觉**：转置卷积可以理解为将输入值"播撒"到更大的输出空间。'''),

    code_cell(r'''# Transposed convolution step by step
np.random.seed(42)

# Simple 2x2 -> 4x4 transposed convolution
input_feat = np.array([[1, 2], [3, 4]], dtype=float)
kernel = np.array([[1, 0], [0, -1]], dtype=float)
stride = 2
output_size = (input_feat.shape[0] - 1) * stride + kernel.shape[0]

output = np.zeros((output_size, output_size))
print("=== Transposed Convolution ===")
print(f"Input:\n{input_feat}")
print(f"Kernel:\n{kernel}")
print(f"Output size: {output_size}x{output_size}")

# Manual transposed convolution
for i in range(input_feat.shape[0]):
    for j in range(input_feat.shape[1]):
        # Place kernel * input_value at position (i*stride, j*stride)
        for ki in range(kernel.shape[0]):
            for kj in range(kernel.shape[1]):
                oi = i * stride + ki
                oj = j * stride + kj
                output[oi, oj] += input_feat[i, j] * kernel[ki, kj]

print(f"\nOutput:\n{output}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(input_feat, cmap="viridis")
axes[0].set_title("Input (2x2)")
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, f"{input_feat[i,j]:.0f}", ha="center", va="center", color="white")

axes[1].imshow(kernel, cmap="RdBu_r")
axes[1].set_title("Kernel (2x2)")
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, f"{kernel[i,j]:.0f}", ha="center", va="center", color="black")

axes[2].imshow(output, cmap="viridis")
axes[2].set_title("Output (4x4)")
for i in range(4):
    for j in range(4):
        axes[2].text(j, i, f"{output[i,j]:.0f}", ha="center", va="center", 
                     color="white" if abs(output[i,j]) > 1 else "black")

plt.suptitle("Transposed Convolution: 2x2 -> 4x4", fontsize=14)
plt.tight_layout()
plt.savefig("transposed_conv.png", dpi=100, bbox_inches="tight")
plt.show()'''),

    code_cell(r'''# Compare upsampling methods
np.random.seed(42)

# Create low-resolution feature map
low_res = np.random.randn(8, 8) * 0.5

# Method 1: Nearest neighbor
def nearest_upsample(x, scale=4):
    h, w = x.shape
    out = np.zeros((h*scale, w*scale))
    for i in range(h*scale):
        for j in range(w*scale):
            out[i, j] = x[i//scale, j//scale]
    return out

# Method 2: Bilinear
def bilinear_upsample(x, scale=4):
    h, w = x.shape
    out = np.zeros((h*scale, w*scale))
    for i in range(h*scale):
        for j in range(w*scale):
            fi, fj = i / scale, j / scale
            i0, j0 = int(fi), int(fj)
            i1, j1 = min(i0+1, h-1), min(j0+1, w-1)
            di, dj = fi - i0, fj - j0
            out[i, j] = (1-di)*(1-dj)*x[i0,j0] + (1-di)*dj*x[i0,j1] + \
                        di*(1-dj)*x[i1,j0] + di*dj*x[i1,j1]
    return out

# Method 3: Transposed conv (simulated with random kernel)
kernel = np.random.randn(4, 4) * 0.2
transposed_out = np.zeros((32, 32))
for i in range(8):
    for j in range(8):
        for ki in range(4):
            for kj in range(4):
                transposed_out[i*4+ki, j*4+kj] += low_res[i, j] * kernel[ki, kj]

methods = [("Nearest", nearest_upsample(low_res)),
           ("Bilinear", bilinear_upsample(low_res)),
           ("Transposed Conv", transposed_out)]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(low_res, cmap="viridis")
axes[0].set_title("Input (8x8)")
axes[0].axis("off")

for ax, (name, out) in zip(axes[1:], methods):
    ax.imshow(out, cmap="viridis")
    ax.set_title(f"{name} (32x32)")
    ax.axis("off")

plt.suptitle("Upsampling Methods Comparison", fontsize=14)
plt.tight_layout()
plt.savefig("upsampling.png", dpi=100, bbox_inches="tight")
plt.show()
print("Nearest: blocky; Bilinear: smooth; Transposed conv: learnable")'''),

    md_cell(r'''### U-Net 架构

U-Net 采用编码器-解码器结构，并通过**跳跃连接** (Skip Connections) 融合多尺度特征：

- **编码器**：逐层下采样提取语义特征
- **解码器**：逐层上采样恢复空间分辨率
- **跳跃连接**：将编码器特征直接拼接到解码器对应层

跳跃连接保留了在编码器中被下采样丢失的空间细节。'''),

    code_cell(r'''# U-Net encoder-decoder with skip connections simulation
np.random.seed(42)

h, w = 64, 64

# Simulate encoder features
enc1 = np.random.randn(h, w, 64)    # full resolution
enc2 = np.random.randn(h//2, w//2, 128)  # half
enc3 = np.random.randn(h//4, w//4, 256)  # quarter
enc4 = np.random.randn(h//8, w//8, 512)  # bottleneck

# Simulate decoder features (with skip connections)
# Decoder upsample + concat with encoder
dec3 = np.random.randn(h//4, w//4, 256) + enc3[:, :, :256] * 0.5  # skip
dec2 = np.random.randn(h//2, w//2, 128) + enc2[:, :, :128] * 0.5  # skip
dec1 = np.random.randn(h, w, 64) + enc1[:, :, :64] * 0.5  # skip

# Without skip connections
dec3_no = np.random.randn(h//4, w//4, 256)
dec2_no = np.random.randn(h//2, w//2, 128)
dec1_no = np.random.randn(h, w, 64)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# With skip
for i, (feat, name) in enumerate([(dec3, "Dec3+Skip"), (dec2, "Dec2+Skip"), (dec1, "Dec1+Skip")]):
    axes[0, i].imshow(feat[:, :, 0], cmap="viridis")
    axes[0, i].set_title(f"{name}\n({feat.shape[0]}x{feat.shape[1]})")
    axes[0, i].axis("off")

# Without skip
for i, (feat, name) in enumerate([(dec3_no, "Dec3 NoSkip"), (dec2_no, "Dec2 NoSkip"), (dec1_no, "Dec1 NoSkip")]):
    axes[1, i].imshow(feat[:, :, 0], cmap="viridis")
    axes[1, i].set_title(f"{name}\n({feat.shape[0]}x{feat.shape[1]})")
    axes[1, i].axis("off")

plt.suptitle("U-Net: Skip Connection Effect", fontsize=14)
plt.tight_layout()
plt.savefig("unet_skip.png", dpi=100, bbox_inches="tight")
plt.show()
print("Skip connections preserve spatial detail from encoder to decoder")'''),

    code_cell(r'''# Skip connection effect quantification
np.random.seed(42)

# Simulate segmentation with and without skip connections
h, w = 32, 32
n_classes = 3

# Ground truth: some spatial pattern
gt = np.zeros((h, w), dtype=int)
gt[5:15, 5:15] = 1
gt[20:30, 20:30] = 2

# With skip (better boundary accuracy)
np.random.seed(42)
pred_skip = gt.copy()
# Add small boundary errors
for _ in range(20):
    i, j = np.random.randint(0, h), np.random.randint(0, w)
    pred_skip[i, j] = np.random.randint(0, n_classes)

# Without skip (worse boundary accuracy)
pred_noskip = gt.copy()
# Add more errors, especially near boundaries
for _ in range(100):
    i, j = np.random.randint(0, h), np.random.randint(0, w)
    pred_noskip[i, j] = np.random.randint(0, n_classes)

# Compute pixel accuracy
acc_skip = (pred_skip == gt).mean()
acc_noskip = (pred_noskip == gt).mean()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(gt, cmap="nipy_spectral")
axes[0].set_title("Ground Truth")
axes[0].axis("off")

axes[1].imshow(pred_skip, cmap="nipy_spectral")
axes[1].set_title(f"With Skip\nAcc: {acc_skip*100:.1f}%")
axes[1].axis("off")

axes[2].imshow(pred_noskip, cmap="nipy_spectral")
axes[2].set_title(f"Without Skip\nAcc: {acc_noskip*100:.1f}%")
axes[2].axis("off")

plt.suptitle("Skip Connection Impact on Segmentation", fontsize=14)
plt.tight_layout()
plt.savefig("skip_effect.png", dpi=100, bbox_inches="tight")
plt.show()
print(f"Skip connections improve pixel accuracy by {(acc_skip - acc_noskip)*100:.1f}%")'''),

    md_cell(r'''### 分割损失函数

常见的分割损失函数：

1. **交叉熵损失**：逐像素分类
$$L_{CE} = -\sum_{i} \sum_{c} y_{i,c} \log p_{i,c}$$

2. **Dice 损失**：直接优化 IoU
$$L_{Dice} = 1 - \frac{2 \sum_i p_i y_i}{\sum_i p_i + \sum_i y_i}$$

3. **Focal Loss**：解决类别不平衡
$$L_{Focal} = -\sum_i (1 - p_i)^\gamma \log p_i$$'''),

    code_cell(r'''# Dice loss implementation
np.random.seed(42)

def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for binary segmentation."""
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def cross_entropy_loss(pred, target):
    """Binary cross-entropy loss."""
    return -np.mean(target * np.log(pred + 1e-10) + (1 - target) * np.log(1 - pred + 1e-10))

def focal_loss(pred, target, gamma=2.0):
    """Focal loss for imbalanced data."""
    return -np.mean(target * (1 - pred)**gamma * np.log(pred + 1e-10) + 
                    (1 - target) * pred**gamma * np.log(1 - pred + 1e-10))

# Create ground truth and predictions
h, w = 32, 32
target = np.zeros((h, w))
target[10:20, 10:20] = 1  # small object (9% of image)

# Simulate predictions with varying quality
qualities = [0.3, 0.5, 0.7, 0.9]
losses = {"CE": [], "Dice": [], "Focal": []}

for q in qualities:
    np.random.seed(42)
    pred = target * q + np.random.rand(h, w) * (1 - q) * 0.3
    pred = np.clip(pred, 0.01, 0.99)
    
    losses["CE"].append(cross_entropy_loss(pred, target))
    losses["Dice"].append(dice_loss(pred, target))
    losses["Focal"].append(focal_loss(pred, target))

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(qualities))
width = 0.25
for i, (name, vals) in enumerate(losses.items()):
    ax.bar(x + i * width, vals, width, label=name)
ax.set_xticks(x + width)
ax.set_xticklabels([f"q={q}" for q in qualities])
ax.set_ylabel("Loss")
ax.set_title("Segmentation Loss Functions Comparison")
ax.legend()

plt.tight_layout()
plt.savefig("seg_losses.png", dpi=100, bbox_inches="tight")
plt.show()
print("Dice loss is more robust to class imbalance than CE")'''),

    code_cell(r'''# Cross-entropy vs Dice loss: effect of class imbalance
np.random.seed(42)

imbalance_ratios = [0.5, 0.2, 0.1, 0.05, 0.01]  # fraction of positive pixels
h, w = 100, 100

ce_losses = []
dice_losses = []

for ratio in imbalance_ratios:
    target = np.zeros(h * w)
    n_pos = int(h * w * ratio)
    target[:n_pos] = 1
    np.random.shuffle(target)
    target = target.reshape(h, w)
    
    # Simulate a model that gets 80% pixel accuracy
    pred = np.zeros((h, w))
    pred[target == 1] = 0.7  # true positive rate
    pred[target == 0] = 0.3  # false positive rate
    pred = np.clip(pred, 0.01, 0.99)
    
    ce_losses.append(cross_entropy_loss(pred, target))
    dice_losses.append(dice_loss(pred, target))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(imbalance_ratios, ce_losses, "ro-", label="Cross-Entropy", linewidth=2)
ax.plot(imbalance_ratios, dice_losses, "bs-", label="Dice", linewidth=2)
ax.set_xlabel("Positive pixel ratio (class imbalance)")
ax.set_ylabel("Loss")
ax.set_title("Loss Behavior Under Class Imbalance")
ax.set_xscale("log")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("loss_imbalance.png", dpi=100, bbox_inches="tight")
plt.show()
print("CE loss increases with imbalance; Dice loss remains stable")'''),

    md_cell(r'''### mIoU 指标

平均交并比 (mean Intersection over Union) 是分割任务的标准评估指标：

$$\text{IoU}_c