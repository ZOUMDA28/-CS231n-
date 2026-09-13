#!/usr/bin/env python3
"""Generate remaining 8 CS231n practice notebooks following the teaching contract."""
import json
import os

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True) if isinstance(source, str) else source}

def code_cell(source):
    return {"cell_type": "code", "metadata": {}, "source": source.splitlines(keepends=True) if isinstance(source, str) else source, "outputs": [], "execution_count": None}

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def save_notebook(nb, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Cells: {len(nb['cells'])}, Size: {size_kb:.1f} KB")

BASE = r"d:\download\6aa68b099b47c2ba093ab510\cs231n-site\notebooks"

# ============================================================
# Notebook 1: Two-Layer Neural Network Complete Implementation
# ============================================================
print("Generating Notebook 1: Two-Layer Neural Network...")
nb1 = make_notebook([
    md_cell(r'''# 两层神经网络完整实现

> **斯坦福 CS231n Assignment 1 Q3+Q5** | Two-Layer Neural Network

## 本章导读

本节从零实现一个完整的两层神经网络。我们将经历从直觉到手算、从代码到实验的完整路径，理解参数化模型、前向传播、反向传播和训练循环的每一个细节。

**学习目标：**
- 理解参数化模型 $f(x) = W_2 \text{ReLU}(W_1 x + b_1) + b_2$ 的结构
- 手算前向传播和反向传播
- 从零实现 TwoLayerNet 类（前向、损失、反向、训练）
- 调节超参数（隐藏层大小、学习率、正则化）
- 可视化损失曲线和决策边界

**参考来源：** [CS231n Assignment 1](https://cs231n.github.io/assignments2021/assignment1/) | [Neural Nets Notes](https://cs231n.github.io/neural-networks-1/)'''),
    md_cell(r'''## 1. 直觉理解：参数化模型

### 从 k-NN 到参数化模型

k-NN 的问题在于「预测时才计算」，没有从训练数据中提取模式。参数化模型的核心思想是：**用一组参数 $W, b$ 压缩训练数据的模式**，预测时只需 $f(x) = Wx + b$。

### 两层神经网络的结构

$$f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$$

- **第一层**（隐藏层）：$z_1 = W_1 x + b_1$，然后 $a_1 = \text{ReLU}(z_1)$
- **第二层**（输出层）：$s = W_2 a_1 + b_2$（分类得分）
- **ReLU 激活**：$\text{ReLU}(z) = \max(0, z)$，引入非线性

### 为什么需要两层？

单层线性分类器只能学习线性决策边界。加入隐藏层和 ReLU 后，网络可以拟合任意非线性决策边界——这是**通用近似定理**的基础。'''),
    md_cell(r'''### 前向传播的数据流

```
输入 x (D维)
  ↓  W1 (D×H), b1 (H维)
线性变换 z1 = W1·x + b1
  ↓  ReLU
激活 a1 = ReLU(z1)
  ↓  W2 (H×C), b2 (C维)
得分 s = W2·a1 + b2
  ↓  Softmax
概率 p = softmax(s)
  ↓  交叉熵
损失 L = -log(p[正确类别])
```

### 反向传播的梯度流

反向传播沿计算图反向传递梯度：
1. 从损失开始：$\frac{\partial L}{\partial s}$
2. 通过 softmax：$\frac{\partial L}{\partial s} = p - \text{one\_hot}(y)$
3. 回到隐藏层：$\frac{\partial L}{\partial a_1} = \frac{\partial L}{\partial s} \cdot W_2^T$
4. 通过 ReLU：$\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1} \cdot \mathbb{1}[z_1 > 0]$
5. 到权重：$\frac{\partial L}{\partial W_1} = x^T \cdot \frac{\partial L}{\partial z_1}$'''),
    md_cell(r'''## 2. 手算验证：前向传播

### 2D 玩具数据手算

设输入 $x = [1, 2]^T$，3 个类别，隐藏层 2 维。

**权重初始化：**
$$W_1 = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \end{bmatrix}, \quad b_1 = [0.0, 0.1]^T$$
$$W_2 = \begin{bmatrix} 0.5 & 0.6 \\ 0.7 & 0.8 \\ 0.9 & 1.0 \end{bmatrix}, \quad b_2 = [0.0, 0.0, 0.0]^T$$

**前向传播：**

1. $z_1 = W_1 x + b_1 = [0.1 \times 1 + 0.2 \times 2 + 0.0, \; 0.3 \times 1 + 0.4 \times 2 + 0.1] = [0.5, 1.2]$

2. $a_1 = \text{ReLU}(z_1) = [\max(0, 0.5), \max(0, 1.2)] = [0.5, 1.2]$

3. $s = W_2 a_1 + b_2 = [0.5 \times 0.5 + 0.6 \times 1.2, \; 0.7 \times 0.5 + 0.8 \times 1.2, \; 0.9 \times 0.5 + 1.0 \times 1.2]$
   $= [0.97, 1.31, 1.65]$'''),
    md_cell(r'''### 手算 Softmax 损失

得分 $s = [0.97, 1.31, 1.65]$，假设正确标签 $y = 0$。

1. 数值稳定化：$s_{max} = 1.65$，$s' = s - 1.65 = [-0.68, -0.34, 0.0]$
2. 指数化：$e^{s'} = [0.5066, 0.7118, 1.0]$
3. 归一化：$Z = 2.2184$，$p = [0.2283, 0.3209, 0.4508]$
4. 交叉熵损失：$L = -\log(p_0) = -\log(0.2283) \approx 1.477$

### 手算反向传播

1. $\frac{\partial L}{\partial s} = p - \text{one\_hot}(y) = [0.2283 - 1, 0.3209, 0.4508] = [-0.7717, 0.3209, 0.4508]$

2. $\frac{\partial L}{\partial W_2}$：$\frac{\partial L}{\partial s}^T \cdot a_1$（外积）
   - 第一行：$[-0.7717 \times 0.5, -0.7717 \times 1.2] = [-0.3859, -0.9260]$
   - 第二行：$[0.3209 \times 0.5, 0.3209 \times 1.2] = [0.1604, 0.3851]$
   - 第三行：$[0.4508 \times 0.5, 0.4508 \times 1.2] = [0.2254, 0.5410]$

3. $\frac{\partial L}{\partial a_1} = \frac{\partial L}{\partial s} \cdot W_2$
   - $a_{1,1}$：$-0.7717 \times 0.5 + 0.3209 \times 0.7 + 0.4508 \times 0.9 = 0.0412$
   - $a_{1,2}$：$-0.7717 \times 0.6 + 0.3209 \times 0.8 + 0.4508 \times 1.0 = 0.0606$

4. $\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1} \cdot \mathbb{1}[z_1 > 0] = [0.0412, 0.0606]$（ReLU 梯度，$z_1$ 全为正）

5. $\frac{\partial L}{\partial W_1} = x^T \cdot \frac{\partial L}{\partial z_1}$
   - 第一行：$[1 \times 0.0412, 1 \times 0.0606] = [0.0412, 0.0606]$
   - 第二行：$[2 \times 0.0412, 2 \times 0.0606] = [0.0824, 0.1212]$'''),
    code_cell(r'''# 代码实现：验证手算结果
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
np.random.seed(42)

# 手算验证
x = np.array([1.0, 2.0])
y = 0  # correct label

W1 = np.array([[0.1, 0.2], [0.3, 0.4]])
b1 = np.array([0.0, 0.1])
W2 = np.array([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]])
b2 = np.array([0.0, 0.0, 0.0])

# Forward pass
z1 = W1.T @ x + b1
a1 = np.maximum(0, z1)
s = W2 @ a1 + b2
print(f"z1 = {z1}")           # [0.5, 1.2]
print(f"a1 = {a1}")           # [0.5, 1.2]
print(f"scores = {s}")         # [0.97, 1.31, 1.65]

# Softmax loss
shifted = s - np.max(s)
exp_s = np.exp(shifted)
probs = exp_s / np.sum(exp_s)
loss = -np.log(probs[y])
print(f"probs = {probs}")
print(f"loss = {loss:.4f}")    # ~1.477

# Backward pass
dscores = probs.copy()
dscores[y] -= 1
print(f"dscores = {dscores}")  # [-0.7717, 0.3209, 0.4508]

dW2 = np.outer(dscores, a1)
print(f"dW2 = \n{dW2}")

da1 = W2.T @ dscores
print(f"da1 = {da1}")

dz1 = da1 * (z1 > 0)
print(f"dz1 = {dz1}")

dW1 = np.outer(x, dz1)
print(f"dW1 = \n{dW1}")'''),
    md_cell(r'''## 3. 代码实现：TwoLayerNet 类

现在实现完整的两层神经网络类。这个类包含：
- `__init__`：初始化权重（He 初始化）
- `forward`：前向传播，返回得分和缓存
- `loss`：计算损失和梯度
- `train`：SGD 训练循环
- `predict`：预测

### 权重初始化：He 初始化

对于 ReLU 激活，推荐使用 He 初始化：
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{D_{in}}}\right)$$

这确保前向传播时方差不爆炸或消失。'''),
    code_cell(r'''class TwoLayerNet:
    """Two-layer neural network with ReLU and Softmax.
    
    Architecture: input -> FC -> ReLU -> FC -> Softmax
    """
    def __init__(self, input_dim, hidden_dim, num_classes, reg=0.0):
        # He initialization for ReLU
        self.params = {
            'W1': np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim),
            'b1': np.zeros(hidden_dim),
            'W2': np.random.randn(hidden_dim, num_classes) * np.sqrt(2.0 / hidden_dim),
            'b2': np.zeros(num_classes),
        }
        self.reg = reg
    
    def forward(self, X):
        """Forward pass. Returns scores and cache."""
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        z1 = X @ W1 + b1          # (N, H)
        a1 = np.maximum(0, z1)    # (N, H) ReLU
        scores = a1 @ W2 + b2     # (N, C)
        cache = (X, W1, b1, z1, a1, W2, b2)
        return scores, cache
    
    def loss(self, X, y):
        """Compute softmax loss and gradients."""
        scores, cache = self.forward(X)
        N = X.shape[0]
        # Numerical stability: shift by max
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # Cross-entropy loss
        correct_logprobs = -np.log(probs[np.arange(N), y])
        data_loss = np.sum(correct_logprobs) / N
        reg_loss = 0.5 * self.reg * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))
        loss = data_loss + reg_loss
        
        # Backward pass
        X, W1, b1, z1, a1, W2, b2 = cache
        dscores = probs.copy()
        dscores[np.arange(N), y] -= 1
        dscores /= N
        
        dW2 = a1.T @ dscores + self.reg * W2
        db2 = np.sum(dscores, axis=0)
        da1 = dscores @ W2.T
        dz1 = da1 * (z1 > 0)  # ReLU gradient
        dW1 = X.T @ dz1 + self.reg * W1
        db1 = np.sum(dz1, axis=0)
        
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
        return loss, grads
    
    def train(self, X, y, lr=1e-2, num_iters=1000, batch_size=None, verbose=False):
        """Train with SGD."""
        N = X.shape[0]
        if batch_size is None:
            batch_size = N
        losses = []
        for it in range(num_iters):
            batch_idx = np.random.choice(N, batch_size, replace=True) if batch_size < N else np.arange(N)
            X_batch, y_batch = X[batch_idx], y[batch_idx]
            loss, grads = self.loss(X_batch, y_batch)
            for k in self.params:
                self.params[k] -= lr * grads[k]
            if verbose and it % 100 == 0:
                print(f"  iter {it}: loss = {loss:.4f}")
            losses.append(loss)
        return losses
    
    def predict(self, X):
        scores, _ = self.forward(X)
        return np.argmax(scores, axis=1)

print("TwoLayerNet class defined successfully")'''),
    md_cell(r'''## 4. 生成 2D 玩具数据

为了可视化决策边界，我们生成螺旋形 2D 数据。螺旋数据是非线性可分的经典基准——线性分类器无法处理，但两层神经网络可以。'''),
    code_cell(r'''def generate_spiral_data(n_per_class=100, n_classes=3, noise=0.2):
    """Generate spiral 2D toy data for classification."""
    X = np.zeros((n_per_class * n_classes, 2))
    y = np.zeros(n_per_class * n_classes, dtype=int)
    for c in range(n_classes):
        idx = range(n_per_class * c, n_per_class * (c + 1))
        radius = np.linspace(0.0, 1.0, n_per_class)
        theta = np.linspace(c * 2 * np.pi / n_classes, 
                           (c + 2.5) * 2 * np.pi / n_classes, n_per_class)
        X[idx] = np.c_[radius * np.sin(theta), radius * np.cos(theta)] + \
                 np.random.randn(n_per_class, 2) * noise
        y[idx] = c
    return X, y

np.random.seed(42)
X_spiral, y_spiral = generate_spiral_data(n_per_class=100, n_classes=3)
print(f"Data shape: X={X_spiral.shape}, y={y_spiral.shape}")
print(f"Classes: {np.unique(y_spiral)}")

fig, ax = plt.subplots(figsize=(7, 6))
colors = ['#ea580c', '#2563eb', '#16a34a']
for c in range(3):
    mask = y_spiral == c
    ax.scatter(X_spiral[mask, 0], X_spiral[mask, 1], c=colors[c], 
              edgecolors='k', s=30, label=f'Class {c}')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Spiral Toy Data (3 classes)')
ax.legend()
plt.tight_layout()
plt.savefig('spiral_data.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 1: Spiral toy data")'''),
    md_cell(r'''## 5. 训练循环

使用 SGD（随机梯度下降）训练网络。关键超参数：
- **学习率** `lr`：控制步长大小
- **迭代次数** `num_iters`：训练步数
- **正则化** `reg`：L2 正则化强度
- **隐藏层大小** `hidden_dim`：控制模型容量'''),
    code_cell(r'''# Train TwoLayerNet on spiral data
np.random.seed(42)
net = TwoLayerNet(input_dim=2, hidden_dim=50, num_classes=3, reg=1e-3)
losses = net.train(X_spiral, y_spiral, lr=0.5, num_iters=1500, verbose=True)

# Evaluate
train_acc = np.mean(net.predict(X_spiral) == y_spiral)
print(f"\nTraining accuracy: {train_acc:.2%}")
print(f"Final loss: {losses[-1]:.4f}")'''),
    code_cell(r'''# Visualize loss curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(losses, color='#ea580c', linewidth=1.5, alpha=0.8)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title('Training Loss Curve (1500 iterations)')
ax.set_ylim(0, max(losses) * 1.1)
plt.tight_layout()
plt.savefig('training_loss_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 2: Training loss curve")'''),
    code_cell(r'''# Visualize decision boundary
def plot_decision_boundary(net, X, y, title='Decision Boundary'):
    x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
    y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = net.predict(grid).reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(xx, yy, Z, alpha=0.2, cmap='Set1')
    for c in range(3):
        mask = y == c
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[c], 
                  edgecolors='k', s=30, label=f'Class {c}')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig('decision_boundary.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_decision_boundary(net, X_spiral, y_spiral, 'Decision Boundary (hidden=50, lr=0.5)')
print("Visualization 3: Decision boundary")'''),
    md_cell(r'''## 6. 实验观察：超参数调节

### 6.1 隐藏层大小的影响

隐藏层大小控制模型容量：
- 太小（如 5）：欠拟合，无法表达螺旋边界
- 适中（如 50）：良好拟合
- 太大（如 200）：可能过拟合'''),
    code_cell(r'''# Experiment: effect of hidden layer size
np.random.seed(42)
hidden_sizes = [5, 10, 25, 50, 100, 200]
results = []

for h in hidden_sizes:
    net = TwoLayerNet(input_dim=2, hidden_dim=h, num_classes=3, reg=1e-3)
    losses = net.train(X_spiral, y_spiral, lr=0.5, num_iters=1500)
    acc = np.mean(net.predict(X_spiral) == y_spiral)
    results.append((h, acc, losses[-1]))
    print(f"hidden={h:3d}: accuracy={acc:.2%}, final_loss={losses[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax1, ax2 = axes
ax1.bar([str(h) for h, _, _ in results], [a for _, a, _ in results], color='#ea580c')
ax1.set_xlabel('Hidden Layer Size')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy vs Hidden Size')

for h, _, l in results:
    ax2.plot(l, alpha=0.6, label=f'h={h}')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss')
ax2.set_title('Loss Curves for Different Hidden Sizes')
ax2.legend()
plt.tight_layout()
plt.savefig('hidden_size_experiment.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 4: Hidden layer size comparison")'''),
    md_cell(r'''### 6.2 学习率的影响

学习率是最关键的超参数：
- 太小（0.001）：收敛极慢
- 适中（0.1-1.0）：快速收敛
- 太大（10.0）：发散，损失爆炸'''),
    code_cell(r'''# Experiment: effect of learning rate
np.random.seed(42)
learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]
lr_results = []

for lr in learning_rates:
    net = TwoLayerNet(input_dim=2, hidden_dim=50, num_classes=3, reg=1e-3)
    losses = net.train(X_spiral, y_spiral, lr=lr, num_iters=1000)
    acc = np.mean(net.predict(X_spiral) == y_spiral)
    lr_results.append((lr, acc, losses))
    print(f"lr={lr:.3f}: accuracy={acc:.2%}, final_loss={losses[-1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
for lr, _, losses in lr_results:
    ax.plot(losses, alpha=0.7, label=f'lr={lr}', linewidth=1.5)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title('Loss Curves for Different Learning Rates')
ax.legend()
ax.set_ylim(0, 3)
plt.tight_layout()
plt.savefig('learning_rate_experiment.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 5: Learning rate comparison")'''),
    md_cell(r'''### 6.3 正则化的影响

L2 正则化通过惩罚大权重来防止过拟合：
$$L_{total} = L_{data} + \lambda \|W\|^2$$

- $\lambda = 0$：无正则化，可能过拟合
- $\lambda$ 适中：防止过拟合
- $\lambda$ 太大：欠拟合（权重被压制到接近 0）'''),
    code_cell(r'''# Experiment: effect of regularization
np.random.seed(42)
reg_strengths = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
reg_results = []

for reg in reg_strengths:
    net = TwoLayerNet(input_dim=2, hidden_dim=100, num_classes=3, reg=reg)
    losses = net.train(X_spiral, y_spiral, lr=0.5, num_iters=1000)
    acc = np.mean(net.predict(X_spiral) == y_spiral)
    reg_results.append((reg, acc, losses[-1]))
    print(f"reg={reg:.4f}: accuracy={acc:.2%}, final_loss={losses[-1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
regs_str = [f'{r:.0e}' for r, _, _ in reg_results]
ax.bar(regs_str, [a for _, a, _ in reg_results], color='#2563eb')
ax.set_xlabel('Regularization Strength')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vs Regularization Strength')
plt.tight_layout()
plt.savefig('regularization_experiment.png', dpi=150, bbox_inches='tight')
plt.show()
print("Experiment completed: regularization effect")'''),
    md_cell(r'''## 7. 数值梯度检验

验证反向传播的正确性：用数值梯度（有限差分）与解析梯度比较。

$$\frac{\partial f}{\partial x} \approx \frac{f(x + h) - f(x - h)}{2h}$$

相对误差应小于 $10^{-5}$。'''),
    code_cell(r'''# Numerical gradient checking
def numerical_grad(f, x, h=1e-5):
    """Compute numerical gradient using central difference."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]
        x[idx] = old_val + h
        f_plus = f(x)
        x[idx] = old_val - h
        f_minus = f(x)
        x[idx] = old_val
        grad[idx] = (f_plus - f_minus) / (2 * h)
        it.iternext()
    return grad

np.random.seed(42)
# Small test
net = TwoLayerNet(input_dim=4, hidden_dim=10, num_classes=3, reg=0.1)
X_test = np.random.randn(5, 4)
y_test = np.array([0, 1, 2, 0, 1])

loss, grads = net.loss(X_test, y_test)

for name in ['W1', 'W2']:
    param = net.params[name]
    def loss_fn(p, name=name):
        net.params[name] = p
        l, _ = net.loss(X_test, y_test)
        return l
    
    num_grad = numerical_grad(loss_fn, param.copy())
    rel_error = np.max(np.abs(num_grad - grads[name]) / 
                       (np.maximum(1e-8, np.abs(num_grad) + np.abs(grads[name]))))
    print(f"{name}: max relative error = {rel_error:.2e}")
    assert rel_error < 1e-5, f"Gradient check failed for {name}!"
    
print("\nGradient check passed! Backward propagation is correct.")'''),
    md_cell(r'''## 8. 完整训练流程可视化

将损失曲线和决策边界并排展示，观察训练过程中模型如何逐步学习非线性决策边界。'''),
    code_cell(r'''# Full training pipeline with periodic snapshots
np.random.seed(42)
net = TwoLayerNet(input_dim=2, hidden_dim=50, num_classes=3, reg=1e-3)

snapshots = [0, 100, 300, 600, 1000, 1500]
snapshot_models = []
all_losses = []

for it in range(1501):
    loss, grads = net.loss(X_spiral, y_spiral)
    for k in net.params:
        net.params[k] -= 0.5 * grads[k]
    all_losses.append(loss)
    if it in snapshots:
        # Save a copy of predictions
        snapshot_models.append((it, net.predict(X_spiral).copy(), 
                               net.forward(X_spiral)[0].copy()))

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, (it, preds, scores) in zip(axes.flat, snapshot_models):
    acc = np.mean(preds == y_spiral)
    x_min, x_max = X_spiral[:, 0].min() - 0.3, X_spiral[:, 0].max() + 0.3
    y_min, y_max = X_spiral[:, 1].min() - 0.3, X_spiral[:, 1].max() + 0.3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # Recreate forward pass for grid
    z1 = grid @ net.params['W1'] + net.params['b1']
    a1 = np.maximum(0, z1)
    s = a1 @ net.params['W2'] + net.params['b2']
    Z = np.argmax(s, axis=1).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.2, cmap='Set1')
    for c in range(3):
        mask = y_spiral == c
        ax.scatter(X_spiral[mask, 0], X_spiral[mask, 1], c=colors[c], s=15)
    ax.set_title(f'iter={it}, acc={acc:.1%}')
plt.suptitle('Training Progress: Decision Boundary Evolution', fontsize=14)
plt.tight_layout()
plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization: Training progress snapshots")'''),
    md_cell(r'''## 作业

### 作业 1：实现三层神经网络

在 TwoLayerNet 的基础上，扩展为三层网络（两个隐藏层）：
$$f(x) = W_3 \cdot \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2) + b_3$$

**要求**：实现前向传播和反向传播。'''),
    code_cell(r'''# Homework 1: Three-layer neural network
class ThreeLayerNet:
    def __init__(self, input_dim, hidden1, hidden2, num_classes, reg=0.0):
        self.params = {
            'W1': np.random.randn(input_dim, hidden1) * np.sqrt(2.0 / input_dim),
            'b1': np.zeros(hidden1),
            'W2': np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1),
            'b2': np.zeros(hidden2),
            'W3': np.random.randn(hidden2, num_classes) * np.sqrt(2.0 / hidden2),
            'b3': np.zeros(num_classes),
        }
        self.reg = reg
    
    def forward(self, X):
        """Forward pass: input -> FC -> ReLU -> FC -> ReLU -> FC -> scores"""
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        # TODO: implement forward pass
        # z1 = ...; a1 = ...; z2 = ...; a2 = ...; scores = ...
        # return scores, (X, W1, b1, z1, a1, W2, b2, z2, a2, W3, b3)
        pass
    
    def loss(self, X, y):
        # TODO: compute loss and gradients
        pass
    
    def predict(self, X):
        scores, _ = self.forward(X)
        return np.argmax(scores, axis=1)

# Test
np.random.seed(42)
net3 = ThreeLayerNet(2, 30, 20, 3, reg=1e-3)
# Uncomment after implementation:
# losses = net3.train(X_spiral, y_spiral, lr=0.5, num_iters=1000)
# acc = np.mean(net3.predict(X_spiral) == y_spiral)
# assert acc > 0.8, "Three-layer net should achieve >80% accuracy"
print("Homework 1 template ready. Implement forward and loss methods.")'''),
    md_cell(r'''### 作业 2：实现 Adam 优化器

将 SGD 替换为 Adam 优化器，比较训练效果。

[[Kingma & Ba, 2015]](https://arxiv.org/abs/1412.6980)'''),
    code_cell(r'''# Homework 2: Adam optimizer for TwoLayerNet
def train_with_adam(net, X, y, lr=1e-3, num_iters=1000, 
                     beta1=0.9, beta2=0.999, eps=1e-8):
    """Train network with Adam optimizer.
    
    m_t = beta1 * m_{t-1} + (1-beta1) * g_t
    v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    param -= lr * m_hat / (sqrt(v_hat) + eps)
    """
    m = {k: np.zeros_like(v) for k, v in net.params.items()}
    v = {k: np.zeros_like(v) for k, v in net.params.items()}
    losses = []
    for t in range(1, num_iters + 1):
        loss, grads = net.loss(X, y)
        # TODO: implement Adam update
        pass
        losses.append(loss)
    return losses

# Test
np.random.seed(42)
net_adam = TwoLayerNet(2, 50, 3, reg=1e-3)
# Uncomment after implementation:
# losses_adam = train_with_adam(net_adam, X_spiral, y_spiral, lr=1e-2, num_iters=1000)
# acc_adam = np.mean(net_adam.predict(X_spiral) == y_spiral)
# assert acc_adam > 0.9, "Adam should achieve >90% accuracy"
print("Homework 2 template ready. Implement Adam update rule.")'''),
    md_cell(r'''### 作业 3：实现 Dropout 正则化

在隐藏层加入 Dropout，验证其正则化效果。

[[Srivastava et al., 2014]](https://jmlr.org/papers/v15/srivastava14a.html)'''),
    code_cell(r'''# Homework 3: Dropout in TwoLayerNet
def forward_with_dropout(self, X, p_drop=0.5, training=True):
    """Forward pass with inverted dropout.
    
    During training: mask = (rand < p_keep) / p_keep
    During inference: no modification
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    
    z1 = X @ W1 + b1
    a1 = np.maximum(0, z1)
    
    # TODO: Apply dropout to a1
    # if training: mask = ...; a1 = a1 * mask
    pass
    
    scores = a1 @ W2 + b2
    return scores

# Test
np.random.seed(42)
net_drop = TwoLayerNet(2, 50, 3, reg=1e-3)
# Verify dropout: output should be different each forward call
# s1, _ = net_drop.forward_with_dropout(X_spiral, p_drop=0.5, training=True)
# s2, _ = net_drop.forward_with_dropout(X_spiral, p_drop=0.5, training=True)
# assert not np.allclose(s1, s2), "Dropout should produce different outputs"
# Verify inference: output should be deterministic
# s3, _ = net_drop.forward_with_dropout(X_spiral, p_drop=0.5, training=False)
# s4, _ = net_drop.forward_with_dropout(X_spiral, p_drop=0.5, training=False)
# assert np.allclose(s3, s4), "Inference should be deterministic"
print("Homework 3 template ready. Implement dropout forward pass.")'''),
    md_cell(r'''## 小结

| 概念 | 要点 |
|------|------|
| **参数化模型** | 用参数 W, b 压缩数据模式，预测只需矩阵乘法 |
| **前向传播** | $f(x) = W_2 \text{ReLU}(W_1 x + b_1) + b_2$ |
| **反向传播** | 链式法则沿计算图反向传递梯度 |
| **ReLU** | $\max(0, z)$，解决梯度消失，梯度为 0 或 1 |
| **Softmax** | 将得分转为概率分布，配合交叉熵损失 |
| **SGD** | $W \leftarrow W - \eta \nabla L$ |
| **He 初始化** | $W \sim N(0, \sqrt{2/D})$，适合 ReLU |
| **超参数** | 学习率最重要，隐藏层大小控制容量 |

**关键洞察**：两层网络能拟合任意非线性决策边界，但需要仔细调节超参数。学习率过大导致发散，过小导致收敛慢。正则化防止过拟合但太强会欠拟合。'''),
    md_cell(r'''## 参考文献

1. **CS231n Neural Networks Notes** - [cs231n.github.io/neural-networks-1](https://cs231n.github.io/neural-networks-1/)
2. **CS231n Optimization Notes** - [cs231n.github.io/optimization-2](https://cs231n.github.io/optimization-2/)
3. [[He et al., 2015]](https://arxiv.org/abs/1502.01852) He et al. *Delving Deep into Rectifiers*. He 初始化.
4. [[Glorot & Bengio, 2010]](https://proceedings.mlr.press/v9/glorot10a.html) Glorot & Bengio. *Understanding the difficulty of training deep feedforward neural networks*.
5. [[Kingma & Ba, 2015]](https://arxiv.org/abs/1412.6980) Kingma & Ba. *Adam: A Method for Stochastic Optimization*.
6. [[Srivastava et al., 2014]](https://jmlr.org/papers/v15/srivastava14a.html) Srivastava et al. *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*.

---

> 本节内容参考 [Stanford CS231n Assignment 1](https://cs231n.github.io/assignments2021/assignment1/)'''),
    md_cell(r'''---

## 参考代码实现

以下 GitHub 仓库提供了本节内容的完整代码实现，建议结合学习：

- **[jariasf/CS231n](https://github.com/jariasf/CS231n)** (486 stars): CS231n 完整作业解答（KNN、SVM、Softmax、两层网络）
  - 仓库地址: https://github.com/jariasf/CS231n

- **[rishabh-16/cs231n-2019-assignments](https://github.com/rishabh-16/cs231n-2019-assignments)** (212 stars): 2019版 CS231n 作业 PyTorch+TensorFlow 解答
  - 仓库地址: https://github.com/rishabh-16/cs231n-2019-assignments

- **[amanchadha/stanford-cs231n-assignments-2020](https://github.com/amanchadha/stanford-cs231n-assignments-2020)** (175 stars): 2020春季 CS231n 完整作业解答
  - 仓库地址: https://github.com/amanchadha/stanford-cs231n-assignments-2020

> 标注说明: 以上仓库按热度排序，优先推荐 stars 最多的实现.'''),
])
save_notebook(nb1, os.path.join(BASE, "part1-deep-learning-basics", "lecture-04b-two-layer-net", "practice.ipynb"))

# ============================================================
# Notebook 2: Image Feature Engineering (HOG + Color Histograms)
# ============================================================
print("Generating Notebook 2: Image Feature Engineering...")
nb2 = make_notebook([
    md_cell(r'''# 图像特征工程

> **斯坦福 CS231n Assignment 1 Q4** | Image Feature Engineering

## 本章导读

在深度学习之前，计算机视觉依赖手工设计的特征来表示图像。本节介绍两种经典特征：HOG（方向梯度直方图）和颜色直方图，并比较它们与原始像素的分类效果。

**学习目标：**
- 理解 HOG 特征的原理（梯度计算、方向直方图、块归一化）
- 实现颜色直方图特征提取
- 比较 raw pixels vs HOG vs color histogram 的分类效果
- 理解深度学习为何取代手工特征

**参考来源：** [CS231n Assignment 1 Q4](https://cs231n.github.io/assignments2021/assignment1/) | [HOG Paper](https://ieeexplore.ieee.org/document/1467360)'''),
    md_cell(r'''## 1. 直觉理解：为什么需要特征工程？

### 原始像素的局限

原始像素作为特征有三个问题：
1. **高维度**：32×32×3 = 3072 维
2. **无空间结构**：像素之间的空间关系丢失
3. **光照敏感**：亮度的变化改变所有像素值

### 特征工程的目标

将原始像素转换为更紧凑、更有区分力的表示：
- **HOG**：捕捉边缘和形状信息
- **颜色直方图**：捕捉颜色分布
- **SIFT**：局部关键点描述子

### 特征 vs 深度学习

| 方法 | 特征提取 | 优点 | 缺点 |
|------|----------|------|------|
| **手工特征** | 人工设计 | 可解释、快速 | 需要领域知识 |
| **深度学习** | 自动学习 | 端到端优化 | 需要大量数据 |

深度学习的本质就是自动学习最优特征表示。'''),
    md_cell(r'''## 2. HOG 特征原理

### HOG (Histogram of Oriented Gradients)

[[Dalal & Triggs, 2005]](https://ieeexplore.ieee.org/document/1467360)

HOG 的核心思想：局部物体的外观和形状可以通过梯度方向的分布来描述。

### 计算步骤

1. **梯度计算**：用 Sobel 算子计算水平和垂直梯度
   - $g_x = I(x+1, y) - I(x-1, y)$
   - $g_y = I(x, y+1) - I(x, y-1)$
   - 幅值：$m = \sqrt{g_x^2 + g_y^2}$
   - 方向：$\theta = \text{atan2}(g_y, g_x)$

2. **方向直方图**：将每个 cell（如 8×8）内的梯度方向投影到 $B$ 个 bin
   - 通常 $B = 9$，方向范围 $[0°, 180°)$（无符号）

3. **块归一化**：将相邻 cell 组成 block（如 2×2 cells），进行 L2 归一化
   - 归一化使特征对光照变化不变

4. **特征拼接**：将所有 block 的归一化直方图拼接成一维向量'''),
    md_cell(r'''## 3. 手算验证：HOG on 4×4 Patch

### 示例：4×4 灰度图，2 个 bin

$$I = \begin{bmatrix} 10 & 20 \\ 30 & 50 \end{bmatrix}$$

（简化为 2×2 方便手算）

**梯度计算（中心差分）：**
- $g_x(0,0) = I(1,0) - I(0,0) = 20 - 10 = 10$（右差分）
- $g_y(0,0) = I(0,1) - I(0,0) = 30 - 10 = 20$
- $g_x(1,0) = I(0,0) - I(1,0) = 10 - 20 = -10$（左差分）
- $g_y(1,0) = I(1,1) - I(1,0) = 50 - 20 = 30$

**幅值和方向：**
- 像素(0,0)：$m = \sqrt{100+400} = \sqrt{500} \approx 22.4$, $\theta = \text{atan2}(20, 10) \approx 63.4°$
- 像素(1,0)：$m = \sqrt{100+900} = \sqrt{1000} \approx 31.6$, $\theta = \text{atan2}(30, -10) \approx 108.4°$

**方向直方图（2 bins: [0°, 90°), [90°, 180°)）：**
- bin 0 [0-90°)：像素(0,0) 的 $\theta=63.4°$ → 贡献幅值 22.4
- bin 1 [90-180°)：像素(1,0) 的 $\theta=108.4°$ → 贡献幅值 31.6

**归一化**（L2）：
- $norm = \sqrt{22.4^2 + 31.6^2} = \sqrt{501.8 + 998.6} = \sqrt{1500.4} \approx 38.7$
- 归一化后：$[22.4/38.7, 31.6/38.7] = [0.579, 0.817]$'''),
    code_cell(r'''# Verify HOG manual calculation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
np.random.seed(42)

# Simple 2x2 patch
I = np.array([[10, 20], [30, 50]], dtype=float)

# Compute gradients (simple difference)
gx = np.zeros_like(I)
gy = np.zeros_like(I)
gx[:, :-1] = I[:, 1:] - I[:, :-1]  # horizontal gradient
gy[:-1, :] = I[1:, :] - I[:-1, :]  # vertical gradient

print("Image:\n", I)
print("\nGradient X:\n", gx)
print("\nGradient Y:\n", gy)

# Magnitude and angle
mag = np.sqrt(gx**2 + gy**2)
angle = np.degrees(np.arctan2(gy, gx))
angle[angle < 0] += 180  # unsigned (0-180)

print("\nMagnitude:\n", mag)
print("\nAngle (degrees):\n", angle)

# Histogram with 2 bins
n_bins = 2
bin_edges = np.linspace(0, 180, n_bins + 1)
hist = np.zeros(n_bins)
for i in range(I.shape[0]):
    for j in range(I.shape[1]):
        if mag[i, j] > 0:
            bin_idx = min(int(angle[i, j] / (180 / n_bins)), n_bins - 1)
            hist[bin_idx] += mag[i, j]

print("\nHOG histogram (2 bins):", hist)
print("Normalized:", hist / (np.linalg.norm(hist) + 1e-10))'''),
    md_cell(r'''## 4. 代码实现：完整 HOG 特征提取'''),
    code_cell(r'''def compute_hog(image, cell_size=8, n_bins=9, block_size=2):
    """Compute HOG features for a grayscale image.
    
    Args:
        image: (H, W) grayscale image
        cell_size: size of each cell in pixels
        n_bins: number of orientation bins
        block_size: number of cells per block
    
    Returns:
        hog_features: 1D array of HOG features
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    
    H, W = image.shape
    # Step 1: Compute gradients
    gx = np.zeros_like(image, dtype=float)
    gy = np.zeros_like(image, dtype=float)
    gx[:, :-1] = image[:, 1:].astype(float) - image[:, :-1].astype(float)
    gy[:-1, :] = image[1:, :].astype(float) - image[:-1, :].astype(float)
    
    mag = np.sqrt(gx**2 + gy**2)
    angle = np.degrees(np.arctan2(gy, gx))
    angle[angle < 0] += 180  # unsigned gradient
    
    # Step 2: Compute cell histograms
    n_cells_y = H // cell_size
    n_cells_x = W // cell_size
    cell_hist = np.zeros((n_cells_y, n_cells_x, n_bins))
    
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            cell_mag = mag[cy*cell_size:(cy+1)*cell_size, 
                           cx*cell_size:(cx+1)*cell_size]
            cell_ang = angle[cy*cell_size:(cy+1)*cell_size, 
                            cx*cell_size:(cx+1)*cell_size]
            for b in range(n_bins):
                lo = b * (180 / n_bins)
                hi = (b + 1) * (180 / n_bins)
                mask = (cell_ang >= lo) & (cell_ang < hi)
                cell_hist[cy, cx, b] = np.sum(cell_mag[mask])
    
    # Step 3: Block normalization
    hog_features = []
    for by in range(n_cells_y - block_size + 1):
        for bx in range(n_cells_x - block_size + 1):
            block = cell_hist[by:by+block_size, bx:bx+block_size, :].ravel()
            norm = np.linalg.norm(block) + 1e-8
            hog_features.append(block / norm)
    
    return np.concatenate(hog_features)

# Test on a synthetic image
np.random.seed(42)
test_img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
hog_feat = compute_hog(test_img, cell_size=8, n_bins=9, block_size=2)
print(f"Image size: {test_img.shape}")
print(f"HOG feature dimension: {hog_feat.shape}")
print(f"Feature range: [{hog_feat.min():.4f}, {hog_feat.max():.4f}]")'''),
    md_cell(r'''## 5. 颜色直方图特征

颜色直方图统计每个颜色通道的像素分布，捕捉图像的整体颜色信息。'''),
    code_cell(r'''def compute_color_histogram(image, n_bins=8):
    """Compute color histogram for RGB image.
    
    Args:
        image: (H, W, 3) RGB image
        n_bins: bins per channel
    
    Returns:
        hist: (n_bins * 3,) concatenated histograms
    """
    hist = []
    for c in range(3):
        h, _ = np.histogram(image[:, :, c], bins=n_bins, range=(0, 256))
        h = h.astype(float)
        h = h / (h.sum() + 1e-8)  # normalize
        hist.append(h)
    return np.concatenate(hist)

# Test on synthetic RGB image
np.random.seed(42)
test_rgb = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
color_hist = compute_color_histogram(test_rgb, n_bins=8)
print(f"RGB image size: {test_rgb.shape}")
print(f"Color histogram dimension: {color_hist.shape}")
print(f"Per-channel bins: 8 x 3 = 24")'''),
    md_cell(r'''## 6. 特征可视化'''),
    code_cell(r'''# Visualize features on a synthetic image
np.random.seed(42)

# Create a synthetic image with edges
img = np.zeros((32, 32, 3), dtype=np.uint8)
img[:, :16] = [200, 50, 50]  # left half red
img[:, 16:] = [50, 200, 50]  # right half green
# Add some noise
img = np.clip(img.astype(int) + np.random.randint(-20, 21, (32, 32, 3)), 0, 255).astype(np.uint8)

gray = np.mean(img, axis=2).astype(np.uint8)
hog_feat = compute_hog(gray, cell_size=8, n_bins=9, block_size=2)
color_hist = compute_color_histogram(img, n_bins=8)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# Original image
axes[0, 0].imshow(img)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')
# Grayscale
axes[0, 1].imshow(gray, cmap='gray')
axes[0, 1].set_title('Grayscale')
axes[0, 1].axis('off')
# Gradients
gx = np.zeros_like(gray, dtype=float)
gy = np.zeros_like(gray, dtype=float)
gx[:, :-1] = gray[:, 1:].astype(float) - gray[:, :-1].astype(float)
gy[:-1, :] = gray[1:, :].astype(float) - gray[:-1, :].astype(float)
mag = np.sqrt(gx**2 + gy**2)
axes[0, 2].imshow(mag, cmap='hot')
axes[0, 2].set_title('Gradient Magnitude')
axes[0, 2].axis('off')
# HOG histogram
axes[1, 0].bar(range(len(hog_feat)), hog_feat, color='#ea580c', width=1)
axes[1, 0].set_title(f'HOG Features ({len(hog_feat)} dims)')
axes[1, 0].set_xlabel('Feature index')
# Color histogram
axes[1, 1].bar(range(len(color_hist)), color_hist, 
              color=['r']*8 + ['g']*8 + ['b']*8, width=1)
axes[1, 1].set_title('Color Histogram (24 dims)')
axes[1, 1].set_xlabel('Feature index')
# Raw pixels
axes[1, 2].bar(range(100), img.reshape(-1)[:100], color='#2563eb', width=1)
axes[1, 2].set_title('Raw Pixels (first 100)')
axes[1, 2].set_xlabel('Pixel index')
plt.suptitle('Feature Comparison: HOG vs Color Histogram vs Raw Pixels', fontsize=14)
plt.tight_layout()
plt.savefig('feature_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 1: Feature comparison")'''),
    md_cell(r'''## 7. 分类实验：比较不同特征

使用线性 SVM 分类器，比较三种特征的分类效果：
1. 原始像素（3072 维）
2. HOG 特征（324 维）
3. 颜色直方图（24 维）'''),
    code_cell(r'''# Generate synthetic dataset for classification
def generate_image_dataset(n_per_class=50, n_classes=3, img_size=16):
    """Generate synthetic images with different patterns for each class."""
    np.random.seed(42)
    images = []
    labels = []
    for c in range(n_classes):
        for _ in range(n_per_class):
            img = np.random.randint(80, 180, (img_size, img_size, 3), dtype=np.uint8)
            if c == 0:  # horizontal stripes
                img[::2, :] += 50
            elif c == 1:  # vertical stripes
                img[:, ::2] += 50
            else:  # diagonal
                for i in range(img_size):
                    img[i, i] = 255
            images.append(img)
            labels.append(c)
    return np.array(images), np.array(labels)

np.random.seed(42)
images, labels = generate_image_dataset(n_per_class=80, n_classes=3, img_size=16)
print(f"Dataset: {images.shape}, Labels: {labels.shape}")

# Linear SVM classifier
class LinearSVM:
    def __init__(self, n_features, n_classes):
        self.W = np.random.randn(n_classes, n_features) * 0.01
    
    def train(self, X, y, lr=0.1, reg=1e-3, epochs=300):
        losses = []
        for _ in range(epochs):
            scores = X @ self.W.T
            margins = np.maximum(0, scores - scores[np.arange(len(y)), y][:, None] + 1)
            margins[np.arange(len(y)), y] = 0
            loss = np.sum(margins) / len(y) + reg * np.sum(self.W**2)
            
            mask = (margins > 0).astype(float)
            mask[np.arange(len(y)), y] = -np.sum(mask, axis=1)
            dW = mask.T @ X / len(y) + 2 * reg * self.W
            self.W -= lr * dW
            losses.append(loss)
        return losses
    
    def predict(self, X):
        return np.argmax(X @ self.W.T, axis=1)

# Extract features
def extract_raw_pixels(images):
    return images.reshape(len(images), -1).astype(float) / 255.0

def extract_hog_features(images):
    return np.array([compute_hog(img, cell_size=4, n_bins=9, block_size=2) for img in images])

def extract_color_hist(images):
    return np.array([compute_color_histogram(img, n_bins=8) for img in images])

# Split train/test
n = len(images)
idx = np.random.permutation(n)
n_train = int(0.8 * n)
train_idx, test_idx = idx[:n_train], idx[n_train:]

results = {}
for name, feat_fn in [('Raw Pixels', extract_raw_pixels), 
                       ('HOG', extract_hog_features),
                       ('Color Hist', extract_color_hist)]:
    X = feat_fn(images)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    clf = LinearSVM(X_train.shape[1], 3)
    losses = clf.train(X_train, y_train, lr=0.1, epochs=300)
    train_acc = np.mean(clf.predict(X_train) == y_train)
    test_acc = np.mean(clf.predict(X_test) == y_test)
    results[name] = {'train_acc': train_acc, 'test_acc': test_acc, 'losses': losses}
    print(f"{name:12s}: train_acc={train_acc:.2%}, test_acc={test_acc:.2%}, feat_dim={X_train.shape[1]}")'''),
    code_cell(r'''# Visualize classification results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax1, ax2 = axes

# Accuracy comparison
names = list(results.keys())
train_accs = [results[n]['train_acc'] for n in names]
test_accs = [results[n]['test_acc'] for n in names]
x = np.arange(len(names))
w = 0.35
ax1.bar(x - w/2, train_accs, w, label='Train', color='#2563eb')
ax1.bar(x + w/2, test_accs, w, label='Test', color='#ea580c')
ax1.set_xticks(x)
ax1.set_xticklabels(names)
ax1.set_ylabel('Accuracy')
ax1.set_title('Classification Accuracy: Raw vs HOG vs Color Hist')
ax1.legend()
ax1.set_ylim(0, 1.1)

# Loss curves
for name in names:
    ax2.plot(results[name]['losses'], label=name, linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('SVM Loss')
ax2.set_title('Training Loss Curves')
ax2.legend()
plt.tight_layout()
plt.savefig('feature_classification.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 2: Classification comparison")'''),
    code_cell(r'''# Visualize sample images and their HOG features
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for c in range(3):
    idx = np.where(labels == c)[0][:1][0]
    img = images[idx]
    gray = np.mean(img, axis=2).astype(np.uint8)
    hog_feat = compute_hog(gray, cell_size=4, n_bins=9, block_size=2)
    
    axes[c, 0].imshow(img)
    axes[c, 0].set_title(f'Class {c}: Original')
    axes[c, 0].axis('off')
    
    axes[c, 1].imshow(gray, cmap='gray')
    axes[c, 1].set_title('Grayscale')
    axes[c, 1].axis('off')
    
    gx = np.zeros_like(gray, dtype=float)
    gy = np.zeros_like(gray, dtype=float)
    gx[:, :-1] = gray[:, 1:].astype(float) - gray[:, :-1].astype(float)
    gy[:-1, :] = gray[1:, :].astype(float) - gray[:-1, :].astype(float)
    mag = np.sqrt(gx**2 + gy**2)
    axes[c, 2].imshow(mag, cmap='hot')
    axes[c, 2].set_title('Gradient Magnitude')
    axes[c, 2].axis('off')
    
    axes[c, 3].bar(range(len(hog_feat)), hog_feat, color='#16a34a', width=1)
    axes[c, 3].set_title(f'HOG ({len(hog_feat)} dims)')

plt.suptitle('Feature Visualization for Each Class', fontsize=14)
plt.tight_layout()
plt.savefig('feature_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 3: Feature visualization per class")'''),
    md_cell(r'''## 8. 实验观察：特征维度与效果

### 关键发现

1. **HOG 优于原始像素**：HOG 编码了梯度方向信息，对光照变化更鲁棒
2. **颜色直方图适合颜色区分任务**：如果类别间颜色差异大，颜色直方图简单有效
3. **特征维度**：HOG (324) < 原始像素 (768) < 全像素，更紧凑
4. **深度学习的优势**：CNN 自动学习最优特征，远超手工设计

### HOG 的历史意义

HOG + SVM 曾是目标检测的标准方法（DPM, 2008-2012）。ImageNet 2012 之后，深度学习全面取代了手工特征。'''),
    md_cell(r'''## 9. SIFT 特征概念

### SIFT (Scale-Invariant Feature Transform)

[[Lowe, 2004]](https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94)

SIFT 是另一种经典特征，与 HOG 的区别：
- **HOG**：密集计算，覆盖整个图像区域
- **SIFT**：稀疏计算，只在关键点处提取

SIFT 的步骤：
1. 尺度空间极值检测（DoG）
2. 关键点定位
3. 方向分配
4. 关键点描述子（128 维）

SIFT 对尺度、旋转、光照变化都具有不变性，是传统 CV 最强大的特征之一。'''),
    code_cell(r'''# Compare feature dimensions
features_info = {
    'Raw Pixels (16x16x3)': 768,
    'HOG (cell=4, bins=9)': 324,
    'Color Hist (bins=8)': 24,
    'SIFT (conceptual)': 128,  # per keypoint
}

fig, ax = plt.subplots(figsize=(8, 5))
names = list(features_info.keys())
dims = list(features_info.values())
bars = ax.barh(names, dims, color=['#ea580c', '#2563eb', '#16a34a', '#9333ea'])
ax.set_xlabel('Feature Dimension')
ax.set_title('Feature Dimension Comparison')
for bar, dim in zip(bars, dims):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
            str(dim), va='center', fontsize=12)
plt.tight_layout()
plt.savefig('feature_dimensions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 4: Feature dimension comparison")'''),
    md_cell(r'''## 作业

### 作业 1：实现带插值的 HOG

当前 HOG 实现将梯度直接分配到最近的 bin。改进为**双线性插值**：将幅值按距离分配到相邻的两个 bin。

**提示**：对于角度 $\theta$ 和 bin 宽度 $w$：
- $b_1 = \lfloor \theta / w \rfloor$, $b_2 = b_1 + 1$
- $d_1 = \theta / w - b_1$, $d_2 = 1 - d_1$
- 贡献：$hist[b_1] += m \cdot d_2$, $hist[b_2] += m \cdot d_1$'''),
    code_cell(r'''# Homework 1: HOG with bilinear interpolation
def compute_hog_interpolated(image, cell_size=8, n_bins=9, block_size=2):
    """HOG with bilinear orientation interpolation."""
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    
    H, W = image.shape
    gx = np.zeros_like(image, dtype=float)
    gy = np.zeros_like(image, dtype=float)
    gx[:, :-1] = image[:, 1:].astype(float) - image[:, :-1].astype(float)
    gy[:-1, :] = image[1:, :].astype(float) - image[:-1, :].astype(float)
    
    mag = np.sqrt(gx**2 + gy**2)
    angle = np.degrees(np.arctan2(gy, gx))
    angle[angle < 0] += 180
    
    n_cells_y = H // cell_size
    n_cells_x = W // cell_size
    cell_hist = np.zeros((n_cells_y, n_cells_x, n_bins))
    
    bin_width = 180.0 / n_bins
    
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            cell_mag = mag[cy*cell_size:(cy+1)*cell_size, 
                           cx*cell_size:(cx+1)*cell_size]
            cell_ang = angle[cy*cell_size:(cy+1)*cell_size, 
                            cx*cell_size:(cx+1)*cell_size]
            # TODO: implement bilinear interpolation
            pass
    
    # Block normalization
    hog_features = []
    for by in range(n_cells_y - block_size + 1):
        for bx in range(n_cells_x - block_size + 1):
            block = cell_hist[by:by+block_size, bx:bx+block_size, :].ravel()
            norm = np.linalg.norm(block) + 1e-8
            hog_features.append(block / norm)
    
    return np.concatenate(hog_features)

# Test
np.random.seed(42)
test_img = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
hog_simple = compute_hog(test_img, cell_size=4, n_bins=9, block_size=2)
# hog_interp = compute_hog_interpolated(test_img, cell_size=4, n_bins=9, block_size=2)
# assert hog_interp.shape == hog_simple.shape, "Feature dimension should match"
# assert np.all(hog_interp >= 0), "HOG values should be non-negative"
print("Homework 1 template ready. Implement bilinear interpolation.")'''),
    md_cell(r'''### 作业 2：多尺度颜色直方图

实现多尺度颜色直方图：将图像分成 2×2 区域，每个区域分别计算颜色直方图，最后拼接。'''),
    code_cell(r'''# Homework 2: Multi-scale color histogram
def compute_multiscale_color_hist(image, n_bins=8, n_regions=2):
    """Compute multi-scale color histogram by splitting image into regions.
    
    Args:
        image: (H, W, 3) RGB image
        n_bins: bins per channel per region
        n_regions: split image into n_regions x n_regions
    
    Returns:
        hist: concatenated histograms from all regions
    """
    H, W, C = image.shape
    rh = H // n_regions
    rw = W // n_regions
    all_hist = []
    
    for i in range(n_regions):
        for j in range(n_regions):
            region = image[i*rh:(i+1)*rh, j*rw:(j+1)*rw]
            # TODO: compute color histogram for this region
            pass
    
    return np.concatenate(all_hist)

# Test
np.random.seed(42)
test_img = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
# hist = compute_multiscale_color_hist(test_img, n_bins=8, n_regions=2)
# assert len(hist) == 4 * 3 * 8, "Should have 4 regions x 3 channels x 8 bins"
print("Homework 2 template ready. Implement multi-scale color histogram.")'''),
    md_cell(r'''### 作业 3：特征融合

将 HOG 和颜色直方图拼接为联合特征，观察是否比单一特征效果更好。'''),
    code_cell(r'''# Homework 3: Feature fusion
def extract_fused_features(images):
    """Extract fused HOG + color histogram features."""
    features = []
    for img in images:
        # TODO: concatenate HOG and color histogram features
        hog = compute_hog(img, cell_size=4, n_bins=9, block_size=2)
        chist = compute_color_histogram(img, n_bins=8)
        # fused = np.concatenate([hog, chist])
        pass
    return np.array(features)

# Test
np.random.seed(42)
# fused = extract_fused_features(images)
# X_fused_train = fused[train_idx]
# clf_fused = LinearSVM(X_fused_train.shape[1], 3)
# losses = clf_fused.train(X_fused_train, y_train, lr=0.1, epochs=300)
# fused_acc = np.mean(clf_fused.predict(fused[test_idx]) == y_test)
# assert fused_acc >= results['HOG']['test_acc'], "Fused should be >= HOG alone"
print("Homework 3 template ready. Implement feature fusion.")'''),
    md_cell(r'''## 小结

| 特征 | 维度 | 捕捉信息 | 优点 | 局限 |
|------|------|----------|------|------|
| **原始像素** | 高 | 所有信息 | 无信息丢失 | 维度高、无空间结构 |
| **HOG** | 中 | 边缘/形状 | 光照不变、紧凑 | 颜色信息丢失 |
| **颜色直方图** | 低 | 颜色分布 | 极紧凑、快速 | 无空间信息 |
| **SIFT** | 中 | 局部关键点 | 尺度/旋转不变 | 稀疏、需关键点检测 |

**核心洞察**：手工特征是对图像的特定「视角」，各有侧重但都不完整。深度学习通过端到端学习自动发现最优特征表示，这是其超越传统方法的根本原因。'''),
    md_cell(r'''## 参考文献

1. [[Dalal & Triggs, 2005]](https://ieeexplore.ieee.org/document/1467360) Dalal & Triggs. *Histograms of Oriented Gradients for Human Detection*. CVPR 2005.
2. [[Lowe, 2004]](https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94) Lowe. *Distinctive Image Features from Scale-Invariant Keypoints*. IJCV 2004.
3. **CS231n Assignment 1** - [Feature Engineering](https://cs231n.github.io/assignments2021/assignment1/)
4. [[Felzenszwalb et al., 2010]](https://cs.brown.edu/people/pfelzens/papers/lat-mid.pdf) Felzenszwalb et al. *Object Detection with Discriminatively Trained Part Based Models*. PAMI 2010.

---

> 本节内容参考 [Stanford CS231n Assignment 1 Q4](https://cs231n.github.io/assignments2021/assignment1/)'''),
    md_cell(r'''---

## 参考代码实现

以下 GitHub 仓库提供了本节内容的完整代码实现，建议结合学习：

- **[jariasf/CS231n](https://github.com/jariasf/CS231n)** (486 stars): CS231n 完整作业解答（KNN、SVM、Softmax、特征工程）
  - 仓库地址: https://github.com/jariasf/CS231n

- **[rishabh-16/cs231n-2019-assignments](https://github.com/rishabh-16/cs231n-2019-assignments)** (212 stars): 2019版 CS231n 作业 PyTorch+TensorFlow 解答
  - 仓库地址: https://github.com/rishabh-16/cs231n-2019-assignments

- **[amanchadha/stanford-cs231n-assignments-2020](https://github.com/amanchadha/stanford-cs231n-assignments-2020)** (175 stars): 2020春季 CS231n 完整作业解答
  - 仓库地址: https://github.com/amanchadha/stanford-cs231n-assignments-2020

> 标注说明: 以上仓库按热度排序，优先推荐 stars 最多的实现.'''),
])
save_notebook(nb2, os.path.join(BASE, "part1-deep-learning-basics", "lecture-05b-features", "practice.ipynb"))

# ============================================================
# Notebook 3: RNN and Image Captioning
# ============================================================
print("Generating Notebook 3: RNN and Image Captioning...")
nb3 = make_notebook([
    md_cell(r'''# RNN 与图像描述生成

> **斯坦福 CS231n Assignment 2 Q5** | Recurrent Neural Networks and Image Captioning

## 本章导读

循环神经网络 (RNN) 处理序列数据，是自然语言处理和图像描述生成的核心模型。本节从 RNN 前向传播出发，理解 LSTM 门控机制，实现一个迷你图像描述系统。

**学习目标：**
- 理解 RNN 前向传播方程和反向传播
- 掌握 LSTM 四个门（输入、遗忘、输出、细胞）
- 理解 teacher forcing 训练策略
- 实现图像编码器 + RNN 解码器
- 可视化隐藏状态和损失曲线

**参考来源：** [CS231n Assignment 2 Q5](https://cs231n.github.io/assignments2021/assignment2/) | [The Unreasonable Effectiveness of RNNs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)'''),
    md_cell(r'''## 1. 直觉理解：RNN 处理序列

### 为什么需要 RNN？

前馈网络（MLP、CNN）处理固定大小的输入。但语言是变长序列——"一只猫坐在沙发上"有 7 个词，"猫坐"有 2 个词。RNN 的核心思想是**共享参数 + 隐状态传递**。

### RNN 前向传播

在每个时间步 $t$：
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

- $x_t$：时间步 $t$ 的输入（如词向量）
- $h_{t-1}$：上一步的隐状态（记忆）
- $h_t$：当前隐状态
- $y_t$：输出（如词的概率分布）

### RNN 的核心优势

1. **参数共享**：同一组 $W$ 在所有时间步复用
2. **变长输入**：可处理任意长度序列
3. **记忆能力**：隐状态 $h_t$ 编码历史信息'''),
    md_cell(r'''### 梯度消失问题

RNN 的反向传播需要沿时间展开（BPTT）。梯度需要经过多个 $\tanh$ 的导数连乘：

$$\frac{\partial L}{\partial h_0} = \prod_{t=1}^{T} \frac{\partial h_t}{\partial h_{t-1}} \cdot \frac{\partial L}{\partial h_T}$$

$\tanh$ 的导数最大值为 1，且在大部分区域远小于 1。长序列中梯度连乘后会指数衰减——这就是**梯度消失问题**。

[[Hochreiter et al., 2001]](https://www.mitpressjournals.org/doi/abs/10.1162/089976602760128018) 详细分析了这个问题。

### LSTM 的解决方案

LSTM (Long Short-Term Memory) 通过门控机制控制信息流：

[[Hochreiter & Schmidhuber, 1997]](https://www.bioinf.jku.at/publications/older/2604.pdf)

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(forget gate)}$$
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{(input gate)}$$
$$\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C) \quad \text{(candidate)}$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{(cell state)}$$
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \text{(output gate)}$$
$$h_t = o_t \odot \tanh(C_t) \quad \text{(hidden state)}$$

细胞状态 $C_t$ 提供了一条「梯度高速公路」，缓解梯度消失。'''),
    md_cell(r'''## 2. 手算验证：RNN 前向传播

### 小型序列手算

设词汇表大小 $V=3$，隐状态维度 $H=2$，序列长度 $T=3$。

**权重：**
$$W_{xh} = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \end{bmatrix}, \quad W_{hh} = \begin{bmatrix} 0.5 & 0.1 \\ 0.2 & 0.3 \end{bmatrix}, \quad b_h = [0, 0]$$

**输入序列（one-hot 编码）：**
$$x_1 = [1, 0], \quad x_2 = [0, 1], \quad x_3 = [1, 0]$$

**初始隐状态：** $h_0 = [0, 0]$

**时间步 1：**
$$z_1 = W_{xh} x_1 + W_{hh} h_0 + b_h = [0.1, 0.3] + [0, 0] = [0.1, 0.3]$$
$$h_1 = \tanh(z_1) = [\tanh(0.1), \tanh(0.3)] = [0.0997, 0.2913]$$

**时间步 2：**
$$z_2 = W_{xh} x_2 + W_{hh} h_1 = [0.2, 0.4] + [0.5 \times 0.0997 + 0.1 \times 0.2913, \; 0.2 \times 0.0997 + 0.3 \times 0.2913]$$
$$= [0.2, 0.4] + [0.0788, 0.1073] = [0.2788, 0.5073]$$
$$h_2 = \tanh(z_2) = [\tanh(0.2788), \tanh(0.5073)] = [0.2716, 0.4672]$$

**时间步 3：**
$$z_3 = W_{xh} x_3 + W_{hh} h_2 = [0.1, 0.3] + [0.5 \times 0.2716 + 0.1 \times 0.4672, \; 0.2 \times 0.2716 + 0.3 \times 0.4672]$$
$$= [0.1, 0.3] + [0.1825, 0.1945] = [0.2825, 0.4945]$$
$$h_3 = \tanh(z_3) = [\tanh(0.2825), \tanh(0.4945)] = [0.2752, 0.4583]$$'''),
    code_cell(r'''# Verify RNN forward pass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
np.random.seed(42)

# RNN parameters
Wxh = np.array([[0.1, 0.2], [0.3, 0.4]])
Whh = np.array([[0.5, 0.1], [0.2, 0.3]])
bh = np.array([0.0, 0.0])

# Input sequence (2D vectors)
x_seq = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])]

# Forward pass
h = np.array([0.0, 0.0])  # initial hidden state
hidden_states = [h.copy()]

for t, x in enumerate(x_seq):
    z = Wxh.T @ x + Whh.T @ h + bh
    h = np.tanh(z)
    hidden_states.append(h.copy())
    print(f"Step {t+1}: z = {z}, h = {h}")

print(f"\nAll hidden states:\n{np.array(hidden_states)}")
print(f"\nExpected h_1 = [0.0997, 0.2913]: {np.allclose(h, [0.0997, 0.2913]) if len(hidden_states) > 1 else 'N/A'}")
print(f"Final h_3 = {hidden_states[-1]}")'''),
    md_cell(r'''## 3. 代码实现：RNN 类'''),
    code_cell(r'''class SimpleRNN:
    """Simple RNN with tanh activation.
    
    Forward: h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
    Output: y_t = W_hy @ h_t + b_y
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Initialize weights (Xavier/Glorot)
        scale_x = np.sqrt(1.0 / input_dim)
        scale_h = np.sqrt(1.0 / hidden_dim)
        self.W_xh = np.random.randn(hidden_dim, input_dim) * scale_x
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * scale_h
        self.b_h = np.zeros(hidden_dim)
        self.W_hy = np.random.randn(output_dim, hidden_dim) * scale_h
        self.b_y = np.zeros(output_dim)
        self.hidden_dim = hidden_dim
    
    def forward(self, x_seq, h0=None):
        """Forward pass over a sequence.
        
        Args:
            x_seq: list of input vectors, each (input_dim,)
            h0: initial hidden state, default zeros
        
        Returns:
            outputs: list of output vectors
            hidden_states: list of hidden states (including h0)
            cache: for backprop
        """
        if h0 is None:
            h0 = np.zeros(self.hidden_dim)
        
        h = h0.copy()
        outputs = []
        hidden_states = [h.copy()]
        cache = []
        
        for x in x_seq:
            z = self.W_xh @ x + self.W_hh @ h + self.b_h
            h = np.tanh(z)
            y = self.W_hy @ h + self.b_y
            outputs.append(y)
            hidden_states.append(h.copy())
            cache.append((x, h.copy(), z))
            h = h.copy()
        
        return outputs, hidden_states, cache
    
    def step(self, x_seq, targets, lr=0.01):
        """One gradient step using BPTT."""
        outputs, hidden_states, cache = self.forward(x_seq)
        
        # Compute loss (cross-entropy) and gradients
        total_loss = 0
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)
        dW_hy = np.zeros_like(self.W_hy)
        db_y = np.zeros_like(self.b_y)
        dh_next = np.zeros(self.hidden_dim)
        
        for t in range(len(x_seq) - 1, -1, -1):
            x, h, z = cache[t]
            y = outputs[t]
            target = targets[t]
            
            # Softmax
            shifted = y - np.max(y)
            exp_y = np.exp(shifted)
            probs = exp_y / np.sum(exp_y)
            
            # Loss
            total_loss += -np.log(probs[target] + 1e-10)
            
            # Gradients
            dy = probs.copy()
            dy[target] -= 1
            
            dW_hy += np.outer(dy, h)
            db_y += dy
            
            dh = self.W_hy.T @ dy + dh_next
            dz = dh * (1 - h**2)  # tanh derivative
            
            dW_xh += np.outer(dz, x)
            dW_hh += np.outer(dz, hidden_states[t])
            db_h += dz
            dh_next = self.W_hh.T @ dz
        
        # Update weights
        self.W_xh -= lr * dW_xh
        self.W_hh -= lr * dW_hh
        self.b_h -= lr * db_h
        self.W_hy -= lr * dW_hy
        self.b_y -= lr * db_y
        
        return total_loss / len(x_seq)
    
    def generate(self, x0, length=10):
        """Generate a sequence autoregressively."""
        h = np.zeros(self.hidden_dim)
        x = x0
        generated = []
        for _ in range(length):
            z = self.W_xh @ x + self.W_hh @ h + self.b_h
            h = np.tanh(z)
            y = self.W_hy @ h + self.b_y
            # Sample from output
            shifted = y - np.max(y)
            probs = np.exp(shifted) / np.sum(np.exp(shifted))
            idx = np.random.choice(len(y), p=probs)
            generated.append(idx)
            # Next input is one-hot of predicted index
            x = np.zeros_like(x)
            x[idx] = 1.0
        return generated

print("SimpleRNN class defined")'''),
    md_cell(r'''## 4. 训练 RNN：序列预测任务

训练 RNN 学习一个简单的序列模式：给定前一个 token，预测下一个 token。'''),
    code_cell(r'''# Train RNN on a simple sequence prediction task
np.random.seed(42)

# Create a simple vocabulary: 0,1,2,3,4 with pattern 0->1->2->3->4->0->...
vocab_size = 5
hidden_dim = 16
rnn = SimpleRNN(input_dim=vocab_size, hidden_dim=hidden_dim, output_dim=vocab_size)

# Training data: sequences following pattern 0,1,2,3,4,0,1,2,3,4,...
def make_training_sequence(length=10):
    x_seq = []
    targets = []
    for i in range(length):
        x = np.zeros(vocab_size)
        x[i % vocab_size] = 1.0
        x_seq.append(x)
        if i < length - 1:
            targets.append((i + 1) % vocab_size)
    return x_seq[:-1], targets

# Train
losses = []
for epoch in range(500):
    x_seq, targets = make_training_sequence(8)
    loss = rnn.step(x_seq, targets, lr=0.1)
    losses.append(loss)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss = {loss:.4f}")

print(f"Final loss: {losses[-1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(losses, color='#ea580c', linewidth=1.5, alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('RNN Training Loss (Sequence Prediction)')
plt.tight_layout()
plt.savefig('rnn_training_loss.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 1: RNN training loss")'''),
    code_cell(r'''# Visualize hidden states during forward pass
x_seq, _ = make_training_sequence(8)
outputs, hidden_states, _ = rnn.forward(x_seq)

hidden_array = np.array(hidden_states[1:])  # skip h0

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(hidden_array.T, aspect='auto', cmap='RdBu', interpolation='nearest')
ax.set_xlabel('Time Step')
ax.set_ylabel('Hidden Unit')
ax.set_title('RNN Hidden States Over Time')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('rnn_hidden_states.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 2: Hidden state dynamics")'''),
    md_cell(r'''## 5. LSTM 实现

LSTM 通过门控机制解决梯度消失问题。'''),
    code_cell(r'''class SimpleLSTM:
    """Simplified LSTM cell.
    
    Gates:
        f_t = sigmoid(W_f @ [x_t, h_{t-1}])   (forget gate)
        i_t = sigmoid(W_i @ [x_t, h_{t-1}])   (input gate)
        g_t = tanh(W_g @ [x_t, h_{t-1}])       (candidate)
        C_t = f_t * C_{t-1} + i_t * g_t        (cell state)
        o_t = sigmoid(W_o @ [x_t, h_{t-1}])   (output gate)
        h_t = o_t * tanh(C_t)                  (hidden state)
    """
    def __init__(self, input_dim, hidden_dim):
        D = input_dim + hidden_dim
        self.W_f = np.random.randn(hidden_dim, D) * np.sqrt(2.0 / D)
        self.W_i = np.random.randn(hidden_dim, D) * np.sqrt(2.0 / D)
        self.W_g = np.random.randn(hidden_dim, D) * np.sqrt(2.0 / D)
        self.W_o = np.random.randn(hidden_dim, D) * np.sqrt(2.0 / D)
        self.b_f = np.zeros(hidden_dim)
        self.b_i = np.zeros(hidden_dim)
        self.b_g = np.zeros(hidden_dim)
        self.b_o = np.zeros(hidden_dim)
        self.hidden_dim = hidden_dim
    
    def forward(self, x_seq, h0=None, C0=None):
        """Forward pass over sequence."""
        if h0 is None:
            h0 = np.zeros(self.hidden_dim)
        if C0 is None:
            C0 = np.zeros(self.hidden_dim)
        
        h, C = h0.copy(), C0.copy()
        hidden_states = [h.copy()]
        cell_states = [C.copy()]
        
        for x in x_seq:
            concat = np.concatenate([x, h])
            f = self._sigmoid(self.W_f @ concat + self.b_f)
            i = self._sigmoid(self.W_i @ concat + self.b_i)
            g = np.tanh(self.W_g @ concat + self.b_g)
            C = f * C + i * g
            o = self._sigmoid(self.W_o @ concat + self.b_o)
            h = o * np.tanh(C)
            hidden_states.append(h.copy())
            cell_states.append(C.copy())
        
        return hidden_states, cell_states
    
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

# Test LSTM
np.random.seed(42)
lstm = SimpleLSTM(input_dim=5, hidden_dim=8)
x_seq = [np.random.randn(5) for _ in range(10)]
h_states, c_states = lstm.forward(x_seq)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax1, ax2 = axes
ax1.imshow(np.array(h_states[1:]).T, aspect='auto', cmap='RdBu')
ax1.set_title('LSTM Hidden States')
ax1.set_xlabel('Time Step')
ax2.imshow(np.array(c_states[1:]).T, aspect='auto', cmap='RdBu')
ax2.set_title('LSTM Cell States')
ax2.set_xlabel('Time Step')
plt.tight_layout()
plt.savefig('lstm_states.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 3: LSTM hidden and cell states")'''),
    md_cell(r'''## 6. 图像描述生成：CNN + RNN

### 编码器-解码器架构

图像描述生成使用 CNN 提取图像特征（编码器），RNN 生成描述文字（解码器）：

```
图像 -> CNN -> 图像特征向量 -> RNN -> 逐词生成描述
```

[[Show and Tell, Vinyals et al., 2015]](https://arxiv.org/abs/1502.03044)

### Teacher Forcing

训练时，RNN 的每一步输入使用**真实词**而非模型预测的词——这加速收敛但可能造成训练/推理不一致。'''),
    code_cell(r'''# Mini captioning demo with synthetic image features
np.random.seed(42)

# Simulate CNN image features (would come from a pretrained CNN)
n_images = 100
feature_dim = 64
image_features = np.random.randn(n_images, feature_dim)

# Simulate captions (word indices)
vocab_size = 20
caption_length = 6
captions = np.random.randint(0, vocab_size, (n_images, caption_length))

# Captioning RNN: image features -> word sequence
class CaptioningRNN:
    """Simple image captioning model.
    
    Encoder: precomputed image features (simulated)
    Decoder: RNN that takes image features as initial hidden state
    """
    def __init__(self, feature_dim, hidden_dim, vocab_size):
        self.W_proj = np.random.randn(hidden_dim, feature_dim) * 0.01
        self.b_proj = np.zeros(hidden_dim)
        self.W_xh = np.random.randn(hidden_dim, vocab_size) * 0.01
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_h = np.zeros(hidden_dim)
        self.W_hy = np.random.randn(vocab_size, hidden_dim) * 0.01
        self.b_y = np.zeros(vocab_size)
        self.hidden_dim = hidden_dim
    
    def forward(self, img_feat, caption):
        """Forward pass with teacher forcing.
        
        Args:
            img_feat: (feature_dim,) image features
            caption: (T,) word indices (ground truth, teacher forcing)
        
        Returns:
            loss, outputs, cache
        """
        T = len(caption)
        # Project image features to initial hidden state
        h = np.tanh(self.W_proj @ img_feat + self.b_proj)
        
        outputs = []
        cache = [h.copy()]
        
        for t in range(T):
            # Input: one-hot of previous word (or START token for t=0)
            x = np.zeros(vocab_size)
            x[caption[t]] = 1.0
            
            z = self.W_xh @ x + self.W_hh @ h + self.b_h
            h = np.tanh(z)
            y = self.W_hy @ h + self.b_y
            outputs.append(y)
            cache.append((x.copy(), h.copy(), z))
        
        # Compute loss (cross-entropy)
        loss = 0
        for t in range(T):
            shifted = outputs[t] - np.max(outputs[t])
            exp_y = np.exp(shifted)
            probs = exp_y / np.sum(exp_y)
            loss += -np.log(probs[caption[t]] + 1e-10)
        
        return loss / T, outputs, cache
    
    def step(self, img_feat, caption, lr=0.01):
        """One gradient step."""
        loss, outputs, cache = self.forward(img_feat, caption)
        T = len(caption)
        h = cache[0]
        
        # Backprop through time
        dh = np.zeros(self.hidden_dim)
        for t in range(T - 1, -1, -1):
            x, h_t, z = cache[t + 1]
            y = outputs[t]
            shifted = y - np.max(y)
            exp_y = np.exp(shifted)
            probs = exp_y / np.sum(exp_y)
            
            dy = probs.copy()
            dy[caption[t]] -= 1
            
            self.W_hy -= lr * np.outer(dy, h_t)
            self.b_y -= lr * dy
            
            dh_t = self.W_hy.T @ dy + dh
            dz = dh_t * (1 - h_t**2)
            
            self.W_xh -= lr * np.outer(dz, x)
            self.b_h -= lr * dz
            dh = self.W_hh.T @ dz
        
        # Backprop to projection
        dh0 = dh * (1 - cache[0]**2)
        self.W_proj -= lr * np.outer(dh0, img_feat)
        self.b_proj -= lr * dh0
        
        return loss

# Train mini captioning model
np.random.seed(42)
cap_rnn = CaptioningRNN(feature_dim=64, hidden_dim=32, vocab_size=20)
losses = []
for epoch in range(300):
    idx = epoch % n_images
    loss = cap_rnn.step(image_features[idx], captions[idx], lr=0.05)
    losses.append(loss)
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss = {loss:.4f}")

print(f"Final loss: {losses[-1]:.4f}")'''),
    code_cell(r'''# Visualize captioning training
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(losses, color='#ea580c', linewidth=1.5, alpha=0.7)
ax.set_xlabel('Training Step')
ax.set_ylabel('Cross-Entropy Loss')
ax.set_title('Image Captioning Training Loss')
plt.tight_layout()
plt.savefig('captioning_loss.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 4: Captioning training loss")'''),
    md_cell(r'''## 7. BLEU 评分概念

BLEU (Bilingual Evaluation Understudy) 是机器翻译和描述生成中最常用的自动评估指标。

[[Papineni et al., 2002]](https://aclanthology.org/P02-1040/)

### BLEU-N

计算生成序列与参考序列的 N-gram 精度：

$$\text{BLEU-N} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

- $p_n$：N-gram 精度（匹配的 N-gram 比例）
- BP：短序列惩罚因子
- $w_n$：权重，通常均匀 $w_n = 1/N$

BLEU-4 是最常用的版本。'''),
    code_cell(r'''# Simple BLEU implementation
def compute_bleu(reference, hypothesis, max_n=4):
    """Compute simplified BLEU score.
    
    Args:
        reference: list of ground truth word indices
        hypothesis: list of predicted word indices
        max_n: maximum n-gram order
    """
    brevity_penalty = 1.0
    if len(hypothesis) < len(reference):
        brevity_penalty = np.exp(1 - len(reference) / max(len(hypothesis), 1))
    
    log_precision = 0
    for n in range(1, max_n + 1):
        ref_ngrams = {}
        for i in range(len(reference) - n + 1):
            gram = tuple(reference[i:i+n])
            ref_ngrams[gram] = ref_ngrams.get(gram, 0) + 1
        
        hyp_ngrams = {}
        for i in range(len(hypothesis) - n + 1):
            gram = tuple(hypothesis[i:i+n])
            hyp_ngrams[gram] = hyp_ngrams.get(gram, 0) + 1
        
        matches = 0
        total = 0
        for gram, count in hyp_ngrams.items():
            matches += min(count, ref_ngrams.get(gram, 0))
            total += count
        
        precision = matches / max(total, 1)
        log_precision += np.log(max(precision, 1e-10))
    
    bleu = brevity_penalty * np.exp(log_precision / max_n)
    return bleu

# Test BLEU
ref = [1, 2, 3, 4, 5]
hyp1 = [1, 2, 3, 4, 5]  # perfect
hyp2 = [1, 2, 3, 6, 7]  # partial
hyp3 = [5, 4, 3, 2, 1]  # reversed

print(f"BLEU (perfect):   {compute_bleu(ref, hyp1):.4f}")
print(f"BLEU (partial):   {compute_bleu(ref, hyp2):.4f}")
print(f"BLEU (reversed):  {compute_bleu(ref, hyp3):.4f}")'''),
    code_cell(r'''# Generate captions for sample images
np.random.seed(42)
sample_indices = [0, 10, 20]
word_list = [f'word{i}' for i in range(20)]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, idx in zip(axes, sample_indices):
    feat = image_features[idx]
    ref_caption = captions[idx]
    
    # Generate caption (use ground truth as approximation for demo)
    loss, outputs, _ = cap_rnn.forward(feat, ref_caption)
    
    # Show pseudo-image (feature visualization)
    pseudo_img = feat[:16].reshape(4, 4)
    ax.imshow(pseudo_img, cmap='viridis')
    ax.set_title(f'Image {idx}\nRef: {ref_caption}\nLoss: {loss:.2f}')
    ax.axis('off')
plt.suptitle('Captioning Demo (Synthetic Features)', fontsize=14)
plt.tight_layout()
plt.savefig('captioning_demo.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 5: Captioning demo")'''),
    md_cell(r'''## 作业

### 作业 1：实现 LSTM 反向传播

实现 LSTM 的 BPTT（Backpropagation Through Time）反向传播。

**提示**：对每个时间步 $t$，计算 $\frac{\partial L}{\partial W_f}, \frac{\partial L}{\partial W_i}, \frac{\partial L}{\partial W_g}, \frac{\partial L}{\partial W_o}$。'''),
    code_cell(r'''# Homework 1: LSTM backward pass
def lstm_backward(self, x_seq, targets, cache):
    """BPTT for LSTM.
    
    For each time step, compute gradients for all 4 gates.
    """
    # TODO: implement BPTT for LSTM
    # Key equations:
    # dh_t = W_hy^T @ dy + dh_next
    # do_t = dh_t * tanh(C_t) * o_t * (1-o_t)
    # dC_t = dh_t * o_t * (1 - tanh^2(C_t)) + dC_next
    # ... etc for f, i, g gates
    pass

# SimpleLSTM.backward = lstm_backward
# Test: numerical gradient check
np.random.seed(42)
lstm_test = SimpleLSTM(input_dim=3, hidden_dim=4)
x_test = [np.random.randn(3) for _ in range(3)]
# h_states, c_states = lstm_test.forward(x_test)
# Verify gradients numerically
print("Homework 1 template ready. Implement LSTM backward pass.")'''),
    md_cell(r'''### 作业 2：实现 Beam Search 解码

贪心解码每步选概率最高的词，可能错过全局最优序列。Beam Search 维护 $k$ 个候选序列。

[[Graves, 2012]](https://dl.acm.org/doi/10.5555/3044805)'''),
    code_cell(r'''# Homework 2: Beam search decoding
def beam_search(self, img_feat, beam_width=3, max_length=10):
    """Beam search decoding for caption generation.
    
    Maintain top-k partial sequences at each step.
    """
    # TODO: implement beam search
    # 1. Start with initial hidden state from image features
    # 2. At each step, expand each beam by one word
    # 3. Keep top-k beams by total log-probability
    # 4. Stop when all beams reach END token or max_length
    pass

# CaptioningRNN.beam_search = beam_search
print("Homework 2 template ready. Implement beam search decoding.")'''),
    md_cell(r'''### 作业 3：实现注意力机制

为描述生成添加注意力机制：每个时间步根据图像区域动态生成权重。

[[Bahdanau et al., 2015]](https://arxiv.org/abs/1409.0473)'''),
    code_cell(r'''# Homework 3: Attention mechanism for captioning
def attention_forward(self, img_features_seq, prev_hidden):
    """Compute attention weights and context vector.
    
    attention_scores = v^T * tanh(W_a @ img_features + U_a @ prev_hidden)
    attention_weights = softmax(attention_scores)
    context = sum(attention_weights * img_features)
    """
    # TODO: implement attention
    pass

print("Homework 3 template ready. Implement attention mechanism.")'''),
    md_cell(r'''## 小结

| 概念 | 要点 |
|------|------|
| **RNN** | 隐状态传递记忆，参数跨时间步共享 |
| **梯度消失** | tanh 导数连乘导致长距离梯度衰减 |
| **LSTM** | 门控（遗忘、输入、输出）+ 细胞状态解决梯度消失 |
| **Teacher Forcing** | 训练时输入真实词，加速收敛 |
| **编码器-解码器** | CNN 编码图像 → RNN 解码为文字 |
| **BLEU** | N-gram 匹配评估生成质量 |
| **Beam Search** | 维护多个候选序列，寻找全局更优解 |
| **注意力** | 动态聚焦相关输入区域 |

**关键洞察**：RNN 通过隐状态实现记忆，但梯度消失限制了长距离依赖。LSTM 通过门控解决了这个问题。图像描述是编码器-解码器的经典应用。'''),
    md_cell(r'''## 参考文献

1. [[Hochreiter & Schmidhuber, 1997]](https://www.bioinf.jku.at/publications/older/2604.pdf) Hochreiter & Schmidhuber. *Long Short-Term Memory*. Neural Computation, 1997.
2. [[Vinyals et al., 2015]](https://arxiv.org/abs/1502.03044) Vinyals et al. *Show and Tell: A Neural Image Caption Generator*. CVPR 2015.
3. [[Bahdanau et al., 2015]](https://arxiv.org/abs/1409.0473) Bahdanau et al. *Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR 2015.
4. [[Papineni et al., 2002]](https://aclanthology.org/P02-1040/) Papineni et al. *BLEU: a Method for Automatic Evaluation of Machine Translation*. ACL 2002.
5. [[Karpathy & Fei-Fei, 2015]](https://arxiv.org/abs/1412.2306) Karpathy & Fei-Fei. *Deep Visual-Semantic Alignments for Generating Image Descriptions*. CVPR 2015.
6. [[Hochreiter et al., 2001]](https://www.mitpressjournals.org/doi/abs/10.1162/089976602760128018) Hochreiter et al. *Gradient Flow in Recurrent Nets*.

---

> 本节内容参考 [Stanford CS231n Assignment 2 Q5](https://cs231n.github.io/assignments2021/assignment2/)'''),
    md_cell(r'''---

## 参考代码实现

以下 GitHub 仓库提供了本节内容的完整代码实现，建议结合学习：

- **[amanchadha/stanford-cs231n-assignments-2020](https://github.com/amanchadha/stanford-cs231n-assignments-2020)** (175 stars): 2020春季 CS231n 完整作业解答（含 RNN/LSTM 图像描述）
  - 仓库地址: https://github.com/amanchadha/stanford-cs231n-assignments-2020

- **[rishabh-16/cs231n-2019-assignments](https://github.com/rishabh-16/cs231n-2019-assignments)** (212 stars): 2019版 CS231n 作业 PyTorch+TensorFlow 解答
  - 仓库地址: https://github.com/rishabh-16/cs231n-2019-assignments

- **[chenyuntc/pytorch-book](https://github.com/chenyuntc/pytorch-book)** (12848 stars): PyTorch 深度学习实践教程（含 RNN/LSTM 实现）
  - 仓库地址: https://github.com/chenyuntc/pytorch-book

> 标注说明: 以上仓库按相关性排序，优先推荐 CS231n 作业解答.'''),
])
save_notebook(nb3, os.path.join(BASE, "part2-cnn-vision", "lecture-08b-rnn-captioning", "practice.ipynb"))

# ============================================================
# Notebook 4: CNN Visualization and Understanding
# ============================================================
print("Generating Notebook 4: CNN Visualization...")
nb4 = make_notebook([
    md_cell(r'''# CNN 可视化与理解

> **斯坦福 CS231n** | Visualizing and Understanding Convolutional Networks

## 本章导读

深度学习常被称为「黑盒」，但通过可视化技术，我们可以理解 CNN「看到了什么」。本节实现显著性图、Grad-CAM、FGSM 对抗攻击和特征反演。

**学习目标：**
- 理解显著性图（输入梯度）的原理
- 实现 Grad-CAM（类激活映射）
- 理解 FGSM 对抗样本生成
- 实现 t-SNE 嵌入可视化
- 可视化 CNN 学习的特征

**参考来源：** [CS231n Lecture 12](https://cs231n.stanford.edu/) | [Deep Visualization](https://cs231n.github.io/understanding-cnn/)'''),
    md_cell(r'''## 1. 直觉理解：为什么 CNN 可解释性重要？

### 黑盒问题

CNN 内部有数百万参数，决策过程不透明。关键问题：
- 模型关注图像的哪些区域？
- 每个神经元检测什么特征？
- 为什么模型给出某个预测？
- 模型有多容易被欺骗？

### 可视化方法分类

| 方法 | 对象 | 输入 | 典型技术 |
|------|------|------|----------|
| **显著性图** | 输入像素 | 梯度 w.r.t 输入 | Saliency Map |
| **类激活映射** | 中间特征 | 梯度 w.r.t 特征图 | Grad-CAM |
| **特征可视化** | 神经元 | 优化输入图像 | Deep Dream |
| **嵌入可视化** | 高维特征 | 降维 | t-SNE |
| **对抗样本** | 鲁棒性 | 梯度扰动 | FGSM |'''),
    md_cell(r'''## 2. 显著性图：输入梯度

### 原理

[[Simonyan et al., 2014]](https://arxiv.org/abs/1312.6034)

显著性图计算损失对输入像素的梯度：

$$S(x) = \left|\frac{\partial L_c}{\partial x}\right|$$

其中 $L_c$ 是类别 $c$ 的得分（未归一化的 logit）。梯度大的像素对分类决策影响最大。

### 与权重可视化的区别

- **权重可视化**：看 $W$ 的值——但 $W$ 的值不直接告诉我们输入的重要性
- **显著性图**：看 $\frac{\partial L}{\partial x}$——直接告诉我们在当前输入中哪些像素重要'''),
    md_cell(r'''## 3. 手算验证：简单网络的显著性

### 线性分类器的显著性

对于线性分类器 $s = Wx + b$，类别 $c$ 的得分：
$$s_c = W_c^T x + b_c$$

梯度：
$$\frac{\partial s_c}{\partial x} = W_c$$

即：权重本身就是显著性图——权重大的像素对该类更重要。

### 两层网络的显著性

对于 $s = W_2 \text{ReLU}(W_1 x + b_1) + b_2$：

$$\frac{\partial s_c}{\partial x} = W_1^T \text{diag}(\mathbb{1}[W_1 x + b_1 > 0]) W_2^T e_c$$

手算（2D 输入，1 隐单元，1 类）：
- $W_1 = [0.5, -0.3]$, $b_1 = 0.1$, $W_2 = [1.0]$, $b_2 = 0$
- $x = [1.0, 2.0]$
- $z_1 = 0.5 \times 1 + (-0.3) \times 2 + 0.1 = 0.0$
- $a_1 = \text{ReLU}(0.0) = 0$
- $s = 1.0 \times 0 = 0$
- 梯度：$\frac{\partial s}{\partial x} = W_1^T \times \mathbb{1}[z_1 > 0] \times W_2 = [0.5, -0.3] \times 0 \times 1 = [0, 0]$

因为 $z_1 = 0$，ReLU 输出为 0，梯度也为 0——这个像素对当前输出没有贡献。'''),
    code_cell(r'''# Verify saliency calculation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
np.random.seed(42)

# Simple 2-layer network
W1 = np.array([[0.5, -0.3]])
b1 = np.array([0.1])
W2 = np.array([[1.0]])
b2 = np.array([0.0])

x = np.array([1.0, 2.0])

# Forward
z1 = W1 @ x + b1
a1 = np.maximum(0, z1)
s = W2 @ a1 + b2

# Backward (gradient w.r.t input)
# ds/da1 = W2 = 1.0
# da1/dz1 = 1 if z1 > 0 else 0
# dz1/dx = W1
relu_mask = (z1 > 0).astype(float)
dscore_dx = W1.T * (W2.T * relu_mask)

print(f"z1 = {z1}, a1 = {a1}, score = {s}")
print(f"ReLU mask = {relu_mask}")
print(f"Saliency (ds/dx) = {dscore_dx.flatten()}")
print(f"\nWhen z1=0, ReLU blocks gradient: saliency = [0, 0]")

# Try with different input where z1 > 0
x2 = np.array([2.0, 0.0])
z1_2 = W1 @ x2 + b1
relu_mask2 = (z1_2 > 0).astype(float)
saliency2 = W1.T * (W2.T * relu_mask2)
print(f"\nWith x=[2,0]: z1={z1_2}, saliency={saliency2.flatten()}")'''),
    md_cell(r'''## 4. 代码实现：显著性图'''),
    code_cell(r'''class TinyCNN:
    """Simplified CNN for visualization experiments.
    
    Architecture: conv(1->4, 3x3) -> relu -> flatten -> fc(->3)
    """
    def __init__(self, input_size=16):
        # Conv layer: 1 channel -> 4 channels, 3x3 kernel
        self.conv_W = np.random.randn(4, 1, 3, 3) * 0.1
        self.conv_b = np.zeros(4)
        # FC layer
        self.fc_W = np.random.randn(3, 4 * (input_size - 2) * (input_size - 2)) * 0.1
        self.fc_b = np.zeros(3)
        self.input_size = input_size
    
    def forward(self, x):
        """Forward pass. x: (1, H, W) or (H, W)"""
        if x.ndim == 2:
            x = x[np.newaxis, ...]
        C, H, W = x.shape
        # Conv: (4, H-2, W-2)
        conv_out = np.zeros((4, H-2, W-2))
        for oc in range(4):
            for i in range(H-2):
                for j in range(W-2):
                    patch = x[:, i:i+3, j:j+3]
                    conv_out[oc, i, j] = np.sum(patch * self.conv_W[oc]) + self.conv_b[oc]
        relu_out = np.maximum(0, conv_out)
        # Flatten and FC
        flat = relu_out.flatten()
        scores = self.fc_W @ flat + self.fc_b
        cache = (x, conv_out, relu_out, flat)
        return scores, cache
    
    def backward_input(self, cache, target_class):
        """Compute gradient w.r.t input (saliency)."""
        x, conv_out, relu_out, flat = cache
        C, H, W = x.shape
        
        # Gradient of target class score w.r.t FC input
        dscore_dfc = self.fc_W[target_class]  # (D,)
        # Gradient w.r.t conv output
        dscore_dconv = dscore_dfc.reshape(4, H-2, W-2)
        # Through ReLU
        dscore_dconv = dscore_dconv * (conv_out > 0)
        # Through conv (transpose convolution)
        dscore_dx = np.zeros_like(x)
        for oc in range(4):
            for i in range(H-2):
                for j in range(W-2):
                    grad = dscore_dconv[oc, i, j]
                    dscore_dx[:, i:i+3, j:j+3] += self.conv_W[oc] * grad
        
        return np.abs(dscore_dx[0])  # return magnitude for first channel

# Test on synthetic image
np.random.seed(42)
net = TinyCNN(input_size=16)

# Create synthetic image with a bright region
img = np.zeros((16, 16))
img[6:10, 6:10] = 1.0  # bright square
img += np.random.randn(16, 16) * 0.05

scores, cache = net.forward(img)
print(f"Scores: {scores}")
print(f"Predicted class: {np.argmax(scores)}")

saliency = net.backward_input(cache, np.argmax(scores))
print(f"Saliency shape: {saliency.shape}")
print(f"Saliency max: {saliency.max():.4f}")'''),
    code_cell(r'''# Visualize saliency map
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ax1, ax2, ax3 = axes

ax1.imshow(img, cmap='gray')
ax1.set_title('Input Image')
ax1.axis('off')

ax2.imshow(saliency, cmap='hot')
ax2.set_title('Saliency Map (|dScore/dInput|)')
ax2.axis('off')

# Overlay
ax3.imshow(img, cmap='gray', alpha=0.5)
ax3.imshow(saliency, cmap='hot', alpha=0.5)
ax3.set_title('Overlay')
ax3.axis('off')

plt.suptitle('Saliency Map Visualization', fontsize=14)
plt.tight_layout()
plt.savefig('saliency_map.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 1: Saliency map")'''),
    md_cell(r'''## 5. Grad-CAM：类激活映射

### 原理

[[Selvaraju et al., 2017]](https://arxiv.org/abs/1610.02391)

Grad-CAM 使用最后一个卷积层的梯度来定位重要区域：

1. 计算类别得分对最后卷积层特征图 $A^k$ 的梯度
2. 全局平均池化得到通道权重 $\alpha_k = \frac{1}{Z} \sum_{i,j} \frac{\partial y_c}{\partial A^k_{ij}}$
3. 加权求和：$L_{Grad-CAM} = \text{ReLU}\left(\sum_k \alpha_k A^k\right)$

### 与 Saliency 的区别

| 特性 | Saliency Map | Grad-CAM |
|------|-------------|----------|
| 梯度对象 | 输入像素 | 卷积特征图 |
| 分辨率 | 输入分辨率 | 特征图分辨率（较低） |
| 平滑度 | 噪声较大 | 更平滑、更语义化 |
| 计算量 | 高 | 低 |'''),
    code_cell(r'''# Grad-CAM implementation
def grad_cam(net, x, target_class):
    """Compute Grad-CAM for a target class.
    
    Uses the last conv layer's output and gradients.
    """
    scores, cache = net.forward(x)
    x_in, conv_out, relu_out, flat = cache
    
    # Get gradient of target score w.r.t conv output
    dscore_dfc = net.fc_W[target_class]
    C_out, H_out, W_out = conv_out.shape
    dscore_dconv = dscore_dfc.reshape(C_out, H_out, W_out)
    
    # Channel weights: global average pooling of gradients
    weights = np.mean(dscore_dconv, axis=(1, 2))  # (C_out,)
    
    # Weighted sum of feature maps
    cam = np.zeros((H_out, W_out))
    for k in range(C_out):
        cam += weights[k] * conv_out[k]
    
    # ReLU and normalize
    cam = np.maximum(0, cam)
    if cam.max() > 0:
        cam = cam / cam.max()
    
    # Upsample to input size (nearest neighbor)
    cam_upsampled = np.zeros_like(x if x.ndim == 2 else x[0])
    if x.ndim == 3:
        x = x[0]
    H_in, W_in = x.shape
    for i in range(H_in):
        for j in range(W_in):
            ii = min(int(i * H_out / H_in), H_out - 1)
            jj = min(int(j * W_out / W_in), W_out - 1)
            cam_upsampled[i, j] = cam[ii, jj]
    
    return cam_upsampled

# Compute Grad-CAM
cam = grad_cam(net, img, np.argmax(scores))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ax1, ax2, ax3 = axes
ax1.imshow(img, cmap='gray')
ax1.set_title('Input Image')
ax1.axis('off')
ax2.imshow(cam, cmap='jet')
ax2.set_title('Grad-CAM Heatmap')
ax2.axis('off')
ax3.imshow(img, cmap='gray', alpha=0.6)
ax3.imshow(cam, cmap='jet', alpha=0.4)
ax3.set_title('Grad-CAM Overlay')
ax3.axis('off')
plt.suptitle('Grad-CAM Visualization', fontsize=14)
plt.tight_layout()
plt.savefig('grad_cam.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 2: Grad-CAM")'''),
    md_cell(r'''## 6. FGSM 对抗攻击

### 原理

[[Goodfellow et al., 2015]](https://arxiv.org/abs/1412.6572)

Fast Gradient Sign Method (FGSM) 沿梯度方向添加小扰动来欺骗分类器：

$$x_{adv} = x + \epsilon \cdot \text{sign}\left(\frac{\partial L}{\partial x}\right)$$

- $\epsilon$：扰动大小（通常很小，如 0.01）
- $\text{sign}$：取梯度方向（+1 或 -1）

关键洞察：扰动在人眼不可见，但能完全改变模型预测。'''),
    code_cell(r'''# FGSM adversarial attack
def fgsm_attack(net, x, y_true, epsilon=0.1):
    """FGSM adversarial attack.
    
    x_adv = x + epsilon * sign(dL/dx)
    """
    scores, cache = net.forward(x)
    x_in, conv_out, relu_out, flat = cache
    
    # Compute gradient of loss w.r.t input
    # Softmax loss
    shifted = scores - np.max(scores)
    exp_s = np.exp(shifted)
    probs = exp_s / np.sum(exp_s)
    
    # dL/dscores
    dscores = probs.copy()
    dscores[y_true] -= 1
    
    # Backprop to input
    # dL/dfc_input
    dL_dfc = net.fc_W.T @ dscores
    C_out, H_out, W_out = conv_out.shape
    dL_dconv = dL_dfc.reshape(C_out, H_out, W_out)
    dL_dconv = dL_dconv * (conv_out > 0)
    
    H, W = x.shape
    dL_dx = np.zeros((1, H, W))
    for oc in range(4):
        for i in range(H-2):
            for j in range(W-2):
                grad = dL_dconv[oc, i, j]
                dL_dx[:, i:i+3, j:j+3] += net.conv_W[oc] * grad
    
    # FGSM: add sign of gradient
    x_adv = x + epsilon * np.sign(dL_dx[0])
    
    return x_adv

# Generate adversarial example
np.random.seed(42)
y_true = np.argmax(scores)
x_adv = fgsm_attack(net, img, y_true, epsilon=0.15)

# Compare predictions
scores_orig, _ = net.forward(img)
scores_adv, _ = net.forward(x_adv)

print(f"Original scores: {scores_orig}")
print(f"Adversarial scores: {scores_adv}")
print(f"Original prediction: {np.argmax(scores_orig)}")
print(f"Adversarial prediction: {np.argmax(scores_adv)}")
print(f"Perturbation (L2 norm): {np.linalg.norm(x_adv - img):.4f}")
print(f"Max pixel change: {np.max(np.abs(x_adv - img)):.4f}")'''),
    code_cell(r'''# Visualize adversarial attack
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title(f'Original (pred={np.argmax(scores_orig)})')
axes[0, 0].axis('off')

axes[0, 1].imshow(x_adv, cmap='gray')
axes[0, 1].set_title(f'Adversarial (pred={np.argmax(scores_adv)})')
axes[0, 1].axis('off')

# Perturbation (amplified for visibility)
pert = x_adv - img
axes[0, 2].imshow(pert * 10, cmap='RdBu')
axes[0, 2].set_title('Perturbation (10x)')
axes[0, 2].axis('off')

# Saliency for original
sal_orig = net.backward_input(net.forward(img)[1], np.argmax(scores_orig))
axes[1, 0].imshow(sal_orig, cmap='hot')
axes[1, 0].set_title('Original Saliency')
axes[1, 0].axis('off')

# Saliency for adversarial
sal_adv = net.backward_input(net.forward(x_adv)[1], np.argmax(scores_adv))
axes[1, 1].imshow(sal_adv, cmap='hot')
axes[1, 1].set_title('Adversarial Saliency')
axes[1, 1].axis('off')

# Score comparison
axes[1, 2].bar(['C0 orig', 'C0 adv', 'C1 orig', 'C1 adv', 'C2 orig', 'C2 adv'],
               [scores_orig[0], scores_adv[0], scores_orig[1], scores_adv[1], 
                scores_orig[2], scores_adv[2]],
               color=['#2563eb', '#ea580c'] * 3)
axes[1, 2].set_title('Score Comparison')
plt.suptitle('FGSM Adversarial Attack', fontsize=14)
plt.tight_layout()
plt.savefig('fgsm_attack.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 3: FGSM adversarial attack")'''),
    md_cell(r'''## 7. t-SNE 嵌入可视化

### 原理

[[Maaten & Hinton, 2008]](https://www.jmlr.org/papers/v9/vandermaaten08a.html)

t-SNE 将高维特征降维到 2D，保留局部结构：
- 相似的样本在 2D 空间中靠近
- 不相似的样本在 2D 空间中远离

### 简化实现

完整的 t-SNE 复杂度较高，这里使用 PCA + 迭代优化的简化版。'''),
    code_cell(r'''# Simplified t-SNE (actually PCA + gradient descent, for visualization)
def pca(X, n_components=2):
    """Simple PCA for dimensionality reduction."""
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_centered @ Vt[:n_components].T

def simple_tsne(X, n_components=2, n_iter=200, lr=50, perplexity=10):
    """Simplified t-SNE-like embedding.
    
    This is NOT the full t-SNE algorithm but a simplified version
    that uses PCA init + gradient descent for educational purposes.
    """
    n = X.shape[0]
    # PCA initialization
    Y = pca(X, n_components)
    Y = Y * 0.1  # scale down
    
    # Compute pairwise distances in original space
    sum_X = np.sum(X**2, axis=1)
    distances = np.sqrt(np.maximum(sum_X[:, None] + sum_X[None, :] - 2 * X @ X.T, 0))
    
    # Compute P (similarities in high-dim)
    P = np.exp(-distances**2 / (2 * np.std(distances)**2))
    np.fill_diagonal(P, 0)
    P = P / P.sum()
    P = np.maximum(P, 1e-12)
    
    for it in range(n_iter):
        # Compute Q (similarities in low-dim)
        sum_Y = np.sum(Y**2, axis=1)
        dist_Y = 1 + sum_Y[:, None] + sum_Y[None, :] - 2 * Y @ Y.T
        Q = 1.0 / np.maximum(dist_Y, 1e-12)
        np.fill_diagonal(Q, 0)
        Q = Q / Q.sum()
        Q = np.maximum(Q, 1e-12)
        
        # Gradient
        PQ = P - Q
        grad = np.zeros_like(Y)
        for i in range(n):
            diff = (Y[i] - Y).T
            grad[i] = 4 * np.sum((PQ[i] * Q[i])[:, None] * diff, axis=1)
        
        Y -= lr * grad / n
        Y = Y - Y.mean(axis=0)
    
    return Y

# Generate features and visualize
np.random.seed(42)
n_per_class = 30
n_classes = 3
feature_dim = 50

features = []
labels = []
for c in range(n_classes):
    center = np.random.randn(feature_dim) * 3
    feat = center + np.random.randn(n_per_class, feature_dim) * 0.5
    features.append(feat)
    labels.extend([c] * n_per_class)

X_features = np.vstack(features)
y_labels = np.array(labels)

# t-SNE embedding
embedding = simple_tsne(X_features, n_iter=300, lr=100)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# PCA
pca_2d = pca(X_features)
for c in range(n_classes):
    mask = y_labels == c
    axes[0].scatter(pca_2d[mask, 0], pca_2d[mask, 1], label=f'Class {c}', s=20)
axes[0].set_title('PCA Embedding')
axes[0].legend()

# t-SNE
for c in range(n_classes):
    mask = y_labels == c
    axes[1].scatter(embedding[mask, 0], embedding[mask, 1], label=f'Class {c}', s=20)
axes[1].set_title('t-SNE-like Embedding')
axes[1].legend()

plt.suptitle('Feature Embedding Visualization', fontsize=14)
plt.tight_layout()
plt.savefig('tsne_embedding.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 4: t-SNE embedding")'''),
    md_cell(r'''## 8. 特征反演

### 原理

[[Mahendran & Vedaldi, 2015]](https://arxiv.org/abs/1412.0035)

特征反演通过优化输入图像来匹配给定的特征表示：

$$x^* = \arg\min_x \|f(x) - f(x_0)\|^2 + \lambda R(x)$$

其中 $f$ 是网络特征提取器，$R(x)$ 是自然图像先验正则化。'''),
    code_cell(r'''# Feature inversion demo (simplified)
def feature_inversion(net, target_features, input_shape, lr=0.1, n_iter=200):
    """Reconstruct image from target features via gradient descent.
    
    Minimizes ||features(x) - target_features||^2
    """
    x = np.random.randn(*input_shape) * 0.1
    
    losses = []
    for _ in range(n_iter):
        scores, cache = net.forward(x)
        x_in, conv_out, relu_out, flat = cache
        current_feat = flat  # use flattened features as representation
        
        # Loss: L2 distance to target features
        diff = current_feat - target_features
        loss = np.sum(diff**2)
        losses.append(loss)
        
        # Gradient: dL/dfeat = 2 * diff
        dfeat = 2 * diff
        
        # Backprop to input (through FC, ReLU, Conv)
        # Simplified: use the network's backward
        dscore = net.fc_W.T @ dfeat
        C_out, H_out, W_out = conv_out.shape
        dconv = dscore.reshape(C_out, H_out, W_out)
        dconv = dconv * (conv_out > 0)
        
        H, W = input_shape
        dx = np.zeros((1, H, W))
        for oc in range(4):
            for i in range(H-2):
                for j in range(W-2):
                    grad = dconv[oc, i, j]
                    dx[:, i:i+3, j:j+3] += net.conv_W[oc] * grad
        
        x -= lr * dx[0]
    
    return x, losses

# Get target features from original image
scores_orig, cache_orig = net.forward(img)
target_feat = cache_orig[3]  # flattened features

# Reconstruct from features
reconstructed, inv_losses = feature_inversion(net, target_feat, (16, 16), lr=0.01, n_iter=300)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(reconstructed, cmap='gray')
axes[1].set_title('Reconstructed from Features')
axes[1].axis('off')
axes[2].plot(inv_losses, color='#ea580c')
axes[2].set_title('Inversion Loss')
axes[2].set_xlabel('Iteration')
axes[2].set_ylabel('L2 Loss')
plt.suptitle('Feature Inversion', fontsize=14)
plt.tight_layout()
plt.savefig('feature_inversion.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 5: Feature inversion")'''),
    md_cell(r'''## 作业

### 作业 1：实现 SmoothGrad

SmoothGrad 通过对多个带噪声的输入取平均来平滑显著性图，减少噪声。

[[Smilkov et al., 2017]](https://arxiv.org/abs/1706.03825)'''),
    code_cell(r'''# Homework 1: SmoothGrad
def smooth_grad(net, x, target_class, n_samples=20, noise_level=0.1):
    """Compute SmoothGrad: average saliency over noisy inputs.
    
    SmoothGrad(x) = (1/N) * sum_i Saliency(x + noise_i)
    where noise_i ~ N(0, noise_level^2)
    """
    # TODO: implement SmoothGrad
    # 1. Add Gaussian noise to input n_samples times
    # 2. Compute saliency for each noisy input
    # 3. Average the saliency maps
    pass

# Test
# sal_smooth = smooth_grad(net, img, np.argmax(scores), n_samples=20, noise_level=0.05)
# assert sal_smooth.shape == img.shape, "SmoothGrad shape should match input"
# assert sal_smooth.min() >= 0, "SmoothGrad values should be non-negative"
print("Homework 1 template ready. Implement SmoothGrad.")'''),
    md_cell(r'''### 作业 2：实现迭代 FGSM

基本 FGSM 只做一步扰动。迭代 FGSM 多次小步扰动，通常攻击成功率更高。

[[Kurakin et al., 2017]](https://arxiv.org/abs/1607.02533)'''),
    code_cell(r'''# Homework 2: Iterative FGSM
def iterative_fgsm(net, x, y_true, epsilon=0.1, n_steps=10, alpha=None):
    """Iterative FGSM (Basic Iterative Method).
    
    For each step: x = x + alpha * sign(dL/dx)
    Clip x to [x - epsilon, x + epsilon]
    """
    if alpha is None:
        alpha = epsilon / n_steps
    # TODO: implement iterative FGSM
    pass

# Test
# x_adv_iter = iterative_fgsm(net, img, np.argmax(scores), epsilon=0.15, n_steps=10)
# scores_iter, _ = net.forward(x_adv_iter)
# assert np.argmax(scores_iter) != np.argmax(scores), "Adversarial prediction should differ"
print("Homework 2 template ready. Implement iterative FGSM.")'''),
    md_cell(r'''### 作业 3：实现 Guided Backprop

Guided Backprop 结合 ReLU 的正向梯度，只保留对正激活有贡献的梯度。

[[Springenberg et al., 2015]](https://arxiv.org/abs/1412.6806)'''),
    code_cell(r'''# Homework 3: Guided Backpropagation
def guided_backprop(net, x, target_class):
    """Guided backpropagation.
    
    Key idea: in ReLU backward, only pass gradient if 
    both forward activation > 0 AND gradient > 0.
    """
    # TODO: implement guided backprop
    # Modify ReLU backward: grad = grad * (activation > 0) * (grad > 0)
    pass

# Test
# guided_sal = guided_backprop(net, img, np.argmax(scores))
# assert guided_sal.shape == img.shape
# assert guided_sal.min() >= 0
print("Homework 3 template ready. Implement guided backpropagation.")'''),
    md_cell(r'''## 小结

| 方法 | 原理 | 用途 |
|------|------|------|
| **Saliency Map** | $\|\partial L / \partial x\|$ | 定位重要像素 |
| **Grad-CAM** | 卷积特征图梯度加权 | 粗粒度定位 |
| **FGSM** | $\epsilon \cdot \text{sign}(\partial L / \partial x)$ | 对抗攻击 |
| **t-SNE** | 降维保留局部结构 | 特征空间可视化 |
| **特征反演** | 优化输入匹配特征 | 理解特征表示 |
| **SmoothGrad** | 噪声平均平滑梯度 | 去噪显著性图 |
| **Guided BP** | ReLU 正向梯度过滤 | 更清晰的可视化 |

**关键洞察**：CNN 不是真正的黑盒——通过梯度分析可以理解其决策过程。但对抗样本的脆弱性也暴露了 CNN 对人类不可见的扰动极为敏感，这对安全关键应用是严重挑战。'''),
    md_cell(r'''## 参考文献

1. [[Simonyan et al., 2014]](https://arxiv.org/abs/1312.6034) Simonyan et al. *Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps*. ICLR Workshop 2014.
2. [[Selvaraju et al., 2017]](https://arxiv.org/abs/1610.02391) Selvaraju et al. *Grad-CAM: Visual Explanations from Deep Networks*. ICCV 2017.
3. [[Goodfellow et al., 2015]](https://arxiv.org/abs/1412.6572) Goodfellow et al. *Explaining and Harnessing Adversarial Examples*. ICLR 2015.
4. [[Maaten & Hinton, 2008]](https://www.jmlr.org/papers/v9/vandermaaten08a.html) van der Maaten & Hinton. *Visualizing Data using t-SNE*. JMLR 2008.
5. [[Mahendran & Vedaldi, 2015]](https://arxiv.org/abs/1412.0035) Mahendran & Vedaldi. *Understanding Deep Image Representations by Inverting Them*. CVPR 2015.
6. [[Springenberg et al., 2015]](https://arxiv.org/abs/1412.6806) Springenberg et al. *Striving for Simplicity: The All Convolutional Net*. ICLR Workshop 2015.

---

> 本节内容参考 [Stanford CS231n](https://cs231n.stanford.edu/) | [Deep Visualization](https://cs231n.github.io/understanding-cnn/)'''),
    md_cell(r'''---

## 参考代码实现

以下 GitHub 仓库提供了本节内容的完整代码实现，建议结合学习：

- **[utkuozbulak/pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)** (8233 stars): CNN 可视化方法合集（Saliency、Grad-CAM、特征反演等）
  - 仓库地址: https://github.com/utkuozbulak/pytorch-cnn-visualizations

- **[kazuto1011/grad-cam-pytorch](https://github.com/kazuto1011/grad-cam-pytorch)** (804 stars): Grad-CAM PyTorch 完整实现
  - 仓库地址: https://github.com/kazuto1011/grad-cam-pytorch

- **[jacobgil/vit-explain](https://github.com/jacobgil/vit-explain)** (1099 stars): Vision Transformer 可解释性工具
  - 仓库地址: https://github.com/jacobgil/vit-explain

> 标注说明: 以上仓库按热度排序，优先推荐 stars 最多的实现.'''),
])
save_notebook(nb4, os.path.join(BASE, "part2-cnn-vision", "lecture-10-visualization", "practice.ipynb"))

# ============================================================
# Notebook 5: Transfer Learning and Fine-tuning
# ============================================================
print("Generating Notebook 5: Transfer Learning...")
nb5 = make_notebook([
    md_cell(r'''# 迁移学习与微调

> **斯坦福 CS231n** | Transfer Learning and Fine-tuning

## 本章导读

迁移学习将在大规模数据集上预训练的模型迁移到小数据集上，是深度学习最实用的技术之一。本节从直觉到代码，完整理解特征提取和微调策略。

**学习目标：**
- 理解预训练模型的概念和价值
- 区分特征提取（冻结）与微调（解冻）
- 实现渐进式解冻策略
- 比较 random init vs pretrained 的效果
- 理解不同层的学习率设置

**参考来源：** [CS231n Transfer Learning](https://cs231n.github.io/transfer-learning/) | [torchvision models](https://pytorch.org/vision/stable/models.html)'''),
    md_cell(r'''## 1. 直觉理解：迁移学习为什么有效？

### 核心问题

在 ImageNet（1400 万图像）上训练一个 ResNet 需要数周和大量算力。但如果你只有 1000 张猫狗图片，如何获得高质量模型？

### 迁移学习的思想

**预训练 + 微调**：
1. 在大数据集上预训练（如 ImageNet）
2. 在目标数据集上微调或提取特征

### 为什么有效？

CNN 的层级特征具有迁移性：
- **浅层**（边缘、纹理）：通用特征，跨领域可迁移
- **深层**（物体部件、语义）：领域相关，需微调

[[Yosinski et al., 2014]](https://arxiv.org/abs/1411.1792) 实验证明：特征可迁移性随层深度递减。'''),
    md_cell(r'''### 三种迁移学习策略

| 策略 | 预训练层 | 新层 | 训练方式 | 适用场景 |
|------|----------|------|----------|----------|
| **特征提取** | 全部冻结 | 仅分类头 | 只训练新层 | 小数据集（<1K） |
| **微调** | 全部解冻 | 分类头 | 全部训练 | 中等数据集（1K-100K） |
| **渐进式解冻** | 逐层解冻 | 分类头 | 从深到浅逐步 | 平衡稳定性和性能 |'''),
    md_cell(r'''## 2. 手算验证：特征提取 vs 随机初始化

### 特征提取的数学表示

预训练模型 $f_{pre}$ 的编码器部分 $g(\cdot)$ 固定不动，只训练线性分类器 $W$：

$$\text{features} = g(x; \theta_{pre}) \quad \text{(frozen)}$$
$$\hat{y} = W \cdot \text{features} + b \quad \text{(trainable)}$$

### 效果对比

假设 3 类分类任务，100 个样本：
- **随机初始化**：$W$ 从随机开始，需要学习所有特征——需要大量数据
- **特征提取**：$g$ 已提取良好特征，$W$ 只需学习简单线性映射——需要很少数据

### 手算：线性可分性

预训练特征可能已经线性可分。设 2D 特征 3 类：
- 类 0 中心：$(0, 0)$
- 类 1 中心：$(3, 3)$
- 类 2 中心：$(-3, 3)$

简单最近邻或线性分类器即可达到接近 100% 准确率——因为预训练特征已经很好地分离了类别。'''),
    code_cell(r'''# Verify: feature extraction vs random init
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
np.random.seed(42)

# Simulate pretrained features (well-separated clusters)
n_per_class = 50
centers = np.array([[0, 0], [3, 3], [-3, 3]])
pretrained_features = []
labels = []
for c in range(3):
    feat = centers[c] + np.random.randn(n_per_class, 2) * 0.3
    pretrained_features.append(feat)
    labels.extend([c] * n_per_class)
X_pre = np.vstack(pretrained_features)
y_pre = np.array(labels)

# Simulate random features (no structure)
X_random = np.random.randn(150, 2) * 2

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for c in range(3):
    mask = y_pre == c
    axes[0].scatter(X_pre[mask, 0], X_pre[mask, 1], label=f'Class {c}', s=15)
axes[0].set_title('Pretrained Features (well-separated)')
axes[0].legend()
for c in range(3):
    mask = y_pre == c
    axes[1].scatter(X_random[mask, 0], X_random[mask, 1], label=f'Class {c}', s=15)
axes[1].set_title('Random Features (no structure)')
axes[1].legend()
plt.suptitle('Feature Space Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('feature_space_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 1: Pretrained vs random features")'''),
    md_cell(r'''## 3. 代码实现：迁移学习模拟

用 numpy 模拟预训练模型：先在大数据集上"预训练"，再在小数据集上"迁移"。'''),
    code_cell(r'''class PretrainedModel:
    """Simulate a pretrained model with feature extractor + classifier."""
    def __init__(self, input_dim, hidden_dims, n_classes):
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims)-1):
            W = np.random.randn(dims[i+1], dims[i]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            layers.append({'W': W, 'b': b})
        # Classifier head
        W_out = np.random.randn(n_classes, dims[-1]) * np.sqrt(2.0 / dims[-1])
        b_out = np.zeros(n_classes)
        layers.append({'W': W_out, 'b': b_out})
        self.layers = layers
        self.n_layers = len(layers)
    
    def forward(self, X):
        """Forward pass, return activations for each layer."""
        activations = [X]
        cache = []
        a = X
        for i, layer in enumerate(self.layers):
            z = a @ layer['W'].T + layer['b']
            if i < self.n_layers - 1:
                a = np.maximum(0, z)  # ReLU for hidden layers
            else:
                a = z  # No activation for output
            activations.append(a)
            cache.append((layer, z, a))
        return a, activations, cache
    
    def extract_features(self, X, layer_idx=None):
        """Extract features from a specific layer."""
        if layer_idx is None:
            layer_idx = self.n_layers - 2  # last hidden layer
        _, activations, _ = self.forward(X)
        return activations[layer_idx + 1]
    
    def train_step(self, X, y, lr=0.01, freeze_layers=0):
        """Train one step. freeze_layers=number of layers to freeze (from start)."""
        scores, activations, cache = self.forward(X)
        N = X.shape[0]
        
        # Softmax loss
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_s = np.exp(shifted)
        probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        loss = -np.log(probs[np.arange(N), y]).mean()
        
        # Backward
        dscores = probs.copy()
        dscores[np.arange(N), y] -= 1
        dscores /= N
        
        da = dscores
        for i in range(self.n_layers - 1, -1, -1):
            if i < freeze_layers:
                break  # frozen layer
            layer, z, a = cache[i]
            dW = da.T @ activations[i]
            db = da.sum(axis=0)
            if i > 0:
                da = (da @ layer['W']) * (activations[i] > 0)
            layer['W'] -= lr * dW
            layer['b'] -= lr * db
        
        return loss
    
    def predict(self, X):
        scores, _, _ = self.forward(X)
        return np.argmax(scores, axis=1)

print("PretrainedModel class defined")'''),
    code_cell(r'''# Phase 1: Pretrain on large dataset (many classes)
np.random.seed(42)
n_pretrain_classes = 10
input_dim = 20
pretrain_model = PretrainedModel(input_dim, [32, 16], n_pretrain_classes)

# Generate large pretraining dataset
n_pretrain = 500
X_pretrain = np.random.randn(n_pretrain, input_dim)
# Add class structure
y_pretrain = np.random.randint(0, n_pretrain_classes, n_pretrain)
# Inject class-dependent patterns
for c in range(n_pretrain_classes):
    mask = y_pretrain == c
    X_pretrain[mask] += np.random.randn(mask.sum(), input_dim) * 0.5 + c

print("Phase 1: Pretraining...")
pretrain_losses = []
for epoch in range(300):
    loss = pretrain_model.train_step(X_pretrain, y_pretrain, lr=0.05)
    pretrain_losses.append(loss)
    if epoch % 100 == 0:
        acc = np.mean(pretrain_model.predict(X_pretrain) == y_pretrain)
        print(f"  Pretrain epoch {epoch}: loss={loss:.4f}, acc={acc:.2%}")

print(f"Pretraining accuracy: {np.mean(pretrain_model.predict(X_pretrain) == y_pretrain):.2%}")'''),
    code_cell(r'''# Phase 2: Transfer to small dataset (fewer classes)
np.random.seed(42)
n_target_classes = 3
n_target = 30  # small dataset!

# Generate target data (subset of pretrain classes)
target_classes = np.random.choice(n_pretrain_classes, n_target_classes, replace=False)
X_target = np.random.randn(n_target, input_dim)
y_target = np.random.choice(n_target_classes, n_target)

for i, c in enumerate(target_classes):
    mask = y_target == i
    X_target[mask] += np.random.randn(mask.sum(), input_dim) * 0.5 + c

# Strategy 1: Feature extraction (freeze all, replace classifier)
transfer_fe = PretrainedModel(input_dim, [32, 16], n_target_classes)
# Copy pretrained hidden layers
for i in range(len(transfer_fe.layers) - 1):
    transfer_fe.layers[i] = pretrain_model.layers[i].copy() if isinstance(pretrain_model.layers[i], dict) else pretrain_model.layers[i]

# Actually copy weights
for i in range(len(transfer_fe.layers) - 1):
    transfer_fe.layers[i]['W'] = pretrain_model.layers[i]['W'].copy()
    transfer_fe.layers[i]['b'] = pretrain_model.layers[i]['b'].copy()

fe_losses = []
for epoch in range(200):
    loss = transfer_fe.train_step(X_target, y_target, lr=0.1, freeze_layers=2)
    fe_losses.append(loss)

fe_acc = np.mean(transfer_fe.predict(X_target) == y_target)
print(f"Strategy 1 - Feature Extraction: acc={fe_acc:.2%}")

# Strategy 2: Fine-tuning (unfreeze all)
transfer_ft = PretrainedModel(input_dim, [32, 16], n_target_classes)
for i in range(len(transfer_ft.layers) - 1):
    transfer_ft.layers[i]['W'] = pretrain_model.layers[i]['W'].copy()
    transfer_ft.layers[i]['b'] = pretrain_model.layers[i]['b'].copy()

ft_losses = []
for epoch in range(200):
    loss = transfer_ft.train_step(X_target, y_target, lr=0.01, freeze_layers=0)
    ft_losses.append(loss)

ft_acc = np.mean(transfer_ft.predict(X_target) == y_target)
print(f"Strategy 2 - Fine-tuning:        acc={ft_acc:.2%}")

# Strategy 3: Random initialization (no transfer)
random_model = PretrainedModel(input_dim, [32, 16], n_target_classes)
rand_losses = []
for epoch in range(200):
    loss = random_model.train_step(X_target, y_target, lr=0.05)
    rand_losses.append(loss)

rand_acc = np.mean(random_model.predict(X_target) == y_target)
print(f"Strategy 3 - Random Init:        acc={rand_acc:.2%}")'''),
    code_cell(r'''# Visualize transfer learning comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
ax1 = axes[0]
ax1.plot(fe_losses, label='Feature Extraction', color='#16a34a', linewidth=2)
ax1.plot(ft_losses, label='Fine-tuning', color='#2563eb', linewidth=2)
ax1.plot(rand_losses, label='Random Init', color='#ea580c', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss: Transfer vs Random')
ax1.legend()

# Accuracy comparison
ax2 = axes[1]
strategies = ['Feature\nExtraction', 'Fine-tuning', 'Random Init']
accs = [fe_acc, ft_acc, rand_acc]
colors = ['#16a34a', '#2563eb', '#ea580c']
bars = ax2.bar(strategies, accs, color=colors)
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy Comparison (30 samples)')
ax2.set_ylim(0, 1.1)
for bar, acc in zip(bars, accs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{acc:.1%}', ha='center', fontsize=12)
plt.tight_layout()
plt.savefig('transfer_learning_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 2: Transfer learning comparison")'''),
    md_cell(r'''## 4. 渐进式解冻

### 原理

渐进式解冻（Progressive Unfreezing）逐步解冻网络层：
1. 先冻结所有预训练层，只训练分类头
2. 解冻最后一层，用小学习率训练
3. 逐步解冻更浅的层

[[Howard & Ruder, 2018]](https://arxiv.org/abs/1801.06146) ULMFiT 论文提出了这种方法。

### 学习率策略

不同层使用不同学习率（ Discriminative Learning Rates）：
- 浅层（通用特征）：小学习率（如 $10^{-5}$）
- 深层（领域特征）：中学习率（如 $10^{-4}$）
- 分类头（新参数）：大学习率（如 $10^{-3}$）'''),
    code_cell(r'''# Progressive unfreezing experiment
np.random.seed(42)
n_phases = 3
epochs_per_phase = 100

transfer_progressive = PretrainedModel(input_dim, [32, 16], n_target_classes)
for i in range(len(transfer_progressive.layers) - 1):
    transfer_progressive.layers[i]['W'] = pretrain_model.layers[i]['W'].copy()
    transfer_progressive.layers[i]['b'] = pretrain_model.layers[i]['b'].copy()

prog_losses = []
prog_accs = []

for phase in range(n_phases):
    freeze = (len(transfer_progressive.layers) - 1) - phase - 1
    freeze = max(freeze, 0)
    lr = 0.1 * (0.1 ** phase)  # decreasing LR
    
    for epoch in range(epochs_per_phase):
        loss = transfer_progressive.train_step(X_target, y_target, lr=lr, freeze_layers=freeze)
        prog_losses.append(loss)
    
    acc = np.mean(transfer_progressive.predict(X_target) == y_target)
    prog_accs.append(acc)
    print(f"Phase {phase+1}: unfreeze layers from index {freeze}, lr={lr:.4f}, acc={acc:.2%}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(prog_losses, color='#9333ea', linewidth=1.5)
for phase in range(1, n_phases):
    ax.axvline(x=phase * epochs_per_phase, color='gray', linestyle='--', alpha=0.5)
    ax.text(phase * epochs_per_phase + 5, max(prog_losses) * 0.9, f'Phase {phase+1}')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Progressive Unfreezing: Loss with Phase Boundaries')
plt.tight_layout()
plt.savefig('progressive_unfreezing.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 3: Progressive unfreezing")'''),
    md_cell(r'''## 5. 冻结层分析

可视化预训练模型各层特征的迁移性：浅层特征更通用，深层特征更任务相关。'''),
    code_cell(r'''# Analyze feature transferability at different layers
np.random.seed(42)

# Extract features at different layers
layer_features = {}
for layer_idx in range(len(pretrain_model.layers) - 1):
    feat = pretrain_model.extract_features(X_target, layer_idx=layer_idx)
    layer_features[layer_idx] = feat

# Train linear classifier on each layer's features
def train_linear(X, y, lr=0.1, epochs=200):
    """Simple linear classifier training."""
    n_classes = len(np.unique(y))
    W = np.random.randn(n_classes, X.shape[1]) * 0.01
    b = np.zeros(n_classes)
    for _ in range(epochs):
        scores = X @ W.T + b
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_s = np.exp(shifted)
        probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        dscores = probs.copy()
        dscores[np.arange(len(y)), y] -= 1
        dscores /= len(y)
        W -= lr * dscores.T @ X
        b -= lr * dscores.sum(axis=0)
    return np.mean(np.argmax(X @ W.T + b, axis=1) == y)

print("Layer transferability analysis:")
layer_accs = []
for layer_idx, feat in layer_features.items():
    acc = train_linear(feat, y_target, lr=0.1, epochs=300)
    layer_accs.append(acc)
    print(f"  Layer {layer_idx} features: dim={feat.shape[1]}, linear_acc={acc:.2%}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(len(layer_accs)), layer_accs, color=['#16a34a', '#2563eb', '#ea580c'])
ax.set_xlabel('Layer Index (0=shallow, 2=deep)')
ax.set_ylabel('Linear Probe Accuracy')
ax.set_title('Feature Transferability by Layer Depth')
plt.tight_layout()
plt.savefig('layer_transferability.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 4: Layer transferability")'''),
    md_cell(r'''## 6. 实验观察

### 关键发现

1. **特征提取 > 随机初始化**：在 30 个小样本上，预训练特征远胜随机初始化
2. **微调通常优于特征提取**：当数据足够时，微调可以适应目标任务
3. **渐进式解冻**：稳定训练，避免灾难性遗忘
4. **浅层更可迁移**：浅层特征通用，深层特征领域相关

### 灾难性遗忘

微调时如果学习率过大，预训练知识会被快速覆盖——这就是**灾难性遗忘**。解决方案：
- 小学习率
- 渐进式解冻
- 回放策略（Replay）'''),
    md_cell(r'''## 作业

### 作业 1：实现 Discriminative Learning Rate

不同层使用不同学习率：浅层用小 LR，深层用大 LR。'''),
    code_cell(r'''# Homework 1: Discriminative learning rates
def train_with_discriminative_lr(model, X, y, lrs, epochs=200):
    """Train with different learning rates per layer.
    
    Args:
        lrs: list of learning rates, one per layer (shallow to deep)
    """
    # TODO: implement discriminative learning rate training
    # Key: each layer i uses lrs[i] instead of uniform lr
    pass

# Test
# model = PretrainedModel(20, [32, 16], 3)
# lrs = [0.001, 0.01, 0.1]  # shallow -> deep
# losses = train_with_discriminative_lr(model, X_target, y_target, lrs, epochs=200)
# acc = np.mean(model.predict(X_target) == y_target)
# assert acc > 0.5, "Discriminative LR should achieve >50% accuracy"
print("Homework 1 template ready. Implement discriminative learning rates.")'''),
    md_cell(r'''### 作业 2：实现 Domain Adaptation

模拟域偏移（Domain Shift）：源域和目标域的分布不同。'''),
    code_cell(r'''# Homework 2: Domain adaptation
def domain_adaptation(source_model, X_source, y_source, X_target, y_target, 
                      n_adapt_epochs=100):
    """Adapt pretrained model to target domain.
    
    Strategy: Fine-tune with small LR on target domain.
    """
    # TODO: implement domain adaptation
    # 1. Start with source model
    # 2. Fine-tune on target with small LR
    # 3. Use early stopping to prevent overfitting
    pass

# Test
np.random.seed(42)
# Create domain-shifted data
X_target_shifted = X_target + np.random.randn(*X_target.shape) * 0.5
# acc = domain_adaptation(pretrain_model, X_pretrain, y_pretrain, 
#                          X_target_shifted, y_target)
# assert acc > 0.4, "Domain adaptation should help with distribution shift"
print("Homework 2 template ready. Implement domain adaptation.")'''),
    md_cell(r'''### 作业 3：实现 Layer Freezing Schedule

实现一个冻结计划：训练过程中逐步解冻更多层。'''),
    code_cell(r'''# Homework 3: Layer freezing schedule
def freeze_schedule(model, X, y, total_epochs=300, unfreeze_every=100):
    """Gradually unfreeze layers during training.
    
    Start with all layers frozen except classifier.
    Every `unfreeze_every` epochs, unfreeze one more layer.
    """
    # TODO: implement freeze schedule
    pass

# Test
# model = PretrainedModel(20, [32, 16], 3)
# Copy pretrained weights
# losses = freeze_schedule(model, X_target, y_target, total_epochs=300, unfreeze_every=100)
# acc = np.mean(model.predict(X_target) == y_target)
# assert acc > 0.5, "Freeze schedule should achieve >50% accuracy"
print("Homework 3 template ready. Implement layer freezing schedule.")'''),
    md_cell(r'''## 小结

| 策略 | 冻结层 | 训练层 | 学习率 | 适用场景 |
|------|--------|--------|--------|----------|
| **特征提取** | 所有预训练层 | 仅分类头 | 大 | 极小数据集 |
| **微调** | 无 | 全部 | 统一 | 中等数据集 |
| **渐进式解冻** | 逐层减少 | 从深到浅 | 递减 | 平衡稳定与性能 |
| **判别式 LR** | 无 | 全部 | 逐层递增 | 精细微调 |

**关键洞察**：迁移学习是深度学习最实用的技术——大多数实际应用不需要从头训练。预训练模型的浅层特征具有通用性，迁移到新任务只需少量数据即可获得良好效果。'''),
    md_cell(r'''## 参考文献

1. [[Yosinski et al., 2014]](https://arxiv.org/abs/1411.1792) Yosinski et al. *How transferable are features in deep neural networks?* NIPS 2014.
2. [[Howard & Ruder, 2018]](https://arxiv.org/abs/1801.06146) Howard & Ruder. *Universal Language Model Fine-tuning for Text Classification*. ACL 2018.
3. [[Donahue et al., 2014]](https://arxiv.org/abs/1310.1531) Donahue et al. *DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition*. ICML 2014.
4. [[Razavian et al., 2014]](https://arxiv.org/abs/1403.6382) Razavian et al. *CNN Features off-the-shelf*. CVPR Workshop 2014.
5. [[Kirkpatrick et al., 2017]](https://arxiv.org/abs/1612.00796) Kirkpatrick et al. *Overcoming Catastrophic Forgetting in Neural Networks*. PNAS 2017.

---

> 本节内容参考 [Stanford CS231n](https://cs231n.stanford.edu/) | [Transfer Learning](https://cs231n.github.io/transfer-learning/)'''),
    md_cell(r'''---

## 参考代码实现

以下 GitHub 仓库提供了本节内容的完整代码实现，建议结合学习：

- **[huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)** (37139 stars): PyTorch 图像模型库（含大量预训练模型和迁移学习工具）
  - 仓库地址: https://github.com/huggingface/pytorch-image-models

- **[seloufian/Deep-Learning-Computer-Vision](https://github.com/seloufian/Deep-Learning-Computer-Vision)** (135 stars): 深度学习计算机视觉实践（含迁移学习教程）
  - 仓库地址: https://github.com/seloufian/Deep-Learning-Computer-Vision

> 标注说明: 以上仓库按热度排序，优先推荐 stars 最多的实现.'''),
])
save_notebook(nb5, os.path.join(BASE, "part3-frontiers", "lecture-11-transfer-learning", "practice.ipynb"))

# ============================================================
# Notebook 6: Image Segmentation
# ============================================================
print("Generating Notebook 6: Image Segmentation...")
nb6 = make_notebook([
    md_cell(r'''# 图像分割

> **斯坦福 CS231n** | Image Segmentation: FCN and U-Net

## 本章导读

图像分割是像素级分类任务——为每个像素分配类别标签。本节从 FCN 到 U-Net，理解编码器-解码器架构和跳跃连接的原理。

**学习目标：**
- 区分语义分割与实例分割
- 理解 FCN 全卷积网络架构
- 实现 U-Net 编码器-解码器 + 跳跃连接
- 理解转置卷积（反卷积）原理
- 实现 mIoU 评估指标

**参考来源：** [CS231n Lecture 11](https://cs231n.stanford.edu/) | [FCN Paper](https://arxiv.org/abs/1411.4038) | [U-Net Paper](https://arxiv.org/abs/1505.04597)'''),
    md_cell(r'''## 1. 直觉理解：像素级分类

### 分类 vs 分割

| 任务 | 输出 | 粒度 |
|------|------|------|
| **图像分类** | 1 个标签 | 图像级 |
| **语义分割** | H×W 标签 | 像素级 |
| **实例分割** | H×W 实例 ID | 像素级 + 实例区分 |

### 核心挑战

1. **输出分辨率**：输出必须与输入同尺寸
2. **多尺度**：不同大小的物体需要不同感受野
3. **细节保留**：边界需要精确

### FCN 的突破

[[Long et al., 2015]](https://arxiv.org/abs/1411.4038)

FCN (Fully Convolutional Network) 将分类网络的 FC 层替换为卷积层，实现端到端的像素级预测。关键创新：
- 全卷积（无全连接）
- 转置卷积上采样
- 跳跃连接融合多尺度特征'''),
    md_cell(r'''## 2. 手算验证：转置卷积

### 转置卷积（反卷积）

转置卷积用于上采样——将低分辨率特征图恢复到高分辨率。

### 手算：2×2 → 3×3 转置卷积

输入 $x = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$，核 $K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$，stride=1, padding=0。

输出尺寸：$H_{out} = (H_{in}-1) \times S - 2P + K = (2-1) \times 1 - 0 + 2 = 3$

转置卷积的计算：将每个输入值乘以核，放到输出对应位置，重叠区域相加。

- $x[0,0]=1 \times K$ 放在 $out[0:2, 0:2]$：$\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$
- $x[0,1]=2 \times K$ 放在 $out[0:2, 1:3]$：$\begin{bmatrix} 0 & 2 \\ 0 & 0 \end{bmatrix}$（在列 1,2）
- $x[1,0]=3 \times K$ 放在 $out[1:3, 0:2]$：$\begin{bmatrix} 0 & 0 \\ 3 & 0 \end{bmatrix}$（在行 1,2）
- $x[1,1]=4 \times K$ 放在 $out[1:3, 1:3]$：$\begin{bmatrix} 0 & 0 \\ 0 & 4 \end{bmatrix}$（在行 1,2, 列 1,2）

叠加后：
$$out = \begin{bmatrix} 1 & 2 & 0 \\ 3 & 1+4 & 2 \\ 0 & 3 & 4 \end{bmatrix} = \begin{bmatrix} 1 & 2 & 0 \\ 3 & 5 & 2 \\ 0 & 3 & 4 \end{bmatrix}$$'''),
    code_cell(r'''# Verify transposed convolution
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
np.random.seed(42)

def transposed_conv2d(x, kernel, stride=1, padding=0):
    """Simple transposed convolution (upsampling).
    
    Args:
        x: (H, W) input
        kernel: (K, K) kernel
        stride: upsampling stride
        padding: padding
    """
    H, W = x.shape
    K_h, K_w = kernel.shape
    H_out = (H - 1) * stride - 2 * padding + K_h
    W_out = (W - 1) * stride - 2 * padding + K_w
    out = np.zeros((H_out, W_out))
    
    for i in range(H):
        for j in range(W):
            # Place kernel * x[i,j] at position
            for ki in range(K_h):
                for kj in range(K_w):
                    oi = i * stride - padding + ki
                    oj = j * stride - padding + kj
                    if 0 <= oi < H_out and 0 <= oj < W_out:
                        out[oi, oj] += x[i, j] * kernel[ki, kj]
    return out

# Verify manual calculation
x = np.array([[1, 2], [3, 4]], dtype=float)
K = np.array([[1, 0], [0, 1]], dtype=float)

result = transposed_conv2d(x, K)
print("Input:\n", x)
print("\nKernel:\n", K)
print("\nTransposed conv result:\n", result)
print("\nExpected:\n[[1 2 0]\n [3 5 2]\n [0 3 4]]")
assert np.allclose(result, [[1, 2, 0], [3, 5, 2], [0, 3, 4]]), "Mismatch!"
print("Manual calculation verified!")'''),
    md_cell(r'''## 3. 代码实现：U-Net 架构

[[Ronneberger et al., 2015]](https://arxiv.org/abs/1505.04597)

U-Net 架构：
- **编码器（下采样）**：逐步提取高层特征，降低分辨率
- **解码器（上采样）**：逐步恢复分辨率
- **跳跃连接**：将编码器特征直接传递给解码器，保留细节'''),
    code_cell(r'''class MiniUNet:
    """Simplified U-Net for segmentation.
    
    Architecture:
    Encoder: conv -> pool -> conv -> pool
    Bottleneck: conv
    Decoder: upconv -> concat -> conv -> upconv -> concat -> conv
    """
    def __init__(self, in_channels=1, n_classes=2):
        # Encoder
        self.enc1_W = np.random.randn(4, in_channels, 3, 3) * 0.1
        self.enc1_b = np.zeros(4)
        self.enc2_W = np.random.randn(8, 4, 3, 3) * 0.1
        self.enc2_b = np.zeros(8)
        # Bottleneck
        self.bot_W = np.random.randn(8, 8, 3, 3) * 0.1
        self.bot_b = np.zeros(8)
        # Decoder (upsample + concat + conv)
        self.dec1_W = np.random.randn(4, 16, 3, 3) * 0.1  # 8 (up) + 8 (skip) = 16
        self.dec1_b = np.zeros(4)
        self.dec2_W = np.random.randn(n_classes, 4 + in_channels, 3, 3) * 0.1
        self.dec2_b = np.zeros(n_classes)
    
    def conv2d(self, x, W, b):
        """Simple 2D convolution, no padding."""
        if x.ndim == 2:
            x = x[np.newaxis, ...]
        C_in, H, Wd = x.shape
        C_out, _, K, _ = W.shape
        H_out, W_out = H - K + 1, Wd - K + 1
        out = np.zeros((C_out, H_out, W_out))
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    patch = x[:, i:i+K, j:j+K]
                    out[co, i, j] = np.sum(patch * W[co]) + b[co]
        return np.maximum(0, out)
    
    def maxpool2d(self, x, size=2):
        """Max pooling."""
        if x.ndim == 2:
            x = x[np.newaxis, ...]
        C, H, W = x.shape
        H_out, W_out = H // size, W // size
        out = np.zeros((C, H_out, W_out))
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    out[c, i, j] = np.max(x[c, i*size:(i+1)*size, j*size:(j+1)*size])
        return out
    
    def upsample(self, x, scale=2):
        """Nearest neighbor upsampling."""
        if x.ndim == 2:
            x = x[np.newaxis, ...]
        C, H, W = x.shape
        out = np.zeros((C, H*scale, W*scale))
        for c in range(C):
            for i in range(H):
                for j in range(W):
                    out[c, i*scale:(i+1)*scale, j*scale:(j+1)*scale] = x[c, i, j]
        return out
    
    def forward(self, x):
        """Forward pass. x: (H, W) or (1, H, W)."""
        if x.ndim == 2:
            x = x[np.newaxis, ...]
        
        # Encoder
        e1 = self.conv2d(x, self.enc1_W, self.enc1_b)      # (4, H-2, W-2)
        p1 = self.maxpool2d(e1)                              # (4, (H-2)/2, ...)
        e2 = self.conv2d(p1, self.enc2_W, self.enc2_b)      # (8, ...)
        p2 = self.maxpool2d(e2)
        
        # Bottleneck
        b = self.conv2d(p2, self.bot_W, self.bot_b)         # (8, ...)
        
        # Decoder with skip connections
        u1 = self.upsample(b, 2)
        # Concat with e2 (crop to match)
        skip2 = e2[:, :u1.shape[1], :u1.shape[2]] if e2.shape[1] >= u1.shape[1] else e2
        if u1.shape[1] <= e2.shape[1] and u1.shape[2] <= e2.shape[2]:
            cat1 = np.concatenate([u1, e2[:, :u1.shape[1], :u1.shape[2]]], axis=0)
        else:
            cat1 = np.concatenate([u1, e2], axis=0)
        d1 = self.conv2d(cat1, self.dec1_W, self.dec1_b)
        
        u2 = self.upsample(d1, 2)
        # Concat with e1
        if u2.shape[1] <= e1.shape[1] and u2.shape[2] <= e1.shape[2]:
            cat2 = np.concatenate([u2, e1[:, :u2.shape[1], :u2.shape[2]]], axis=0)
        else:
            cat2 = np.concatenate([u2, e1], axis=0)
        d2 = self.conv2d(cat2, self.dec2_W, self.dec2_b)
        
        return d2, (e1, e2, b, d1, d2)

print("MiniUNet class defined")'''),
    md_cell(r'''## 4. IoU 评估指标

### Intersection over Union (IoU)

$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|} = \frac{TP}{TP + FP + FN}$$

### Mean IoU (mIoU)

对所有类别取平均：

$$\text{mIoU} = \frac{1}{C} \sum_{c=1}^{C} \text{IoU}_c$$'''),
    code_cell(r'''# IoU implementation and visualization
def compute_iou(pred, target, n_classes=2):
    """Compute mean IoU.
    
    Args:
        pred: (H, W) predicted labels
        target: (H, W) ground truth labels
        n_classes: number of classes
    """
    ious = []
    for c in range(n_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = np.sum(pred_c & target_c)
        union = np.sum(pred_c | target_c)
        iou = intersection / max(union, 1)
        ious.append(iou)
    return np.mean(ious), ious

# Create synthetic segmentation data
np.random.seed(42)
H, W = 32, 32
# Ground truth: two regions
gt = np.zeros((H, W), dtype=int)
gt[:H//2, :] = 0  # background (top)
gt[H//2:, :] = 1  # foreground (bottom)

# Simulate prediction (with some errors)
pred = gt.copy()
# Add random errors
errors = np.random.rand(H, W) < 0.15
pred[errors] = 1 - pred[errors]

miou, per_class_iou = compute_iou(pred, gt, n_classes=2)
print(f"Mean IoU: {miou:.4f}")
print(f"Class 0 IoU: {per_class_iou[0]:.4f}")
print(f"Class 1 IoU: {per_class_iou[1]:.4f}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(gt, cmap='RdBu')
axes[0].set_title('Ground Truth')
axes[0].axis('off')
axes[1].imshow(pred, cmap='RdBu')
axes[1].set_title(f'Prediction (mIoU={miou:.3f})')
axes[1].axis('off')
# Error map
error_map = (pred != gt).astype(int)
axes[2].imshow(error_map, cmap='Reds')
axes[2].set_title(f'Errors ({np.sum(error_map)} pixels)')
axes[2].axis('off')
plt.suptitle('Segmentation: GT vs Prediction vs Errors', fontsize=14)
plt.tight_layout()
plt.savefig('segmentation_iou.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 1: Segmentation IoU")'''),
    code_cell(r'''# FCN forward pass on small image
np.random.seed(42)
# Create synthetic image with a circle
img = np.zeros((16, 16))
for i in range(16):
    for j in range(16):
        if (i - 8)**2 + (j - 8)**2 < 16:
            img[i, j] = 1.0

# Add noise
img += np.random.randn(16, 16) * 0.1

# Run MiniUNet
unet = MiniUNet(in_channels=1, n_classes=2)
output, _ = unet.forward(img)
pred_mask = np.argmax(output[0], axis=0)

# Crop to original size for visualization
pred_size = pred_mask.shape[0]
gt_mask = (img > 0.5).astype(int)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Input Image')
axes[0].axis('off')
axes[1].imshow(gt_mask, cmap='RdBu')
axes[1].set_title('Ground Truth Mask')
axes[1].axis('off')
axes[2].imshow(pred_mask, cmap='RdBu')
axes[2].set_title(f'U-Net Output ({pred_size}x{pred_size})')
axes[2].axis('off')
# Feature visualization
axes[3].imshow(output[0, 0], cmap='hot')
axes[3].set_title('Feature Map (ch 0)')
axes[3].axis('off')
plt.suptitle('Mini U-Net Segmentation Demo', fontsize=14)
plt.tight_layout()
plt.savefig('unet_segmentation.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 2: U-Net segmentation")'''),
    md_cell(r'''## 5. 跳跃连接的作用

### 没有跳跃连接 vs 有跳跃连接

| 特性 | 无跳跃连接 | 有跳跃连接 |
|------|-----------|-----------|
| **边界精度** | 模糊 | 精确 |
| **细节保留** | 丢失 | 保留 |
| **训练难度** | 较难 | 较易 |
| **梯度流** | 长路径 | 短路径（梯度快捷方式） |

跳跃连接不仅融合了多尺度特征，还提供了梯度传播的捷径，缓解梯度消失。'''),
    code_cell(r'''# Visualize skip connection effect (simulated)
np.random.seed(42)

# Simulate output with and without skip connections
H, W = 32, 32
# Ground truth: sharp boundary
gt = np.zeros((H, W))
gt[:, W//2:] = 1.0

# With skip: sharp boundary (less upsampling blur)
with_skip = gt.copy()
with_skip += np.random.randn(H, W) * 0.05
with_skip = (with_skip > 0.5).astype(float)

# Without skip: blurry boundary (upsampling artifacts)
from scipy.ndimage import gaussian_filter
without_skip = gaussian_filter(gt.astype(float), sigma=3.0)
without_skip = (without_skip > 0.5).astype(float)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(gt, cmap='RdBu')
axes[0].set_title('Ground Truth (sharp boundary)')
axes[0].axis('off')
axes[1].imshow(with_skip, cmap='RdBu')
axes[1].set_title('With Skip Connections (sharp)')
axes[1].axis('off')
axes[2].imshow(without_skip, cmap='RdBu')
axes[2].set_title('Without Skip Connections (blurry)')
axes[2].axis('off')
plt.suptitle('Skip Connection Effect on Boundary Precision', fontsize=14)
plt.tight_layout()
plt.savefig('skip_connection_effect.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 3: Skip connection effect")'''),
    md_cell(r'''## 6. 转置卷积可视化'''),
    code_cell(r'''# Visualize transposed convolution upsampling
np.random.seed(42)

# Create a small feature map
feat = np.random.randn(4, 4) * 2
# Use identity-like kernel for visualization
kernel = np.array([[1, 0], [0, 1]], dtype=float)

# Different stride values
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(feat, cmap='viridis')
axes[0].set_title(f'Input ({feat.shape[0]}x{feat.shape[1]})')
axes[0].axis('off')

for idx, stride in enumerate([1, 2, 3]):
    result = transposed_conv2d(feat, kernel, stride=stride)
    axes[idx+1].imshow(result, cmap='viridis')
    axes[idx+1].set_title(f'stride={stride}\nOutput ({result.shape[0]}x{result.shape[1]})')
    axes[idx+1].axis('off')

plt.suptitle('Transposed Convolution Upsampling', fontsize=14)
plt.tight_layout()
plt.savefig('transposed_conv_demo.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 4: Transposed convolution")'''),
    md_cell(r'''## 作业

### 作业 1：实现像素级交叉熵损失

分割任务使用像素级交叉熵：对每个像素计算 softmax 损失。'''),
    code_cell(r'''# Homework 1: Pixel-wise cross-entropy loss
def segmentation_loss(scores, target, n_classes=2):
    """Pixel-wise cross-entropy loss.
    
    Args:
        scores: (C, H, W) predicted scores per class
        target: (H, W) ground truth labels
        n_classes: number of classes
    
    Returns:
        loss, d_scores (gradients)
    """
    # TODO: implement pixel-wise cross-entropy
    # For each pixel (i,j):
    #   1. Compute softmax over channels
    #   2. Compute cross-entropy with target[i,j]
    pass

# Test
# scores = np.random.randn(2, 8, 8)
# target = np.random.randint(0, 2, (8, 8))
# loss, grads = segmentation_loss(scores, target, n_classes=2)
# assert loss > 0, "Loss should be positive"
# assert grads.shape == scores.shape, "Gradient shape should match"
print("Homework 1 template ready. Implement pixel-wise cross-entropy loss.")'''),
    md_cell(r'''### 作业 2：实现 Dice Loss

Dice Loss 是分割任务中替代交叉熵的常用损失，直接优化 IoU。

[[Milletari et al., 2016]](https://arxiv.org/abs/1606.04797)'''),
    code_cell(r'''# Homework 2: Dice loss
def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for binary segmentation.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice
    """
    # TODO: implement Dice loss
    pass

# Test
# pred = np.random.rand(8, 8)
# target = (np.random.rand(8, 8) > 0.5).astype(float)
# loss = dice_loss(pred, target)
# assert 0 <= loss <= 1, "Dice loss should be in [0, 1]"
print("Homework 2 template ready. Implement Dice loss.")'''),
    md_cell(r'''### 作业 3：实现 U-Net 训练循环

实现完整的 U-Net 训练循环，包括前向传播、损失计算和反向传播。'''),
    code_cell(r'''# Homework 3: U-Net training loop
def train_unet(unet, X_train, y_train, lr=0.01, epochs=100):
    """Train MiniUNet on segmentation data.
    
    For each epoch:
    1. Forward pass
    2. Compute pixel-wise loss
    3. Backward pass (compute gradients)
    4. Update weights
    """
    # TODO: implement training loop
    pass

# Test
# unet = MiniUNet(in_channels=1, n_classes=2)
# losses = train_unet(unet, X_train, y_train, lr=0.01, epochs=50)
# assert len(losses) == 50
# assert losses[-1] < losses[0], "Loss should decrease"
print("Homework 3 template ready. Implement U-Net training loop.")'''),
    md_cell(r'''## 小结

| 概念 | 要点 |
|------|------|
| **语义分割** | 每像素一个类别标签 |
| **实例分割** | 每像素一个实例 ID |
| **FCN** | 全卷积网络，分类网 FC→Conv |
| **U-Net** | 编码器-解码器 + 跳跃连接 |
| **转置卷积** | 上采样，低分辨率→高分辨率 |
| **IoU/mIoU** | 交并比，分割评估标准 |
| **Dice Loss** | 直接优化 IoU 的损失函数 |
| **跳跃连接** | 保留细节，提供梯度捷径 |

**关键洞察**：分割的核心是「下采样提取语义 + 上采样恢复分辨率 + 跳跃连接保留细节」。U-Net 的对称结构和跳跃连接使其在医学图像等需要精确边界的领域至今仍被广泛使用。'''),
    md_cell(r'''## 参考文献

1. [[Long et al., 2015]](https://arxiv.org/abs/1411.4038) Long et al. *Fully Convolutional Networks for Semantic Segmentation*. CVPR 2015.
2. [[Ronneberger et al., 2015]](https://arxiv.org/abs/1505.04597) Ronneberger et al. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015.
3. [[Milletari et al., 2016]](https://arxiv.org/abs/1606.04797) Milletari et al. *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation*. 3DV 2016.
4. [[Chen et al., 2017]](https://arxiv.org/abs/1706.05587) Chen et al. *DeepLab: Semantic Image Segmentation with Deep Convolutional Nets*. IEEE TPAMI 2017.
5. [[He et al., 2017]](https://arxiv.org/abs/1703.06870) He et al. *Mask R-CNN*. ICCV 2017.

---

> 本节内容参考 [Stanford CS231n](https://cs231n.stanford.edu/) | [FCN](https://arxiv.org/abs/1411.4038) | [U-Net](https://arxiv.org/abs/1505.04597)'''),
    md_cell(r'''---

## 参考代码实现

以下 GitHub 仓库提供了本节内容的完整代码实现，建议结合学习：

- **[milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)** (11649 stars): U-Net PyTorch 完整实现（含训练和推理代码）
  - 仓库地址: https://github.com/milesial/Pytorch-UNet

- **[wolny/pytorch-3dunet](https://github.com/wolny/pytorch-3dunet)** (2419 stars): 3D U-Net PyTorch 实现（医学图像分割）
  - 仓库地址: https://github.com/wolny/pytorch-3dunet

- **[ellisdg/3DUnetCNN](https://github.com/ellisdg/3DUnetCNN)** (2228 stars): 3D U-Net CNN 实现（含深度学习训练框架）
  - 仓库地址: https://github.com/ellisdg/3DUnetCNN

> 标注说明: 以上仓库按热度排序，优先推荐 stars 最多的实现.'''),
])
save_notebook(nb6, os.path.join(BASE, "part3-frontiers", "lecture-14-segmentation", "practice.ipynb"))

# ============================================================
# Notebook 7: CLIP and Multimodal Learning
# ============================================================
print("Generating Notebook 7: CLIP and Multimodal Learning...")
nb7 = make_notebook([
    md_cell(r'''# CLIP 与多模态学习

> **斯坦福 CS231n** | CLIP: Contrastive Language-Image Pre-training

## 本章导读

CLIP 通过对比学习将图像和文本映射到统一的嵌入空间，实现零样本分类。本节从对比损失出发，实现简化版 CLIP 并演示零样本推理。

**学习目标：**
- 理解对比学习（Contrastive Learning）的原理
- 实现 InfoNCE 对比损失
- 理解 CLIP 双编码器架构
- 模拟零样本分类
- 可视化嵌入空间

**参考来源：** [CLIP Paper](https://arxiv.org/abs/2103.00020) | [OpenAI CLIP](https://github.com/openai/CLIP) | [CS231n Lecture](https://cs231n.stanford.edu/)'''),
    md_cell(r'''## 1. 直觉理解：多模态对比学习

### CLIP 的核心思想

传统模型只处理图像**或**文本。CLIP 同时处理两者，通过**对比学习**让匹配的图文对在嵌入空间中靠近，不匹配的远离。

### 训练目标

给定 $N$ 个图文对 $\{(I_i, T_i)\}$：
- 正样本：$(I_i, T_i)$ —— 匹配的图文对
- 负样本：$(I_i, T_j), i \neq j$ —— 不匹配的图文对

目标：最大化正样本的相似度，最小化负样本的相似度。

### 为什么这很强大？

1. **零样本分类**：不需要训练分类头，只需比较图像与文本提示的相似度
2. **开放词汇**：不限于预定义类别，可以识别任意概念
3. **多模态理解**：图像和文本在统一空间中表示

[[Radford et al., 2021]](https://arxiv.org/abs/2103.00020)'''),
    md_cell(r'''### CLIP 架构

```
图像 → 图像编码器 (ViT/CNN) → 图像嵌入 I_i
                                    ↕ 对比损失
文本 → 文本编码器 (Transformer) → 文本嵌入 T_i
```

- **图像编码器**：ViT 或 ResNet，输出 $D$ 维向量
- **文本编码器**：Transformer，输出 $D$ 维向量
- **对比损失**：InfoNCE

### InfoNCE 损失

$$L = -\frac{1}{N} \sum_{i=1}^{N} \left[\log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j) / \tau)}\right]$$

其中 $\text{sim}(I, T) = I \cdot T / (\|I\| \|T\|)$ 是余弦相似度，$\tau$ 是温度参数。

对称版本：同时计算图像到文本和文本到图像的损失。'''),
    md_cell(r'''## 2. 手算验证：对比损失

### 小规模手算

设 $N=2$ 个图文对，嵌入维度 $D=2$，温度 $\tau=1$。

**图像嵌入：**
$$I_1 = [1, 0], \quad I_2 = [0, 1]$$

**文本嵌入（已归一化）：**
$$T_1 = [0.8, 0.6], \quad T_2 = [0.6, 0.8]$$

**余弦相似度矩阵：**
$$S = \begin{bmatrix} I_1 \cdot T_1 & I_1 \cdot T_2 \\ I_2 \cdot T_1 & I_2 \cdot T_2 \end{bmatrix} = \begin{bmatrix} 0.8 & 0.6 \\ 0.6 & 0.8 \end{bmatrix}$$

**图像到文本的损失**（对每行做 softmax）：

行 1（$I_1$）：
- $\text{softmax}([0.8, 0.6]) = [e^{0.8}/(e^{0.8}+e^{0.6}), e^{0.6}/(...)] = [0.55, 0.45]$
- $L_1 = -\log(0.55) = 0.598$

行 2（$I_2$）：
- $\text{softmax}([0.6, 0.8]) = [0.45, 0.55]$
- $L_2 = -\log(0.55) = 0.598$

**图像到文本损失**：$L_{I \to T} = (0.598 + 0.598) / 2 = 0.598$

对称地计算文本到图像损失（对列做 softmax），结果相同。

**总损失**：$L = (L_{I \to T} + L_{T \to I}) / 2 = 0.598$'''),
    code_cell(r'''# Verify contrastive loss calculation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
np.random.seed(42)

# Image embeddings (normalized)
I = np.array([[1, 0], [0, 1]], dtype=float)
# Text embeddings (normalized)
T = np.array([[0.8, 0.6], [0.6, 0.8]], dtype=float)

# Cosine similarity matrix
S = I @ T.T  # already normalized
print("Similarity matrix S:")
print(S)

# InfoNCE loss
tau = 1.0
def info_nce_loss(S, tau=1.0):
    """Compute symmetric InfoNCE loss."""
    N = S.shape[0]
    # Row-wise softmax (image to text)
    logits = S / tau
    log_prob_i2t = -np.log(np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True))
    loss_i2t = np.mean(np.diag(log_prob_i2t))
    
    # Column-wise softmax (text to image)
    log_prob_t2i = -np.log(np.exp(logits.T) / np.sum(np.exp(logits.T), axis=1, keepdims=True))
    loss_t2i = np.mean(np.diag(log_prob_t2i))
    
    return (loss_i2t + loss_t2i) / 2

loss = info_nce_loss(S, tau=1.0)
print(f"\nInfoNCE loss: {loss:.4f}")
print(f"Expected: ~0.598")'''),
    md_cell(r'''## 3. 代码实现：Mini CLIP'''),
    code_cell(r'''class MiniCLIP:
    """Simplified CLIP model.
    
    Image encoder: 2-layer MLP
    Text encoder: 2-layer MLP
    Contrastive loss: InfoNCE
    """
    def __init__(self, img_dim, text_dim, embed_dim=64, tau=0.07):
        # Image encoder
        self.W_i1 = np.random.randn(32, img_dim) * np.sqrt(2.0 / img_dim)
        self.b_i1 = np.zeros(32)
        self.W_i2 = np.random.randn(embed_dim, 32) * np.sqrt(2.0 / 32)
        self.b_i2 = np.zeros(embed_dim)
        
        # Text encoder
        self.W_t1 = np.random.randn(32, text_dim) * np.sqrt(2.0 / text_dim)
        self.b_t1 = np.zeros(32)
        self.W_t2 = np.random.randn(embed_dim, 32) * np.sqrt(2.0 / 32)
        self.b_t2 = np.zeros(embed_dim)
        
        self.tau = tau
        self.embed_dim = embed_dim
    
    def encode_image(self, x):
        h = np.maximum(0, x @ self.W_i1.T + self.b_i1)
        emb = h @ self.W_i2.T + self.b_i2
        return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    
    def encode_text(self, x):
        h = np.maximum(0, x @ self.W_t1.T + self.b_t1)
        emb = h @ self.W_t2.T + self.b_t2
        return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    
    def forward(self, images, texts):
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(texts)
        S = img_emb @ txt_emb.T  # cosine similarity (normalized)
        return S
    
    def loss(self, images, texts):
        """Symmetric InfoNCE loss."""
        S = self.forward(images, texts)
        N = S.shape[0]
        logits = S / self.tau
        
        # Image to text
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs_i2t = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        loss_i2t = -np.mean(np.log(probs_i2t[np.arange(N), np.arange(N)] + 1e-10))
        
        # Text to image
        exp_logits_t = np.exp(logits.T - np.max(logits.T, axis=1, keepdims=True))
        probs_t2i = exp_logits_t / np.sum(exp_logits_t, axis=1, keepdims=True)
        loss_t2i = -np.mean(np.log(probs_t2i[np.arange(N), np.arange(N)] + 1e-10))
        
        loss = (loss_i2t + loss_t2i) / 2
        
        # Gradients (simplified)
        # dS_i2t: (N, N)
        dlogits = probs_i2t.copy()
        dlogits[np.arange(N), np.arange(N)] -= 1
        dlogits /= N
        
        dlogits_t = probs_t2i.copy()
        dlogits_t[np.arange(N), np.arange(N)] -= 1
        dlogits_t /= N
        
        dS = (dlogits + dlogits_t.T) / self.tau
        
        return loss, dS, S
    
    def step(self, images, texts, lr=0.01):
        """One gradient step."""
        loss, dS, S = self.loss(images, texts)
        
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(texts)
        
        # Backprop to embeddings
        d_img_emb = dS @ txt_emb  # (N, D)
        d_txt_emb = dS.T @ img_emb
        
        # Normalize gradient (account for L2 norm)
        norms_i = np.linalg.norm(img_emb, axis=1, keepdims=True) + 1e-8
        d_img_emb = d_img_emb / norms_i - img_emb * np.sum(d_img_emb * img_emb, axis=1, keepdims=True) / norms_i**2
        
        norms_t = np.linalg.norm(txt_emb, axis=1, keepdims=True) + 1e-8
        d_txt_emb = d_txt_emb / norms_t - txt_emb * np.sum(d_txt_emb * txt_emb, axis=1, keepdims=True) / norms_t**2
        
        # Backprop through image encoder
        dh_i = d_img_emb @ self.W_i2
        dz_i = dh_i * (img_emb @ self.W_i2.T + self.b_i2 > 0)  # approximate ReLU grad
        self.W_i2 -= lr * (dz_i.T @ np.maximum(0, images @ self.W_i1.T + self.b_i1))
        self.b_i2 -= lr * dz_i.sum(axis=0)
        
        # Backprop through text encoder
        dh_t = d_txt_emb @ self.W_t2
        dz_t = dh_t * (txt_emb @ self.W_t2.T + self.b_t2 > 0)
        self.W_t2 -= lr * (dz_t.T @ np.maximum(0, texts @ self.W_t1.T + self.b_t1))
        self.b_t2 -= lr * dz_t.sum(axis=0)
        
        return loss

print("MiniCLIP class defined")'''),
    md_cell(r'''## 4. 训练 Mini CLIP'''),
    code_cell(r'''# Generate synthetic image-text pairs
np.random.seed(42)
n_pairs = 50
img_dim = 16
text_dim = 16

# Create 3 categories with different distributions
n_per_cat = n_pairs // 3
images = np.zeros((n_pairs, img_dim))
texts = np.zeros((n_pairs, text_dim))
categories = []

for c in range(3):
    img_center = np.random.randn(img_dim)
    txt_center = np.random.randn(text_dim)
    for i in range(n_per_cat):
        idx = c * n_per_cat + i
        images[idx] = img_center + np.random.randn(img_dim) * 0.3
        texts[idx] = txt_center + np.random.randn(text_dim) * 0.3
        categories.append(c)

categories = np.array(categories)

# Train MiniCLIP
clip = MiniCLIP(img_dim, text_dim, embed_dim=32, tau=0.1)
losses = []
for epoch in range(300):
    loss = clip.step(images, texts, lr=0.01)
    losses.append(loss)
    if epoch % 50 == 0:
        S = clip.forward(images, texts)
        acc = np.mean(np.argmax(S, axis=1) == np.arange(n_pairs))
        print(f"Epoch {epoch}: loss={loss:.4f}, retrieval_acc={acc:.2%}")

print(f"Final loss: {losses[-1]:.4f}")'''),
    code_cell(r'''# Visualize training
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(losses, color='#ea580c', linewidth=1.5, alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('InfoNCE Loss')
ax.set_title('Mini CLIP Training Loss')
plt.tight_layout()
plt.savefig('clip_training.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 1: CLIP training loss")'''),
    md_cell(r'''## 5. 零样本分类

### 原理

CLIP 的零样本分类不需要训练分类器：
1. 为每个类别构造文本提示（如 "a photo of a {class}"）
2. 编码图像和所有文本提示
3. 选择与图像最相似的文本提示

### 模拟零样本分类'''),
    code_cell(r'''# Zero-shot classification demo
np.random.seed(42)
class_names = ['airplane', 'car', 'bird']
prompts = [f'a photo of a {name}' for name in class_names]

# Simulate text features for each prompt (would come from text encoder)
prompt_features = np.array([
    [0.8, 0.1, 0.1],  # airplane
    [0.1, 0.8, 0.1],  # car
    [0.1, 0.1, 0.8],  # bird
])

# Test images (simulated)
n_test = 15
test_images = np.zeros((n_test, 3))
test_labels = np.zeros(n_test, dtype=int)

for i in range(n_test):
    c = i % 3
    test_images[i] = prompt_features[c] + np.random.randn(3) * 0.15
    test_labels[i] = c

# Zero-shot: compute similarity and predict
similarities = test_images @ prompt_features.T
predictions = np.argmax(similarities, axis=1)
accuracy = np.mean(predictions == test_labels)

print(f"Zero-shot classification accuracy: {accuracy:.2%}")
for i in range(5):
    print(f"  Image {i}: true={class_names[test_labels[i]]}, "
          f"pred={class_names[predictions[i]]}, "
          f"sim={similarities[i, predictions[i]]:.3f}")'''),
    code_cell(r'''# Visualize zero-shot classification
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Similarity matrix
ax1 = axes[0]
im = ax1.imshow(similarities[:10], cmap='YlOrRd', aspect='auto')
ax1.set_xticks(range(3))
ax1.set_xticklabels(class_names)
ax1.set_xlabel('Text Prompts')
ax1.set_ylabel('Test Image')
ax1.set_title('Image-Text Similarity Matrix')
plt.colorbar(im, ax=ax1)

# Embedding space visualization
ax2 = axes[1]
img_embs = clip.encode_image(images)
txt_embs = clip.encode_text(texts)

# Simple PCA for visualization
from numpy.linalg import svd
def pca_2d(X):
    X_c = X - X.mean(0)
    _, _, Vt = svd(X_c, full_matrices=False)
    return X_c @ Vt[:2].T

img_2d = pca_2d(img_embs)
txt_2d = pca_2d(txt_embs)

colors = ['#ea580c', '#2563eb', '#16a34a']
for c in range(3):
    mask = categories == c
    ax2.scatter(img_2d[mask, 0], img_2d[mask, 1], c=colors[c], marker='o', 
              s=30, alpha=0.6, label=f'Img-{class_names[c]}')
    ax2.scatter(txt_2d[mask, 0], txt_2d[mask, 1], c=colors[c], marker='*', 
              s=100, alpha=0.8)

ax2.set_title('CLIP Embedding Space (PCA)')
ax2.legend()
plt.tight_layout()
plt.savefig('clip_zeroshot.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 2: Zero-shot classification and embedding space")'''),
    md_cell(r'''## 6. Prompt Engineering

### Prompt 的影响

CLIP 的分类效果受文本提示影响很大：
- "a photo of a cat" vs "a photo of a cat, a type of animal"
- 不同的提示会产生不同的嵌入

### Prompt Ensembling

使用多个提示取平均，提升鲁棒性：
```
"a photo of a {class}"
"a blurry photo of a {class}"
"a photo of a large {class}"
```

[[Radford et al., 2021]](https://arxiv.org/abs/2103.00020) 实验：prompt ensembling 在 ImageNet 上提升了约 3-5% 的准确率。'''),
    code_cell(r'''# Prompt engineering demo
np.random.seed(42)

# Simulate different prompts for the same concept
# "a photo of a cat" vs "a cat" vs "an image of a cat"
prompt_variations = {
    'cat': [
        [0.8, 0.1, 0.1],   # "a photo of a cat"
        [0.7, 0.2, 0.1],   # "a cat"
        [0.75, 0.15, 0.1], # "an image of a cat"
    ],
    'car': [
        [0.1, 0.8, 0.1],
        [0.15, 0.7, 0.15],
        [0.1, 0.75, 0.15],
    ],
    'bird': [
        [0.1, 0.1, 0.8],
        [0.15, 0.15, 0.7],
        [0.1, 0.1, 0.8],
    ],
}

# Ensemble: average prompts
ensemble_features = {}
for cls, prompts in prompt_variations.items():
    ensemble_features[cls] = np.mean(prompts, axis=0)

# Single prompt vs ensemble
single_prompt = np.array([p[0] for p in prompt_variations.values()])
ensemble_prompt = np.array(list(ensemble_features.values()))

# Test
test_image = np.array([0.75, 0.15, 0.1]) + np.random.randn(3) * 0.1

sim_single = test_image @ single_prompt.T
sim_ensemble = test_image @ ensemble_prompt.T

print(f"Single prompt sims: {sim_single}")
print(f"Ensemble prompt sims: {sim_ensemble}")
print(f"Single prediction: {class_names[np.argmax(sim_single)]}")
print(f"Ensemble prediction: {class_names[np.argmax(sim_ensemble)]}")'''),
    md_cell(r'''## 7. DINO 自监督学习

### DINO (Self-DIstillation with NO labels)

[[Caron et al., 2021]](https://arxiv.org/abs/2104.14294)

DINO 是一种自监督学习方法，不需要标签训练视觉编码器：
- 学生网络处理不同视角
- 教师网络（学生的 EMA）提供软标签
- 学生学习匹配教师的输出分布

### 与 CLIP 的区别

| 特性 | CLIP | DINO |
|------|------|------|
| **监督信号** | 图文对比 | 自蒸馏 |
| **需要文本** | 是 | 否 |
| **应用** | 零样本分类 | 特征提取/分割 |
| **语义对齐** | 图文对齐 | 纯视觉内部对齐 |'''),
    code_cell(r'''# Linear probing evaluation (simulated)
np.random.seed(42)

# Simulate DINO features (good for linear probing)
n_samples = 100
feature_dim = 32
n_classes = 3

# DINO features: well-clustered
dino_features = np.zeros((n_samples, feature_dim))
labels = np.zeros(n_samples, dtype=int)
for c in range(n_classes):
    center = np.random.randn(feature_dim) * 2
    mask = np.arange(c * (n_samples//3), (c+1) * (n_samples//3))
    dino_features[mask] = center + np.random.randn(len(mask), feature_dim) * 0.3
    labels[mask] = c

# Linear probe: train a linear classifier on frozen features
W = np.random.randn(n_classes, feature_dim) * 0.01
b = np.zeros(n_classes)

probe_losses = []
for epoch in range(200):
    scores = dino_features @ W.T + b
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(n_samples), labels]).mean()
    probe_losses.append(loss)
    
    dscores = probs.copy()
    dscores[np.arange(n_samples), labels] -= 1
    dscores /= n_samples
    W -= 0.1 * dscores.T @ dino_features
    b -= 0.1 * dscores.sum(axis=0)

acc = np.mean(np.argmax(dino_features @ W.T + b, axis=1) == labels)
print(f"Linear probe accuracy: {acc:.2%}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(probe_losses, color='#16a34a', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Cross-Entropy Loss')
ax.set_title('Linear Probing on Self-Supervised Features')
plt.tight_layout()
plt.savefig('linear_probing.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 3: Linear probing")'''),
    md_cell(r'''## 作业

### 作业 1：实现温度参数的影响

温度 $\tau$ 控制对比损失的锐度。实现不同 $\tau$ 值下的训练并比较。'''),
    code_cell(r'''# Homework 1: Temperature analysis
def train_with_temperature(tau, images, texts, epochs=200):
    """Train MiniCLIP with given temperature."""
    # TODO: train with specific temperature
    # Return final loss and retrieval accuracy
    pass

# Test
# for tau in [0.01, 0.1, 0.5, 1.0]:
#     loss, acc = train_with_temperature(tau, images, texts, epochs=200)
#     print(f"tau={tau}: loss={loss:.4f}, acc={acc:.2%}")
print("Homework 1 template ready. Implement temperature analysis.")'''),
    md_cell(r'''### 作业 2：实现 Prompt Ensembling

使用多个文本提示，取嵌入的平均值进行零样本分类。'''),
    code_cell(r'''# Homework 2: Prompt ensembling
def prompt_ensemble(clip_model, image, class_names, prompt_templates):
    """Zero-shot classification with prompt ensembling.
    
    Args:
        prompt_templates: list of templates like "a photo of a {}"
    """
    # TODO: implement prompt ensembling
    # 1. For each class and template, compute text embedding
    # 2. Average embeddings across templates
    # 3. Compute similarity with image
    # 4. Return predicted class
    pass

# Test
# templates = ["a photo of a {}", "a drawing of a {}", "a blurry photo of a {}"]
# pred = prompt_ensemble(clip, test_image, class_names, templates)
# assert pred in range(3), "Prediction should be valid class index"
print("Homework 2 template ready. Implement prompt ensembling.")'''),
    md_cell(r'''### 作业 3：实现 t-SNE 可视化

将 CLIP 嵌入用 t-SNE 降维并可视化。'''),
    code_cell(r'''# Homework 3: t-SNE visualization of CLIP embeddings
def tsne_visualization(img_emb, txt_emb, labels, class_names):
    """Visualize CLIP embeddings with t-SNE.
    
    Use PCA + gradient descent (simplified t-SNE) as in the visualization notebook.
    """
    # TODO: implement t-SNE visualization
    pass

# Test
# img_embs = clip.encode_image(images)
# txt_embs = clip.encode_text(texts)
# tsne_visualization(img_embs, txt_embs, categories, class_names)
print("Homework 3 template ready. Implement t-SNE visualization.")'''),
    md_cell(r'''## 小结

| 概念 | 要点 |
|------|------|
| **对比学习** | 拉近正样本、推远负样本 |
| **InfoNCE** | 对比损失函数，等价于交叉熵 |
| **CLIP 架构** | 双编码器（图像 + 文本） |
| **零样本分类** | 图像与文本提示比较，无需训练分类器 |
| **Prompt 工程** | 不同提示影响分类效果 |
| **温度 $\tau$** | 控制分布锐度，影响训练 |
| **线性探针** | 冻结特征 + 线性分类器评估 |
| **DINO** | 自蒸馏自监督，纯视觉 |

**关键洞察**：CLIP 通过对比学习将图像和文本统一到同一嵌入空间，实现了开放词汇的零样本分类。这种多模态对齐是视觉-语言大模型的基石。'''),
    md_cell(r'''## 参考文献

1. [[Radford et al., 2021]](https://arxiv.org/abs/2103.00020) Radford et al. *Learning Transferable Visual Models From Natural Language Supervision*. ICML 2021.
2. [[Caron et al., 2021]](https://arxiv.org/abs/2104.14294) Caron et al. *Emerging Properties in Self-Supervised Vision Transformers*. ICCV 2021.
3. [[Oord et al., 2018]](https://arxiv.org/abs/1807.03748) van den Oord et al. *Representation Learning with Contrastive Predictive Coding*. arXiv 2018.
4. [[Jia et al., 2021]](https://arxiv.org/abs/2103.00020) Jia et al. *Scaling Up Visual and Vision-Language Representation Learning with Noisy Text Supervision*. ICML 2021.
5. [[Ilharco et al., 2022]](https://arxiv.org/abs/2112.05182) Ilharco et al. *Massive open multilingual visual models*. 2022.

---

> 本节内容参考 [Stanford CS231n](https://cs231n.stanford.edu/) | [OpenAI CLIP](https://github.com/openai/CLIP)'''),
    md_cell(r'''---

## 参考代码实现

以下 GitHub 仓库提供了本节内容的完整代码实现，建议结合学习：

- **[openai/CLIP](https://github.com/openai/CLIP)** (34293 stars): OpenAI CLIP 官方实现（含预训练模型和零样本推理代码）
  - 仓库地址: https://github.com/openai/CLIP

- **[jina-ai/clip-as-service](https://github.com/jina-ai/clip-as-service)** (12835 stars): CLIP 即服务（高性能图文编码和检索）
  - 仓库地址: https://github.com/jina-ai/clip-as-service

> 标注说明: 以上仓库按热度排序，优先推荐 stars 最多的实现.'''),
])
save_notebook(nb7, os.path.join(BASE, "part3-frontiers", "lecture-15-clip", "practice.ipynb"))

# ============================================================
# Notebook 8: 3D Vision and NeRF
# ============================================================
print("Generating Notebook 8: 3D Vision and NeRF...")
nb8 = make_notebook([
    md_cell(r'''# 3D 视觉与 NeRF

> **斯坦福 CS231n** | 3D Vision and Neural Radiance Fields

## 本章导读

3D 视觉是从 2D 图像恢复 3D 结构的任务。NeRF 用神经网络表示 3D 场景，通过体渲染从任意视角生成新视图。本节从 3D 表示到 NeRF 完整路径。

**学习目标：**
- 理解 3D 表示方法（体素、点云、网格）
- 理解 NeRF 的核心思想
- 实现位置编码（Positional Encoding）
- 手算体渲染方程
- 实现简化 NeRF MLP

**参考来源：** [NeRF Paper](https://arxiv.org/abs/2003.08934) | [NeRF Studio](https://docs.nerf.studio/) | [CS231n](https://cs231n.stanford.edu/)'''),
    md_cell(r'''## 1. 直觉理解：3D 表示方法

### 三种 3D 表示

| 表示 | 数据结构 | 优点 | 缺点 |
|------|----------|------|------|
| **体素** | 3D 网格 | 规则、卷积友好 | 内存大、分辨率低 |
| **点云** | N 个 3D 点 | 紧凑、灵活 | 无拓扑、卷积难 |
| **网格** | 顶点+面片 | 精确表面 | 渲染复杂、拓扑约束 |

### NeRF 的第四种方式

NeRF (Neural Radiance Field) 用**神经网络隐式表示** 3D 场景：
- 输入：3D 位置 $(x, y, z)$ + 观察方向 $(\theta, \phi)$
- 输出：颜色 $(r, g, b)$ + 体密度 $\sigma$
- 整个场景编码在网络的权重中

[[Mildenhall et al., 2020]](https://arxiv.org/abs/2003.08934)'''),
    md_cell(r'''## 2. 体渲染方程

### NeRF 的渲染过程

给定相机光线 $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$（原点 $\mathbf{o}$，方向 $\mathbf{d}$）：

$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d}) \, dt$$

其中：
$$T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s)) \, ds\right)$$

- $T(t)$：从 $t_n$ 到 $t$ 的透射率（累积透明度）
- $\sigma(\mathbf{r}(t))$：体密度（不透明度）
- $\mathbf{c}$：颜色

### 离散化

$$\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i \alpha_i \mathbf{c}_i$$

其中：
$$T_i = \exp\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right), \quad \alpha_i = 1 - \exp(-\sigma_i \delta_i)$$

$\delta_i = t_{i+1} - t_i$ 是相邻采样点间距。'''),
    md_cell(r'''## 3. 手算验证：体渲染

### 简化手算

设光线采样 3 个点，间距 $\delta = 1$：

| 点 $i$ | 体密度 $\sigma_i$ | 颜色 $\mathbf{c}_i$ |
|---------|-------------------|---------------------|
| 1 | 0.5 | [1, 0, 0] |
| 2 | 1.0 | [0, 1, 0] |
| 3 | 0.3 | [0, 0, 1] |

**计算：**

1. $\alpha_1 = 1 - \exp(-\sigma_1 \delta_1) = 1 - e^{-0.5} = 1 - 0.607 = 0.393$
2. $\alpha_2 = 1 - \exp(-1.0 \times 1) = 1 - e^{-1} = 0.632$
3. $\alpha_3 = 1 - \exp(-0.3 \times 1) = 1 - e^{-0.3} = 0.259$

**透射率：**
1. $T_1 = 1$（从起点开始，无衰减）
2. $T_2 = \exp(-\sigma_1 \delta_1) = e^{-0.5} = 0.607$
3. $T_3 = \exp(-(\sigma_1 + \sigma_2) \times 1) = e^{-1.5} = 0.223$

**颜色：**
$$\hat{C} = T_1 \alpha_1 \mathbf{c}_1 + T_2 \alpha_2 \mathbf{c}_2 + T_3 \alpha_3 \mathbf{c}_3$$
$$= 1 \times 0.393 \times [1,0,0] + 0.607 \times 0.632 \times [0,1,0] + 0.223 \times 0.259 \times [0,0,1]$$
$$= [0.393, 0, 0] + [0, 0.384, 0] + [0, 0, 0.058]$$
$$= [0.393, 0.384, 0.058]$$

颜色以红色和绿色为主，蓝色贡献最小（因为点 3 被前面的点遮挡）。'''),
    code_cell(r'''# Verify volume rendering
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
np.random.seed(42)

def volume_render(sigmas, colors, deltas):
    """Volume rendering equation.
    
    Args:
        sigmas: (N,) volume density at each point
        colors: (N, 3) RGB color at each point
        deltas: (N,) distance between adjacent samples
    
    Returns:
        rendered_color: (3,) RGB color
    """
    N = len(sigmas)
    # Alpha (opacity) at each point
    alphas = 1 - np.exp(-sigmas * deltas)
    
    # Transmittance (cumulative)
    T = np.ones(N)
    for i in range(1, N):
        T[i] = T[i-1] * np.exp(-sigmas[i-1] * deltas[i-1])
    
    # Rendered color
    weights = T * alphas  # (N,)
    rendered = np.sum(weights[:, None] * colors, axis=0)
    
    return rendered, weights

# Verify manual calculation
sigmas = np.array([0.5, 1.0, 0.3])
colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
deltas = np.array([1.0, 1.0, 1.0])

rendered, weights = volume_render(sigmas, colors, deltas)
print(f"Alphas: {1 - np.exp(-sigmas * deltas)}")
print(f"Transmittance: {weights / (1 - np.exp(-sigmas * deltas))}")
print(f"Weights: {weights}")
print(f"Rendered color: {rendered}")
print(f"Expected: [0.393, 0.384, 0.058]")
assert np.allclose(rendered, [0.393, 0.384, 0.058], atol=0.01), "Mismatch!"
print("Volume rendering verified!")'''),
    md_cell(r'''## 4. 位置编码（Positional Encoding）

### 原理

NeRF 使用位置编码将低维坐标映射到高维空间，使网络能拟合高频细节：

$$\gamma(x) = \left(x, \sin(2^0 \pi x), \cos(2^0 \pi x), \sin(2^1 \pi x), \cos(2^1 \pi x), \ldots, \sin(2^{L-1} \pi x), \cos(2^{L-1} \pi x)\right)$$

### 为什么需要位置编码？

MLP 倾向于学习低频函数（Rahaman et al., 2019）。位置编码通过将坐标映射到高频空间，使网络能学习高频细节（如锐利边缘、纹理）。

[[Vaswani et al., 2017]](https://arxiv.org/abs/1706.03762) 在 Transformer 中首次使用位置编码。
[[Tancik et al., 2020]](https://arxiv.org/abs/2006.10739) 分析了位置编码的 Fourier 特征。'''),
    code_cell(r'''# Positional encoding implementation
def positional_encoding(x, L=10):
    """Positional encoding for NeRF.
    
    gamma(x) = [x, sin(2^0 * pi * x), cos(2^0 * pi * x), 
                  sin(2^1 * pi * x), cos(2^1 * pi * x), ...]
    
    Args:
        x: (..., D) input coordinates
        L: number of frequency bands
    
    Returns:
        encoded: (..., D * (1 + 2*L)) encoded features
    """
    original = [x]
    for l in range(L):
        freq = 2**l * np.pi
        original.append(np.sin(freq * x))
        original.append(np.cos(freq * x))
    return np.concatenate(original, axis=-1)

# Test: 1D positional encoding
x = np.linspace(-1, 1, 100).reshape(-1, 1)
enc_low = positional_encoding(x, L=2)
enc_high = positional_encoding(x, L=10)

print(f"Input shape: {x.shape}")
print(f"Encoded (L=2): {enc_low.shape} (1 + 2*2 = 5 dims per input)")
print(f"Encoded (L=10): {enc_high.shape} (1 + 2*10 = 21 dims per input)")

# Visualize positional encoding
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
for l in range(5):
    freq = 2**l * np.pi
    axes[0].plot(x.flatten(), np.sin(freq * x.flatten()), 
                label=f'sin(2^{l}*pi*x)', alpha=0.8)
axes[0].set_title('Sinusoidal Positional Encoding (different frequencies)')
axes[0].legend()
axes[0].set_xlabel('x')

# Show how encoding helps with high-frequency functions
def high_freq_function(x):
    return np.sin(5 * np.pi * x) * np.cos(3 * np.pi * x)

y = high_freq_function(x.flatten())
axes[1].plot(x.flatten(), y, 'k-', linewidth=2, label='Target high-freq function')
axes[1].set_title('High-Frequency Function (hard for plain MLP, easy with PE)')
axes[1].legend()
axes[1].set_xlabel('x')
plt.tight_layout()
plt.savefig('positional_encoding.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 1: Positional encoding")'''),
    md_cell(r'''## 5. 代码实现：Mini NeRF'''),
    code_cell(r'''class MiniNeRF:
    """Simplified NeRF: MLP that maps (x,y,z) -> (rgb, sigma).
    
    Uses positional encoding for input coordinates.
    Architecture: PE -> FC(256) -> ReLU -> FC(256) -> ReLU -> FC(4)
    """
    def __init__(self, pos_L=6, hidden_dim=64):
        self.pos_L = pos_L
        # Input dim: 3 * (1 + 2*L) = 3 * 13 = 39 for L=6
        input_dim = 3 * (1 + 2 * pos_L)
        
        # Layer 1
        self.W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        # Layer 2
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        # Output: RGB (3) + sigma (1) = 4
        self.W3 = np.random.randn(4, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(4)
    
    def forward(self, points):
        """Forward pass.
        
        Args:
            points: (N, 3) 3D coordinates
        
        Returns:
            rgb: (N, 3) colors in [0, 1]
            sigma: (N,) volume density
        """
        # Positional encoding
        x_enc = positional_encoding(points, L=self.pos_L)
        
        # MLP
        h1 = np.maximum(0, x_enc @ self.W1.T + self.b1)
        h2 = np.maximum(0, h1 @ self.W2.T + self.b2)
        out = h2 @ self.W3.T + self.b3
        
        rgb = 1 / (1 + np.exp(-out[:, :3]))  # sigmoid for color
        sigma = np.maximum(0, out[:, 3])  # ReLU for density
        
        return rgb, sigma
    
    def render_rays(self, origins, directions, near=0, far=4, n_samples=32):
        """Render rays through the volume.
        
        Args:
            origins: (N_rays, 3) ray origins
            directions: (N_rays, 3) ray directions (normalized)
            near, far: near and far bounds
            n_samples: number of points to sample along each ray
        """
        N_rays = origins.shape[0]
        # Sample points along rays
        t_vals = np.linspace(near, far, n_samples)
        t_vals = t_vals + np.random.uniform(0, (far-near)/n_samples, (N_rays, n_samples))
        
        # Points: (N_rays, n_samples, 3)
        points = origins[:, None, :] + t_vals[..., None] * directions[:, None, :]
        
        # Flatten and forward
        points_flat = points.reshape(-1, 3)
        rgb, sigma = self.forward(points_flat)
        rgb = rgb.reshape(N_rays, n_samples, 3)
        sigma = sigma.reshape(N_rays, n_samples)
        
        # Deltas
        deltas = np.diff(t_vals, append=t_vals[:, -1:] + 0.01)
        
        # Volume rendering per ray
        colors = np.zeros((N_rays, 3))
        for i in range(N_rays):
            rendered, _ = volume_render(sigma[i], rgb[i], deltas[i])
            colors[i] = rendered
        
        return colors

print("MiniNeRF class defined")'''),
    md_cell(r'''## 6. 3D 点云可视化'''),
    code_cell(r'''# Visualize 3D point cloud and NeRF rendering
np.random.seed(42)

# Generate a 3D scene: a sphere
n_points = 1000
theta = np.random.uniform(0, 2*np.pi, n_points)
phi = np.random.uniform(0, np.pi, n_points)
r = 1.0 + np.random.randn(n_points) * 0.05

x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)

# Color by position
colors = np.zeros((n_points, 3))
colors[:, 0] = (x + 1) / 2  # R from x
colors[:, 1] = (y + 1) / 2  # G from y
colors[:, 2] = (z + 1) / 2  # B from z

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# Front view
axes[0].scatter(x, y, c=colors, s=1)
axes[0].set_title('Front View (XY)')
axes[0].set_aspect('equal')
# Side view
axes[1].scatter(z, y, c=colors, s=1)
axes[1].set_title('Side View (ZY)')
axes[1].set_aspect('equal')
# Top view
axes[2].scatter(x, z, c=colors, s=1)
axes[2].set_title('Top View (XZ)')
axes[2].set_aspect('equal')
plt.suptitle('3D Point Cloud: Sphere from Different Views', fontsize=14)
plt.tight_layout()
plt.savefig('point_cloud_views.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 2: 3D point cloud views")'''),
    code_cell(r'''# NeRF rendering demo
np.random.seed(42)
nerf = MiniNeRF(pos_L=4, hidden_dim=32)

# Create camera rays
n_rays = 64
image_size = 32

# Camera setup
camera_origin = np.array([0, 0, -3])
# Ray directions (perspective)
fx = fy = 32
pixel_coords = np.linspace(-image_size/2, image_size/2, image_size)
xx, yy = np.meshgrid(pixel_coords, pixel_coords)
dirs = np.stack([xx.flatten(), -yy.flatten(), np.ones(image_size**2) * fx], axis=-1)
dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

# Render
colors = nerf.render_rays(np.tile(camera_origin, (len(dirs), 1)), dirs, 
                          near=2, far=4, n_samples=16)
image = colors.reshape(image_size, image_size, 3)
image = np.clip(image, 0, 1)

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(image)
ax.set_title('NeRF Rendered View (untrained)')
ax.axis('off')
plt.tight_layout()
plt.savefig('nerf_render.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 3: NeRF rendering (untrained model)")'''),
    md_cell(r'''## 7. NeRF 训练流程

### 训练步骤

1. **数据准备**：多视角图像 + 相机位姿
2. **光线采样**：从每个像素发射光线
3. **点采样**：沿光线采样 3D 点
4. **位置编码**：对 3D 坐标做 PE
5. **MLP 前向**：预测颜色和密度
6. **体渲染**：从颜色和密度合成像素颜色
7. **损失计算**：渲染图像与真实图像的 MSE
8. **反向传播**：更新 MLP 权重

### 训练特点

- **逐场景训练**：每个场景单独训练一个 NeRF
- **训练时间长**：通常需要 100K-300K 次迭代
- **渲染速度慢**：每个像素需要多次 MLP 前向

[[Mildenhall et al., 2020]](https://arxiv.org/abs/2003.08934) 原始 NeRF 在单张 V100 上渲染一张图需要约 30 秒。'''),
    code_cell(r'''# Simulate NeRF training (simplified)
np.random.seed(42)

# Create a target function: a colored sphere
def target_color_and_density(points):
    """Simulate a sphere with color and density."""
    r = np.linalg.norm(points, axis=1)
    inside = r < 1.0
    
    colors = np.zeros((len(points), 3))
    colors[inside, 0] = 1.0  # red inside
    colors[inside, 1] = 0.5
    colors[inside, 2] = 0.0
    
    sigma = np.where(inside, 5.0, 0.0)
    return colors, sigma

# Training: minimize MSE between rendered and target
nerf_train = MiniNeRF(pos_L=4, hidden_dim=32)
losses = []

for epoch in range(200):
    # Sample random points in 3D space
    points = np.random.randn(256, 3) * 1.5
    
    # Target
    target_rgb, target_sigma = target_color_and_density(points)
    
    # Predict
    pred_rgb, pred_sigma = nerf_train.forward(points)
    
    # MSE loss
    loss = np.mean((pred_rgb - target_rgb)**2) + np.mean((pred_sigma - target_sigma)**2)
    losses.append(loss)
    
    # Simple gradient step (numerical approximation)
    # In practice, use autograd
    if epoch < 100:  # Only train for first 100 steps
        lr = 0.01
        # Perturb weights slightly to minimize loss
        for attr in ['W1', 'W2', 'W3']:
            grad = np.random.randn(*getattr(nerf_train, attr).shape) * 0.001
            setattr(nerf_train, attr, getattr(nerf_train, attr) - lr * grad)
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss = {loss:.4f}")

print(f"Final loss: {losses[-1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(losses, color='#ea580c', linewidth=1.5, alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('NeRF Training Loss (Simplified)')
plt.tight_layout()
plt.savefig('nerf_training.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 4: NeRF training loss")'''),
    md_cell(r'''## 8. 3D 表示比较'''),
    code_cell(r'''# Compare 3D representations
np.random.seed(42)

# Voxel representation (3D grid)
voxel_size = 16
voxels = np.zeros((voxel_size, voxel_size, voxel_size))
# Create a sphere in voxel space
for i in range(voxel_size):
    for j in range(voxel_size):
        for k in range(voxel_size):
            dx, dy, dz = i - voxel_size/2, j - voxel_size/2, k - voxel_size/2
            if dx**2 + dy**2 + dz**2 < (voxel_size/3)**2:
                voxels[i, j, k] = 1.0

# Point cloud
n_pts = 500
theta = np.random.uniform(0, 2*np.pi, n_pts)
phi = np.random.uniform(0, np.pi, n_pts)
pc = np.stack([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)], axis=1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Voxel slices
axes[0].imshow(voxels[:, :, voxel_size//2], cmap='gray')
axes[0].set_title(f'Voxel (slice, {voxel_size}^3={voxel_size**3} voxels)')
axes[0].axis('off')

# Point cloud
axes[1].scatter(pc[:, 0], pc[:, 1], c=pc[:, 2], cmap='viridis', s=3)
axes[1].set_title(f'Point Cloud ({n_pts} points)')
axes[1].set_aspect('equal')

# Memory comparison
methods = ['Voxels\n(16^3)', 'Point Cloud\n(500 pts)', 'NeRF\n(MLP weights)']
memories = [16**3 * 4 / 1024, 500 * 3 * 4 / 1024, 32*39 + 32*32 + 4*32]  # rough KB
axes[2].bar(methods, memories, color=['#ea580c', '#2563eb', '#16a34a'])
axes[2].set_ylabel('Memory (KB, approx)')
axes[2].set_title('Memory Comparison')

plt.suptitle('3D Representation Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('3d_representation_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization 5: 3D representation comparison")'''),
    md_cell(r'''## 作业

### 作业 1：实现分层采样

NeRF 使用分层采样（Stratified Sampling）和精细采样（Hierarchical Sampling）提高渲染质量。'''),
    code_cell(r'''# Homework 1: Stratified and hierarchical sampling
def stratified_sample(near, far, n_samples, n_rays=1):
    """Stratified sampling along rays.
    
    Instead of evenly spaced, add random noise to each bin.
    """
    # TODO: implement stratified sampling
    # t_i = near + (far - near) * (i + u_i) / n_samples
    # where u_i ~ Uniform(0, 1)
    pass

# Test
# samples = stratified_sample(near=2, far=4, n_samples=32, n_rays=100)
# assert samples.shape == (100, 32)
# assert np.all(samples >= 2) and np.all(samples <= 4)
print("Homework 1 template ready. Implement stratified sampling.")'''),
    md_cell(r'''### 作业 2：实现 NeRF 渲染损失

实现完整的渲染损失：从光线采样到体渲染到 MSE 损失。'''),
    code_cell(r'''# Homework 2: NeRF rendering loss
def nerf_loss(nerf, target_image, camera_params):
    """Compute NeRF rendering loss.
    
    1. Generate rays from camera
    2. Sample points along rays
    3. Forward through NeRF
    4. Volume render
    5. Compare with target image
    """
    # TODO: implement full rendering loss
    pass

# Test
# loss = nerf_loss(nerf_train, image, camera_origin)
# assert loss >= 0, "Loss should be non-negative"
print("Homework 2 template ready. Implement NeRF rendering loss.")'''),
    md_cell(r'''### 作业 3：实现 View Synthesis

从不同视角渲染同一 3D 场景。'''),
    code_cell(r'''# Homework 3: Novel view synthesis
def render_novel_views(nerf, scene_center, n_views=8, radius=3):
    """Render the scene from multiple viewpoints.
    
    Args:
        nerf: trained NeRF model
        scene_center: (3,) center of the scene
        n_views: number of views to render
        radius: camera distance from center
    """
    # TODO: implement novel view synthesis
    # 1. Generate camera positions on a circle around the scene
    # 2. For each position, create rays and render
    # 3. Return list of rendered images
    pass

# Test
# views = render_novel_views(nerf_train, np.zeros(3), n_views=8, radius=3)
# assert len(views) == 8
print("Homework 3 template ready. Implement novel view synthesis.")'''),
    md_cell(r'''## 小结

| 概念 | 要点 |
|------|------|
| **体素** | 3D 网格，规则但内存大 |
| **点云** | 3D 点集，紧凑但无拓扑 |
| **网格** | 顶点+面片，精确但复杂 |
| **NeRF** | 神经网络隐式表示 3D 场景 |
| **体渲染** | $C = \int T(t) \sigma(t) \mathbf{c}(t) dt$ |
| **位置编码** | $\gamma(x) = [x, \sin, \cos, ...]$ 提升高频 |
| **分层采样** | 沿光线随机采样，减少偏差 |
| **逐场景训练** | 每个场景独立训练 |

**关键洞察**：NeRF 将 3D 场景编码在神经网络权重中，通过可微体渲染实现从任意视角生成新视图。位置编码是 NeRF 成功的关键——它让 MLP 能够拟合高频细节。后续工作（Instant-NGP, Plenoxels）大幅加速了 NeRF 训练和渲染。'''),
    md_cell(r'''## 参考文献

1. [[Mildenhall et al., 2020]](https://arxiv.org/abs/2003.08934) Mildenhall et al. *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*. ECCV 2020.
2. [[Vaswani et al., 2017]](https://arxiv.org/abs/1706.03762) Vaswani et al. *Attention Is All You Need*. NeurIPS 2017.
3. [[Tancik et al., 2020]](https://arxiv.org/abs/2006.10739) Tancik et al. *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains*. NeurIPS 2020.
4. [[Muller et al., 2022]](https://arxiv.org/abs/2201.05989) Muller et al. *Instant Neural Graphics Primitives with a Multiresolution Hash Encoding*. ACM TOG 2022.
5. [[Chen et al., 2022]](https://arxiv.org/abs/2112.05126) Chen et al. *TensoRF: Tensorial Radiance Fields*. ECCV 2022.
6. [[Rahaman et al., 2019]](https://arxiv.org/abs/1806.06837) Rahaman et al. *On the Spectral Bias of Neural Networks*. ICML 2019.

---

> 本节内容参考 [Stanford CS231n](https://cs231n.stanford.edu/) | [NeRF Paper](https://arxiv.org/abs/2003.08934) | [NeRF Studio](https://docs.nerf.studio/)'''),
    md_cell(r'''---

## 参考代码实现

以下 GitHub 仓库提供了本节内容的完整代码实现，建议结合学习：

- **[nerfstudio-project/nerfstudio](https://github.com/nerfstudio-project/nerfstudio)** (11996 stars): NeRF 全栈工具箱（训练、渲染、评估一站式）
  - 仓库地址: https://github.com/nerfstudio-project/nerfstudio

- **[MaximeVandegar/Papers-in-100-Lines-of-Code](https://github.com/MaximeVandegar/Papers-in-100-Lines-of-Code)** (2883 stars): 论文精简实现（含 NeRF 100 行代码版）
  - 仓库地址: https://github.com/MaximeVandegar/Papers-in-100-Lines-of-Code

- **[NVlabs/tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)** (4535 stars): NVIDIA 高效神经网络框架（Instant-NGP 基础）
  - 仓库地址: https://github.com/NVlabs/tiny-cuda-nn

> 标注说明: 以上仓库按热度排序，优先推荐 stars 最多的实现.'''),
])
save_notebook(nb8, os.path.join(BASE, "part3-frontiers", "lecture-16-3d-vision", "practice.ipynb"))

print("\n========================================")
print("All 8 notebooks generated successfully!")
print("========================================")
