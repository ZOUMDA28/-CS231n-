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

print("Notebooks 1-4 done. Continuing with notebooks 5-8...")
