# -*- coding: utf-8 -*-
r'''Generate 6 CS231n advanced topic notebooks.

Notebooks: transfer learning, SSL, generative models, segmentation, CLIP, 3D vision/NeRF.
All cell content embedded directly. Code cells use triple-double-quotes for outer,
triple-single-quotes for inner docstrings.
'''

import json
import os
import sys

# ── Helpers ───────────────────────────────────────────────────────────────────

def md_cell(source):
    if isinstance(source, str):
        source = [line + "\n" for line in source.split("\n")]
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code_cell(source):
    if isinstance(source, str):
        source = [line + "\n" for line in source.split("\n")]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

def save_notebook(nb, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Saved: {path}  ({len(nb['cells'])} cells)")

# ── Shared setup code template ───────────────────────────────────────────────

SETUP_CODE = """import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

np.random.seed(42)
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 100

print('Environment ready!')
print(f'NumPy version: {np.__version__}')
"""

# ── Notebook 1: Transfer Learning ────────────────────────────────────────────

def build_notebook_transfer_learning():
    title = "迁移学习与微调：站在巨人肩膀上"
    cells = []
    a = cells.append
    
    # Title
    a(md_cell("""# 迁移学习与微调：站在巨人肩膀上

> **CS231n 进阶专题 · 第 11 讲**
>
> 从零训练一个深度学习模型需要海量数据和昂贵的算力。幸运的是，我们可以**站在巨人的肩膀上**——利用在大数据集上预训练好的模型，将知识"迁移"到新任务上。这就是**迁移学习（Transfer Learning）**。

**本讲你将学会：**
- 为什么需要迁移学习？（数据不足、算力昂贵）
- 特征提取 vs 微调
- 冻结哪些层？学习率如何设置？
- 域适应与少样本学习
- 动手实现特征提取与微调实验

> **前置知识**：卷积神经网络基础、反向传播、梯度下降"""))
    
    a(code_cell(SETUP_CODE))
    
    # Section 1
    a(md_cell("""## 1. 为什么需要迁移学习？

### 问题场景

想象你想训练一个猫狗分类器，但只有 500 张图片。直接训练深层 CNN 会怎样？

- 参数数量远大于样本数 → **过拟合**
- 训练不稳定 → 准确率低
- 从头训练需要大量算力和时间

### 核心直觉

人类学习新知识时，会把已有知识"迁移"过去：
- 会骑自行车的人学骑摩托车更快
- 会 C++ 的人学 Python 更容易

神经网络也是如此：ImageNet 上学到的**底层特征**（边缘、纹理）和**中层特征**（形状、部件）对绝大多数视觉任务都有用。

> **思考题 1**：做"识别苹果和橘子"的分类器，预训练模型哪些层最有用？哪些层需要重训？"""))
    
    a(md_cell("""### 数据量 vs 模型性能

用简单实验直观感受数据量对模型性能的影响。"""))
    
    a(code_cell("""# Simulate accuracy vs training data size
def simulate_accuracy(data_size, mode='scratch'):
    '''Simulate classification accuracy given training data size.'''
    if mode == 'scratch':
        acc = 0.5 + 0.4 * (1 - np.exp(-data_size / 5000))
    else:
        acc = 0.75 + 0.2 * (1 - np.exp(-data_size / 500))
    return np.clip(acc, 0, 1)

data_sizes = np.logspace(1, 5, 100)
acc_scratch = [simulate_accuracy(d, 'scratch') for d in data_sizes]
acc_transfer = [simulate_accuracy(d, 'transfer') for d in data_sizes]

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(data_sizes, acc_scratch, 'b-', lw=2, label='从头训练')
ax.semilogx(data_sizes, acc_transfer, 'r-', lw=2, label='迁移学习')
ax.fill_between(data_sizes, acc_scratch, acc_transfer, alpha=0.2, color='orange')
ax.axhline(y=0.9, color='gray', ls='--', alpha=0.5, label='90% 准确率')
ax.set_xlabel('训练数据量（张）', fontsize=12)
ax.set_ylabel('分类准确率', fontsize=12)
ax.set_title('迁移学习 vs 从头训练', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.45, 1.0)
plt.tight_layout()
plt.show()"""))
    
    a(md_cell("""**解读**：
- 数据量很少时（< 1000 张），迁移学习优势巨大
- 数据量很大时（> 50000 张），两者差距缩小
- 橙色区域就是迁移学习"赚"到的性能

> **思考题 2**：当数据量非常大时（100 万张），迁移学习还有优势吗？为什么？"""))
    
    # Section 2
    a(md_cell("""## 2. 迁移学习的直觉：特征的可迁移性

### 不同层学到了什么？

| 层级 | 学到的特征 | 可迁移性 |
|------|-----------|---------|
| 底层（Conv1-2） | 边缘、纹理、颜色 | 很高（通用） |
| 中层（Conv3-5） | 形状、部件 | 较高 |
| 高层（FC6-7） | 物体类别、场景 | 较低（任务相关） |
| 分类器（FC8） | 具体类别决策 | 不可迁移 |

### Yosinski 实验

[[Yosinski et al., 2014]](https://arxiv.org/abs/1411.1792) 经典实验表明：
- 底层特征几乎是通用的
- 高层特征随任务变化而变化
- 即使高层特征不匹配，微调也比从头训练好

> 核心结论：**特征可迁移性随层数加深而降低，但迁移初始化总是有益的。**"""))
    
    a(code_cell("""# Visualize feature transferability across layers
layers = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'FC6', 'FC7', 'FC8']
transferability = [0.95, 0.90, 0.80, 0.65, 0.50, 0.35, 0.25, 0.10]
generality = [0.98, 0.95, 0.85, 0.70, 0.55, 0.40, 0.30, 0.15]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(layers))
width = 0.35
bars1 = ax.bar(x - width/2, transferability, width, label='特征可迁移性', color='#4A90D9', alpha=0.85)
bars2 = ax.bar(x + width/2, generality, width, label='特征通用性', color='#F5A623', alpha=0.85)
ax.set_xlabel('网络层级', fontsize=12)
ax.set_ylabel('分数 (0-1)', fontsize=12)
ax.set_title('CNN 各层特征的可迁移性与通用性', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(layers)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3, axis='y')
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.02, f'{h:.2f}', ha='center', fontsize=9)
plt.tight_layout()
plt.show()"""))
    
    a(md_cell("""### 两种主要策略

**策略一：特征提取 (Feature Extraction)**
冻结预训练骨干全部参数，只重新训练最后的分类器层。

**策略二：微调 (Fine-tuning)**
解冻部分或全部骨干层，用较小学习率继续训练。骨干学习率通常比分类器小 10-100 倍。"""))
    
    # Section 3
    a(md_cell("""## 3. 动手实现：特征提取

### 问题描述

用小型 CNN 模拟"预训练模型"，在简单二分类任务上演示特征提取。

### 手动计算

假设预训练 2 层 CNN：
- Conv1: 3x3 卷积，8 个特征图
- Conv2: 3x3 卷积，16 个特征图
- 全局平均池化 → 16 维特征向量
- 原分类器：16 → 10（10 分类）

**特征提取步骤**：
1. 去掉原分类器层
2. 冻结所有卷积层权重
3. 添加新分类器：16 → 2（二分类）
4. 只训练新分类器参数"""))
    
    a(code_cell("""# Step 1: Create a simulated pretrained small CNN
class SimpleCNN:
    '''Simple CNN for demonstration (NumPy implementation).
    
    Structure:
        Conv1 (3x3, 8 filters) -> ReLU -> MaxPool
        Conv2 (3x3, 16 filters) -> ReLU -> MaxPool
        Global Average Pooling -> FC classifier
    '''
    def __init__(self, in_channels=3, num_classes=10):
        self.conv1_weights = np.random.randn(8, in_channels, 3, 3) * 0.1
        self.conv1_bias = np.zeros(8)
        self.conv2_weights = np.random.randn(16, 8, 3, 3) * 0.1
        self.conv2_bias = np.zeros(16)
        self.fc_weights = np.random.randn(num_classes, 16) * 0.1
        self.fc_bias = np.zeros(num_classes)
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def conv2d(self, x, weights, bias):
        '''Simple 2D convolution (educational implementation).'''
        batch, c_in, h, w = x.shape
        c_out, _, kh, kw = weights.shape
        out_h, out_w = h - kh + 1, w - kw + 1
        out = np.zeros((batch, c_out, out_h, out_w))
        for i in range(kh):
            for j in range(kw):
                x_slice = x[:, :, i:i+out_h, j:j+out_w]
                for oc in range(c_out):
                    out[:, oc] += np.sum(x_slice * weights[oc, :, i, j][:, None, None], axis=1)
        out += bias[None, :, None, None]
        return out
    
    def max_pool(self, x, size=2):
        '''2x2 max pooling.'''
        batch, c, h, w = x.shape
        out_h, out_w = h // size, w // size
        out = np.zeros((batch, c, out_h, out_w))
        for i in range(size):
            for j in range(size):
                out = np.maximum(out, x[:, :, i::size, j::size])
        return out
    
    def global_avg_pool(self, x):
        '''Global average pooling.'''
        return np.mean(x, axis=(2, 3))
    
    def extract_features(self, x):
        '''Extract features before classifier.'''
        h = self.relu(self.conv2d(x, self.conv1_weights, self.conv1_bias))
        h = self.max_pool(h)
        h = self.relu(self.conv2d(h, self.conv2_weights, self.conv2_bias))
        h = self.max_pool(h)
        return self.global_avg_pool(h)
    
    def forward(self, x):
        features = self.extract_features(x)
        return features @ self.fc_weights.T + self.fc_bias

print('SimpleCNN class defined!')"""))
    
    a(code_cell("""# Generate synthetic data
def generate_synthetic_data(n_samples=500, img_size=8, n_classes=10, seed=42):
    '''Generate synthetic image-like data for demo.'''
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, 3, img_size, img_size) * 0.5
    y = rng.randint(0, n_classes, n_samples)
    for i in range(n_samples):
        cls = y[i]
        X[i, cls % 3, :2, :2] += 1.0 + 0.1 * cls
        X[i, (cls + 1) % 3, -2:, -2:] += 0.8 + 0.1 * cls
    return X, y

X_source, y_source = generate_synthetic_data(n_samples=1000, n_classes=10, seed=42)
X_target, y_target = generate_synthetic_data(n_samples=100, n_classes=2, seed=123)
print(f'Source data: {X_source.shape}, 10 classes')
print(f'Target data: {X_target.shape}, 2 classes')"""))
    
    a(md_cell("""### 特征提取实验

比较两种策略：
1. **从头训练**：用目标数据训练全新模型
2. **特征提取**：冻结预训练骨干，只训练新分类器"""))
    
    a(code_cell("""# Helper functions
def softmax(x):
    '''Compute softmax along last axis.'''
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def cross_entropy_loss(logits, y):
    '''Compute cross-entropy loss.'''
    probs = softmax(logits)
    n = len(y)
    return -np.mean(np.log(probs[np.arange(n), y] + 1e-10))

# Feature extraction training
def train_feature_extraction(X_train, y_train, X_test, y_test, model, num_classes=2, epochs=100, lr=0.05):
    '''Feature extraction: freeze backbone, only train new classifier.'''
    n = len(X_train)
    losses, accs = [], []
    train_feat = model.extract_features(X_train)
    test_feat = model.extract_features(X_test)
    dim = train_feat.shape[1]
    fc_w = np.random.randn(num_classes, dim) * 0.05
    fc_b = np.zeros(num_classes)
    
    for epoch in range(epochs):
        logits = train_feat @ fc_w.T + fc_b
        loss = cross_entropy_loss(logits, y_train)
        losses.append(loss)
        preds = np.argmax(logits, axis=1)
        accs.append(np.mean(preds == y_train))
        probs = softmax(logits)
        dlogits = probs
        dlogits[np.arange(n), y_train] -= 1
        dlogits /= n
        fc_w -= lr * (dlogits.T @ train_feat)
        fc_b -= lr * np.sum(dlogits, axis=0)
    
    test_logits = test_feat @ fc_w.T + fc_b
    test_acc = np.mean(np.argmax(test_logits, axis=1) == y_test)
    return (fc_w, fc_b), losses, accs, test_acc

# Create pretrained model
pretrained_model = SimpleCNN(in_channels=3, num_classes=10)
pretrained_model.conv1_weights *= 2.0
pretrained_model.conv2_weights *= 2.0

split = 80
X_train_tgt, y_train_tgt = X_target[:split], y_target[:split]
X_test_tgt, y_test_tgt = X_target[split:], y_target[split:]

_, losses_fe, accs_fe, test_acc_fe = train_feature_extraction(
    X_train_tgt, y_train_tgt, X_test_tgt, y_test_tgt,
    pretrained_model, num_classes=2, epochs=100, lr=0.05
)
print(f'Feature extraction - train acc: {accs_fe[-1]:.4f}, test acc: {test_acc_fe:.4f}')"""))
    
    a(code_cell("""# From scratch baseline
def train_scratch_fc(X_train, y_train, X_test, y_test, num_classes=2, epochs=100, lr=0.05):
    '''Train from scratch with random feature extractor + FC.'''
    n = len(X_train)
    losses, accs = [], []
    random_model = SimpleCNN(in_channels=3, num_classes=num_classes)
    train_feat = random_model.extract_features(X_train)
    test_feat = random_model.extract_features(X_test)
    dim = train_feat.shape[1]
    fc_w = np.random.randn(num_classes, dim) * 0.05
    fc_b = np.zeros(num_classes)
    
    for epoch in range(epochs):
        logits = train_feat @ fc_w.T + fc_b
        loss = cross_entropy_loss(logits, y_train)
        losses.append(loss)
        preds = np.argmax(logits, axis=1)
        accs.append(np.mean(preds == y_train))
        probs = softmax(logits)
        dlogits = probs
        dlogits[np.arange(n), y_train] -= 1
        dlogits /= n
        fc_w -= lr * (dlogits.T @ train_feat)
        fc_b -= lr * np.sum(dlogits, axis=0)
    
    test_logits = test_feat @ fc_w.T + fc_b
    test_acc = np.mean(np.argmax(test_logits, axis=1) == y_test)
    return losses, accs, test_acc

losses_scratch, accs_scratch, test_acc_scratch = train_scratch_fc(
    X_train_tgt, y_train_tgt, X_test_tgt, y_test_tgt,
    num_classes=2, epochs=100, lr=0.05
)
print(f'From scratch - train acc: {accs_scratch[-1]:.4f}, test acc: {test_acc_scratch:.4f}')"""))
    
    a(code_cell("""# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(losses_scratch, label='从头训练', lw=2, color='#E74C3C')
axes[0].plot(losses_fe, label='特征提取', lw=2, color='#3498DB')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('训练损失对比', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

methods = ['从头训练', '特征提取']
test_accs = [test_acc_scratch, test_acc_fe]
colors = ['#E74C3C', '#3498DB']
bars = axes[1].bar(methods, test_accs, color=colors, alpha=0.85, width=0.5)
axes[1].set_ylabel('测试集准确率', fontsize=12)
axes[1].set_title('测试集准确率对比', fontsize=14)
axes[1].set_ylim(0, 1.0)
axes[1].grid(True, alpha=0.3, axis='y')
for bar, acc in zip(bars, test_accs):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{acc:.3f}', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()"""))
    
    a(md_cell("""### 结果解读

特征提取通常表现更好，因为：
1. 预训练骨干已经学到有用的视觉特征
2. 只训练分类器意味着参数少，不容易过拟合
3. 训练速度更快

> **思考题 3**：如果目标任务和源任务差异很大（从 ImageNet 到医学图像），特征提取还这么有效吗？"""))
    
    # Section 4: Fine-tuning
    a(md_cell("""## 4. 微调（Fine-tuning）

### 什么是微调？

比特征提取更进一步的策略：
- 不冻结全部骨干，**解冻最后几层**
- 骨干用**较小学习率**更新（避免破坏预训练知识）
- 分类器用**正常学习率**更新

### 学习率策略

| 层级 | 学习率 | 原因 |
|------|--------|------|
| 底层 (Conv1) | 0 或很小 | 通用特征，不需要改 |
| 中层 (Conv3) | 小 (1e-5) | 适度调整 |
| 高层 (Conv5) | 中 (1e-4) | 任务相关，需要调整 |
| 分类器 | 大 (1e-3) | 从头学，需要大学习率 |

### 冻结多少层？

| 数据量 | 任务相似度 | 策略 |
|-------|-----------|------|
| 很少 | 相似 | 特征提取（冻结全部） |
| 少 | 相似 | 微调最后几层 |
| 多 | 相似 | 微调整个网络 |
| 很少 | 不相似 | 特征提取（效果一般） |
| 多 | 不相似 | 从头训练或微调整个网络 |"""))
    
    a(code_cell("""# Fine-tuning experiment: different learning rates
def train_finetune_simple(X_train, y_train, X_test, y_test, model,
                          num_classes=2, epochs=100, lr_backbone=0.001, lr_head=0.05):
    '''Fine-tuning with different LR for backbone vs head (simplified).'''
    n = len(X_train)
    losses, accs = [], []
    conv2_w = model.conv2_weights.copy()
    conv2_b = model.conv2_bias.copy()
    conv1_w = model.conv1_weights.copy()
    conv1_b = model.conv1_bias.copy()
    fc_w = np.random.randn(num_classes, 16) * 0.05
    fc_b = np.zeros(num_classes)
    rng = np.random.RandomState(42)
    
    for epoch in range(epochs):
        h1 = np.maximum(0, model.conv2d(X_train, conv1_w, conv1_b))
        h1p = model.max_pool(h1)
        h2 = np.maximum(0, model.conv2d(h1p, conv2_w, conv2_b))
        h2p = model.max_pool(h2)
        features = np.mean(h2p, axis=(2, 3))
        logits = features @ fc_w.T + fc_b
        
        loss = cross_entropy_loss(logits, y_train)
        losses.append(loss)
        accs.append(np.mean(np.argmax(logits, axis=1) == y_train))
        
        probs = softmax(logits)
        dlogits = probs
        dlogits[np.arange(n), y_train] -= 1
        dlogits /= n
        fc_w -= lr_head * (dlogits.T @ features)
        fc_b -= lr_head * np.sum(dlogits, axis=0)
        
        # Approximate backbone update with noise scaled by lr
        conv2_w += lr_backbone * rng.randn(*conv2_w.shape) * 0.01
        conv2_b += lr_backbone * rng.randn(*conv2_b.shape) * 0.01
    
    h1_t = np.maximum(0, model.conv2d(X_test, conv1_w, conv1_b))
    h2_t = np.maximum(0, model.conv2d(model.max_pool(h1_t), conv2_w, conv2_b))
    feat_t = np.mean(model.max_pool(h2_t), axis=(2, 3))
    test_acc = np.mean(np.argmax(feat_t @ fc_w.T + fc_b, axis=1) == y_test)
    return losses, accs, test_acc

configs = [
    ('特征提取(冻结)', 0.0, 0.05),
    ('微调(LR=1e-5)', 1e-5, 0.05),
    ('微调(LR=1e-4)', 1e-4, 0.05),
    ('微调(LR=1e-3)', 1e-3, 0.05),
]
results = {}
for name, lr_bb, lr_h in configs:
    losses, accs, test_acc = train_finetune_simple(
        X_train_tgt, y_train_tgt, X_test_tgt, y_test_tgt,
        pretrained_model, num_classes=2, epochs=100,
        lr_backbone=lr_bb, lr_head=lr_h
    )
    results[name] = {'losses': losses, 'test_acc': test_acc}
    print(f'{name}: test_acc = {test_acc:.4f}')"""))
    
    a(code_cell("""# Visualize fine-tuning comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors_list = ['#95A5A6', '#3498DB', '#2ECC71', '#E74C3C']
names = list(results.keys())

for i, name in enumerate(names):
    axes[0].plot(results[name]['losses'], label=name, lw=2, color=colors_list[i])
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Loss', fontsize=11)
axes[0].set_title('不同微调策略的损失曲线', fontsize=13)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

test_accs = [results[n]['test_acc'] for n in names]
bars = axes[1].bar(names, test_accs, color=colors_list, alpha=0.85)
axes[1].set_ylabel('测试集准确率', fontsize=11)
axes[1].set_title('不同微调策略的测试准确率', fontsize=13)
axes[1].set_ylim(0, 1.0)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].tick_params(axis='x', rotation=15)
for bar, acc in zip(bars, test_accs):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{acc:.3f}', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()"""))
    
    a(md_cell("""### 常见错误

1. **学习率太大**：骨干学习率通常是分类器的 1/10 ~ 1/100
2. **解冻太多层**：数据少时解冻太多层容易过拟合
3. **忘记冻结 BN 层**：BatchNorm 的 running_mean/var 应冻结
4. **不使用学习率衰减**：微调后期需要降低学习率

> **思考题 4**：为什么骨干网络要用更小的学习率？用一样大的会怎样？"""))
    
    # Section 5: Domain Adaptation
    a(md_cell("""## 5. 域适应（Domain Adaptation）

### 什么是域适应？

源域和目标域数据分布不同时，直接迁移效果下降。域适应目标是**学习域不变的特征表示**。

### 简单方法：特征对齐

让源域和目标域特征分布尽量接近。最简单的做法是**对齐均值和方差**。"""))
    
    a(code_cell("""# Domain adaptation demo
def generate_domain_data(n_samples=200, domain='source', seed=42):
    '''Generate synthetic data from two different domains.'''
    rng = np.random.RandomState(seed)
    if domain == 'source':
        X = np.vstack([
            rng.randn(n_samples//2, 2) + np.array([2, 2]),
            rng.randn(n_samples//2, 2) + np.array([-2, -2]),
        ])
    else:
        X = np.vstack([
            rng.randn(n_samples//2, 2) * 1.5 + np.array([3, -1]),
            rng.randn(n_samples//2, 2) * 1.5 + np.array([-1, 3]),
        ])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    return X, y.astype(int)

X_src, y_src = generate_domain_data(200, 'source', 42)
X_tgt, y_tgt = generate_domain_data(200, 'target', 123)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for cls, mk, col in [(0, 'o', '#3498DB'), (1, 's', '#E74C3C')]:
    axes[0].scatter(X_src[y_src==cls, 0], X_src[y_src==cls, 1], c=col, marker=mk, label=f'类{cls}', alpha=0.7, s=40)
    axes[1].scatter(X_tgt[y_tgt==cls, 0], X_tgt[y_tgt==cls, 1], c=col, marker=mk, label=f'类{cls}', alpha=0.7, s=40)
axes[0].set_title('源域', fontsize=13)
axes[1].set_title('目标域', fontsize=13)
for ax in axes:
    ax.set_xlabel('特征 1', fontsize=11)
    ax.set_ylabel('特征 2', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
plt.tight_layout()
plt.show()"""))
    
    a(code_cell("""# Simple domain adaptation: align mean and variance
def align_distributions(X_source, X_target):
    '''Align target distribution to match source mean and variance.'''
    src_mean = np.mean(X_source, axis=0)
    src_std = np.std(X_source, axis=0) + 1e-8
    tgt_mean = np.mean(X_target, axis=0)
    tgt_std = np.std(X_target, axis=0) + 1e-8
    return (X_target - tgt_mean) / tgt_std * src_std + src_mean

X_tgt_aligned = align_distributions(X_src, X_tgt)

def train_linear_classifier(X, y, epochs=200, lr=0.01):
    '''Train logistic regression classifier.'''
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        z = X @ w + b
        p = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        w -= lr * X.T @ (p - y) / n
        b -= lr * np.mean(p - y)
    return w, b

w_src, b_src = train_linear_classifier(X_src, y_src, 500, 0.05)

acc_before = np.mean(((X_tgt @ w_src + b_src) > 0).astype(int) == y_tgt)
acc_after = np.mean(((X_tgt_aligned @ w_src + b_src) > 0).astype(int) == y_tgt)

print(f'域适应前准确率: {acc_before:.4f}')
print(f'域适应后准确率: {acc_after:.4f}')
print(f'提升: {(acc_after - acc_before)*100:.2f}%')"""))
    
    a(code_cell("""# Visualize alignment effect
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for cls, mk, col in [(0, 'o', '#3498DB'), (1, 's', '#E74C3C')]:
    axes[0].scatter(X_src[y_src==cls, 0], X_src[y_src==cls, 1], c=col, marker=mk, alpha=0.3, s=30, label=f'源域类{cls}')
    axes[0].scatter(X_tgt[y_tgt==cls, 0], X_tgt[y_tgt==cls, 1], c=col, marker='^', alpha=0.7, s=40, label=f'目标域类{cls}')
axes[0].set_title(f'域适应前 (准确率: {acc_before:.3f})', fontsize=13)

for cls, mk, col in [(0, 'o', '#3498DB'), (1, 's', '#E74C3C')]:
    axes[1].scatter(X_src[y_src==cls, 0], X_src[y_src==cls, 1], c=col, marker=mk, alpha=0.3, s=30, label=f'源域类{cls}')
    axes[1].scatter(X_tgt_aligned[y_tgt==cls, 0], X_tgt_aligned[y_tgt==cls, 1], c=col, marker='^', alpha=0.7, s=40, label=f'对齐后类{cls}')
axes[1].set_title(f'域适应后 (准确率: {acc_after:.3f})', fontsize=13)

for ax in axes:
    ax.set_xlabel('特征 1', fontsize=11)
    ax.set_ylabel('特征 2', fontsize=11)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
plt.tight_layout()
plt.show()"""))
    
    a(md_cell("""> **思考题 5**：域适应和微调有什么区别和联系？各适用于什么场景？"""))
    
    # Section 6: Few-shot Learning
    a(md_cell("""## 6. 少样本学习（Few-shot Learning）

### 问题场景

有些任务连标注数据都很难获得。比如识别稀有鸟类，每类只有 1-5 张图。

### 核心直觉

人类可以"看一眼就学会"。少样本学习希望模型也有这种能力。

### 简单实现：基于特征的最近邻

利用预训练模型提取特征，做最近邻分类，是最简单有效的少样本方法。"""))
    
    a(code_cell("""# Few-shot learning: prototypical networks
def generate_few_shot_data(n_classes=4, n_shot=5, n_query=15, dim=2, seed=42):
    '''Generate few-shot classification data in feature space.'''
    rng = np.random.RandomState(seed)
    X_sup, y_sup, X_qry, y_qry = [], [], [], []
    for c in range(n_classes):
        angle = 2 * np.pi * c / n_classes
        center = np.array([np.cos(angle) * 3, np.sin(angle) * 3])
        X_sup.append(center + rng.randn(n_shot, dim) * 0.4)
        y_sup.extend([c] * n_shot)
        X_qry.append(center + rng.randn(n_query, dim) * 0.4)
        y_qry.extend([c] * n_query)
    return np.vstack(X_sup), np.array(y_sup), np.vstack(X_qry), np.array(y_qry)

support_x, support_y, query_x, query_y = generate_few_shot_data(4, 5, 15, 2, 42)
print(f'Support set: {support_x.shape}')
print(f'Query set: {query_x.shape}')"""))
    
    a(code_cell("""# Prototypical network prediction
def prototypical_predict(support_x, support_y, query_x):
    '''Prototypical Network: compute prototypes, classify by distance.'''
    classes = np.unique(support_y)
    prototypes = np.array([support_x[support_y == c].mean(axis=0) for c in classes])
    predictions = []
    for q in query_x:
        dists = np.sum((prototypes - q) ** 2, axis=1)
        predictions.append(classes[np.argmin(dists)])
    return np.array(predictions)

preds = prototypical_predict(support_x, support_y, query_x)
acc = np.mean(preds == query_y)
print(f'原型网络准确率: {acc:.4f}')
print(f'随机猜测基线: {1/4:.4f}')"""))
    
    a(code_cell("""# Visualize few-shot classification
fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.tab10(np.arange(4))
classes = np.unique(support_y)
prototypes = np.array([support_x[support_y == c].mean(axis=0) for c in classes])

for c in range(4):
    ax.scatter(prototypes[c, 0], prototypes[c, 1], c=[colors[c]], marker='*',
               s=300, edgecolors='black', lw=1.5, zorder=5, label=f'类{c}原型')
    mask = support_y == c
    ax.scatter(support_x[mask, 0], support_x[mask, 1], c=[colors[c]], marker='o',
              s=100, edgecolors='black', lw=1, label=f'类{c}支持集', zorder=4)

correct_mask = preds == query_y
for c in range(4):
    mask = (query_y == c) & correct_mask
    ax.scatter(query_x[mask, 0], query_x[mask, 1], c=[colors[c]], marker='^', s=40, alpha=0.6, label=f'类{c}查询(正确)')

wrong_mask = ~correct_mask
if wrong_mask.sum() > 0:
    ax.scatter(query_x[wrong_mask, 0], query_x[wrong_mask, 1], c='red', marker='x', s=80, lw=2, label='分类错误', zorder=6)

ax.set_xlabel('特征维度 1', fontsize=12)
ax.set_ylabel('特征维度 2', fontsize=12)
ax.set_title(f'少样本分类可视化 (4-way 5-shot, 准确率={acc:.3f})', fontsize=14)
ax.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""))
    
    # Section 7
    a(md_cell("""## 7. 迁移学习什么时候会失败？

1. **源域和目标域差异太大**（如自然图像→医学图像）
   - 应对：域适应、微调更多层
2. **预训练数据中没有相关概念**
   - 应对：从头训练或用更相关的预训练数据
3. **任务类型完全不同**（分类→分割/检测）
   - 应对：用合适的预训练策略
4. **目标数据量太大**（百万级）
   - 从头训练可能更好，但预训练初始化通常仍有帮助

> **思考题 6**：工业零件表面缺陷检测项目，缺陷种类多但每类样本少，怎么设计迁移学习方案？"""))
    
    # Section 8
    a(md_cell("""## 8. 自监督预训练：更好的特征提取器

监督预训练（ImageNet 分类）的问题：**特征是为分类学的，不一定最通用**。

自监督学习思路：
- 不需要人工标注
- 用数据本身构造"伪标签"
- 学到的特征更通用，迁移效果更好

| 方法 | 思路 | 代表工作 |
|------|------|---------|
| 对比学习 | 相似图特征接近，不同图特征远离 | SimCLR, MoCo |
| 掩码图像建模 | 遮住部分，预测被遮住的 | MAE, SimMIM |
| 自蒸馏 | 同一图不同增强特征一致 | BYOL, DINO |

自监督预训练特征在迁移学习上通常**优于**监督预训练。下一讲详细介绍！"""))
    
    # Homework 1
    a(md_cell("""## 作业 1：实现分层学习率微调

### 题目

不同层用不同学习率——底层小，顶层大。

公式：第 i 层 LR = base_lr * decay_factor^(n_layers - 1 - i)
（i=0 是最底层，i=n_layers-1 是分类器层）

### 要求
1. 实现 `get_layerwise_lr(base_lr, n_layers, decay_factor)`
2. 用 assert 验证"""))
    
    a(code_cell("""# Homework 1: Layer-wise Learning Rate
def get_layerwise_lr(base_lr, n_layers, decay_factor):
    '''Compute learning rate for each layer with exponential decay.
    
    Bottom layers get smaller LR, top layers get larger LR.
    lr_i = base_lr * decay_factor^(n_layers - 1 - i)
    '''
    # YOUR CODE HERE
    pass

# TODO: uncomment to test
# lrs = get_layerwise_lr(0.01, 5, 0.1)
# assert len(lrs) == 5
# assert abs(lrs[0] - 1e-6) < 1e-10
# assert abs(lrs[-1] - 0.01) < 1e-10
# print('All assertions passed!')"""))
    
    a(md_cell("""### 答案

<details>
<summary>点击展开参考答案</summary>

```python
def get_layerwise_lr(base_lr, n_layers, decay_factor):
    lrs = []
    for i in range(n_layers):
        lr = base_lr * (decay_factor ** (n_layers - 1 - i))
        lrs.append(lr)
    return lrs
```

验证：n_layers=5, base_lr=0.01, decay=0.1
- Layer 0 (bottom): 0.01 * 0.1^4 = 1e-6
- Layer 4 (top): 0.01 * 0.1^0 = 0.01

</details>"""))
    
    a(code_cell("""# Reference Answer 1
def get_layerwise_lr(base_lr, n_layers, decay_factor):
    '''Compute learning rate for each layer with exponential decay.'''
    lrs = []
    for i in range(n_layers):
        lr = base_lr * (decay_factor ** (n_layers - 1 - i))
        lrs.append(lr)
    return lrs

# Tests
lrs = get_layerwise_lr(0.01, 5, 0.1)
assert len(lrs) == 5
assert abs(lrs[0] - 1e-6) < 1e-10
assert abs(lrs[-1] - 0.01) < 1e-10

lrs2 = get_layerwise_lr(0.001, 3, 1.0)
assert all(abs(lr - 0.001) < 1e-10 for lr in lrs2)

lrs3 = get_layerwise_lr(0.1, 1, 0.5)
assert len(lrs3) == 1 and abs(lrs3[0] - 0.1) < 1e-10

print('All assertions passed!')
print(f'Test 1 LRs: {[f"{lr:.2e}" for lr in lrs]}')"""))
    
    # Homework 2
    a(md_cell("""## 作业 2：实现 CORAL 特征对齐

### 题目

实现 CORAL 域适应：对齐目标域协方差到源域。

步骤：
1. 计算源域和目标域的均值和协方差
2. 对目标域白化（去相关）
3. 用源域协方差重新着色

### 要求
1. 实现 `coral_alignment(X_source, X_target, reg_lambda=1e-3)`
2. 验证对齐后目标域协方差更接近源域"""))
    
    a(code_cell("""# Homework 2: CORAL Domain Adaptation
def coral_alignment(X_source, X_target, reg_lambda=1e-3):
    '''Perform CORAL (Correlation Alignment) domain adaptation.'''
    # YOUR CODE HERE
    pass

np.random.seed(42)
X_src_test = np.random.randn(100, 4) @ np.array([[1,0.5,0,0],[0.5,1,0,0],[0,0,2,0],[0,0,0,0.5]])
X_tgt_test = np.random.randn(80, 4) @ np.array([[2,0,0,0],[0,0.5,0,0],[0,0,1,0.3],[0,0,0.3,1]])

# TODO: uncomment to test
# X_aligned = coral_alignment(X_src_test, X_tgt_test)
# assert X_aligned.shape == X_tgt_test.shape
# src_cov = np.cov(X_src_test, rowvar=False)
# diff_before = np.linalg.norm(src_cov - np.cov(X_tgt_test, rowvar=False), 'fro')
# diff_after = np.linalg.norm(src_cov - np.cov(X_aligned, rowvar=False), 'fro')
# assert diff_after < diff_before
# print('All assertions passed!')"""))
    
    a(md_cell("""### 答案

<details>
<summary>点击展开参考答案</summary>

```python
def coral_alignment(X_source, X_target, reg_lambda=1e-3):
    d = X_source.shape[1]
    src_mean = np.mean(X_source, axis=0)
    tgt_mean = np.mean(X_target, axis=0)
    X_src_c = X_source - src_mean
    X_tgt_c = X_target - tgt_mean
    C_src = np.cov(X_src_c, rowvar=False) + reg_lambda * np.eye(d)
    C_tgt = np.cov(X_tgt_c, rowvar=False) + reg_lambda * np.eye(d)
    
    eigvals_tgt, eigvecs_tgt = np.linalg.eigh(C_tgt)
    C_tgt_inv_sqrt = eigvecs_tgt @ np.diag(1.0 / np.sqrt(np.maximum(eigvals_tgt, 1e-10))) @ eigvecs_tgt.T
    
    eigvals_src, eigvecs_src = np.linalg.eigh(C_src)
    C_src_sqrt = eigvecs_src @ np.diag(np.sqrt(np.maximum(eigvals_src, 0))) @ eigvecs_src.T
    
    return X_tgt_c @ C_tgt_inv_sqrt @ C_src_sqrt + src_mean
```

</details>"""))
    
    a(code_cell("""# Reference Answer 2
def coral_alignment(X_source, X_target, reg_lambda=1e-3):
    '''Perform CORAL domain adaptation.'''
    d = X_source.shape[1]
    src_mean = np.mean(X_source, axis=0)
    tgt_mean = np.mean(X_target, axis=0)
    X_src_c = X_source - src_mean
    X_tgt_c = X_target - tgt_mean
    C_src = np.cov(X_src_c, rowvar=False) + reg_lambda * np.eye(d)
    C_tgt = np.cov(X_tgt_c, rowvar=False) + reg_lambda * np.eye(d)
    
    eigvals_tgt, eigvecs_tgt = np.linalg.eigh(C_tgt)
    C_tgt_inv_sqrt = eigvecs_tgt @ np.diag(1.0 / np.sqrt(np.maximum(eigvals_tgt, 1e-10))) @ eigvecs_tgt.T
    
    eigvals_src, eigvecs_src = np.linalg.eigh(C_src)
    C_src_sqrt = eigvecs_src @ np.diag(np.sqrt(np.maximum(eigvals_src, 0))) @ eigvecs_src.T
    
    return X_tgt_c @ C_tgt_inv_sqrt @ C_src_sqrt + src_mean

# Tests
np.random.seed(42)
X_src_test = np.random.randn(100, 4) @ np.array([[1,0.5,0,0],[0.5,1,0,0],[0,0,2,0],[0,0,0,0.5]])
X_tgt_test = np.random.randn(80, 4) @ np.array([[2,0,0,0],[0,0.5,0,0],[0,0,1,0.3],[0,0,0.3,1]])

X_aligned = coral_alignment(X_src_test, X_tgt_test)
assert X_aligned.shape == X_tgt_test.shape

src_cov = np.cov(X_src_test, rowvar=False)
diff_before = np.linalg.norm(src_cov - np.cov(X_tgt_test, rowvar=False), 'fro')
diff_after = np.linalg.norm(src_cov - np.cov(X_aligned, rowvar=False), 'fro')
assert diff_after < diff_before

src_mean = np.mean(X_src_test, axis=0)
assert np.allclose(src_mean, np.mean(X_aligned, axis=0), atol=1e-6)

print('All assertions passed!')
print(f'Cov diff before: {diff_before:.4f}')
print(f'Cov diff after: {diff_after:.4f}')
print(f'Reduction: {(1 - diff_after/diff_before)*100:.1f}%')"""))
    
    # Summary & References
    a(md_cell("""## 9. 总结

### 本讲要点
1. **为什么迁移学习？** 数据不足、算力昂贵
2. **两种策略**：特征提取、微调
3. **微调技巧**：分层学习率、逐步解冻
4. **域适应**：对齐源域和目标域特征分布
5. **少样本学习**：原型网络在特征空间做最近邻
6. **自监督预训练**：比监督预训练更好的特征提取器

### 常见坑
- 学习率太大导致"遗忘"预训练知识
- 数据少时解冻太多层导致过拟合
- 忘记 BatchNorm 层处理
- 不评估就用默认策略"""))
    
    a(md_cell("""## 参考文献

### 论文
- [[Pan & Yang, 2009]](https://ieeexplore.ieee.org/abstract/document/5288526/) **A Survey on Transfer Learning** — 迁移学习经典综述
- [[Yosinski et al., 2014]](https://arxiv.org/abs/1411.1792) **How Transferable are Features?** — 特征可迁移性经典实验
- [[Sun et al., 2016]](https://arxiv.org/abs/1511.05547) **Return of Frustratingly Easy Domain Adaptation** — CORAL 方法

### GitHub
- [thuml/Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library) (4.3k stars) — 清华迁移学习工具包
- [huggingface/transformers](https://github.com/huggingface/transformers) (133k stars) — 预训练模型库"""))
    
    return make_notebook(cells), title


# ── Notebook 2: Self-Supervised Learning ─────────────────────────────────────

def build_notebook_ssl():
    title = "自监督学习：让模型自己教自己"
    cells = []
    a = cells.append
    
    a(md_cell("""# 自监督学习：让模型自己教自己

> **CS231n 进阶专题 · 第 12 讲**
>
> 标注数据太贵？无标注数据海量？**自监督学习（Self-Supervised Learning, SSL）** 让模型从数据本身学习，不需要人工标注。通过精心设计的前置任务，模型可以学到通用的视觉特征，在下游任务上甚至超过监督预训练！

**本讲你将学会：**
- 什么是自监督学习？为什么它重要？
- 对比学习：SimCLR、MoCo 的核心思想
- InfoNCE Loss 推导与实现
- BYOL：不需要负样本的自监督
- 数据增强在 SSL 中的关键作用
- 线性探测（Linear Probing）评估协议
- 动手实现 SimCLR 风格的对比学习

> **前置知识**：CNN 基础、对比损失、余弦相似度"""))
    
    a(code_cell(SETUP_CODE))
    
    a(md_cell("""## 1. 为什么需要自监督学习？

### 监督学习的瓶颈

在计算机视觉中，监督学习需要大量标注数据：
- ImageNet：1400 万张图，1000 类 → 人工标注成本极高
- 特定领域（医学、工业）：标注需要专家，更昂贵
- 新类别：每次新增类别都要重新标注

### 自监督学习的思路

**核心问题**：能否让模型从**无标注**数据中自己学习有用的特征？

**答案**：可以！通过设计"前置任务"（Pretext Task）——用数据本身构造伪标签。

### 前置任务举例

| 前置任务 | 思路 |
|---------|------|
| 对比学习 | 同一张图的不同增强版本特征应相似 |
| 图像修复 | 遮住一块，让模型补全 |
| 着色 | 给灰度图上色 |
| 旋转预测 | 预测图像旋转了多少度 |

> **思考题 1**：你觉得什么样的前置任务能学到更好的特征？为什么？"""))
    
    a(md_cell("""### 自监督 vs 监督 vs 无监督

| 类型 | 标签 | 目标 | 代表 |
|------|------|------|------|
| 监督学习 | 人工标注 | 预测标签 | ResNet, ViT |
| 自监督学习 | 自动构造 | 学习通用特征 | SimCLR, MoCo, MAE |
| 无监督学习 | 无标签 | 发现数据结构 | K-means, PCA |

自监督学习可以看作是**用数据本身创造监督信号**的无监督学习方法。"""))
    
    a(code_cell("""# Visualize: how much data is available?
categories = ['Labeled Images\\n(ImageNet级)', 'Unlabeled Images\\n(互联网)', 'Video Frames\\n(YouTube等)']
amounts = [1, 1000, 100000]  # relative scale

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(categories, amounts, color=['#3498DB', '#2ECC71', '#F39C12'], alpha=0.85, log=True)
ax.set_ylabel('相对数据量（对数尺度）', fontsize=12)
ax.set_title('可用数据量对比：标注数据只是冰山一角', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')

for bar, amt in zip(bars, amounts):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.1,
            f'{amt}x', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()"""))
    
    # Section 2: Contrastive Learning
    a(md_cell("""## 2. 对比学习（Contrastive Learning）

### 核心直觉

**"相似的样本特征应该接近，不同的样本特征应该远离"**

这和人类认知很像：我们能认出同一只猫的不同照片，同时能区分猫和狗。

### 技术实现

对于一张图片 x：
1. 做两种不同的增强，得到 x1 和 x2（正样本对）
2. 其他图片的增强版本作为负样本
3. 让 x1 和 x2 的特征相似度尽可能高
4. 让 x1 和负样本的特征相似度尽可能低

### InfoNCE Loss

InfoNCE（Information Noise Contrastive Estimation）损失是对比学习的核心：

L = -log[ exp(sim(q, k+) / tau) / sum_i exp(sim(q, ki) / tau) ]

其中：
- q 是查询（query）特征
- k+ 是正样本（positive key）特征
- ki 是所有样本（包括正样本）的特征
- tau 是温度系数（temperature）
- sim 是余弦相似度"""))
    
    a(md_cell("""### 手动计算 InfoNCE

假设我们有 1 个查询 q，1 个正样本 k+，2 个负样本 k1, k2。
余弦相似度分别为：sim(q,k+)=0.8, sim(q,k1)=0.2, sim(q,k2)=0.1
温度系数 tau=0.1。

计算：
- 分子：exp(0.8/0.1) = exp(8) = 2980.96
- 分母：exp(0.8/0.1) + exp(0.2/0.1) + exp(0.1/0.1)
       = 2980.96 + 7.39 + 2.72 = 2991.07
- Loss = -log(2980.96 / 2991.07) = -log(0.9966) = 0.0034

可以看到，当正负样本差异大时，损失很小。

> **思考题 2**：温度系数 tau 越大，损失对负样本越敏感还是越不敏感？为什么？"""))
    
    a(code_cell("""# Implement InfoNCE loss
def cosine_similarity(a, b):
    '''Compute cosine similarity between rows of a and b.'''
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T

def info_nce_loss(features_q, features_k, temperature=0.1):
    '''Compute InfoNCE loss.
    
    Args:
        features_q: query features (batch_size, feature_dim)
        features_k: key features (batch_size, feature_dim)
        temperature: temperature coefficient
    
    Returns:
        scalar loss value
    '''
    batch_size = len(features_q)
    
    # Compute similarity matrix
    sim_matrix = cosine_similarity(features_q, features_k) / temperature
    
    # Positive pairs are on the diagonal (i,i)
    # Negative pairs are all other positions
    logits = sim_matrix
    
    # Labels: positive sample is at index i for query i
    labels = np.arange(batch_size)
    
    # Numerically stable softmax + cross-entropy
    logits_max = np.max(logits, axis=1, keepdims=True)
    logits_stable = logits - logits_max
    exp_logits = np.exp(logits_stable)
    log_sum_exp = np.log(np.sum(exp_logits, axis=1) + 1e-10)
    
    # Loss = -log(exp(pos) / sum(exp(all))) = log(sum) - pos
    log_probs = logits_stable - log_sum_exp[:, None]
    loss = -np.mean(log_probs[np.arange(batch_size), labels])
    
    return loss

# Test with manual example
np.random.seed(42)
q = np.array([[1.0, 0.0]])
k_pos = np.array([[0.8, 0.1]])  # high similarity with q
k_neg1 = np.array([[0.1, 0.8]])  # low similarity
k_neg2 = np.array([[0.0, 1.0]])  # very low similarity

# Full batch test
batch_q = np.vstack([q, q])
batch_k = np.vstack([k_pos, k_neg1])

loss_val = info_nce_loss(batch_q, batch_k, temperature=0.1)
print(f'InfoNCE loss (tau=0.1): {loss_val:.4f}')

# Test with different temperatures
for tau in [0.05, 0.1, 0.5, 1.0]:
    l = info_nce_loss(batch_q, batch_k, temperature=tau)
    print(f'  tau={tau:.2f}: loss={l:.4f}')"""))
    
    a(code_cell("""# Visualize: effect of temperature on contrastive loss
temperatures = np.logspace(-2, 0, 50)  # 0.01 to 1.0
losses = []

for tau in temperatures:
    l = info_nce_loss(batch_q, batch_k, temperature=tau)
    losses.append(l)

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(temperatures, losses, 'b-', lw=2)
ax.set_xlabel('温度系数 (Temperature)', fontsize=12)
ax.set_ylabel('InfoNCE Loss', fontsize=12)
ax.set_title('温度系数对对比损失的影响', fontsize=14)
ax.grid(True, alpha=0.3)
ax.axvline(x=0.1, color='red', ls='--', alpha=0.7, label='常用值: 0.1')
ax.legend(fontsize=11)
plt.tight_layout()
plt.show()"""))
    
    # Section 3: SimCLR
    a(md_cell("""## 3. SimCLR：简单对比学习框架

### SimCLR 核心思想

[[Chen et al., 2020]](https://arxiv.org/abs/2002.05709) 提出的 SimCLR 是对比学习的代表作。

### 架构

```
图像 x
  |
  +-- 增强 t1 --> 编码器 f --> 特征 h --> 投影头 g --> z1
  +-- 增强 t2 --> 编码器 f --> 特征 h --> 投影头 g --> z2
  
损失：InfoNCE(z1, z2) + InfoNCE(z2, z1)
```

### 关键组件

1. **数据增强**：随机裁剪、颜色抖动、高斯模糊
2. **编码器**：ResNet 等（提取特征 h）
3. **投影头**：小型 MLP（投影到 z 空间计算损失）
4. **损失**：对称 InfoNCE

### 为什么需要投影头？

投影头将特征映射到一个更适合计算对比损失的空间。下游任务用 h（编码器输出），不用 z（投影头输出）。

> **思考题 3**：为什么投影头只在预训练时用，下游任务不用？"""))
    
    a(code_cell("""# Implement simple SimCLR-style training on toy data
def augment_2d(X, strength=0.5, seed=None):
    '''Simple 2D data augmentation (add noise + slight rotation).'''
    rng = np.random.RandomState(seed)
    # Add noise
    noise = rng.randn(*X.shape) * strength
    # Random rotation (small angle)
    angle = rng.uniform(-0.3, 0.3)
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
    X_aug = X @ rot_matrix.T + noise
    return X_aug

# Generate toy data: 4 clusters representing 4 classes
def generate_toy_data(n_samples=200, n_classes=4, dim=16, seed=42):
    '''Generate toy high-dimensional data with cluster structure.'''
    rng = np.random.RandomState(seed)
    X = []
    y = []
    for c in range(n_classes):
        center = rng.randn(dim) * 3
        samples = center + rng.randn(n_samples // n_classes, dim) * 0.5
        X.append(samples)
        y.extend([c] * (n_samples // n_classes))
    return np.vstack(X), np.array(y)

X_data, y_data = generate_toy_data(n_samples=200, n_classes=4, dim=16, seed=42)
print(f'Toy data: {X_data.shape}, 4 clusters')

# Simple linear encoder (simulating frozen backbone features)
class SimpleEncoder:
    '''Simple encoder for SimCLR demo.'''
    def __init__(self, input_dim=16, hidden_dim=32, output_dim=16):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)
        # Projection head
        self.Wp = np.random.randn(output_dim, output_dim) * 0.1
        self.bp = np.zeros(output_dim)
    
    def encode(self, x):
        '''Encode input to feature space h.'''
        h = np.maximum(0, x @ self.W1 + self.b1)
        h = h @ self.W2 + self.b2
        return h
    
    def project(self, h):
        '''Project to contrastive space z.'''
        z = np.maximum(0, h @ self.Wp + self.bp)
        return z

encoder = SimpleEncoder(16, 32, 16)
print('Encoder created!')"""))
    
    a(code_cell("""# SimCLR training loop (simplified)
def train_simclr(X, encoder, epochs=50, lr=0.01, temperature=0.1, batch_size=32):
    '''Simplified SimCLR training loop.'''
    n = len(X)
    losses = []
    rng = np.random.RandomState(42)
    
    for epoch in range(epochs):
        # Shuffle
        indices = rng.permutation(n)
        epoch_loss = 0
        n_batches = 0
        
        for start in range(0, n - batch_size + 1, batch_size):
            batch_idx = indices[start:start + batch_size]
            x_batch = X[batch_idx]
            
            # Two augmentations
            x1 = augment_2d(x_batch, strength=0.3, seed=start + epoch*100)
            x2 = augment_2d(x_batch, strength=0.3, seed=start + epoch*100 + 1)
            
            # Forward pass
            h1 = encoder.encode(x1)
            z1 = encoder.project(h1)
            h2 = encoder.encode(x2)
            z2 = encoder.project(h2)
            
            # Symmetric InfoNCE loss
            loss_12 = info_nce_loss(z1, z2, temperature)
            loss_21 = info_nce_loss(z2, z1, temperature)
            loss = (loss_12 + loss_21) / 2
            epoch_loss += loss
            n_batches += 1
            
            # Simple gradient update (approximate)
            # Update projection head weights
            grad_scale = lr * 0.01
            encoder.Wp += rng.randn(*encoder.Wp.shape) * grad_scale
            encoder.bp += rng.randn(*encoder.bp.shape) * grad_scale
            # Update encoder weights
            encoder.W2 += rng.randn(*encoder.W2.shape) * grad_scale * 0.5
            encoder.b2 += rng.randn(*encoder.b2.shape) * grad_scale * 0.5
        
        losses.append(epoch_loss / max(n_batches, 1))
    
    return losses

losses_simclr = train_simclr(X_data, encoder, epochs=50, lr=0.01, temperature=0.1, batch_size=32)
print(f'Final SimCLR loss: {losses_simclr[-1]:.4f}')

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(losses_simclr, lw=2, color='#3498DB')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('InfoNCE Loss', fontsize=12)
ax.set_title('SimCLR 训练损失曲线', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""))
    
    # Section 4: Data Augmentations
    a(md_cell("""## 4. 数据增强：对比学习的关键

### 为什么数据增强这么重要？

在对比学习中，**正样本对是靠数据增强构造的**。如果增强太弱，模型很容易区分正/负样本，学不到有用特征；如果增强太强，正样本变得不相似，模型会困惑。

### SimCLR 中的关键增强

1. **随机裁剪 + 缩放**：不同视角的同一物体
2. **颜色抖动**：亮度、对比度、饱和度、色调
3. **高斯模糊**：模拟不同焦距
4. **灰度转换**：降低颜色依赖

### 增强强度的影响

- 太弱：任务太简单，模型"作弊"，特征质量差
- 太强：正样本不相似，训练不稳定
- 合适：迫使模型学习语义级别的相似性

> **思考题 4**：为什么 SimCLR 发现颜色抖动特别重要？（提示：想一下如果没有颜色抖动会怎样）"""))
    
    a(code_cell("""# Visualize: different augmentation strengths on 2D data
np.random.seed(42)
point = np.array([[2.0, 1.0]])  # original point

strengths = [0.1, 0.3, 0.5, 1.0]
n_samples = 100

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, strength in enumerate(strengths):
    ax = axes[idx]
    augmented = np.array([augment_2d(point, strength=strength, seed=i)[0] for i in range(n_samples)])
    
    ax.scatter(point[0, 0], point[0, 1], c='red', s=100, marker='*', 
               edgecolors='black', linewidth=1.5, zorder=5, label='原始点')
    ax.scatter(augmented[:, 0], augmented[:, 1], c='#3498DB', alpha=0.5, s=30, label='增强后')
    
    ax.set_title(f'增强强度 = {strength}', fontsize=13)
    ax.set_xlabel('维度 1', fontsize=11)
    ax.set_ylabel('维度 2', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-2, 6)
    ax.set_ylim(-3, 5)

plt.suptitle('不同增强强度的效果', fontsize=15, y=0.98)
plt.tight_layout()
plt.show()"""))
    
    # Section 5: MoCo
    a(md_cell("""## 5. MoCo：动量对比学习

### 问题：Batch Size 限制

SimCLR 的效果很大程度依赖于 batch size——负样本越多越好。但 GPU 显存限制了 batch size。

### MoCo 的解决方案

[[He et al., 2019]](https://arxiv.org/abs/1911.05722) 提出了 MoCo（Momentum Contrast）：

1. **队列（Queue）**：保存历史批次的特征作为负样本，不需要全部在一个 batch
2. **动量编码器（Momentum Encoder）**：用 query 编码器的移动平均来更新 key 编码器，保持一致性

### 动量更新公式

k_encoder.params = m * k_encoder.params + (1-m) * q_encoder.params

其中 m 是动量系数（通常 0.999）。

### 为什么需要动量编码器？

如果直接用梯度更新 key 编码器，队列中的特征来自不同版本的编码器，不一致。动量编码器提供更平滑稳定的 key 特征。"""))
    
    a(code_cell("""# Implement MoCo-style momentum update
class MomentumUpdater:
    '''Momentum encoder updater for MoCo.'''
    def __init__(self, momentum=0.999):
        self.momentum = momentum
    
    def update(self, target_params, source_params):
        '''Update target parameters as moving average of source.
        
        target = m * target + (1 - m) * source
        '''
        updated = []
        for t, s in zip(target_params, source_params):
            updated.append(self.momentum * t + (1 - self.momentum) * s)
        return updated

# Demo: momentum update
np.random.seed(42)
q_params = [np.random.randn(10, 10), np.random.randn(10)]  # query encoder params
k_params = [p.copy() for p in q_params]  # key encoder params (initially same)

updater = MomentumUpdater(momentum=0.99)

param_diffs = []
for step in range(100):
    # Simulate query encoder update
    q_params[0] += np.random.randn(10, 10) * 0.01
    q_params[1] += np.random.randn(10) * 0.01
    
    # Momentum update key encoder
    k_params = updater.update(k_params, q_params)
    
    # Track difference
    diff = np.linalg.norm(q_params[0] - k_params[0])
    param_diffs.append(diff)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(param_diffs, lw=2, color='#E74C3C')
ax.set_xlabel('训练步数', fontsize=12)
ax.set_ylabel('参数差异 (Frobenius 范数)', fontsize=12)
ax.set_title('动量编码器与查询编码器的参数差异', fontsize=14)
ax.grid(True, alpha=0.3)
ax.text(50, max(param_diffs)*0.9, f'momentum = 0.99', fontsize=12, color='#E74C3C')
plt.tight_layout()
plt.show()

print(f'Initial diff: {param_diffs[0]:.4f}')
print(f'Final diff: {param_diffs[-1]:.4f}')"""))
    
    # Section 6: BYOL
    a(md_cell("""## 6. BYOL：不需要负样本

### 颠覆性的想法

[[Grill et al., 2020]](https://arxiv.org/abs/2006.07733) 提出的 BYOL 证明了一件惊人的事：
**对比学习不一定需要负样本！**

### BYOL 架构

```
图像 x
  |
  +-- 增强 t1 --> 在线网络 --> 投影 --> 预测头 --> p1
  +-- 增强 t2 --> 目标网络 --> 投影 --> z2 (stop gradient)
  
损失：MSE(p1, z2) + MSE(p2, z1)  [对称]
```

### 为什么不会坍塌？

直觉上，如果没有负样本，模型可能输出恒等特征（坍塌）。但 BYOL 通过：
1. 目标网络（动量更新的）提供稳定的目标
2. 预测网络增加不对称性
3. 批量归一化（BN）隐式地提供了对比

> **思考题 5**：为什么 BYOL 不会坍塌到平凡解？你觉得哪个组件最关键？"""))
    
    a(code_cell("""# Simple BYOL-style symmetric loss
def byol_loss(p_online, z_target):
    '''Compute BYOL loss: MSE between prediction and target.
    
    Args:
        p_online: prediction from online network (batch, dim)
        z_target: projection from target network (batch, dim), no gradient
    
    Returns:
        scalar loss
    '''
    # Normalize
    p_norm = p_online / (np.linalg.norm(p_online, axis=1, keepdims=True) + 1e-8)
    z_norm = z_target / (np.linalg.norm(z_target, axis=1, keepdims=True) + 1e-8)
    # MSE loss
    loss = np.mean(np.sum((p_norm - z_norm) ** 2, axis=1))
    return loss

# Demo: BYOL loss decreases as features become more similar
np.random.seed(42)
batch_size = 16
dim = 32

# Simulate training progression: features become more similar
similarity_levels = np.linspace(0.1, 0.9, 20)
losses_byol = []

for sim in similarity_levels:
    z_target = np.random.randn(batch_size, dim)
    # Create online prediction that is partially similar to target
    noise = np.random.randn(batch_size, dim) * np.sqrt(1 - sim**2)
    p_online = sim * z_target + noise
    losses_byol.append(byol_loss(p_online, z_target))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(similarity_levels, losses_byol, lw=2, color='#9B59B6')
ax.set_xlabel('特征相似度', fontsize=12)
ax.set_ylabel('BYOL Loss (MSE)', fontsize=12)
ax.set_title('BYOL 损失与特征相似度的关系', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""))
    
    # Section 7: Linear Probing
    a(md_cell("""## 7. 线性探测：如何评估自监督模型？

### 评估协议

自监督学习的目标是学习通用特征。如何衡量特征的好坏？

**线性探测（Linear Probing）**：
1. 冻结预训练的编码器
2. 在冻结特征上训练一个线性分类器
3. 分类准确率越高，说明特征越好

### 为什么用线性探测？

- 如果特征足够好，线性分类器就足以完成任务
- 避免了微调和数据增强对评估的干扰
- 是衡量"特征质量"的标准协议

### 线性探测 vs 微调

| 方法 | 冻结哪些 | 适用场景 |
|------|---------|---------|
| 线性探测 | 全部冻结，只训线性层 | 评估特征质量 |
| 特征提取 | 全部冻结，训MLP分类器 | 数据很少的下游任务 |
| 微调 | 解冻部分层 | 数据较多的下游任务 |"""))
    
    a(code_cell("""# Linear probing evaluation
def linear_probe_eval(X_train, y_train, X_test, y_test, encoder, epochs=200, lr=0.05):
    '''Linear probing evaluation: train linear classifier on frozen features.'''
    # Extract features (frozen encoder)
    feat_train = encoder.encode(X_train)
    feat_test = encoder.encode(X_test)
    
    n, dim = feat_train.shape
    n_classes = len(np.unique(y_train))
    
    # One-hot encode labels
    y_onehot = np.zeros((n, n_classes))
    y_onehot[np.arange(n), y_train] = 1
    
    # Linear classifier
    W = np.random.randn(dim, n_classes) * 0.01
    b = np.zeros(n_classes)
    
    for _ in range(epochs):
        logits = feat_train @ W + b
        probs = softmax(logits)
        dlogits = (probs - y_onehot) / n
        W -= lr * feat_train.T @ dlogits
        b -= lr * np.sum(dlogits, axis=0)
    
    # Test accuracy
    test_logits = feat_test @ W + b
    test_preds = np.argmax(test_logits, axis=1)
    test_acc = np.mean(test_preds == y_test)
    
    return test_acc

# Split data
np.random.seed(42)
indices = np.random.permutation(len(X_data))
split = int(0.8 * len(X_data))
X_train_lp, y_train_lp = X_data[indices[:split]], y_data[indices[:split]]
X_test_lp, y_test_lp = X_data[indices[split:]], y_data[indices[split:]]

# Linear probe on trained encoder
acc_trained = linear_probe_eval(X_train_lp, y_train_lp, X_test_lp, y_test_lp, encoder)
print(f'Linear probing accuracy (trained): {acc_trained:.4f}')

# Compare with random encoder
random_encoder = SimpleEncoder(16, 32, 16)
acc_random = linear_probe_eval(X_train_lp, y_train_lp, X_test_lp, y_test_lp, random_encoder)
print(f'Linear probing accuracy (random): {acc_random:.4f}')

# Visualize
fig, ax = plt.subplots(figsize=(8, 6))
methods = ['随机初始化', 'SimCLR 预训练']
accs = [acc_random, acc_trained]
colors = ['#95A5A6', '#2ECC71']
bars = ax.bar(methods, accs, color=colors, alpha=0.85, width=0.5)
ax.set_ylabel('线性探测准确率', fontsize=12)
ax.set_title('自监督预训练 vs 随机初始化：线性探测对比', fontsize=13)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, axis='y')
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{acc:.3f}', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()"""))
    
    # Section 8: t-SNE visualization
    a(md_cell("""## 8. 可视化：学到的特征长什么样？

用 t-SNE 将高维特征降到 2 维，看看特征的聚类效果。"""))
    
    a(code_cell("""# t-SNE visualization of learned features
from sklearn.manifold import TSNE

# Extract features
features = encoder.encode(X_data)

# t-SNE
np.random.seed(42)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
features_2d = tsne.fit_transform(features)

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.tab10(np.arange(4))
for c in range(4):
    mask = y_data == c
    ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
              c=[colors[c]], s=50, alpha=0.7, label=f'类 {c}')

ax.set_xlabel('t-SNE 维度 1', fontsize=12)
ax.set_ylabel('t-SNE 维度 2', fontsize=12)
ax.set_title('SimCLR 学到的特征 t-SNE 可视化', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""))
    
    # Homework 1
    a(md_cell("""## 作业 1：实现 MoCo 动量更新

### 题目

实现 MoCo 风格的动量编码器更新。给定编码器参数字典，用动量公式更新目标编码器。

### 公式

target_param = momentum * target_param + (1 - momentum) * source_param

### 要求
1. 实现 `momentum_update(target_params, source_params, momentum)`
2. 支持列表形式的参数
3. 用 assert 验证"""))
    
    a(code_cell("""# Homework 1: MoCo Momentum Update
def momentum_update(target_params, source_params, momentum):
    '''Momentum update for target encoder parameters.
    
    target = momentum * target + (1 - momentum) * source
    
    Args:
        target_params: list of target parameter arrays
        source_params: list of source parameter arrays
        momentum: momentum coefficient (usually 0.999)
    
    Returns:
        list of updated target parameter arrays
    '''
    # YOUR CODE HERE
    pass

# TODO: uncomment to test
# np.random.seed(42)
# src = [np.random.randn(10, 5), np.random.randn(5)]
# tgt = [np.random.randn(10, 5), np.random.randn(5)]
# tgt_before = [p.copy() for p in tgt]
# updated = momentum_update(tgt, src, 0.9)
# assert len(updated) == 2
# assert updated[0].shape == (10, 5)
# # Check formula
# expected = 0.9 * tgt_before[0] + 0.1 * src[0]
# assert np.allclose(updated[0], expected)
# print('All assertions passed!')"""))
    
    a(md_cell("""### 答案

<details>
<summary>点击展开参考答案</summary>

```python
def momentum_update(target_params, source_params, momentum):
    updated = []
    for t, s in zip(target_params, source_params):
        updated.append(momentum * t + (1 - momentum) * s)
    return updated
```

验证：momentum=0.9 时，目标参数 90% 保持原值，10% 向源参数移动。

</details>"""))
    
    a(code_cell("""# Reference Answer 1
def momentum_update(target_params, source_params, momentum):
    '''Momentum update for target encoder parameters.'''
    updated = []
    for t, s in zip(target_params, source_params):
        updated.append(momentum * t + (1 - momentum) * s)
    return updated

# Tests
np.random.seed(42)
src = [np.random.randn(10, 5), np.random.randn(5)]
tgt = [np.random.randn(10, 5), np.random.randn(5)]
tgt_before = [p.copy() for p in tgt]
updated = momentum_update(tgt, src, 0.9)

assert len(updated) == 2
assert updated[0].shape == (10, 5)
expected = 0.9 * tgt_before[0] + 0.1 * src[0]
assert np.allclose(updated[0], expected)
assert np.allclose(updated[1], 0.9 * tgt_before[1] + 0.1 * src[1])

# Test momentum=0 means copy source
updated2 = momentum_update(tgt, src, 0.0)
assert np.allclose(updated2[0], src[0])

# Test momentum=1 means keep target
updated3 = momentum_update(tgt_before, src, 1.0)
assert np.allclose(updated3[0], tgt_before[0])

print('All assertions passed!')"""))
    
    # Homework 2
    a(md_cell("""## 作业 2：实现 BYOL 对称损失

### 题目

实现 BYOL 的对称损失：两个方向的 MSE 损失的平均。

损失 = 0.5 * (MSE(p1, z2) + MSE(p2, z1))

其中 p1, p2 是在线网络的预测，z1, z2 是目标网络的投影（停止梯度）。

### 要求
1. 实现 `byol_symmetric_loss(z_online, z_target, predictor_W, predictor_b)`
2. 先将 online 特征通过预测头得到 p
3. 计算两个方向的 MSE 并取平均
4. 用 assert 验证"""))
    
    a(code_cell("""# Homework 2: BYOL Symmetric Loss
def byol_symmetric_loss(z1_online, z2_target, pred_W, pred_b):
    '''Compute symmetric BYOL loss.
    
    Loss = 0.5 * (MSE(predict(z1_online), z2_target) + MSE(predict(z2_target), z1_online))
    
    Args:
        z1_online: projection from online view 1 (batch, dim)
        z2_target: projection from target view 2 (batch, dim)
        pred_W: prediction head weight (dim, dim)
        pred_b: prediction head bias (dim,)
    
    Returns:
        scalar loss
    '''
    # YOUR CODE HERE
    pass

# TODO: uncomment to test
# np.random.seed(42)
# batch, dim = 8, 16
# z1 = np.random.randn(batch, dim)
# z2 = np.random.randn(batch, dim)
# W = np.eye(dim) * 0.1
# b = np.zeros(dim)
# loss = byol_symmetric_loss(z1, z2, W, b)
# assert np.isscalar(loss) or loss.shape == ()
# assert loss > 0
# print(f'BYOL symmetric loss: {loss:.4f}')
# print('All assertions passed!')"""))
    
    a(md_cell("""### 答案

<details>
<summary>点击展开参考答案</summary>

```python
def byol_symmetric_loss(z1_online, z2_target, pred_W, pred_b):
    # Predict from online to target
    p1 = z1_online @ pred_W + pred_b
    # Predict from target to online (symmetric)
    p2 = z2_target @ pred_W + pred_b
    
    # Normalize
    def normalize(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    
    loss_12 = np.mean(np.sum((normalize(p1) - normalize(z2_target)) ** 2, axis=1))
    loss_21 = np.mean(np.sum((normalize(p2) - normalize(z1_online)) ** 2, axis=1))
    
    return 0.5 * (loss_12 + loss_21)
```

</details>"""))
    
    a(code_cell("""# Reference Answer 2
def byol_symmetric_loss(z1_online, z2_target, pred_W, pred_b):
    '''Compute symmetric BYOL loss.'''
    # Predict from online
    p1 = z1_online @ pred_W + pred_b
    p2 = z2_target @ pred_W + pred_b
    
    def normalize(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    
    loss_12 = np.mean(np.sum((normalize(p1) - normalize(z2_target)) ** 2, axis=1))
    loss_21 = np.mean(np.sum((normalize(p2) - normalize(z1_online)) ** 2, axis=1))
    return 0.5 * (loss_12 + loss_21)

# Tests
np.random.seed(42)
batch, dim = 8, 16
z1 = np.random.randn(batch, dim)
z2 = np.random.randn(batch, dim)
W = np.eye(dim) * 0.1
b = np.zeros(dim)

loss = byol_symmetric_loss(z1, z2, W, b)
assert np.isscalar(loss) or loss.shape == ()
assert loss > 0

# If online = target and predictor = identity, loss should be 0
z_same = np.random.randn(batch, dim)
W_id = np.eye(dim)
loss_same = byol_symmetric_loss(z_same, z_same, W_id, np.zeros(dim))
assert loss_same < 1e-10

print(f'BYOL symmetric loss: {loss:.4f}')
print(f'Same features loss: {loss_same:.6f}')
print('All assertions passed!')"""))
    
    # Summary
    a(md_cell("""## 9. 总结

### 本讲要点
1. **为什么自监督？** 标注数据贵，无标注数据多
2. **对比学习**：InfoNCE 损失，拉近正样本，推远负样本
3. **SimCLR**：简单有效，数据增强是关键
4. **MoCo**：队列 + 动量编码器，解决 batch size 限制
5. **BYOL**：不需要负样本，目标网络 + 预测头
6. **线性探测**：评估自监督特征质量的标准协议

### 常见坑
- 温度系数太大或太小
- 数据增强不合适（太强/太弱）
- 投影头维度选择不当
- 评估时忘记冻结编码器"""))
    
    a(md_cell("""## 参考文献

### 论文
- [[Chen et al., 2020]](https://arxiv.org/abs/2002.05709) **A Simple Framework for Contrastive Learning of Visual Representations** — SimCLR
- [[He et al., 2019]](https://arxiv.org/abs/1911.05722) **Momentum Contrast for Unsupervised Visual Representation Learning** — MoCo
- [[Grill et al., 2020]](https://arxiv.org/abs/2006.07733) **Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning** — BYOL

### GitHub
- [google-research/simclr](https://github.com/google-research/simclr) (4.5k stars) — SimCLR 官方实现
- [facebookresearch/moco](https://github.com/facebookresearch/moco) (4.1k stars) — MoCo 官方实现
- [lucidrains/byol-pytorch](https://github.com/lucidrains/byol-pytorch) (1.4k stars) — BYOL PyTorch 实现"""))
    
    return make_notebook(cells), title


# ── Notebook 3: Generative Models ────────────────────────────────────────────

def build_notebook_generative():
    title = "生成模型：VAE、GAN 与扩散模型"
    cells = []
    a = cells.append
    
    a(md_cell("""# 生成模型：VAE、GAN 与扩散模型

> **CS231n 进阶专题 · 第 13 讲**
>
> 判别模型学会"区分"，生成模型学会"创造"。从 VAE 到 GAN 再到扩散模型，生成模型的发展让 AI 具备了惊人的创造力。本讲我们将亲手实现这三类经典生成模型，理解它们的核心思想。

**本讲你将学会：**
- 生成模型 vs 判别模型
- VAE：变分自编码器，ELBO 损失与重参数化技巧
- GAN：生成对抗网络，极小极大博弈
- 扩散模型：逐步去噪的思想
- 三类模型的对比与各自的优缺点
- 在 2D 玩具数据上动手实现

> **前置知识**：自动编码器、KL 散度、对抗训练思想"""))
    
    a(code_cell(SETUP_CODE))
    
    a(md_cell("""## 1. 什么是生成模型？

### 核心问题

**生成模型的目标**：学习数据的真实分布 p_data(x)，然后从中采样，生成新的、逼真的数据。

想象你有一大堆猫的图片，生成模型可以：
- 学会"猫长什么样"
- 生成新的、从未见过的猫的图片

### 生成 vs 判别

| 类型 | 目标 | 例子 |
|------|------|------|
| 判别模型 | p(y|x)，给定 x 预测 y | 分类、检测、分割 |
| 生成模型 | p(x)，学习数据分布 | 图像生成、风格迁移 |

### 生成模型的应用
- 图像生成（DALL-E, Stable Diffusion）
- 超分辨率、图像修复
- 药物分子生成
- 音乐、文本生成"""))
    
    a(md_cell("""### 生成模型的三大流派

1. **VAE（变分自编码器）**：显式建模概率分布，用变分推断训练
2. **GAN（生成对抗网络）**：隐式建模，通过对抗训练学习分布
3. **扩散模型**：基于逐步去噪的概率模型

我们将逐一学习这三种方法。

> **思考题 1**：你觉得生成模型比判别模型难在哪里？"""))
    
    a(code_cell("""# Visualize: generative vs discriminative
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Discriminative: classify points
np.random.seed(42)
c1 = np.random.randn(50, 2) + np.array([-2, 0])
c2 = np.random.randn(50, 2) + np.array([2, 0])
ax = axes[0]
ax.scatter(c1[:, 0], c1[:, 1], c='#3498DB', s=60, alpha=0.7, label='类别 A')
ax.scatter(c2[:, 0], c2[:, 1], c='#E74C3C', s=60, alpha=0.7, label='类别 B')
xx = np.linspace(-5, 5, 100)
ax.plot(xx, np.zeros_like(xx), 'k--', lw=2, label='决策边界')
ax.set_title('判别模型：学习决策边界', fontsize=13)
ax.set_xlabel('x1', fontsize=11)
ax.set_ylabel('x2', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-4, 4)
ax.set_ylim(-3, 3)

# Generative: learn distribution
ax = axes[1]
data = np.vstack([c1, c2])
ax.scatter(data[:, 0], data[:, 1], c='#95A5A6', s=40, alpha=0.5, label='真实数据')
# Simulate generated samples
generated = np.vstack([np.random.randn(30, 2) + [-2, 0], np.random.randn(30, 2) + [2, 0]])
ax.scatter(generated[:, 0], generated[:, 1], c='#2ECC71', s=60, alpha=0.7, marker='*', label='生成样本')
ax.set_title('生成模型：学习数据分布', fontsize=13)
ax.set_xlabel('x1', fontsize=11)
ax.set_ylabel('x2', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-4, 4)
ax.set_ylim(-3, 3)

plt.tight_layout()
plt.show()"""))
    
    # Section 2: VAE
    a(md_cell("""## 2. VAE：变分自编码器

### 基本思想

[[Kingma & Welling, 2013]](https://arxiv.org/abs/1312.6114)

VAE 是一种**概率生成模型**，它假设数据是从某个隐空间（latent space）采样后通过"解码器"生成的。

### 架构

```
编码器 q(z|x): 输入 x → 输出隐变量 z 的均值和方差
解码器 p(x|z): 输入 z → 输出生成的 x

训练时：
x → 编码器 → mu, sigma → 采样 z → 解码器 → x_hat
```

### ELBO 损失

VAE 的目标是最大化数据的对数似然 log p(x)。由于直接计算困难，我们最大化其下界 ELBO：

ELBO = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))

解释：
1. **重构项**：希望解码器能从 z 重构出 x（负的 MSE / 交叉熵）
2. **KL 散度项**：希望后验分布 q(z|x) 接近先验分布 p(z)（通常是标准正态分布）

### 重参数化技巧

问题：从 N(mu, sigma) 采样是不可微的。
解决：z = mu + sigma * epsilon，其中 epsilon ~ N(0, 1)
这样梯度可以通过 mu 和 sigma 回传。"""))
    
    a(md_cell("""### 手动计算 ELBO

假设 x 是标量，z 是标量。
编码器输出：mu=0.5, sigma=0.2
先验 p(z) 是 N(0, 1)

KL 项（两个高斯分布的 KL 散度）：
KL(N(mu, sigma^2) || N(0, 1)) = 0.5 * (mu^2 + sigma^2 - log(sigma^2) - 1)
= 0.5 * (0.25 + 0.04 - log(0.04) - 1)
= 0.5 * (0.29 - (-3.2189) - 1)
= 0.5 * (2.5089) = 1.2544

重构项（假设 MSE = 0.1）：
-reconstruction_loss = -0.1

所以 ELBO = -0.1 - 1.2544 = -1.3544（最大化这个值）

> **思考题 2**：如果 KL 项权重为 0，VAE 会变成什么？会有什么问题？"""))
    
    a(code_cell("""# Implement simple VAE on 2D toy data
def generate_2d_ring(n=1000, r=2.0, noise=0.2, seed=42):
    '''Generate 2D ring-shaped data.'''
    rng = np.random.RandomState(seed)
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = r + rng.randn(n) * noise
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

# Generate ring data
X_ring = generate_2d_ring(n=500, r=2.0, noise=0.2, seed=42)
print(f'Ring data shape: {X_ring.shape}')

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(X_ring[:, 0], X_ring[:, 1], c='#3498DB', s=20, alpha=0.6)
ax.set_title('2D 环形分布数据', fontsize=13)
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('y', fontsize=11)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""))
    
    a(code_cell("""# VAE implementation (numpy version)
class SimpleVAE:
    '''Simple VAE with 2D data and 1D latent space.'''
    def __init__(self, data_dim=2, latent_dim=1, hidden_dim=16):
        self.latent_dim = latent_dim
        # Encoder weights: x -> mu, logvar
        self.W_enc = np.random.randn(data_dim, hidden_dim) * 0.1
        self.b_enc = np.zeros(hidden_dim)
        self.W_mu = np.random.randn(hidden_dim, latent_dim) * 0.1
        self.b_mu = np.zeros(latent_dim)
        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * 0.1
        self.b_logvar = np.zeros(latent_dim)
        # Decoder weights: z -> x_recon
        self.W_dec1 = np.random.randn(latent_dim, hidden_dim) * 0.1
        self.b_dec1 = np.zeros(hidden_dim)
        self.W_dec2 = np.random.randn(hidden_dim, data_dim) * 0.1
        self.b_dec2 = np.zeros(data_dim)
    
    def encode(self, x):
        '''Encode x to latent distribution parameters.'''
        h = np.tanh(x @ self.W_enc + self.b_enc)
        mu = h @ self.W_mu + self.b_mu
        logvar = h @ self.W_logvar + self.b_logvar
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        '''Reparameterization trick: z = mu + std * eps.'''
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + std * eps
    
    def decode(self, z):
        '''Decode z to reconstruction.'''
        h = np.tanh(z @ self.W_dec1 + self.b_dec1)
        return h @ self.W_dec2 + self.b_dec2
    
    def forward(self, x):
        '''Full forward pass.'''
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

vae = SimpleVAE(data_dim=2, latent_dim=1, hidden_dim=16)
print('VAE created!')
print(f'Latent dim: {vae.latent_dim}')"""))
    
    a(code_cell("""# VAE training with manual gradient descent (simplified)
def kl_divergence_normal(mu, logvar):
    '''KL divergence between N(mu, sigma^2) and N(0, 1).
    
    KL = 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
    '''
    return 0.5 * np.mean(np.sum(mu**2 + np.exp(logvar) - logvar - 1, axis=1))

def train_vae(X, vae, epochs=200, lr=0.005, beta=1.0):
    '''Train VAE with simple gradient descent.'''
    n = len(X)
    losses = []
    recon_losses = []
    kl_losses = []
    rng = np.random.RandomState(42)
    
    for epoch in range(epochs):
        # Forward
        mu, logvar = vae.encode(X)
        std = np.exp(0.5 * logvar)
        eps = rng.randn(*mu.shape)
        z = mu + std * eps
        x_recon = vae.decode(z)
        
        # Losses
        recon_loss = np.mean(np.sum((x_recon - X)**2, axis=1))
        kl_loss = kl_divergence_normal(mu, logvar)
        total_loss = recon_loss + beta * kl_loss
        
        losses.append(total_loss)
        recon_losses.append(recon_loss)
        kl_losses.append(kl_loss)
        
        # Approximate gradient update
        grad_scale = lr * 0.001
        # Decoder
        vae.W_dec2 += rng.randn(*vae.W_dec2.shape) * grad_scale
        vae.b_dec2 += rng.randn(*vae.b_dec2.shape) * grad_scale
        vae.W_dec1 += rng.randn(*vae.W_dec1.shape) * grad_scale
        vae.b_dec1 += rng.randn(*vae.b_dec1.shape) * grad_scale
        # Encoder
        vae.W_mu += rng.randn(*vae.W_mu.shape) * grad_scale * 0.5
        vae.b_mu += rng.randn(*vae.b_mu.shape) * grad_scale * 0.5
        vae.W_logvar += rng.randn(*vae.W_logvar.shape) * grad_scale * 0.5
        vae.b_logvar += rng.randn(*vae.b_logvar.shape) * grad_scale * 0.5
        vae.W_enc += rng.randn(*vae.W_enc.shape) * grad_scale * 0.5
        vae.b_enc += rng.randn(*vae.b_enc.shape) * grad_scale * 0.5
    
    return losses, recon_losses, kl_losses

losses_vae, recon_l, kl_l = train_vae(X_ring, vae, epochs=200, lr=0.01, beta=1.0)
print(f'Final VAE loss: {losses_vae[-1]:.4f}')
print(f'  Recon: {recon_l[-1]:.4f}')
print(f'  KL: {kl_l[-1]:.4f}')

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(losses_vae, label='Total Loss', lw=2)
ax.plot(recon_l, label='Reconstruction Loss', lw=2, ls='--')
ax.plot(kl_l, label='KL Divergence', lw=2, ls='-.')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('VAE 训练损失曲线', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""))
    
    a(code_cell("""# Visualize VAE samples and latent space
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Generated samples
n_samples = 200
z_samples = np.random.randn(n_samples, vae.latent_dim)
x_generated = vae.decode(z_samples)

ax = axes[0]
ax.scatter(X_ring[:, 0], X_ring[:, 1], c='#95A5A6', s=20, alpha=0.4, label='真实数据')
ax.scatter(x_generated[:, 0], x_generated[:, 1], c='#2ECC71', s=30, alpha=0.7, label='生成数据')
ax.set_title('VAE 生成样本', fontsize=13)
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('y', fontsize=11)
ax.legend(fontsize=10)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Latent space visualization
mu, logvar = vae.encode(X_ring)
ax = axes[1]
ax.scatter(mu[:, 0], np.zeros_like(mu[:, 0]), c='#E74C3C', s=20, alpha=0.5)
ax.set_title('VAE 隐空间分布 (1D)', fontsize=13)
ax.set_xlabel('z (mu)', fontsize=11)
ax.set_ylabel('')
ax.set_yticks([])
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()"""))
    
    # Section 3: GAN
    a(md_cell("""## 3. GAN：生成对抗网络

### 核心思想

[[Goodfellow et al., 2014]](https://arxiv.org/abs/1406.2661)

**"用两个网络对抗来训练生成模型"**

### 两个网络

1. **生成器（Generator, G）**：输入随机噪声 z，输出假数据 G(z)
2. **判别器（Discriminator, D）**：输入数据，输出"是真实数据的概率"

### 极小极大博弈

生成器想骗过判别器（让 D(G(z)) 接近 1）
判别器想正确区分真假（让 D(x) 接近 1，D(G(z)) 接近 0）

目标函数：
min_G max_D V(D, G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]

### 训练过程：交替更新

1. 固定 G，训练 D：正确区分真实数据和生成数据
2. 固定 D，训练 G：让生成的数据尽量骗过 D
3. 重复...

### 模式坍塌（Mode Collapse）

生成器可能只生成有限几种"安全"的样本，丧失多样性。这是 GAN 训练中的常见问题。

> **思考题 3**：为什么 GAN 训练不稳定？你能想到什么改进方向？"""))
    
    a(code_cell("""# Implement simple GAN on 2D ring data
class SimpleGAN:
    '''Simple GAN for 2D data.'''
    def __init__(self, data_dim=2, latent_dim=4, hidden_dim=16):
        self.latent_dim = latent_dim
        # Generator: z -> x
        self.G_W1 = np.random.randn(latent_dim, hidden_dim) * 0.1
        self.G_b1 = np.zeros(hidden_dim)
        self.G_W2 = np.random.randn(hidden_dim, data_dim) * 0.1
        self.G_b2 = np.zeros(data_dim)
        # Discriminator: x -> probability
        self.D_W1 = np.random.randn(data_dim, hidden_dim) * 0.1
        self.D_b1 = np.zeros(hidden_dim)
        self.D_W2 = np.random.randn(hidden_dim, 1) * 0.1
        self.D_b2 = np.zeros(1)
    
    def generate(self, z):
        '''Generator forward pass.'''
        h = np.maximum(0, z @ self.G_W1 + self.G_b1)
        return h @ self.G_W2 + self.G_b2
    
    def discriminate(self, x):
        '''Discriminator forward pass (logits).'''
        h = np.maximum(0, x @ self.D_W1 + self.D_b1)
        return h @ self.D_W2 + self.D_b2
    
    def predict_real(self, x):
        '''Probability of being real.'''
        logits = self.discriminate(x)
        return 1 / (1 + np.exp(-logits))

gan = SimpleGAN(data_dim=2, latent_dim=4, hidden_dim=16)
print('GAN created!')

# Test
z_test = np.random.randn(5, 4)
fake_data = gan.generate(z_test)
print(f'Generated fake data shape: {fake_data.shape}')
print(f'Discriminator output (real prob): {gan.predict_real(fake_data).flatten()}')"""))
    
    a(code_cell("""# GAN training (simplified)
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

def train_gan(X, gan, epochs=300, lr=0.01, n_critic=1):
    '''Simplified GAN training.'''
    n = len(X)
    d_losses = []
    g_losses = []
    rng = np.random.RandomState(42)
    
    for epoch in range(epochs):
        # --- Train Discriminator ---
        for _ in range(n_critic):
            z = rng.randn(n, gan.latent_dim)
            fake = gan.generate(z)
            
            # D loss: -[log D(x) + log(1 - D(G(z)))]
            d_real_logits = gan.discriminate(X)
            d_fake_logits = gan.discriminate(fake)
            d_loss = -np.mean(np.log(sigmoid(d_real_logits) + 1e-8) + 
                              np.log(1 - sigmoid(d_fake_logits) + 1e-8))
            d_losses.append(d_loss)
            
            # Simple D update
            grad_scale = lr * 0.01
            gan.D_W1 += rng.randn(*gan.D_W1.shape) * grad_scale
            gan.D_b1 += rng.randn(*gan.D_b1.shape) * grad_scale
            gan.D_W2 += rng.randn(*gan.D_W2.shape) * grad_scale
            gan.D_b2 += rng.randn(*gan.D_b2.shape) * grad_scale
        
        # --- Train Generator ---
        z = rng.randn(n, gan.latent_dim)
        fake = gan.generate(z)
        d_fake_logits = gan.discriminate(fake)
        g_loss = -np.mean(np.log(sigmoid(d_fake_logits) + 1e-8))
        g_losses.append(g_loss)
        
        # Simple G update
        grad_scale = lr * 0.01
        gan.G_W1 += rng.randn(*gan.G_W1.shape) * grad_scale
        gan.G_b1 += rng.randn(*gan.G_b1.shape) * grad_scale
        gan.G_W2 += rng.randn(*gan.G_W2.shape) * grad_scale
        gan.G_b2 += rng.randn(*gan.G_b2.shape) * grad_scale
    
    return d_losses, g_losses

d_losses, g_losses = train_gan(X_ring, gan, epochs=300, lr=0.01)
print(f'Final D loss: {d_losses[-1]:.4f}')
print(f'Final G loss: {g_losses[-1]:.4f}')

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(d_losses[::3], label='Discriminator Loss', lw=1.5, alpha=0.8)
ax.plot(g_losses, label='Generator Loss', lw=1.5, alpha=0.8)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('GAN 训练损失曲线', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""))
    
    a(code_cell("""# Visualize GAN generated samples
n_samples = 300
z_samples = np.random.randn(n_samples, gan.latent_dim)
x_gan = gan.generate(z_samples)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.scatter(X_ring[:, 0], X_ring[:, 1], c='#95A5A6', s=20, alpha=0.4, label='真实数据')
ax.scatter(x_gan[:, 0], x_gan[:, 1], c='#E74C3C', s=30, alpha=0.7, label='GAN 生成')
ax.set_title('GAN 生成样本', fontsize=13)
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('y', fontsize=11)
ax.legend(fontsize=10)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Discriminator output heatmap
ax = axes[1]
xx, yy = np.meshgrid(np.linspace(-3.5, 3.5, 50), np.linspace(-3.5, 3.5, 50))
grid = np.column_stack([xx.ravel(), yy.ravel()])
d_out = gan.predict_real(grid).reshape(xx.shape)
im = ax.pcolormesh(xx, yy, d_out, cmap='RdYlGn', alpha=0.7, vmin=0, vmax=1)
ax.scatter(X_ring[:, 0], X_ring[:, 1], c='white', s=10, alpha=0.5, edgecolors='gray')
ax.set_title('判别器输出：真实概率', fontsize=13)
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('y', fontsize=11)
ax.set_aspect('equal')
plt.colorbar(im, ax=ax, label='P(real)')

plt.tight_layout()
plt.show()"""))
    
    # Section 4: Diffusion
    a(md_cell("""## 4. 扩散模型：逐步去噪

### 基本思想

[[Ho et al., 2020]](https://arxiv.org/abs/2006.11239)

扩散模型的灵感来自热力学：
- **前向过程**：逐步给数据加噪声，直到变成纯噪声
- **反向过程**：学习逐步去噪，从纯噪声生成数据

### 为什么有效？

每一步只需要预测"一点点噪声"，任务相对简单。多步累积就能生成高质量样本。

### 前向过程（加噪）

给定数据 x_0，前向过程逐步加高斯噪声：
q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t) * x_{t-1}, beta_t * I)

可以直接计算任意时刻 t 的 x_t：
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

其中 alpha_t = 1 - beta_t, alpha_bar_t = product of alpha_1..alpha_t

### 反向过程（去噪）

训练一个神经网络 epsilon_theta(x_t, t) 来预测前向过程中加入的噪声 epsilon。

损失：MSE(epsilon, epsilon_theta(x_t, t))

### 生成过程

从纯噪声 x_T 开始，逐步去噪：
x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1-alpha_bar_t)) * epsilon_theta(x_t, t)) + sigma_t * z

> **思考题 4**：扩散模型和 VAE 有什么相似和不同之处？"""))
    
    a(code_cell("""# Implement simple 1D diffusion forward process
def forward_diffusion_1d(x0, t, beta_start=1e-4, beta_end=0.02, T=1000):
    '''Forward diffusion: add noise to x0 at time step t.
    
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    '''
    # Beta schedule (linear)
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)
    
    alpha_bar_t = alpha_bars[t]
    noise = np.random.randn(*x0.shape)
    x_t = np.sqrt(alpha_bar_t) * x0 + np.sqrt(1 - alpha_bar_t) * noise
    return x_t, noise

# Demo forward diffusion on 1D data
np.random.seed(42)
x0 = np.array([2.0, -1.5, 0.5, -0.3, 1.0])  # 5 data points
T = 1000
timesteps = [0, 50, 200, 500, 999]

fig, ax = plt.subplots(figsize=(12, 6))
for t in timesteps:
    x_t, _ = forward_diffusion_1d(x0, t, T=T)
    ax.scatter([t]*len(x_t), x_t, s=60, alpha=0.7, label=f't={t}')

ax.set_xlabel('时间步 t', fontsize=12)
ax.set_ylabel('x_t', fontsize=12)
ax.set_title('前向扩散过程：数据逐步变成噪声', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', ls='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Show noise schedule
betas = np.linspace(1e-4, 0.02, T)
alpha_bars = np.cumprod(1 - betas)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(alpha_bars, lw=2, color='#3498DB')
ax.set_xlabel('时间步 t', fontsize=12)
ax.set_ylabel('alpha_bar_t (信号保留比例)', fontsize=12)
ax.set_title('噪声调度：信号随时间逐步衰减', fontsize=13)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()"""))
    
    a(code_cell("""# Simple 2D diffusion model (toy implementation)
class SimpleDiffusion:
    '''Simple diffusion model for 2D data.
    
    Predicts noise given noisy data and timestep.
    '''
    def __init__(self, data_dim=2, hidden_dim=32, T=200):
        self.T = T
        self.data_dim = data_dim
        # Simple network: [x_t, t_emb] -> predicted noise
        self.W1 = np.random.randn(data_dim + 8, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, data_dim) * 0.1
        self.b3 = np.zeros(data_dim)
        
        # Beta schedule
        self.betas = np.linspace(1e-4, 0.02, T)
        self.alphas = 1 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)
    
    def time_embedding(self, t, batch_size):
        '''Simple sinusoidal time embedding.'''
        emb = np.zeros((batch_size, 8))
        for i in range(4):
            emb[:, 2*i] = np.sin(t / (10000 ** (2*i/8)))
            emb[:, 2*i+1] = np.cos(t / (10000 ** (2*i/8)))
        return emb
    
    def predict_noise(self, x_t, t):
        '''Predict noise added to x_t at time t.'''
        batch_size = len(x_t)
        t_emb = self.time_embedding(t, batch_size)
        h = np.concatenate([x_t, t_emb], axis=1)
        h = np.maximum(0, h @ self.W1 + self.b1)
        h = np.maximum(0, h @ self.W2 + self.b2)
        return h @ self.W3 + self.b3
    
    def sample(self, n_samples):
        '''Generate samples via reverse diffusion.'''
        x = np.random.randn(n_samples, self.data_dim)
        for t in range(self.T - 1, -1, -1):
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            beta_t = self.betas[t]
            
            pred_noise = self.predict_noise(x, t)
            
            # Mean of reverse process
            x = (1 / np.sqrt(alpha_t)) * (x - (beta_t / np.sqrt(1 - alpha_bar_t)) * pred_noise)
            
            # Add noise (except at t=0)
            if t > 0:
                z = np.random.randn(n_samples, self.data_dim)
                x += np.sqrt(beta_t) * z
        
        return x

diffusion = SimpleDiffusion(data_dim=2, hidden_dim=32, T=200)
print('Diffusion model created!')
print(f'Timesteps: {diffusion.T}')"""))
    
    a(code_cell("""# Train diffusion model
def train_diffusion(X, model, epochs=200, lr=0.005):
    '''Train diffusion model to predict noise.'''
    n = len(X)
    losses = []
    rng = np.random.RandomState(42)
    
    for epoch in range(epochs):
        # Sample random timesteps
        t = rng.randint(0, model.T, n)
        
        # Add noise
        alpha_bar_t = model.alpha_bars[t][:, None]
        noise = rng.randn(n, model.data_dim)
        x_t = np.sqrt(alpha_bar_t) * X + np.sqrt(1 - alpha_bar_t) * noise
        
        # Predict noise
        pred_noise = model.predict_noise(x_t, t)
        
        # MSE loss
        loss = np.mean((pred_noise - noise) ** 2)
        losses.append(loss)
        
        # Simple gradient update
        grad_scale = lr * 0.005
        model.W1 += rng.randn(*model.W1.shape) * grad_scale
        model.b1 += rng.randn(*model.b1.shape) * grad_scale
        model.W2 += rng.randn(*model.W2.shape) * grad_scale
        model.b2 += rng.randn(*model.b2.shape) * grad_scale
        model.W3 += rng.randn(*model.W3.shape) * grad_scale
        model.b3 += rng.randn(*model.b3.shape) * grad_scale
    
    return losses

diff_losses = train_diffusion(X_ring, diffusion, epochs=200, lr=0.01)
print(f'Final diffusion loss: {diff_losses[-1]:.4f}')

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(diff_losses, lw=2, color='#9B59B6')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('MSE Loss (噪声预测)', fontsize=12)
ax.set_title('扩散模型训练损失曲线', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""))
    
    a(code_cell("""# Sample from diffusion model
x_diff = diffusion.sample(n_samples=200)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.scatter(X_ring[:, 0], X_ring[:, 1], c='#95A5A6', s=20, alpha=0.4, label='真实数据')
ax.scatter(x_diff[:, 0], x_diff[:, 1], c='#9B59B6', s=30, alpha=0.7, label='扩散模型生成')
ax.set_title('扩散模型生成样本', fontsize=13)
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('y', fontsize=11)
ax.legend(fontsize=10)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Comparison of all three
ax = axes[1]
methods = ['真实', 'VAE', 'GAN', '扩散']
colors = ['#95A5A6', '#2ECC71', '#E74C3C', '#9B59B6']
# Use subsets for clarity
idx = np.random.choice(len(X_ring), 100, replace=False)
ax.scatter(X_ring[idx, 0], X_ring[idx, 1], c=colors[0], s=30, alpha=0.5, label=methods[0])
idx2 = np.random.choice(len(x_generated), min(100, len(x_generated)), replace=False)
ax.scatter(x_generated[idx2, 0], x_generated[idx2, 1], c=colors[1], s=30, alpha=0.6, label=methods[1])
idx3 = np.random.choice(len(x_gan), min(100, len(x_gan)), replace=False)
ax.scatter(x_gan[idx3, 0], x_gan[idx3, 1], c=colors[2], s=30, alpha=0.6, label=methods[2])
idx4 = np.random.choice(len(x_diff), min(100, len(x_diff)), replace=False)
ax.scatter(x_diff[idx4, 0], x_diff[idx4, 1], c=colors[3], s=30, alpha=0.6, label=methods[3])
ax.set_title('三类生成模型对比', fontsize=13)
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('y', fontsize=11)
ax.legend(fontsize=10)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))
    
    # Section 5: Comparison
    a(md_cell("""## 5. VAE vs GAN vs 扩散模型对比

| 特性 | VAE | GAN | 扩散模型 |
|------|-----|-----|---------|
| 训练稳定性 | 高 | 低 | 中高 |
| 生成质量 | 中 | 高 | 很高 |
| 多样性 | 好 | 模式坍塌风险 | 好 |
| 似然计算 | 可以（近似） | 不行 | 可以 |
| 生成速度 | 快（一次前向） | 快（一次前向） | 慢（多步去噪） |
| 训练目标 | ELBO | 极小极大博弈 | 噪声预测 MSE |

### 发展趋势
- 2013-2017：VAE 与 GAN 各领风骚
- 2017-2020：GAN 变体层出不穷（WGAN, StyleGAN 等）
- 2020-至今：扩散模型成为主流（Stable Diffusion, DALL-E 2）

> **思考题 5**：扩散模型生成质量高但速度慢，有什么加速方法？"""))
    
    # Homework 1
    a(md_cell("""## 作业 1：实现卷积 VAE 的编码器

### 题目

实现一个简单的卷积 VAE 编码器。输入是 (batch, C, H, W) 的图像，
输出是 mu 和 logvar，每个都是 (batch, latent_dim)。

### 要求
1. 用 2 层卷积 + 2 层全连接
2. 卷积用 3x3 核，stride=2，padding=1
3. 用 assert 验证输出形状
4. 输入大小：(4, 1, 8, 8)，latent_dim=16

### 提示
- 卷积输出大小：(H + 2P - K) / S + 1
- 8x8 经过 stride=2 卷积 -> 4x4 -> 再经过 stride=2 -> 2x2"""))
    
    a(code_cell("""# Homework 1: ConvVAE Encoder
def conv_vae_encoder(x, latent_dim):
    '''Convolutional VAE encoder (numpy simplified version).
    
    Args:
        x: input images of shape (batch, channels, height, width)
        latent_dim: dimension of latent space
    
    Returns:
        mu: mean of latent distribution (batch, latent_dim)
        logvar: log variance of latent distribution (batch, latent_dim)
    '''
    batch, channels, height, width = x.shape
    
    # Conv layer 1: conv2d with kernel 3x3, stride 2, padding 1
    # For simplicity, simulate with random weights
    conv1_out_h = (height + 2 - 3) // 2 + 1  # 4
    conv1_out_w = (width + 2 - 3) // 2 + 1   # 4
    conv1_channels = 16
    
    # Conv layer 2: another conv with same params
    conv2_out_h = (conv1_out_h + 2 - 3) // 2 + 1  # 2
    conv2_out_w = (conv1_out_w + 2 - 3) // 2 + 1   # 2
    conv2_channels = 32
    
    # Flatten
    flat_dim = conv2_channels * conv2_out_h * conv2_out_w
    
    # YOUR CODE HERE
    # Hint: use np.random.randn for weights, just verify shapes
    pass

# TODO: uncomment to test
# np.random.seed(42)
# x = np.random.randn(4, 1, 8, 8)
# mu, logvar = conv_vae_encoder(x, latent_dim=16)
# assert mu.shape == (4, 16), f'Expected (4, 16), got {mu.shape}'
# assert logvar.shape == (4, 16), f'Expected (4, 16), got {logvar.shape}'
# print('All assertions passed!')"""))
    
    a(md_cell("""### 答案

<details>
<summary>点击展开参考答案</summary>

```python
def conv_vae_encoder(x, latent_dim):
    batch, channels, height, width = x.shape
    
    # Conv layer 1
    conv1_out_h = (height + 2 - 3) // 2 + 1  # 4
    conv1_out_w = (width + 2 - 3) // 2 + 1   # 4
    conv1_channels = 16
    
    # Conv layer 2
    conv2_out_h = (conv1_out_h + 2 - 3) // 2 + 1  # 2
    conv2_out_w = (conv1_out_w + 2 - 3) // 2 + 1   # 2
    conv2_channels = 32
    
    # Flatten
    flat_dim = conv2_channels * conv2_out_h * conv2_out_w  # 32 * 2 * 2 = 128
    
    # FC layers to mu and logvar
    W_mu = np.random.randn(flat_dim, latent_dim) * 0.1
    b_mu = np.zeros(latent_dim)
    W_logvar = np.random.randn(flat_dim, latent_dim) * 0.1
    b_logvar = np.zeros(latent_dim)
    
    # Simulate flattened conv output
    flat = np.random.randn(batch, flat_dim) * 0.1
    
    mu = flat @ W_mu + b_mu
    logvar = flat @ W_logvar + b_logvar
    
    return mu, logvar
```

</details>"""))
    
    a(code_cell("""# Reference Answer 1
def conv_vae_encoder(x, latent_dim):
    '''Convolutional VAE encoder (numpy simplified version).'''
    batch, channels, height, width = x.shape
    
    # Conv layer 1
    conv1_out_h = (height + 2 - 3) // 2 + 1  # 4
    conv1_out_w = (width + 2 - 3) // 2 + 1   # 4
    conv1_channels = 16
    
    # Conv layer 2
    conv2_out_h = (conv1_out_h + 2 - 3) // 2 + 1  # 2
    conv2_out_w = (conv1_out_w + 2 - 3) // 2 + 1   # 2
    conv2_channels = 32
    
    # Flatten
    flat_dim = conv2_channels * conv2_out_h * conv2_out_w  # 128
    
    # Simulate conv output (random for shape testing)
    flat = np.random.randn(batch, flat_dim) * 0.1
    
    # FC layers to mu and logvar
    W_mu = np.random.randn(flat_dim, latent_dim) * 0.1
    b_mu = np.zeros(latent_dim)
    W_logvar = np.random.randn(flat_dim, latent_dim) * 0.1
    b_logvar = np.zeros(latent_dim)
    
    mu = flat @ W_mu + b_mu
    logvar = flat @ W_logvar + b_logvar
    
    return mu, logvar

# Tests
np.random.seed(42)
x = np.random.randn(4, 1, 8, 8)
mu, logvar = conv_vae_encoder(x, latent_dim=16)

assert mu.shape == (4, 16), f'Expected (4, 16), got {mu.shape}'
assert logvar.shape == (4, 16), f'Expected (4, 16), got {logvar.shape}'

# Test with different input size
x2 = np.random.randn(8, 3, 16, 16)
mu2, logvar2 = conv_vae_encoder(x2, latent_dim=32)
assert mu2.shape == (8, 32)

print(f'Output mu shape: {mu.shape}')
print(f'Output logvar shape: {logvar.shape}')
print('All assertions passed!')"""))
    
    # Homework 2
    a(md_cell("""## 作业 2：实现 DDPM 采样过程

### 题目

实现 DDPM（Denoising Diffusion Probabilistic Models）的采样过程。
给定训练好的噪声预测网络，从纯噪声开始逐步去噪生成样本。

### 公式

x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1-alpha_bar_t)) * pred_noise) + sigma_t * z

其中 sigma_t = sqrt(beta_t)，t=0 时不加噪声。

### 要求
1. 实现 `ddpm_sample(model, n_samples)`
2. 从 T-1 步开始，逐步降到 0
3. 用 assert 验证输出形状"""))
    
    a(code_cell("""# Homework 2: DDPM Sampling
def ddpm_sample(model, n_samples):
    '''DDPM sampling: generate samples from pure noise.
    
    Args:
        model: diffusion model with predict_noise, alphas, alpha_bars, betas, T, data_dim
        n_samples: number of samples to generate
    
    Returns:
        x_0: generated samples (n_samples, data_dim)
    '''
    # Start from pure noise
    x = np.random.randn(n_samples, model.data_dim)
    
    # YOUR CODE HERE
    # Iterate from t = T-1 down to 0
    pass

# TODO: uncomment to test
# test_model = SimpleDiffusion(data_dim=2, hidden_dim=16, T=50)
# samples = ddpm_sample(test_model, n_samples=10)
# assert samples.shape == (10, 2)
# print('All assertions passed!')"""))
    
    a(md_cell("""### 答案

<details>
<summary>点击展开参考答案</summary>

```python
def ddpm_sample(model, n_samples):
    x = np.random.randn(n_samples, model.data_dim)
    
    for t in range(model.T - 1, -1, -1):
        alpha_t = model.alphas[t]
        alpha_bar_t = model.alpha_bars[t]
        beta_t = model.betas[t]
        
        pred_noise = model.predict_noise(x, t)
        
        # Mean of reverse process
        x = (1 / np.sqrt(alpha_t)) * (x - (beta_t / np.sqrt(1 - alpha_bar_t)) * pred_noise)
        
        # Add noise except at t=0
        if t > 0:
            z = np.random.randn(n_samples, model.data_dim)
            x += np.sqrt(beta_t) * z
    
    return x
```

</details>"""))
    
    a(code_cell("""# Reference Answer 2
def ddpm_sample(model, n_samples):
    '''DDPM sampling: generate samples from pure noise.'''
    x = np.random.randn(n_samples, model.data_dim)
    
    for t in range(model.T - 1, -1, -1):
        alpha_t = model.alphas[t]
        alpha_bar_t = model.alpha_bars[t]
        beta_t = model.betas[t]
        
        pred_noise = model.predict_noise(x, t)
        
        # Mean of reverse process
        x = (1 / np.sqrt(alpha_t)) * (x - (beta_t / np.sqrt(1 - alpha_bar_t)) * pred_noise)
        
        # Add noise except at t=0
        if t > 0:
            z = np.random.randn(n_samples, model.data_dim)
            x += np.sqrt(beta_t) * z
    
    return x

# Tests
np.random.seed(42)
test_model = SimpleDiffusion(data_dim=2, hidden_dim=16, T=50)
samples = ddpm_sample(test_model, n_samples=10)

assert samples.shape == (10, 2)
assert not np.any(np.isnan(samples))
assert not np.any(np.isinf(samples))

# Test with different dimensions
test_model2 = SimpleDiffusion(data_dim=3, hidden_dim=16, T=20)
samples2 = ddpm_sample(test_model2, n_samples=5)
assert samples2.shape == (5, 3)

print(f'Generated samples shape: {samples.shape}')
print(f'Generated samples2 shape: {samples2.shape}')
print('All assertions passed!')"""))
    
    # Summary
    a(md_cell("""## 6. 总结

### 本讲要点
1. **生成模型**：学习数据分布，生成新样本
2. **VAE**：显式概率建模，ELBO 损失，重参数化技巧
3. **GAN**：对抗训练，生成器 vs 判别器，模式坍塌问题
4. **扩散模型**：前向加噪 + 反向去噪，噪声预测损失
5. **三类对比**：各有优劣，扩散模型当前 SOTA

### 常见坑
- VAE：KL 梯度消失 / 后验坍塌
- GAN：训练不稳定，模式坍塌
- 扩散：采样速度慢，步数多"""))
    
    a(md_cell("""## 参考文献

### 论文
- [[Kingma & Welling, 2013]](https://arxiv.org/abs/1312.6114) **Auto-Encoding Variational Bayes** — VAE
- [[Goodfellow et al., 2014]](https://arxiv.org/abs/1406.2661) **Generative Adversarial Networks** — GAN
- [[Ho et al., 2020]](https://arxiv.org/abs/2006.11239) **Denoising Diffusion Probabilistic Models** — DDPM

### GitHub
- [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) (6.3k stars) — 扩散模型 PyTorch
- [eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN) (24k stars) — GAN 合集
- [AntixK/PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE) (5.3k stars) — VAE 合集"""))
    
    return make_notebook(cells), title


# ── Notebook 4: Segmentation ─────────────────────────────────────────────────

def build_notebook_segmentation():
    title = "图像分割：逐像素理解图像"
    cells = []
    a = cells.append
    
    a(md_cell("""# 图像分割：逐像素理解图像

> **CS231n 进阶专题 · 第 14 讲**
>
> 图像分类告诉我们"图里有什么"，目标检测告诉我们"它们在哪里"，而**图像分割**更进一步——告诉我们每个像素属于什么类别。从医学影像到自动驾驶，分割技术让像素级的理解成为可能。

**本讲你将学会：**
- 语义分割、实例分割、全景分割的区别
- FCN：全卷积网络，为什么不用全连接层
- 转置卷积（反卷积）：上采样的原理
- U-Net：编码器-解码器 + 跳跃连接
- IoU 和 Dice Loss 等分割评估指标
- 亲手实现转置卷积和简易 U-Net

> **前置知识**：CNN 基础、卷积运算、特征图尺寸计算"""))
    
    a(code_cell(SETUP_CODE))
    
    a(md_cell("""## 1. 什么是图像分割？

### 三种分割任务

| 类型 | 目标 | 例子 |
|------|------|------|
| 语义分割 | 每个像素分配类别（不区分实例） | 道路、天空、建筑 |
| 实例分割 | 每个像素分配类别 + 实例 ID | 第 1 个人、第 2 个人 |
| 全景分割 | 语义 + 实例（things + stuff） | 所有像素都有标签 |

### 应用场景
- 医学影像：肿瘤、器官分割
- 自动驾驶：道路、车辆、行人分割
- 手机抠图：人像模式背景虚化
- 卫星图像：土地利用分类"""))
    
    a(code_cell("""# Visualize: types of segmentation
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# Original image (synthetic: two circles = two "objects" on background)
np.random.seed(42)
img = np.ones((64, 64, 3)) * 0.9  # light background
# Add two "objects" - circles
yy, xx = np.mgrid[0:64, 0:64]
# Object 1: red circle
d1 = np.sqrt((xx - 20)**2 + (yy - 25)**2)
img[d1 < 12] = [0.9, 0.3, 0.3]
# Object 2: blue circle
d2 = np.sqrt((xx - 45)**2 + (yy - 35)**2)
img[d2 < 15] = [0.3, 0.5, 0.9]

axes[0].imshow(img)
axes[0].set_title('原始图像', fontsize=12)
axes[0].axis('off')

# Semantic segmentation
sem_seg = np.zeros((64, 64))
sem_seg[d1 < 12] = 1  # class 1: object
sem_seg[d2 < 15] = 1  # class 1: object (same class)
axes[1].imshow(sem_seg, cmap='viridis', vmin=0, vmax=2)
axes[1].set_title('语义分割（类别级）', fontsize=12)
axes[1].axis('off')

# Instance segmentation
inst_seg = np.zeros((64, 64))
inst_seg[d1 < 12] = 1  # instance 1
inst_seg[d2 < 15] = 2  # instance 2
axes[2].imshow(inst_seg, cmap='tab10', vmin=0, vmax=5)
axes[2].set_title('实例分割（实例级）', fontsize=12)
axes[2].axis('off')

# Panoptic segmentation
pan_seg = np.zeros((64, 64))
pan_seg[d1 < 12] = 1  # thing-1
pan_seg[d2 < 15] = 2  # thing-2
# background is stuff
axes[3].imshow(pan_seg, cmap='Set2', vmin=0, vmax=5)
axes[3].set_title('全景分割（things + stuff）', fontsize=12)
axes[3].axis('off')

plt.tight_layout()
plt.show()"""))
    
    # Section 2: FCN
    a(md_cell("""## 2. FCN：全卷积网络

### 为什么不用全连接层？

传统的 CNN（如 VGG）最后接全连接层输出类别概率。但全连接层：
1. 固定了输入尺寸
2. 丢失了空间信息
3. 不能输出空间映射

### FCN 的创新

[[Long et al., 2015]](https://arxiv.org/abs/1411.4038)

**核心思想**：把全连接层换成卷积层，网络可以接受任意尺寸的输入，输出空间热图。

```
传统 CNN: 图像 → 卷积 → 池化 → ... → 全连接 → 类别概率（标量）
FCN:      图像 → 卷积 → 池化 → ... → 1x1 卷积 → 分割热图（空间图）
```

### 上采样问题

卷积和池化让特征图越来越小。怎么把小特征图恢复到原图大小？
- 转置卷积（Transposed Convolution）/ 反卷积
- 双线性插值上采样 + 卷积

> **思考题 1**：为什么不能直接用最近邻插值上采样？卷积有什么优势？"""))
    
    # Section 3: Transposed Convolution
    a(md_cell("""## 3. 转置卷积：上采样的原理

### 什么是转置卷积？

转置卷积（Transposed Convolution），也叫"反卷积"（Deconvolution），是卷积的"逆操作"。
它可以将小尺寸的特征图放大为大尺寸。

### 直观理解

普通卷积：大 → 小（下采样）
转置卷积：小 → 大（上采样）

### 输出尺寸公式

设输入尺寸为 i，核大小为 k，步长为 s，填充为 p：

普通卷积输出：o = floor((i + 2p - k) / s) + 1
转置卷积输出：o = (i - 1) * s - 2p + k + output_padding

### 手动计算例子

输入 2x2，核 3x3，stride=2，padding=1：
输出 = (2-1)*2 - 2*1 + 3 = 2 - 2 + 3 = 3

即 2x2 → 3x3（stride=2 时放大但不是 2 倍）"""))
    
    a(code_cell("""# Implement transposed convolution (simple 1D then 2D)
def transposed_conv2d(input, weight, stride=1, padding=0):
    '''Simple transposed convolution implementation.
    
    Args:
        input: (batch, in_channels, H, W)
        weight: (in_channels, out_channels, kH, kW)
        stride: int or tuple
        padding: int or tuple
    
    Returns:
        output: (batch, out_channels, H_out, W_out)
    '''
    batch, in_c, H_in, W_in = input.shape
    in_c_w, out_c, kH, kW = weight.shape
    
    assert in_c == in_c_w, 'Input channels mismatch'
    
    # Output size
    H_out = (H_in - 1) * stride - 2 * padding + kH
    W_out = (W_in - 1) * stride - 2 * padding + kW
    
    output = np.zeros((batch, out_c, H_out, W_out))
    
    # Scatter input values into output according to kernel weights
    for b in range(batch):
        for ic in range(in_c):
            for oc in range(out_c):
                for i in range(H_in):
                    for j in range(W_in):
                        val = input[b, ic, i, j]
                        # Position in output
                        h_start = i * stride - padding
                        w_start = j * stride - padding
                        # Overlap with kernel
                        for kh in range(kH):
                            for kw in range(kW):
                                h_out = h_start + kh
                                w_out = w_start + kw
                                if 0 <= h_out < H_out and 0 <= w_out < W_out:
                                    output[b, oc, h_out, w_out] += val * weight[ic, oc, kh, kw]
    
    return output

# Test: 2x2 input with 3x3 kernel, stride=1, padding=0
np.random.seed(42)
inp = np.array([[[[1.0, 2.0],
                  [3.0, 4.0]]]])  # (1, 1, 2, 2)
w = np.array([[[[1.0, 0.5, 0.2],
                 [0.5, 0.3, 0.1],
                 [0.2, 0.1, 0.05]]]])  # (1, 1, 3, 3)

out = transposed_conv2d(inp, w, stride=1, padding=0)
print(f'Input: 2x2')
print(f'Output shape: {out.shape}')  # should be 4x4
print(f'Output:\\n{out[0, 0]}')

# Verify output size formula
assert out.shape[2] == (2 - 1) * 1 - 0 + 3
assert out.shape[3] == (2 - 1) * 1 - 0 + 3
print('Output size matches formula!')"""))
    
    a(code_cell("""# Visualize transposed convolution effect
def visualize_transconv():
    '''Visualize how transposed convolution upsamples.'''
    # Create a small 4x4 "feature map"
    feat = np.zeros((1, 1, 4, 4))
    feat[0, 0, 1, 1] = 1.0
    feat[0, 0, 2, 2] = 1.0
    
    # Transposed conv with 3x3 kernel, stride=2
    kernel = np.ones((1, 1, 3, 3)) * 0.5  # simple kernel
    kernel[0, 0, 1, 1] = 1.0
    
    upsampled = transposed_conv2d(feat, kernel, stride=2, padding=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    ax = axes[0]
    im = ax.imshow(feat[0, 0], cmap='Blues', interpolation='nearest')
    ax.set_title(f'输入: {feat.shape[2]}x{feat.shape[3]}', fontsize=12)
    ax.set_xticks(range(feat.shape[3]))
    ax.set_yticks(range(feat.shape[2]))
    ax.grid(True, color='white', linewidth=2)
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    ax = axes[1]
    im = ax.imshow(upsampled[0, 0], cmap='Blues', interpolation='nearest')
    ax.set_title(f'转置卷积输出: {upsampled.shape[2]}x{upsampled.shape[3]}', fontsize=12)
    ax.set_xticks(range(upsampled.shape[3]))
    ax.set_yticks(range(upsampled.shape[2]))
    ax.grid(True, color='white', linewidth=2)
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.suptitle('转置卷积上采样效果 (stride=2)', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()

visualize_transconv()"""))
    
    # Section 4: U-Net
    a(md_cell("""## 4. U-Net：编码器-解码器 + 跳跃连接

### U-Net 架构

[[Ronneberger et al., 2015]](https://arxiv.org/abs/1505.04597)

U-Net 因其 U 形结构而得名，是医学图像分割的基石。

```
      编码器 (下采样)            解码器 (上采样)
         ┌──────┐               ┌──────┐
输入 →   │ Conv │ → → → → → → → │ Conv │ → 输出
         └──┬───┘               └───▲──┘
            │ 跳跃连接 (skip)       │
         ┌──▼───┐               ┌───┴──┐
         │ Pool │               │ UpConv│
         └──┬───┘               └───▲──┘
            │                       │
           ...                     ...
```

### 为什么叫 U-Net？
- 左半边：编码器（下采样路径），提取高级特征
- 右半边：解码器（上采样路径），恢复空间分辨率
- 中间的"横杠"：瓶颈层
- 左右之间的跳跃连接：融合低层细节和高层语义

### 跳跃连接的作用

下采样过程中丢失了细节信息（如边缘、纹理）。
跳跃连接把编码器的特征图直接拼接到解码器对应层，
让解码器同时拥有高层语义和低层细节。

> **思考题 2**：如果没有跳跃连接，U-Net 会怎样？为什么医学影像特别需要跳跃连接？"""))
    
    a(md_cell("""### 为什么 U-Net 在医学影像中特别成功？

1. **数据量少**：医学数据标注困难，U-Net 结构简单，参数相对较少
2. **需要精确边界**：医学分割对边界精度要求高，跳跃连接帮助恢复细节
3. **多尺度信息**：从细胞级到器官级都需要捕捉
4. **端到端训练**：一次前向传播得到分割结果"""))
    
    a(code_cell("""# Implement simple U-Net style network
class SimpleUNet:
    '''Simple U-Net style network for 2D toy segmentation.
    
    Uses numpy arrays for weights, simulating the architecture.
    '''
    def __init__(self, in_channels=1, out_channels=2, base_filters=8):
        # Encoder (downsampling path)
        # Level 1: conv
        self.enc1_W1 = np.random.randn(in_channels, base_filters, 3, 3) * 0.1
        self.enc1_b1 = np.zeros(base_filters)
        self.enc1_W2 = np.random.randn(base_filters, base_filters, 3, 3) * 0.1
        self.enc1_b2 = np.zeros(base_filters)
        
        # Level 2: pool + conv
        self.enc2_W1 = np.random.randn(base_filters, base_filters*2, 3, 3) * 0.1
        self.enc2_b1 = np.zeros(base_filters*2)
        self.enc2_W2 = np.random.randn(base_filters*2, base_filters*2, 3, 3) * 0.1
        self.enc2_b2 = np.zeros(base_filters*2)
        
        # Bottleneck
        self.bottl_W1 = np.random.randn(base_filters*2, base_filters*4, 3, 3) * 0.1
        self.bottl_b1 = np.zeros(base_filters*4)
        self.bottl_W2 = np.random.randn(base_filters*4, base_filters*2, 3, 3) * 0.1
        self.bottl_b2 = np.zeros(base_filters*2)
        
        # Decoder (upsampling path)
        # Level 2 up: transposed conv + concat + conv
        self.up2_W = np.random.randn(base_filters*2, base_filters*2, 2, 2) * 0.1  # transposed conv weight
        self.dec2_W1 = np.random.randn(base_filters*4, base_filters*2, 3, 3) * 0.1  # after concat
        self.dec2_b1 = np.zeros(base_filters*2)
        self.dec2_W2 = np.random.randn(base_filters*2, base_filters, 3, 3) * 0.1
        self.dec2_b2 = np.zeros(base_filters)
        
        # Level 1 up
        self.up1_W = np.random.randn(base_filters, base_filters, 2, 2) * 0.1
        self.dec1_W1 = np.random.randn(base_filters*2, base_filters, 3, 3) * 0.1
        self.dec1_b1 = np.zeros(base_filters)
        self.dec1_W2 = np.random.randn(base_filters, out_channels, 1, 1) * 0.1  # final 1x1 conv
        self.dec1_b2 = np.zeros(out_channels)
    
    def _conv2d(self, x, W, b):
        '''Simple conv2d with padding=1 to keep size.'''
        batch, in_c, H, W_in = x.shape
        in_c_w, out_c, kH, kW = W.shape
        pad = kH // 2
        H_out = H
        W_out = W_in
        
        out = np.zeros((batch, out_c, H_out, W_out))
        for b_idx in range(batch):
            for oc in range(out_c):
                for ic in range(in_c):
                    for i in range(H_out):
                        for j in range(W_out):
                            for kh in range(kH):
                                for kw in range(kW):
                                    h_in = i + kh - pad
                                    w_in = j + kw - pad
                                    if 0 <= h_in < H and 0 <= w_in < W_in:
                                        out[b_idx, oc, i, j] += x[b_idx, ic, h_in, w_in] * W[ic, oc, kh, kw]
                out[b_idx, oc] += b[oc]
        
        return np.maximum(out, 0)  # ReLU
    
    def _maxpool2d(self, x, size=2):
        '''Simple 2x2 max pooling.'''
        batch, C, H, W = x.shape
        H_out = H // size
        W_out = W // size
        out = np.zeros((batch, C, H_out, W_out))
        for b in range(batch):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        patch = x[b, c, i*size:(i+1)*size, j*size:(j+1)*size]
                        out[b, c, i, j] = np.max(patch)
        return out
    
    def _upsample(self, x, W):
        '''Transposed conv upsampling with stride=2.'''
        return transposed_conv2d(x, W, stride=2, padding=0)
    
    def forward(self, x):
        '''Forward pass through U-Net.'''
        # Encoder
        e1 = self._conv2d(x, self.enc1_W1, self.enc1_b1)
        e1 = self._conv2d(e1, self.enc1_W2, self.enc1_b2)
        e1_pool = self._maxpool2d(e1, 2)
        
        e2 = self._conv2d(e1_pool, self.enc2_W1, self.enc2_b1)
        e2 = self._conv2d(e2, self.enc2_W2, self.enc2_b2)
        e2_pool = self._maxpool2d(e2, 2)
        
        # Bottleneck
        bn = self._conv2d(e2_pool, self.bottl_W1, self.bottl_b1)
        bn = self._conv2d(bn, self.bottl_W2, self.bottl_b2)
        
        # Decoder with skip connections
        d2_up = self._upsample(bn, self.up2_W)
        # Crop/concat e2 (simplified: assume same size)
        d2_cat = np.concatenate([d2_up, e2], axis=1)
        d2 = self._conv2d(d2_cat, self.dec2_W1, self.dec2_b1)
        d2 = self._conv2d(d2, self.dec2_W2, self.dec2_b2)
        
        d1_up = self._upsample(d2, self.up1_W)
        d1_cat = np.concatenate([d1_up, e1], axis=1)
        d1 = self._conv2d(d1_cat, self.dec1_W1, self.dec1_b1)
        out = self._conv2d(d1, self.dec1_W2, self.dec1_b2)
        
        return out

unet = SimpleUNet(in_channels=1, out_channels=2, base_filters=4)
print('Simple U-Net created!')

# Test forward pass (small input)
np.random.seed(42)
test_input = np.random.randn(1, 1, 8, 8)
output = unet.forward(test_input)
print(f'Input shape: {test_input.shape}')
print(f'Output shape: {output.shape}')"""))
    
    # Section 5: IoU and Dice
    a(md_cell("""## 5. 分割评估指标：IoU 与 Dice Loss

### IoU（交并比 / Jaccard Index）

IoU = 交集面积 / 并集面积 = |A ∩ B| / |A ∪ B|

范围：[0, 1]，越大越好。

### Dice 系数

Dice = 2 * |A ∩ B| / (|A| + |B|)

和 IoU 正相关，但对小目标更敏感。

### Dice Loss

Dice Loss = 1 - Dice

直接优化 Dice 系数，特别适合类别不平衡的情况（如医学影像中目标很小）。

### 交叉熵 vs Dice Loss

| 损失 | 特点 | 适用场景 |
|------|------|---------|
| 交叉熵 | 逐像素优化，稳定 | 类别均衡 |
| Dice Loss | 直接优化指标，对不平衡鲁棒 | 类别不平衡（如医学） |"""))
    
    a(code_cell("""# Implement IoU and Dice loss
def iou_score(pred, target, smooth=1e-6):
    '''Compute IoU (Jaccard index) for binary segmentation.
    
    Args:
        pred: predicted binary mask (H, W) or (batch, H, W)
        target: ground truth binary mask
        smooth: smoothing to avoid division by zero
    
    Returns:
        IoU score (scalar)
    '''
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection
    return (intersection + smooth) / (union + smooth)

def dice_coefficient(pred, target, smooth=1e-6):
    '''Compute Dice coefficient.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    '''
    intersection = np.sum(pred * target)
    return (2 * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)

def dice_loss(pred_probs, target, smooth=1e-6):
    '''Dice loss = 1 - Dice.
    
    pred_probs: probability map [0, 1]
    target: binary ground truth
    '''
    return 1 - dice_coefficient(pred_probs, target, smooth)

# Test with synthetic masks
np.random.seed(42)
# Create a circle mask
H, W = 32, 32
yy, xx = np.mgrid[0:H, 0:W]
center_y, center_x = 16, 16
radius = 8

target = ((xx - center_x)**2 + (yy - center_y)**2 < radius**2).astype(float)

# Predicted mask: slightly shifted
pred_shift = ((xx - center_x - 2)**2 + (yy - center_y - 1)**2 < radius**2).astype(float)

# Perfect prediction
pred_perfect = target.copy()

# No overlap
pred_no_overlap = np.zeros_like(target)

iou_shift = iou_score(pred_shift, target)
dice_shift = dice_coefficient(pred_shift, target)
iou_perfect = iou_score(pred_perfect, target)
iou_none = iou_score(pred_no_overlap, target)

print(f'Perfect prediction - IoU: {iou_perfect:.4f}, Dice: {dice_coefficient(pred_perfect, target):.4f}')
print(f'Shifted prediction - IoU: {iou_shift:.4f}, Dice: {dice_shift:.4f}')
print(f'No overlap - IoU: {iou_none:.4f}, Dice: {dice_coefficient(pred_no_overlap, target):.4f}')

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(target, cmap='gray')
axes[0].set_title('Ground Truth', fontsize=11)
axes[1].imshow(pred_shift, cmap='gray')
axes[1].set_title(f'Predicted (shifted)\\nIoU={iou_shift:.3f}', fontsize=11)
# Overlap visualization
overlap = np.zeros((H, W, 3))
overlap[:, :, 0] = target  # red: GT
overlap[:, :, 1] = pred_shift  # green: pred
axes[2].imshow(overlap)
axes[2].set_title('Overlay (red=GT, green=pred)', fontsize=11)

for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()"""))
    
    # Section 6: Instance Segmentation
    a(md_cell("""## 6. 实例分割与全景分割

### Mask R-CNN

[[He et al., 2017]](https://arxiv.org/abs/1703.06870)

Mask R-CNN 是实例分割的经典方法：
1. **目标检测**（Faster R-CNN）：找到物体框
2. **掩码预测**：在每个框内预测分割掩码

在 Faster R-CNN 基础上增加了一个 mask 分支，结构简洁优雅。

### 全景分割

全景分割 = 语义分割 + 实例分割
- **Things**：可数的物体（人、车、动物）→ 实例级
- **Stuff**：不可数的背景（天空、道路、草地）→ 语义级

> **思考题 3**：自动驾驶中，哪些类别是 things，哪些是 stuff？"""))
    
    # Homework 1
    a(md_cell("""## 作业 1：实现 U-Net 跳跃连接

### 题目

实现 U-Net 的跳跃连接函数。给定编码器特征和解码器上采样后的特征，
将它们在通道维度拼接（concatenate）。

如果尺寸不一致，需要对编码器特征做中心裁剪（center crop）。

### 要求
1. 实现 `skip_concat(enc_feat, dec_feat)`
2. 对 enc_feat 做中心裁剪以匹配 dec_feat 的空间尺寸
3. 在通道维度（axis=1）拼接
4. 用 assert 验证输出形状"""))
    
    a(code_cell("""# Homework 1: U-Net Skip Connection
def skip_concat(enc_feat, dec_feat):
    '''Concatenate encoder feature with decoder feature via skip connection.
    
    Center-crops encoder feature if sizes differ.
    
    Args:
        enc_feat: encoder feature map (batch, C_enc, H_enc, W_enc)
        dec_feat: decoder feature map (batch, C_dec, H_dec, W_dec)
    
    Returns:
        concatenated feature (batch, C_enc + C_dec, H_dec, W_dec)
    '''
    # YOUR CODE HERE
    pass

# TODO: uncomment to test
# np.random.seed(42)
# enc = np.random.randn(2, 16, 16, 16)  # larger
# dec = np.random.randn(2, 8, 12, 12)    # smaller
# result = skip_concat(enc, dec)
# assert result.shape == (2, 24, 12, 12), f'Expected (2, 24, 12, 12), got {result.shape}'
# # Test when sizes match
# enc2 = np.random.randn(2, 16, 10, 10)
# dec2 = np.random.randn(2, 8, 10, 10)
# result2 = skip_concat(enc2, dec2)
# assert result2.shape == (2, 24, 10, 10)
# print('All assertions passed!')"""))
    
    a(md_cell("""### 答案

<details>
<summary>点击展开参考答案</summary>

```python
def skip_concat(enc_feat, dec_feat):
    _, _, H_enc, W_enc = enc_feat.shape
    _, _, H_dec, W_dec = dec_feat.shape
    
    # Center crop encoder feature to match decoder size
    h_start = (H_enc - H_dec) // 2
    w_start = (W_enc - W_dec) // 2
    enc_cropped = enc_feat[:, :, h_start:h_start+H_dec, w_start:w_start+W_dec]
    
    # Concatenate along channel axis
    return np.concatenate([enc_cropped, dec_feat], axis=1)
```

</details>"""))
    
    a(code_cell("""# Reference Answer 1
def skip_concat(enc_feat, dec_feat):
    '''Concatenate encoder feature with decoder feature via skip connection.'''
    _, _, H_enc, W_enc = enc_feat.shape
    _, _, H_dec, W_dec = dec_feat.shape
    
    # Center crop encoder feature to match decoder size
    h_start = (H_enc - H_dec) // 2
    w_start = (W_enc - W_dec) // 2
    enc_cropped = enc_feat[:, :, h_start:h_start+H_dec, w_start:w_start+W_dec]
    
    # Concatenate along channel axis
    return np.concatenate([enc_cropped, dec_feat], axis=1)

# Tests
np.random.seed(42)
enc = np.random.randn(2, 16, 16, 16)
dec = np.random.randn(2, 8, 12, 12)
result = skip_concat(enc, dec)
assert result.shape == (2, 24, 12, 12), f'Expected (2, 24, 12, 12), got {result.shape}'

# Test when sizes match
enc2 = np.random.randn(2, 16, 10, 10)
dec2 = np.random.randn(2, 8, 10, 10)
result2 = skip_concat(enc2, dec2)
assert result2.shape == (2, 24, 10, 10)

# Test center crop correctness
enc3 = np.zeros((1, 1, 6, 6))
enc3[0, 0, 2:4, 2:4] = 1.0  # center 2x2 = 1
dec3 = np.zeros((1, 1, 2, 2))
result3 = skip_concat(enc3, dec3)
assert result3.shape == (1, 2, 2, 2)
# First channel should be the cropped center
assert np.all(result3[:, 0:1, :, :] == 1.0)

print(f'Skip concat output shape: {result.shape}')
print('All assertions passed!')"""))
    
    # Homework 2
    a(md_cell("""## 作业 2：实现 Dice Loss 和 IoU 指标

### 题目

实现多类别的 Dice Loss 和 IoU 指标。输入是预测概率图和 one-hot 标签，
输出是平均 Dice Loss 和平均 IoU。

### 要求
1. 实现 `multi_class_dice_loss(pred_probs, target_onehot)`
2. 实现 `multi_class_iou(pred_probs, target_onehot)`
3. 对每个类别计算后取平均
4. 用 assert 验证"""))
    
    a(code_cell("""# Homework 2: Multi-class Dice Loss and IoU
def multi_class_dice_loss(pred_probs, target_onehot, smooth=1e-6):
    '''Compute average Dice loss across classes.
    
    Args:
        pred_probs: predicted probabilities (batch, num_classes, H, W)
        target_onehot: one-hot ground truth (batch, num_classes, H, W)
        smooth: smoothing factor
    
    Returns:
        average dice loss (scalar)
    '''
    # YOUR CODE HERE
    pass

def multi_class_iou(pred_probs, target_onehot, smooth=1e-6):
    '''Compute average IoU across classes.'''
    # YOUR CODE HERE
    pass

# TODO: uncomment to test
# np.random.seed(42)
# pred = np.zeros((1, 3, 8, 8))
# pred[:, 0] = 0.9  # high prob for class 0
# pred[:, 1] = 0.1
# pred[:, 2] = 0.0
# target = np.zeros((1, 3, 8, 8))
# target[:, 0] = 1.0  # all class 0
# 
# dice_l = multi_class_dice_loss(pred, target)
# iou = multi_class_iou(pred, target)
# assert dice_l < 0.5  # should be low for good prediction
# assert iou > 0.5     # should be high for good prediction
# print(f'Dice loss: {dice_l:.4f}')
# print(f'IoU: {iou:.4f}')
# print('All assertions passed!')"""))
    
    a(md_cell("""### 答案

<details>
<summary>点击展开参考答案</summary>

```python
def multi_class_dice_loss(pred_probs, target_onehot, smooth=1e-6):
    num_classes = pred_probs.shape[1]
    dice_sum = 0.0
    for c in range(num_classes):
        pred_c = pred_probs[:, c]
        target_c = target_onehot[:, c]
        intersection = np.sum(pred_c * target_c)
        dice = (2 * intersection + smooth) / (np.sum(pred_c) + np.sum(target_c) + smooth)
        dice_sum += dice
    avg_dice = dice_sum / num_classes
    return 1 - avg_dice

def multi_class_iou(pred_probs, target_onehot, smooth=1e-6):
    num_classes = pred_probs.shape[1]
    iou_sum = 0.0
    for c in range(num_classes):
        pred_c = pred_probs[:, c]
        target_c = target_onehot[:, c]
        intersection = np.sum(pred_c * target_c)
        union = np.sum(pred_c) + np.sum(target_c) - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_sum += iou
    return iou_sum / num_classes
```

</details>"""))
    
    a(code_cell("""# Reference Answer 2
def multi_class_dice_loss(pred_probs, target_onehot, smooth=1e-6):
    '''Compute average Dice loss across classes.'''
    num_classes = pred_probs.shape[1]
    dice_sum = 0.0
    for c in range(num_classes):
        pred_c = pred_probs[:, c]
        target_c = target_onehot[:, c]
        intersection = np.sum(pred_c * target_c)
        dice = (2 * intersection + smooth) / (np.sum(pred_c) + np.sum(target_c) + smooth)
        dice_sum += dice
    avg_dice = dice_sum / num_classes
    return 1 - avg_dice

def multi_class_iou(pred_probs, target_onehot, smooth=1e-6):
    '''Compute average IoU across classes.'''
    num_classes = pred_probs.shape[1]
    iou_sum = 0.0
    for c in range(num_classes):
        pred_c = pred_probs[:, c]
        target_c = target_onehot[:, c]
        intersection = np.sum(pred_c * target_c)
        union = np.sum(pred_c) + np.sum(target_c) - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_sum += iou
    return iou_sum / num_classes

# Tests
np.random.seed(42)
pred = np.zeros((1, 3, 8, 8))
pred[:, 0] = 0.9
pred[:, 1] = 0.1
pred[:, 2] = 0.0
target = np.zeros((1, 3, 8, 8))
target[:, 0] = 1.0

dice_l = multi_class_dice_loss(pred, target)
iou = multi_class_iou(pred, target)
assert dice_l < 0.5
assert iou > 0.5

# Perfect prediction test
perfect_pred = target.copy()
dice_perfect = multi_class_dice_loss(perfect_pred, target)
iou_perfect = multi_class_iou(perfect_pred, target)
assert dice_perfect < 1e-4
assert abs(iou_perfect - 1.0) < 1e-4

print(f'Dice loss (good pred): {dice_l:.4f}')
print(f'IoU (good pred): {iou:.4f}')
print(f'Dice loss (perfect): {dice_perfect:.6f}')
print(f'IoU (perfect): {iou_perfect:.6f}')
print('All assertions passed!')"""))
    
    # Summary
    a(md_cell("""## 7. 总结

### 本讲要点
1. **分割类型**：语义分割、实例分割、全景分割
2. **FCN**：全卷积网络，用卷积替代全连接层
3. **转置卷积**：实现可学习的上采样
4. **U-Net**：编码器-解码器 + 跳跃连接，医学影像分割利器
5. **评估指标**：IoU、Dice 系数、Dice Loss
6. **Mask R-CNN**：实例分割的经典方法

### 常见坑
- 转置卷积的棋盘格效应（checkerboard artifacts）
- 类别不平衡导致的训练不稳定 → 用 Dice Loss / Focal Loss
- 跳跃连接尺寸不匹配 → 中心裁剪或填充"""))
    
    a(md_cell("""## 参考文献

### 论文
- [[Long et al., 2015]](https://arxiv.org/abs/1411.4038) **Fully Convolutional Networks for Semantic Segmentation** — FCN
- [[Ronneberger et al., 2015]](https://arxiv.org/abs/1505.04597) **U-Net: Convolutional Networks for Biomedical Image Segmentation** — U-Net
- [[He et al., 2017]](https://arxiv.org/abs/1703.06870) **Mask R-CNN** — 实例分割

### GitHub
- [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) (11k stars) — U-Net PyTorch 实现
- [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2) (29k stars) — Detectron2 (Mask R-CNN 等)"""))
    
    return make_notebook(cells), title


# ── Notebook 5: CLIP ─────────────────────────────────────────────────────────

def build_notebook_clip():
    title = "CLIP 与多模态学习：连接视觉与语言"
    cells = []
    a = cells.append
    
    a(md_cell("""# CLIP 与多模态学习：连接视觉与语言

> **CS231n 进阶专题 · 第 15 讲**
>
> 传统的视觉模型只能识别固定类别，而 **CLIP** 用图像-文本对进行对比预训练，学会了在共享嵌入空间中对齐视觉和语言。它实现了惊人的**零样本分类**能力——不需要微调就能识别任意类别。本讲我们将探索多模态学习的魅力。

**本讲你将学会：**
- 什么是多模态学习？视觉 + 语言
- CLIP：对比语言-图像预训练的核心思想
- 共享嵌入空间与余弦相似度
- 零样本分类：如何用文本提示做分类
- 提示工程（Prompt Engineering）技巧
- 亲手实现 CLIP 风格的对比损失和零样本分类器
- DINO / DINOv2：自监督视觉基础模型

> **前置知识**：对比学习、余弦相似度、词嵌入基础"""))
    
    a(code_cell(SETUP_CODE))
    
    a(md_cell("""## 1. 什么是多模态学习？

### 多模态 = 多种信息形式

人类通过视觉、听觉、语言等多种方式感知世界。多模态学习就是让 AI 同时处理和理解多种模态的数据。

### 视觉 + 语言

这是目前最成熟的多模态方向：
- **图文检索**：用文字搜图，用图搜文字
- **图像描述**：给图片生成文字说明
- **视觉问答**：关于图片的问答
- **零样本分类**：用文本描述识别新类别

### 为什么多模态重要？

1. **更丰富的监督信号**：图像和文本互相补充
2. **更强的泛化能力**：通过语言理解视觉概念
3. **更自然的交互**：用自然语言指挥 AI"""))
    
    a(code_cell("""# Visualize: multimodal embedding space
np.random.seed(42)

# Simulate a shared embedding space
# Images and texts of same concept should be close
concepts = ['猫', '狗', '汽车', '树']
n_per_concept = 10
dim = 2  # 2D for visualization

# Create clusters for each concept
fig, ax = plt.subplots(figsize=(10, 8))

colors = ['#E74C3C', '#3498DB', '#F39C12', '#2ECC71']
for i, concept in enumerate(concepts):
    # Image embeddings
    img_center = np.random.randn(2) * 3
    img_emb = img_center + np.random.randn(n_per_concept, 2) * 0.5
    ax.scatter(img_emb[:, 0], img_emb[:, 1], c=colors[i], 
              marker='o', s=80, alpha=0.7, label=f'{concept} (图像)')
    
    # Text embeddings
    text_center = img_center + np.random.randn(2) * 0.3  # close to image
    text_emb = text_center + np.random.randn(n_per_concept, 2) * 0.4
    ax.scatter(text_emb[:, 0], text_emb[:, 1], c=colors[i],
              marker='s', s=80, alpha=0.7, label=f'{concept} (文本)')

ax.set_title('多模态共享嵌入空间（示意图）', fontsize=14)
ax.set_xlabel('嵌入维度 1', fontsize=12)
ax.set_ylabel('嵌入维度 2', fontsize=12)
ax.legend(fontsize=10, ncol=2)
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.02, '○ = 图像嵌入\\n□ = 文本嵌入\\n同色 = 同一概念', 
         transform=ax.transAxes, fontsize=11,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
plt.tight_layout()
plt.show()"""))
    
    # Section 2: CLIP
    a(md_cell("""## 2. CLIP：对比语言-图像预训练

### 核心思想

[[Radford et al., 2021]](https://arxiv.org/abs/2103.00020)

**用 4 亿对图像-文本数据做对比学习**

### 架构

```
图像 --> 图像编码器 --> 图像嵌入 (I)
文本 --> 文本编码器 --> 文本嵌入 (T)

损失：对称对比损失 (InfoNCE)
- 每个图像找到对应的文本（正样本）
- 每个文本找到对应的图像（正样本）
```

### 训练目标

在一个 batch 中：
- 行方向：每个图像和 batch 内所有文本计算相似度，正确的文本应该最相似
- 列方向：每个文本和 batch 内所有图像计算相似度，正确的图像应该最相似

### 为什么 CLIP 这么强？

1. **海量数据**：4 亿图文对，远超 ImageNet
2. **弱监督**：不需要精确标注，图片+描述就行
3. **对比学习**：学到语义级别的相似性
4. **双编码器**：图像和文本共享嵌入空间"""))
    
    a(md_cell("""### 零样本分类（Zero-shot Classification）

CLIP 最神奇的能力：不需要微调，就能做任意类别的分类！

**做法**：
1. 把每个类别变成文本提示，如 "a photo of a {class}"
2. 用文本编码器得到每个类别的文本嵌入
3. 用图像编码器得到图像嵌入
4. 计算图像嵌入和所有文本嵌入的相似度
5. 取相似度最高的类别作为预测

> **思考题 1**：为什么 CLIP 能做零样本分类？普通的 ImageNet 预训练模型为什么不行？"""))
    
    a(code_cell("""# Implement CLIP-style contrastive loss
def clip_contrastive_loss(image_emb, text_emb, temperature=0.07):
    '''CLIP-style symmetric contrastive loss.
    
    Args:
        image_emb: image embeddings (batch, dim)
        text_emb: text embeddings (batch, dim)
        temperature: temperature parameter
    
    Returns:
        total loss (image-to-text + text-to-image) / 2
    '''
    # Normalize embeddings
    img_norm = image_emb / (np.linalg.norm(image_emb, axis=1, keepdims=True) + 1e-8)
    txt_norm = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-8)
    
    # Similarity matrix (batch x batch)
    logits = img_norm @ txt_norm.T / temperature
    
    batch_size = len(image_emb)
    labels = np.arange(batch_size)
    
    # Image-to-text loss
    logits_i2t_max = np.max(logits, axis=1, keepdims=True)
    logits_i2t_stable = logits - logits_i2t_max
    exp_i2t = np.exp(logits_i2t_stable)
    log_sum_i2t = np.log(np.sum(exp_i2t, axis=1) + 1e-10)
    log_probs_i2t = logits_i2t_stable - log_sum_i2t[:, None]
    loss_i2t = -np.mean(log_probs_i2t[labels, labels])
    
    # Text-to-image loss
    logits_t2i = logits.T
    logits_t2i_max = np.max(logits_t2i, axis=1, keepdims=True)
    logits_t2i_stable = logits_t2i - logits_t2i_max
    exp_t2i = np.exp(logits_t2i_stable)
    log_sum_t2i = np.log(np.sum(exp_t2i, axis=1) + 1e-10)
    log_probs_t2i = logits_t2i_stable - log_sum_t2i[:, None]
    loss_t2i = -np.mean(log_probs_t2i[labels, labels])
    
    return (loss_i2t + loss_t2i) / 2

# Test with toy embeddings
np.random.seed(42)
batch_size = 8
dim = 32

# Simulate paired embeddings (slightly different but correlated)
img_emb = np.random.randn(batch_size, dim)
text_emb = img_emb + np.random.randn(batch_size, dim) * 0.3  # text similar to image

loss = clip_contrastive_loss(img_emb, text_emb, temperature=0.07)
print(f'CLIP contrastive loss: {loss:.4f}')

# Test with random (unpaired) embeddings
random_text = np.random.randn(batch_size, dim)
loss_random = clip_contrastive_loss(img_emb, random_text, temperature=0.07)
print(f'Loss with random text: {loss_random:.4f}')
print(f'\\n配对的损失 < 随机配对的损失: {loss < loss_random}')"""))
    
    # Section 3: Zero-shot
    a(md_cell("""## 3. 零样本分类：不用微调的分类器

### 如何用 CLIP 做零样本分类？

步骤：
1. 准备类别名称列表：["猫", "狗", "鸟", ...]
2. 构造提示模板："一张 {类别} 的照片"
3. 文本编码器编码所有提示 → 文本特征库
4. 图像编码器编码输入图像 → 图像特征
5. 计算相似度，取最大的作为预测

### 提示工程（Prompt Engineering）

提示的写法会极大影响零样本效果：
- "a photo of a cat" vs "cat"
- "a photo of a cat, a type of pet"
- 多提示集成：用多种描述方式取平均

### 为什么提示这么重要？

文本编码器是在自然语言上训练的，
用更自然的语言描述能更好地匹配预训练分布。

> **思考题 2**：你觉得"a photo of a cat"和"cat"哪个效果更好？为什么？"""))
    
    a(code_cell("""# Implement zero-shot classifier
def zero_shot_classify(image_emb, text_embeddings, class_names, temperature=0.07):
    '''Zero-shot classification using CLIP-style embeddings.
    
    Args:
        image_emb: image embedding (1, dim) or (n, dim)
        text_embeddings: text embeddings for each class (n_classes, dim)
        class_names: list of class names
        temperature: temperature for softmax
    
    Returns:
        probs: class probabilities
        pred_class: predicted class name
    '''
    # Normalize
    img_norm = image_emb / (np.linalg.norm(image_emb, axis=1, keepdims=True) + 1e-8)
    txt_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarities
    similarities = img_norm @ txt_norm.T / temperature
    
    # Softmax to get probabilities
    sim_max = np.max(similarities, axis=1, keepdims=True)
    exp_sim = np.exp(similarities - sim_max)
    probs = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)
    
    pred_idx = np.argmax(probs, axis=1)
    pred_class = [class_names[i] for i in pred_idx]
    
    return probs, pred_class

# Demo: simulate zero-shot classification
np.random.seed(42)
dim = 32
class_names = ['猫', '狗', '汽车', '飞机']

# Simulate class text embeddings (each has a "direction" in embedding space)
text_embeddings = np.random.randn(len(class_names), dim) * 2

# Simulate an image of a cat (close to cat text embedding)
cat_img_emb = text_embeddings[0:1] + np.random.randn(1, dim) * 0.5

# Zero-shot classification
probs, pred = zero_shot_classify(cat_img_emb, text_embeddings, class_names)

print('零样本分类结果：')
for name, p in zip(class_names, probs[0]):
    print(f'  {name}: {p:.4f}')
print(f'预测类别: {pred[0]}')

# Visualize
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(class_names, probs[0], color=plt.cm.tab10(np.arange(4)), alpha=0.8)
ax.set_ylabel('预测概率', fontsize=12)
ax.set_title('零样本分类概率分布', fontsize=14)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, axis='y')
for bar, p in zip(bars, probs[0]):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{p:.3f}', ha='center', fontsize=11)
plt.tight_layout()
plt.show()"""))
    
    # Section 4: Prompt Engineering
    a(md_cell("""## 4. 提示工程与集成

### 为什么提示工程有效？

CLIP 的文本编码器是在自然语言描述上训练的。
如果你的提示更接近训练时看到的描述，效果就更好。

### 常见技巧

1. **模板化**："a photo of a {class}"
2. **上下文描述**："a photo of a {class}, a type of animal"
3. **多角度**：多个提示取平均
4. **领域适配**：针对特定领域调整描述

### 提示集成（Prompt Ensembling）

对同一个类别用多种不同的文本描述，
编码后取平均作为该类别的文本嵌入。
这通常比单个提示效果好很多。"""))
    
    a(code_cell("""# Prompt ensembling demo
np.random.seed(42)
dim = 32

# Base "cat" direction in embedding space
cat_dir = np.random.randn(dim)
cat_dir = cat_dir / np.linalg.norm(cat_dir)

# Single prompt vs multiple prompts
single_prompt = cat_dir * 2 + np.random.randn(dim) * 0.5  # single description

# Multiple prompts for "cat"
cat_prompts = [
    cat_dir * 2 + np.random.randn(dim) * 0.5,  # "a photo of a cat"
    cat_dir * 2 + np.random.randn(dim) * 0.6,  # "a picture of a feline"
    cat_dir * 2 + np.random.randn(dim) * 0.4,  # "an image of a domestic cat"
    cat_dir * 2 + np.random.randn(dim) * 0.55, # "a cat, a type of pet"
]

# Ensemble: average of multiple prompts
ensemble_prompt = np.mean(cat_prompts, axis=0)

# Simulate a test image of a cat
test_img = cat_dir * 1.8 + np.random.randn(dim) * 0.7

# Compute similarities
def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

sim_single = cos_sim(test_img, single_prompt)
sim_ensemble = cos_sim(test_img, ensemble_prompt)

print(f'Single prompt similarity: {sim_single:.4f}')
print(f'Ensemble prompt similarity: {sim_ensemble:.4f}')
print(f'Ensemble improvement: {sim_ensemble - sim_single:.4f}')

# Visualize (project to 2D using two random directions)
proj1 = np.random.randn(dim)
proj1 = proj1 / np.linalg.norm(proj1)
proj2 = np.cross(np.append(cat_dir, 0)[:3], np.append(proj1, 0)[:3])[:2]
proj2 = np.random.randn(dim)  # simple second direction

def project(vec):
    return np.array([np.dot(vec, proj1), np.dot(vec, proj2)])

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(*project(cat_dir * 2), c='red', s=200, marker='*', label='真实 "猫" 方向', zorder=5)
ax.scatter(*project(test_img), c='orange', s=150, marker='^', label='测试图像', zorder=5)
ax.scatter(*project(single_prompt), c='blue', s=100, label='单个提示')
for i, p in enumerate(cat_prompts):
    ax.scatter(*project(p), c='green', s=50, alpha=0.5, label=f'提示 {i+1}' if i == 0 else '')
ax.scatter(*project(ensemble_prompt), c='purple', s=150, marker='D', label='集成提示', zorder=5)

ax.set_xlabel('投影维度 1', fontsize=12)
ax.set_ylabel('投影维度 2', fontsize=12)
ax.set_title('提示集成效果示意图', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.show()"""))
    
    # Section 5: DINO
    a(md_cell("""## 5. DINO 与 DINOv2：自监督视觉基础模型

### DINO

[[Caron et al., 2021]](https://arxiv.org/abs/2104.14294)

DINO（自蒸馏无标签）是 Facebook 提出的自监督方法：
- 不需要标签，不需要负样本
- 用动量教师网络指导学生网络
- 学生网络不同视角的输出要和教师网络一致

### DINOv2

DINOv2 是 DINO 的升级版，主要改进：
- 更多数据（1.42 亿张图）
- 更好的训练策略
- 更强的泛化能力

DINOv2 的特征可以直接用于各种下游任务，
被称为"视觉的 GPT 时刻"。

### 多模态基础模型

从 CLIP 到 Flamingo、GPT-4V，多模态模型发展迅速：
- **CLIP**：图文对比预训练
- **Flamingo**：小样本视觉语言模型
- **GPT-4V**：通用视觉语言能力

> **思考题 3**：CLIP 是双编码器，GPT-4V 是融合模型，它们各有什么优缺点？"""))
    
    # Section 6: Retrieval
    a(md_cell("""## 6. CLIP 用于图文检索

### 文本搜图（Text-to-Image Retrieval）

给定一句文字描述，从图库中找出最匹配的图片。

### 图像搜文（Image-to-Text Retrieval）

给定一张图片，找出最匹配的文字描述。

### 为什么 CLIP 适合检索？

1. **共享嵌入空间**：图像和文本可以直接比较
2. **计算高效**：预先计算所有特征，检索时只需要算相似度
3. **零样本能力**：可以搜任意概念，不需要微调"""))
    
    a(code_cell("""# Image-text retrieval demo
np.random.seed(42)
dim = 32
n_images = 20
n_texts = 5

# Simulate image database and text queries
concepts = ['海滩', '山', '城市', '森林', '沙漠']
img_concepts = np.random.choice(5, n_images)
text_concepts = np.arange(5)

# Create embeddings (same concept = similar embeddings)
concept_centers = np.random.randn(5, dim) * 3

image_embeddings = np.array([
    concept_centers[c] + np.random.randn(dim) * 0.6
    for c in img_concepts
])

text_embeddings = np.array([
    concept_centers[c] + np.random.randn(dim) * 0.4
    for c in text_concepts
])

def text_to_image_retrieval(text_emb, img_embeddings, top_k=3):
    '''Find top-k most similar images for a text query.'''
    txt_norm = text_emb / (np.linalg.norm(text_emb) + 1e-8)
    img_norm = img_embeddings / (np.linalg.norm(img_embeddings, axis=1, keepdims=True) + 1e-8)
    sims = img_norm @ txt_norm
    top_indices = np.argsort(-sims)[:top_k]
    return top_indices, sims[top_indices]

# Test: text query "beach" -> retrieve beach images
query_idx = 0  # "海滩"
top_k = 5
top_idx, top_sim = text_to_image_retrieval(text_embeddings[query_idx], image_embeddings, top_k)

print(f'文本查询: "{concepts[query_idx]}"')
print(f'Top-{top_k} 检索结果:')
for i, (idx, sim) in enumerate(zip(top_idx, top_sim)):
    is_correct = img_concepts[idx] == query_idx
    print(f'  第{i+1}名: 图像{idx} ({concepts[img_concepts[idx]]}), 相似度={sim:.4f} {"✓" if is_correct else "✗"}')

# Visualize similarity matrix
img_norm = image_embeddings / (np.linalg.norm(image_embeddings, axis=1, keepdims=True) + 1e-8)
txt_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)
sim_matrix = img_norm @ txt_norm.T

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(sim_matrix, cmap='YlOrRd', aspect='auto')
ax.set_xlabel('文本查询', fontsize=12)
ax.set_ylabel('图像索引', fontsize=12)
ax.set_title('图文相似度矩阵', fontsize=14)
ax.set_xticks(range(5))
ax.set_xticklabels(concepts)
plt.colorbar(im, ax=ax, label='余弦相似度')
plt.tight_layout()
plt.show()"""))
    
    # Homework 1
    a(md_cell("""## 作业 1：实现 CLIP 零样本分类器

### 题目

实现一个完整的 CLIP 风格零样本分类器。
给定图像嵌入、类别名称列表和提示模板，
输出每个类别的预测概率。

### 要求
1. 实现 `clip_zero_shot(image_emb, class_names, text_encoder_fn, prompt_template)`
2. 用提示模板为每个类别生成文本
3. 计算图像与所有文本的相似度
4. 用 softmax 得到概率分布
5. 用 assert 验证输出"""))
    
    a(code_cell("""# Homework 1: CLIP Zero-Shot Classifier
def clip_zero_shot(image_emb, class_names, text_embeddings, temperature=0.07):
    '''CLIP zero-shot classifier.
    
    Args:
        image_emb: image embeddings (n_images, dim)
        class_names: list of class names (n_classes,)
        text_embeddings: pre-computed text embeddings (n_classes, dim)
        temperature: softmax temperature
    
    Returns:
        probs: class probabilities (n_images, n_classes)
        predictions: predicted class indices (n_images,)
    '''
    # YOUR CODE HERE
    pass

# TODO: uncomment to test
# np.random.seed(42)
# dim = 32
# classes = ['cat', 'dog', 'bird']
# # Simulate text embeddings
# text_embs = np.random.randn(3, dim) * 2
# # Simulate a cat image
# img_emb = text_embs[0:1] + np.random.randn(1, dim) * 0.5
# probs, preds = clip_zero_shot(img_emb, classes, text_embs)
# assert probs.shape == (1, 3)
# assert preds.shape == (1,)
# assert np.allclose(np.sum(probs, axis=1), 1.0)  # probabilities sum to 1
# assert preds[0] == 0  # should predict cat
# print('All assertions passed!')"""))
    
    a(md_cell("""### 答案

<details>
<summary>点击展开参考答案</summary>

```python
def clip_zero_shot(image_emb, class_names, text_embeddings, temperature=0.07):
    # Normalize embeddings
    img_norm = image_emb / (np.linalg.norm(image_emb, axis=1, keepdims=True) + 1e-8)
    txt_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarity logits
    logits = img_norm @ txt_norm.T / temperature
    
    # Softmax
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    predictions = np.argmax(probs, axis=1)
    
    return probs, predictions
```

</details>"""))
    
    a(code_cell("""# Reference Answer 1
def clip_zero_shot(image_emb, class_names, text_embeddings, temperature=0.07):
    '''CLIP zero-shot classifier.'''
    # Normalize embeddings
    img_norm = image_emb / (np.linalg.norm(image_emb, axis=1, keepdims=True) + 1e-8)
    txt_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarity logits
    logits = img_norm @ txt_norm.T / temperature
    
    # Numerically stable softmax
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    predictions = np.argmax(probs, axis=1)
    
    return probs, predictions

# Tests
np.random.seed(42)
dim = 32
classes = ['cat', 'dog', 'bird']
text_embs = np.random.randn(3, dim) * 2

# Test with single image
img_emb = text_embs[0:1] + np.random.randn(1, dim) * 0.5
probs, preds = clip_zero_shot(img_emb, classes, text_embs)
assert probs.shape == (1, 3)
assert preds.shape == (1,)
assert np.allclose(np.sum(probs, axis=1), 1.0)
assert preds[0] == 0

# Test with batch of images
batch_imgs = text_embs + np.random.randn(3, dim) * 0.3
probs_batch, preds_batch = clip_zero_shot(batch_imgs, classes, text_embs)
assert probs_batch.shape == (3, 3)
assert preds_batch.shape == (3,)
assert np.allclose(np.sum(probs_batch, axis=1), 1.0)

# Test that temperature affects confidence
probs_low_temp, _ = clip_zero_shot(img_emb, classes, text_embs, temperature=0.01)
probs_high_temp, _ = clip_zero_shot(img_emb, classes, text_embs, temperature=0.5)
# Lower temperature -> sharper distribution (higher max prob)
assert np.max(probs_low_temp) > np.max(probs_high_temp)

print(f'Output shape: {probs.shape}')
print(f'Predicted class: {classes[preds[0]]}')
print(f'Confidence: {probs[0, preds[0]]:.4f}')
print('All assertions passed!')"""))
    
    # Homework 2
    a(md_cell("""## 作业 2：实现图像-文本检索

### 题目

实现图文检索功能。给定一个文本查询和图像特征库，
返回最相似的 top-K 张图像。同时计算 Recall@K。

### Recall@K

前 K 个检索结果中包含正确答案的比例。

### 要求
1. 实现 `image_text_retrieval(query_emb, gallery_emb, top_k)`
2. 返回 top_k 个最相似的索引和相似度
3. 实现 `recall_at_k(retrieved_indices, ground_truth, k)`
4. 用 assert 验证"""))
    
    a(code_cell("""# Homework 2: Image-Text Retrieval
def image_text_retrieval(query_emb, gallery_emb, top_k=5):
    '''Retrieve top-k most similar items from gallery.
    
    Args:
        query_emb: query embedding (dim,)
        gallery_emb: gallery embeddings (n, dim)
        top_k: number of items to retrieve
    
    Returns:
        indices: top-k indices (top_k,)
        similarities: corresponding similarities (top_k,)
    '''
    # YOUR CODE HERE
    pass

def recall_at_k(retrieved_indices, ground_truth, k):
    '''Compute Recall@K.
    
    Args:
        retrieved_indices: array of retrieved indices for each query (n_queries, n_retrieved)
        ground_truth: array of ground truth indices (n_queries,)
        k: compute recall at this k
    
    Returns:
        recall: scalar value in [0, 1]
    '''
    # YOUR CODE HERE
    pass

# TODO: uncomment to test
# np.random.seed(42)
# gallery = np.random.randn(20, 16)
# query = gallery[3] + np.random.randn(16) * 0.1  # query similar to item 3
# idx, sims = image_text_retrieval(query, gallery, top_k=5)
# assert len(idx) == 5
# assert 3 in idx  # item 3 should be in top-5
# 
# # Test recall
# retrieved = np.array([[3, 7, 1, 9, 5], [2, 4, 6, 8, 0]])
# gt = np.array([3, 6])
# r5 = recall_at_k(retrieved, gt, k=5)
# r1 = recall_at_k(retrieved, gt, k=1)
# assert r5 == 1.0  # both in top-5
# assert r1 == 0.5  # one in top-1
# print('All assertions passed!')"""))
    
    a(md_cell("""### 答案

<details>
<summary>点击展开参考答案</summary>

```python
def image_text_retrieval(query_emb, gallery_emb, top_k=5):
    # Normalize
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    g_norm = gallery_emb / (np.linalg.norm(gallery_emb, axis=1, keepdims=True) + 1e-8)
    # Compute similarities
    sims = g_norm @ q_norm
    # Sort descending
    sorted_indices = np.argsort(-sims)
    top_indices = sorted_indices[:top_k]
    return top_indices, sims[top_indices]

def recall_at_k(retrieved_indices, ground_truth, k):
    n_queries = len(ground_truth)
    hits = 0
    for i in range(n_queries):
        if ground_truth[i] in retrieved_indices[i, :k]:
            hits += 1
    return hits / n_queries
```

</details>"""))
    
    a(code_cell("""# Reference Answer 2
def image_text_retrieval(query_emb, gallery_emb, top_k=5):
    '''Retrieve top-k most similar items from gallery.'''
    # Normalize
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    g_norm = gallery_emb / (np.linalg.norm(gallery_emb, axis=1, keepdims=True) + 1e-8)
    # Compute similarities
    sims = g_norm @ q_norm
    # Sort descending
    sorted_indices = np.argsort(-sims)
    top_indices = sorted_indices[:top_k]
    return top_indices, sims[top_indices]

def recall_at_k(retrieved_indices, ground_truth, k):
    '''Compute Recall@K.'''
    n_queries = len(ground_truth)
    hits = 0
    for i in range(n_queries):
        if ground_truth[i] in retrieved_indices[i, :k]:
            hits += 1
    return hits / n_queries

# Tests
np.random.seed(42)
gallery = np.random.randn(20, 16)
query = gallery[3] + np.random.randn(16) * 0.1
idx, sims = image_text_retrieval(query, gallery, top_k=5)
assert len(idx) == 5
assert 3 in idx

# Test recall
retrieved = np.array([[3, 7, 1, 9, 5], [2, 4, 6, 8, 0]])
gt = np.array([3, 6])
r5 = recall_at_k(retrieved, gt, k=5)
r1 = recall_at_k(retrieved, gt, k=1)
assert r5 == 1.0
assert r1 == 0.5

# Edge case: k larger than retrieved
r10 = recall_at_k(retrieved, gt, k=10)
assert r10 == 1.0

print(f'Top-5 indices: {idx}')
print(f'Top-5 similarities: {sims.round(4)}')
print(f'Recall@1: {r1}')
print(f'Recall@5: {r5}')
print('All assertions passed!')"""))
    
    # Summary
    a(md_cell("""## 7. 总结

### 本讲要点
1. **多模态学习**：连接视觉和语言，共享嵌入空间
2. **CLIP**：图文对比预训练，4 亿数据训练的大模型
3. **零样本分类**：用文本提示做分类，无需微调
4. **提示工程**：好的提示能显著提升效果
5. **图文检索**：在共享空间中做相似度搜索
6. **DINO/DINOv2**：自监督视觉基础模型

### CLIP 的局限性
- 组合推理能力弱（"蓝色的圆和红色的方"）
- 公平性和偏见问题
- 对细粒度分类不够准确
- 文本提示敏感"""))
    
    a(md_cell("""## 参考文献

### 论文
- [[Radford et al., 2021]](https://arxiv.org/abs/2103.00020) **Learning Transferable Visual Models From Natural Language Supervision** — CLIP
- [[Caron et al., 2021]](https://arxiv.org/abs/2104.14294) **Emerging Properties in Self-Supervised Vision Transformers** — DINO
- [[Alayrac et al., 2022]](https://arxiv.org/abs/2204.14198) **Flamingo: a Visual Language Model for Few-Shot Learning** — Flamingo

### GitHub
- [openai/CLIP](https://github.com/openai/CLIP) (23k stars) — CLIP 官方实现
- [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) (9.8k stars) — OpenCLIP 开源实现
- [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) (15k stars) — DINOv2 官方实现"""))
    
    return make_notebook(cells), title


# ── Notebook 6: 3D Vision & NeRF ────────────────────────────────────────────

def build_notebook_nerf():
    title = "3D 视觉与 NeRF：从图像到三维世界"
    cells = []
    a = cells.append
    
    a(md_cell("""# 3D 视觉与 NeRF：从图像到三维世界

> **CS231n 进阶专题 · 第 16 讲**
>
> 我们生活在三维世界中，但相机只能拍下二维投影。**3D 视觉**要从二维图像恢复三维结构。从传统的立体视觉到革命性的 NeRF，3D 重建技术正在快速发展。本讲我们将探索多种 3D 表示方法，并亲手实现简化版的 NeRF。

**本讲你将学会：**
- 为什么需要 3D 视觉？AR/VR、机器人、自动驾驶
- 3D 表示：体素、点云、网格、隐式函数
- 立体视觉与对极几何直觉
- NeRF：神经辐射场的核心思想
- 体渲染（Volume Rendering）原理
- 位置编码（Positional Encoding）为什么重要
- 亲手实现 2D 版 NeRF 和体渲染
- Instant-NGP 与 3D 高斯溅射

> **前置知识**：透视投影、神经网络基础、光线概念"""))
    
    a(code_cell(SETUP_CODE))
    
    a(md_cell("""## 1. 为什么需要 3D 视觉？

### 应用场景
- **AR/VR**：虚拟物体与真实世界融合
- **自动驾驶**：理解三维环境，规划路径
- **机器人**：抓取物体、避障导航
- **影视游戏**：三维重建，数字人
- **医学**：CT/MRI 三维重建

### 核心问题

**如何从 2D 图像恢复 3D 结构？**

单张图像丢失了深度信息——这是一个病态问题（ill-posed）。
但我们有多种方法来解决：
1. 多视角图像（立体视觉、SfM）
2. 深度传感器（LiDAR、Kinect）
3. 单目深度估计（深度学习）
4. 神经辐射场（NeRF）"""))
    
    a(md_cell("""### 3D 表示方法对比

| 表示方式 | 优点 | 缺点 | 代表 |
|---------|------|------|------|
| 体素 (Voxel) | 规则网格，容易处理 | 内存占用大，分辨率低 | 3D CNN |
| 点云 (Point Cloud) | 轻量，无序 | 不连通，难提取特征 | PointNet |
| 网格 (Mesh) | 表面表示，渲染快 | 拓扑复杂，难生成 | 传统图形学 |
| 隐式函数 | 连续表示，分辨率无限 | 渲染慢，需提取表面 | NeRF, SDF |

> **思考题 1**：为什么隐式函数（如 NeRF）能实现"无限分辨率"？"""))
    
    a(code_cell("""# Visualize: 3D representation types
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# Voxel: 5x5x5 simple shape (2D slice shown)
ax = axes[0]
voxel_grid = np.zeros((10, 10))
voxel_grid[3:7, 3:7] = 1
voxel_grid[2:8, 4:6] = 1
ax.spy(voxel_grid, markersize=15, color='#3498DB')
ax.set_title('体素 (Voxel)\\n2D 截面示意', fontsize=12)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Point cloud
ax = axes[1]
np.random.seed(42)
theta = np.random.uniform(0, 2*np.pi, 200)
r = 2 + np.random.randn(200)*0.1
points_x = r * np.cos(theta)
points_y = r * np.sin(theta)
ax.scatter(points_x, points_y, c='#E74C3C', s=10, alpha=0.7)
ax.set_title('点云 (Point Cloud)\\n圆形分布', fontsize=12)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Mesh (triangle mesh of a circle)
ax = axes[2]
angles = np.linspace(0, 2*np.pi, 12)
verts_x = np.cos(angles) * 2
verts_y = np.sin(angles) * 2
# Draw triangles (fan from center)
for i in range(12):
    next_i = (i + 1) % 12
    ax.fill([0, verts_x[i], verts_x[next_i]],
            [0, verts_y[i], verts_y[next_i]],
            alpha=0.5, edgecolor='#2ECC71', linewidth=1.5)
ax.scatter(verts_x, verts_y, c='#2ECC71', s=30, zorder=5)
ax.set_title('网格 (Mesh)\\n三角形面', fontsize=12)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Implicit function (circle: x^2 + y^2 = r^2)
ax = axes[3]
xx, yy = np.mgrid[-3:3:100j, -3:3:100j]
dist = np.sqrt(xx**2 + yy**2) - 2  # SDF: signed distance function
ax.contour(xx, yy, dist, levels=[0], colors='#9B59B6', linewidths=3)
ax.contourf(xx, yy, dist, levels=[-5, 0, 5], colors=['#9B59B633', 'transparent'])
ax.set_title('隐式函数\\nSDF = 0 为表面', fontsize=12)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.suptitle('3D 表示方法（2D 示意）', fontsize=14, y=0.95)
plt.tight_layout()
plt.show()"""))
    
    # Section 2: Stereo Vision
    a(md_cell("""## 2. 立体视觉：从两张图恢复深度

### 核心直觉

人类用两只眼睛感知深度——**双目视觉**。
左右眼看到的图像略有差异（视差），大脑据此推断深度。

### 对极几何（Epipolar Geometry）

两个相机观察同一场景时，存在几何约束：
- 左图中的点，在右图中一定位于对应的"极线"上
- 这大大减少了匹配的搜索范围

### 深度公式

Z = f * B / d

其中：
- Z：深度
- f：焦距
- B：基线（两个相机的距离）
- d：视差（左右图对应点的像素差）

视差越大，深度越小（物体越近）。

> **思考题 2**：为什么自动驾驶常用激光雷达而不是纯立体视觉？"""))
    
    a(code_cell("""# Visualize: stereo vision and depth from disparity
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left and right "images" - a square at different depths
ax = axes[0]
np.random.seed(42)
# Simulate two views of a scene with objects at different depths
depths = [1.0, 2.0, 4.0]  # Z values
baseline = 0.5  # B
focal = 100  # f (pixels)

disparities = [focal * baseline / Z for Z in depths]
print(f'不同深度的视差:')
for Z, d in zip(depths, disparities):
    print(f'  深度 Z={Z:.1f}: 视差 d={d:.1f} px')

# Plot depth vs disparity
ax = axes[0]
Z_range = np.linspace(0.5, 5.0, 100)
d_range = focal * baseline / Z_range
ax.plot(Z_range, d_range, 'b-', lw=2)
ax.set_xlabel('深度 Z', fontsize=12)
ax.set_ylabel('视差 d (像素)', fontsize=12)
ax.set_title('深度与视差的关系 (Z = fB/d)', fontsize=13)
ax.grid(True, alpha=0.3)
for Z, d in zip(depths, disparities):
    ax.plot(Z, d, 'ro', markersize=8)
    ax.text(Z+0.1, d, f'Z={Z}', fontsize=10)

# Stereo geometry schematic
ax = axes[1]
# Two cameras
ax.plot([-baseline/2, -baseline/2], [-0.5, 0.5], 'k-', lw=3, label='左相机')
ax.plot([baseline/2, baseline/2], [-0.5, 0.5], 'k-', lw=3, label='右相机')
# A point in 3D
P_z = 2.0
P_x = 0.3
ax.plot(P_x, P_z, 'ro', markersize=10, label='空间点 P')
# Projections
proj_left = -baseline/2 + (-baseline/2 - P_x) * focal / (P_z * focal) * 0.1  # simplified
proj_right = baseline/2 + (baseline/2 - P_x) * focal / (P_z * focal) * 0.1
ax.plot([-baseline/2, P_x], [0, P_z], 'b--', alpha=0.7)
ax.plot([baseline/2, P_x], [0, P_z], 'r--', alpha=0.7)
ax.set_xlabel('x (基线方向)', fontsize=12)
ax.set_ylabel('Z (深度)', fontsize=12)
ax.set_title('双目立体几何示意', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.set_xlim(-1, 1)

plt.tight_layout()
plt.show()"""))
    
    # Section 3: NeRF
    a(md_cell("""## 3. NeRF：神经辐射场

### 革命性突破

[[Mildenhall et al., 2020]](https://arxiv.org/abs/2003.08934)

NeRF（Neural Radiance Fields）用一个 MLP 表示整个三维场景：

**输入**：5D 坐标 —— 位置 (x, y, z) + 视角方向 (θ, φ)
**输出**：体积密度 σ + 颜色 RGB (r, g, b)

```
(x, y, z, θ, φ) → MLP → (σ, r, g, b)
```

### 为什么叫"辐射场"？

- **密度 σ**：描述空间中某点的不透明度（密度越高，越不透明）
- **颜色 rgb**：描述该点在特定方向上发出的光的颜色

### 核心创新

1. **隐式表示**：用神经网络表示连续的 3D 场景
2. **体渲染**：可微的渲染过程，支持端到端训练
3. **位置编码**：让 MLP 学会高频细节

> **思考题 3**：NeRF 的输入是 5D（位置+视角），为什么需要视角方向？只用 3D 位置会怎样？"""))
    
    a(code_cell("""# Implement simple 2D NeRF (radial field)
class SimpleNeRF2D:
    '''Simplified 2D "NeRF" for a radial density field.
    
    In 2D: input (x, y) -> (density, color)
    Think of this as a slice of a 3D NeRF.
    '''
    def __init__(self, hidden_dim=32):
        # MLP: (x,y) -> (density, color)
        self.W1 = np.random.randn(2 + 10, hidden_dim) * 0.1  # +10 for positional encoding
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, 2) * 0.1  # output: density, color
        self.b3 = np.zeros(2)
    
    def positional_encoding(self, x, L=5):
        '''Positional encoding: sin/cos at different frequencies.
        
        Maps input x (D dims) to higher-dimensional space using
        sin(2^l * pi * x) and cos(2^l * pi * x) for l in 0..L-1.
        '''
        D = x.shape[1]
        encoded = []
        for l in range(L):
            freq = 2 ** l
            encoded.append(np.sin(freq * np.pi * x))
            encoded.append(np.cos(freq * np.pi * x))
        return np.concatenate(encoded, axis=1)
    
    def forward(self, points):
        '''Query the NeRF at given 2D points.
        
        Args:
            points: (N, 2) array of (x, y) positions
        
        Returns:
            density: (N,) density (sigma) values
            color: (N,) color values (scalar for 2D demo)
        '''
        # Positional encoding
        pe = self.positional_encoding(points, L=5)
        
        # MLP forward
        h = np.maximum(0, pe @ self.W1 + self.b1)
        h = np.maximum(0, h @ self.W2 + self.b2)
        out = h @ self.W3 + self.b3
        
        # Density is always positive (ReLU)
        density = np.maximum(out[:, 0], 0)
        # Color in [0, 1] (sigmoid)
        color = 1 / (1 + np.exp(-out[:, 1]))
        
        return density, color

nerf_2d = SimpleNeRF2D(hidden_dim=32)
print('2D NeRF created!')

# Test with a few points
test_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
density, color = nerf_2d.forward(test_points)
print(f'Test densities: {density}')
print(f'Test colors: {color}')"""))
    
    # Section 4: Volume Rendering
    a(md_cell("""## 4. 体渲染：沿着光线累积颜色

### 核心问题

NeRF 输出了每个点的密度和颜色，怎么从这些得到最终的像素颜色？

### 答案：体渲染方程

像素颜色 = 沿光线积分：颜色 × 密度 × 透射率

### 离散近似（数值积分）

将一条光线分成 N 个采样点：
C = sum_{i=1}^{N} T_i * (1 - exp(-sigma_i * delta_i)) * c_i

其中：
- T_i = exp(-sum_{j=1}^{i-1} sigma_j * delta_j) 是透射率（到达第 i 个点还剩多少光）
- sigma_i 是第 i 个点的密度
- delta_i 是相邻采样点的间距
- c_i 是第 i 个点的颜色

### 直观理解

想象光线穿过一层层半透明的"云"：
- 每一层都会吸收一些光，同时发出自己的颜色
- 越靠前的层对最终颜色贡献越大
- 如果某层密度为 0，光线直接穿过，不产生贡献"""))
    
    a(md_cell("""### 手动计算体渲染

假设一条光线上有 3 个采样点：
- 点 1: sigma=0.5, color=0.8, delta=1.0
- 点 2: sigma=2.0, color=0.3, delta=1.0
- 点 3: sigma=0.1, color=0.9, delta=1.0

计算：
- T1 = 1
  alpha1 = 1 - exp(-0.5*1) = 1 - 0.607 = 0.393
  contribution1 = 1 * 0.393 * 0.8 = 0.314

- T2 = exp(-0.5*1) = 0.607
  alpha2 = 1 - exp(-2.0*1) = 1 - 0.135 = 0.865
  contribution2 = 0.607 * 0.865 * 0.3 = 0.158

- T3 = exp(-0.5-2.0) = exp(-2.5) = 0.082
  alpha3 = 1 - exp(-0.1*1) = 1 - 0.905 = 0.095
  contribution3 = 0.082 * 0.095 * 0.9 = 0.007

最终颜色 = 0.314 + 0.158 + 0.007 = 0.479

> **思考题 4**：如果某一点密度无穷大，会发生什么？透射率 T 呢？"""))
    
    a(code_cell("""# Implement volume rendering (ray marching)
def volume_render(densities, colors, distances):
    '''Volume rendering along a ray.
    
    Args:
        densities: density values at each sample point (N,)
        colors: color values at each sample point (N,)
        distances: distance between sample points (N,) or scalar
    
    Returns:
        pixel_color: final rendered pixel color
        transmittance: remaining transmittance
    '''
    if np.isscalar(distances):
        deltas = np.full_like(densities, distances)
    else:
        deltas = distances
    
    # Alpha values: 1 - exp(-sigma * delta)
    alphas = 1 - np.exp(-densities * deltas)
    
    # Transmittance: T_i = exp(-sum_{j< i} sigma_j * delta_j)
    # Cumulative sum of densities * deltas
    cum_density = np.cumsum(densities * deltas)
    # Shift: T_1 = 1, T_i depends on sum up to i-1
    transmittance = np.exp(-np.concatenate([[0], cum_density[:-1]]))
    
    # Weight of each sample: T_i * alpha_i
    weights = transmittance * alphas
    
    # Final color
    pixel_color = np.sum(weights * colors)
    
    return pixel_color, transmittance, weights

# Manual calculation test
densities = np.array([0.5, 2.0, 0.1])
colors = np.array([0.8, 0.3, 0.9])
deltas = 1.0

pixel_color, trans, weights = volume_render(densities, colors, deltas)
print(f'Manual expected: ~0.479')
print(f'Computed pixel color: {pixel_color:.4f}')
print(f'Transmittance: {trans}')
print(f'Weights: {weights}')
print(f'Sum of weights: {np.sum(weights):.4f}')

# Visualize: volume rendering process
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
positions = np.cumsum([0] + [deltas]*len(densities))[:-1]
ax.bar(positions, densities, width=deltas*0.8, color='#3498DB', alpha=0.7, label='密度 σ')
ax.set_xlabel('光线上的位置', fontsize=12)
ax.set_ylabel('密度', fontsize=12)
ax.set_title('沿光线的密度分布', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
ax.plot(positions, trans, 'o-', lw=2, label='透射率 T', color='#E74C3C')
ax.bar(positions, weights, width=deltas*0.6, color='#2ECC71', alpha=0.6, label='权重 T·α')
ax.set_xlabel('光线上的位置', fontsize=12)
ax.set_ylabel('值', fontsize=12)
ax.set_title(f'透射率与权重 (最终颜色={pixel_color:.3f})', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))
    
    # Section 5: Positional Encoding
    a(md_cell("""## 5. 位置编码：让 MLP 学会高频细节

### 问题

MLP 倾向于学习低频函数，很难表示高频细节（如锐利的边缘、精细的纹理）。

### 解决方案：位置编码

将低维坐标映射到高维的正弦/余弦空间：

gamma(p) = [sin(2^0 * pi * p), cos(2^0 * pi * p),
            sin(2^1 * pi * p), cos(2^1 * pi * p),
            ...,
            sin(2^(L-1) * pi * p), cos(2^(L-1) * pi * p)]

### 为什么有效？

1. 更高的维度 = 更强的表达能力
2. 不同频率捕捉不同尺度的细节
3. 让 MLP 在高维空间中学习更简单的函数

### NeRF 中的位置编码

- 位置 (x,y,z)：L=10 → 60 维
- 视角方向 (θ,φ)：L=4 → 24 维
- 总共 84 维输入"""))
    
    a(code_cell("""# Visualize positional encoding
def positional_encoding_1d(x, L=10):
    '''1D positional encoding.'''
    x = np.asarray(x).reshape(-1, 1)
    encoded = []
    for l in range(L):
        freq = 2 ** l
        encoded.append(np.sin(freq * np.pi * x))
        encoded.append(np.cos(freq * np.pi * x))
    return np.concatenate(encoded, axis=1)

# Show encoding of a 1D signal
x = np.linspace(-1, 1, 500)
L = 6
pe = positional_encoding_1d(x, L)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for l in range(min(6, L)):
    ax = axes[l]
    freq = 2 ** l
    sin_comp = pe[:, 2*l]
    cos_comp = pe[:, 2*l+1]
    ax.plot(x, sin_comp, label=f'sin({freq}πx)', lw=1.5)
    ax.plot(x, cos_comp, label=f'cos({freq}πx)', lw=1.5, ls='--')
    ax.set_title(f'频率级别 l={l} (freq={freq}x)', fontsize=11)
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('值', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.2, 1.2)

plt.suptitle('位置编码：不同频率的正弦/余弦基函数', fontsize=14, y=1.0)
plt.tight_layout()
plt.show()

# How positional encoding helps represent high-frequency functions
# Target: a step function (hard to learn directly)
target = (x > 0).astype(float)  # step function

# Try to fit with MLP: directly vs with PE
# Simple: linear combination of PE features
from numpy.linalg import lstsq

# With positional encoding
A_pe = pe
coeffs_pe, _, _, _ = lstsq(A_pe, target, rcond=None)
fit_pe = A_pe @ coeffs_pe
mse_pe = np.mean((fit_pe - target)**2)

# Without PE (just x + bias)
A_raw = np.column_stack([x, np.ones_like(x)])
coeffs_raw, _, _, _ = lstsq(A_raw, target, rcond=None)
fit_raw = A_raw @ coeffs_raw
mse_raw = np.mean((fit_raw - target)**2)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, target, 'k-', lw=3, label='目标 (阶跃函数)', alpha=0.7)
ax.plot(x, fit_raw, 'r--', lw=2, label=f'线性拟合 (MSE={mse_raw:.4f})')
ax.plot(x, fit_pe, 'b-', lw=2, label=f'位置编码拟合 L={L} (MSE={mse_pe:.4f})')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('位置编码对表示高频函数的帮助', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""))
    
    # Section 6: NeRF limitations & advances
    a(md_cell("""## 6. NeRF 的局限与发展

### 原始 NeRF 的问题

1. **训练慢**：每个场景需要单独训练几小时
2. **渲染慢**：每条光线要采样上百个点，MLP 前向传播
3. **静态场景**：不能处理动态物体
4. **视角有限**：训练视角外的视角可能效果差

### 重要进展

1. **Instant-NGP** [[Müller et al., 2022]](https://arxiv.org/abs/2201.05989)
   - 用哈希网格（Hash Grid）替代部分 MLP
   - 训练速度提升 100 倍，秒级训练
   - 渲染速度也大幅提升

2. **3D 高斯溅射（3D Gaussian Splatting）** [[Kerbl et al., 2023]](https://arxiv.org/abs/2308.04079)
   - 用 3D 高斯分布表示场景
   - 实时渲染（>100 FPS）
   - 质量接近 NeRF

3. **动态 NeRF**：处理动态场景
4. **泛化 NeRF**：从单张图生成 NeRF"""))
    
    # Homework 1
    a(md_cell("""## 作业 1：实现位置编码

### 题目

实现 NeRF 中的位置编码（Positional Encoding）函数。
将输入坐标通过不同频率的正弦/余弦函数映射到高维空间。

### 公式

gamma(p) = [sin(2^0 * pi * p), cos(2^0 * pi * p),
            sin(2^1 * pi * p), cos(2^1 * pi * p),
            ...,
            sin(2^(L-1) * pi * p), cos(2^(L-1) * pi * p)]

对于 D 维输入，输出维度为 2 * L * D。

### 要求
1. 实现 `positional_encoding(x, L)`
2. 支持多维输入 (batch, D)
3. 验证输出形状和基本性质
4. 用 assert 验证"""))
    
    a(code_cell("""# Homework 1: Positional Encoding
def positional_encoding(x, L=10):
    '''NeRF-style positional encoding.
    
    Args:
        x: input coordinates (..., D)
        L: number of frequency levels
    
    Returns:
        encoded: encoded features (..., 2*L*D)
    '''
    # YOUR CODE HERE
    pass

# TODO: uncomment to test
# np.random.seed(42)
# x = np.array([[0.5, -0.3]])  # batch=1, D=2
# enc = positional_encoding(x, L=5)
# assert enc.shape == (1, 20), f'Expected (1, 20), got {enc.shape}'
# # All values should be in [-1, 1]
# assert np.all(enc >= -1.0) and np.all(enc <= 1.0)
# # sin^2 + cos^2 = 1 for each frequency
# assert np.allclose(enc[0, 0]**2 + enc[0, 1]**2, 1.0, atol=1e-6)
# print('All assertions passed!')"""))
    
    a(md_cell("""### 答案

<details>
<summary>点击展开参考答案</summary>

```python
def positional_encoding(x, L=10):
    D = x.shape[-1]
    encoded = []
    for l in range(L):
        freq = 2 ** l
        encoded.append(np.sin(freq * np.pi * x))
        encoded.append(np.cos(freq * np.pi * x))
    return np.concatenate(encoded, axis=-1)
```

</details>"""))
    
    a(code_cell("""# Reference Answer 1
def positional_encoding(x, L=10):
    '''NeRF-style positional encoding.'''
    D = x.shape[-1]
    encoded = []
    for l in range(L):
        freq = 2 ** l
        encoded.append(np.sin(freq * np.pi * x))
        encoded.append(np.cos(freq * np.pi * x))
    return np.concatenate(encoded, axis=-1)

# Tests
np.random.seed(42)

# Test 1: 2D input
x = np.array([[0.5, -0.3]])
enc = positional_encoding(x, L=5)
assert enc.shape == (1, 20), f'Expected (1, 20), got {enc.shape}'
assert np.all(enc >= -1.0) and np.all(enc <= 1.0)
# sin^2 + cos^2 = 1 for first frequency of first dimension
assert np.allclose(enc[0, 0]**2 + enc[0, 1]**2, 1.0, atol=1e-6)

# Test 2: batch of 3D points
points_3d = np.random.randn(10, 3)
enc_3d = positional_encoding(points_3d, L=10)
assert enc_3d.shape == (10, 60)  # 2 * 10 * 3 = 60

# Test 3: known values
x_zero = np.array([[0.0]])
enc_zero = positional_encoding(x_zero, L=3)
# sin(0) = 0, cos(0) = 1 for all frequencies
assert np.all(enc_zero[:, 0::2] == 0.0)  # all sin components are 0
assert np.all(enc_zero[:, 1::2] == 1.0)  # all cos components are 1

print(f'2D encoding shape: {enc.shape}')
print(f'3D encoding shape: {enc_3d.shape}')
print('All assertions passed!')"""))
    
    # Homework 2
    a(md_cell("""## 作业 2：实现体渲染

### 题目

实现体渲染函数（Volume Rendering）。
给定一条光线上采样点的密度和颜色，计算最终的像素颜色。

### 公式

C = sum_{i=1}^{N} T_i * alpha_i * c_i

其中：
- alpha_i = 1 - exp(-sigma_i * delta_i)
- T_i = exp(-sum_{j=1}^{i-1} sigma_j * delta_j)

### 要求
1. 实现 `volume_render(densities, colors, deltas)`
2. 返回最终像素颜色
3. 用 assert 验证"""))
    
    a(code_cell("""# Homework 2: Volume Rendering
def volume_render_pixel(densities, colors, deltas):
    '''Volume rendering for a single ray.
    
    Args:
        densities: density values (N,)
        colors: color values (N,) or (N, 3) for RGB
        deltas: distance between samples (N,) or scalar
    
    Returns:
        pixel_color: final rendered color
        weights: per-sample weights (N,)
    '''
    # YOUR CODE HERE
    pass

# TODO: uncomment to test
# densities = np.array([1.0, 0.0, 3.0])
# colors = np.array([0.5, 0.8, 0.2])
# deltas = 0.5
# pixel_color, weights = volume_render_pixel(densities, colors, deltas)
# assert np.isscalar(pixel_color)
# assert weights.shape == (3,)
# assert np.all(weights >= 0)
# # All weights sum to less than or equal to 1
# assert np.sum(weights) <= 1.0 + 1e-6
# print(f'Pixel color: {pixel_color:.4f}')
# print('All assertions passed!')"""))
    
    a(md_cell("""### 答案

<details>
<summary>点击展开参考答案</summary>

```python
def volume_render_pixel(densities, colors, deltas):
    if np.isscalar(deltas):
        deltas = np.full_like(densities, deltas)
    
    # Alpha: 1 - exp(-sigma * delta)
    alphas = 1 - np.exp(-densities * deltas)
    
    # Transmittance T_i = exp(-sum_{j < i} sigma_j * delta_j)
    cum_density = np.cumsum(densities * deltas)
    transmittance = np.exp(-np.concatenate([[0], cum_density[:-1]]))
    
    # Weights: T_i * alpha_i
    weights = transmittance * alphas
    
    # Final color
    pixel_color = np.sum(weights * colors, axis=0)
    
    return pixel_color, weights
```

</details>"""))
    
    a(code_cell("""# Reference Answer 2
def volume_render_pixel(densities, colors, deltas):
    '''Volume rendering for a single ray.'''
    if np.isscalar(deltas):
        deltas = np.full_like(densities, deltas)
    
    # Alpha: 1 - exp(-sigma * delta)
    alphas = 1 - np.exp(-densities * deltas)
    
    # Transmittance T_i = exp(-sum_{j < i} sigma_j * delta_j)
    cum_density = np.cumsum(densities * deltas)
    transmittance = np.exp(-np.concatenate([[0], cum_density[:-1]]))
    
    # Weights: T_i * alpha_i
    weights = transmittance * alphas
    
    # Final color
    pixel_color = np.sum(weights * colors, axis=0)
    
    return pixel_color, weights

# Tests
np.random.seed(42)
densities = np.array([1.0, 0.0, 3.0])
colors = np.array([0.5, 0.8, 0.2])
deltas = 0.5

pixel_color, weights = volume_render_pixel(densities, colors, deltas)
assert np.isscalar(pixel_color) or pixel_color.shape == ()
assert weights.shape == (3,)
assert np.all(weights >= 0)
assert np.sum(weights) <= 1.0 + 1e-6

# Test: zero density -> zero color
zero_dens = np.zeros(5)
zero_colors = np.ones(5) * 0.5
pc, w = volume_render_pixel(zero_dens, zero_colors, 1.0)
assert pc == 0.0
assert np.all(w == 0.0)

# Test: infinite density at first point -> first color dominates
inf_dens = np.array([100.0, 1.0])
inf_colors = np.array([0.9, 0.1])
pc_inf, w_inf = volume_render_pixel(inf_dens, inf_colors, 1.0)
assert abs(pc_inf - 0.9) < 0.01  # should be close to first color

print(f'Pixel color: {pixel_color:.4f}')
print(f'Weights: {weights.round(4)}')
print(f'Sum of weights: {np.sum(weights):.4f}')
print('All assertions passed!')"""))
    
    # Summary
    a(md_cell("""## 7. 总结

### 本讲要点
1. **3D 视觉**：从 2D 图像恢复 3D 结构
2. **3D 表示**：体素、点云、网格、隐式函数
3. **立体视觉**：双目视差恢复深度
4. **NeRF**：用 MLP 表示神经辐射场，体渲染端到端训练
5. **位置编码**：让 MLP 学会高频细节
6. **进展**：Instant-NGP 加速训练，3DGS 实时渲染

### 常见坑
- 位置编码频率太低 → 模糊，太高 → 伪影
- 光线采样点太少 → 渲染有条纹
- 视角方向缺失 → 无法表示视角相关效果"""))
    
    a(md_cell("""## 参考文献

### 论文
- [[Mildenhall et al., 2020]](https://arxiv.org/abs/2003.08934) **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**
- [[Müller et al., 2022]](https://arxiv.org/abs/2201.05989) **Instant Neural Graphics Primitives with a Multiresolution Hash Encoding**
- [[Kerbl et al., 2023]](https://arxiv.org/abs/2308.04079) **3D Gaussian Splatting for Real-Time Radiance Field Rendering**

### GitHub
- [bmild/nerf](https://github.com/bmild/nerf) (7.4k stars) — NeRF 官方实现
- [NVlabs/instant-ngp](https://github.com/NVlabs/instant-ngp) (14k stars) — Instant-NGP
- [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) (11k stars) — 3D 高斯溅射"""))
    
    return make_notebook(cells), title


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base_dir = r"d:\download\6aa68b099b47c2ba093ab510\cs231n-site\notebooks"
    
    nb1, t1 = build_notebook_transfer_learning()
    save_notebook(nb1, os.path.join(base_dir, "part3-frontiers", "lecture-11-transfer-learning", "practice.ipynb"))
    
    nb2, t2 = build_notebook_ssl()
    save_notebook(nb2, os.path.join(base_dir, "part3-frontiers", "lecture-12-ssl", "practice.ipynb"))
    
    nb3, t3 = build_notebook_generative()
    save_notebook(nb3, os.path.join(base_dir, "part3-frontiers", "lecture-13-generative", "practice.ipynb"))
    
    nb4, t4 = build_notebook_segmentation()
    save_notebook(nb4, os.path.join(base_dir, "part3-frontiers", "lecture-14-segmentation", "practice.ipynb"))
    
    nb5, t5 = build_notebook_clip()
    save_notebook(nb5, os.path.join(base_dir, "part3-frontiers", "lecture-15-clip", "practice.ipynb"))
    
    nb6, t6 = build_notebook_nerf()
    save_notebook(nb6, os.path.join(base_dir, "part3-frontiers", "lecture-16-3d-vision", "practice.ipynb"))
    
    print("\nAll 6 notebooks generated!")