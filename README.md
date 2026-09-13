# Stanford CS231n | 深度学习与计算机视觉

基于斯坦福 CS231n 课程的交互式教程网站，通过可运行的 Jupyter Notebook 系统学习计算机视觉与深度学习。

> 参考 [modern-llm-notebook](https://github.com/walkinglabs/modern-llm-notebook) 项目模板构建

## 课程概览

| 模块 | 章节 | 主题 |
|------|------|------|
| **深度学习基础** | L1-L4 | 计算机视觉导论、图像分类与线性分类器、正则化与优化、神经网络与反向传播 |
| **CNN 与视觉** | L5-L9 | 卷积神经网络、CNN 经典架构、注意力机制与 Transformer、目标检测与图像分割 |
| **前沿主题** | L12-L13 | 自监督学习、生成模型 |

## 教学方法

每篇 Notebook 遵循四步教学路径：

1. **直觉理解** - 从直觉出发，理解核心概念
2. **手算验证** - 手动计算小规模示例，确保真正理解
3. **代码实现** - 从零实现核心算法，不依赖封装库
4. **实验观察** - 可视化结果，总结规律

## 功能特性

- **直接渲染 .ipynb 文件** - 浏览器内直接解析，无需后端
- **嵌套目录支持** - Notebook 按课程结构组织
- **代码语法高亮** - Python 代码智能高亮
- **数学公式支持** - KaTeX 渲染 LaTeX 公式
- **浅色/深色主题** - 主题模式切换
- **笔记和书签** - 本地存储学习笔记
- **响应式设计** - 移动端完美适配
- **GitHub Actions 自动部署** - 推送即部署

## 技术栈

| 技术 | 用途 |
|------|------|
| React 19 | 前端框架 |
| Vite 6 | 构建工具 |
| Tailwind CSS 4 | 样式框架 |
| KaTeX | 数学公式渲染 |
| Lucide React | 图标库 |

## 项目结构

```
cs231n-site/
├── notebooks/                          # 教程笔记本
│   ├── part1-deep-learning-basics/     # 深度学习基础
│   │   ├── lecture-01-intro/
│   │   │   └── practice.ipynb
│   │   ├── lecture-02-classification/
│   │   ├── lecture-03-optimization/
│   │   └── lecture-04-backprop/
│   ├── part2-cnn-vision/               # CNN 与视觉
│   │   ├── lecture-05-cnn/
│   │   ├── lecture-06-architectures/
│   │   ├── lecture-08-transformers/
│   │   └── lecture-09-detection/
│   └── part3-frontiers/                # 前沿主题
│       ├── lecture-12-ssl/
│       └── lecture-13-generative/
├── papers/                             # 论文研读笔记
│   ├── lecture-01/
│   │   └── NOTES.md
│   └── ...
├── web/                                # React/Vite 前端
│   ├── src/
│   │   ├── components/
│   │   ├── context/
│   │   ├── data/
│   │   ├── hooks/
│   │   ├── styles/
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── .github/workflows/
│   └── deploy.yml                      # 自动部署工作流
└── README.md
```

## 本地运行

```bash
cd web
npm install
npm run dev
```

浏览器打开 http://127.0.0.1:5274

## 部署

推送到 `main` 分支后，GitHub Actions 自动构建并部署到 GitHub Pages。

## 参考来源

- [Stanford CS231n](https://cs231n.stanford.edu/) - 课程官网
- [CS231n Course Notes](https://cs231n.github.io/) - 课程笔记
- 所有 Notebook 中的论文引用均来自 arXiv 或官方出版物

## 许可

仅用于教学目的。
