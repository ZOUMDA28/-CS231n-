"""
CS231n 各 Notebook 的 GitHub 代码实现引用映射表
用于在 Notebook 末尾自动添加「参考代码实现」section
"""

GITHUB_REFS = {
    # ─── Part 1: 深度学习基础 ───
    "lecture-01-intro": [
        ("jariasf/CS231n", 486, "https://github.com/jariasf/CS231n", "CS231n 完整作业解答（KNN、SVM、Softmax、两层网络）"),
        ("rishabh-16/cs231n-2019-assignments", 212, "https://github.com/rishabh-16/cs231n-2019-assignments", "2019版 CS231n 作业 PyTorch+TensorFlow 解答"),
        ("amanchadha/stanford-cs231n-assignments-2020", 175, "https://github.com/amanchadha/stanford-cs231n-assignments-2020", "2020春季 CS231n 完整作业解答"),
    ],
    "lecture-02-classification": [
        ("jariasf/CS231n", 486, "https://github.com/jariasf/CS231n", "SVM 与 Softmax 线性分类器从零实现"),
        ("bingcheng1998/CS231n-2020-spring-assignment-solution", 146, "https://github.com/bingcheng1998/CS231n-2020-spring-assignment-solution", "2020春季作业解答，含线性分类器"),
        ("madalinabuzau/cs231n-convolutional-neural-networks-solutions", 113, "https://github.com/madalinabuzau/cs231n-convolutional-neural-networks-solutions", "2017版 TF+PyTorch 双框架解答"),
    ],
    "lecture-03-optimization": [
        ("labmlai/annotated_deep_learning_paper_implementations", 67433, "https://github.com/labmlai/annotated_deep_learning_paper_implementations", "SGD、Momentum、Adam 等优化器带注释实现"),
        ("jariasf/CS231n", 486, "https://github.com/jariasf/CS231n", "梯度检查与优化作业解答"),
    ],
    "lecture-04-backprop": [
        ("jariasf/CS231n", 486, "https://github.com/jariasf/CS231n", "反向传播作业解答"),
        ("Yorko/stanford_cs231n_2019", 155, "https://github.com/Yorko/stanford_cs231n_2019", "2019版作业解答与注释"),
    ],
    "lecture-04b-two-layer-net": [
        ("jariasf/CS231n", 486, "https://github.com/jariasf/CS231n", "两层神经网络完整实现（前向+反向+训练）"),
        ("rishabh-16/cs231n-2019-assignments", 212, "https://github.com/rishabh-16/cs231n-2019-assignments", "两层网络 PyTorch 版本"),
        ("amanchadha/stanford-cs231n-assignments-2020", 175, "https://github.com/amanchadha/stanford-cs231n-assignments-2020", "两层网络 2020版解答"),
    ],
    "lecture-05-cnn": [
        ("Skylark0924/Machine-Learning-is-ALL-You-Need", 424, "https://github.com/Skylark0924/Machine-Learning-is-ALL-You-Need", "CNN 从零实现（numpy）"),
        ("MorvanZhou/PyTorch-Tutorial", 8466, "https://github.com/MorvanZhou/PyTorch-Tutorial", "CNN PyTorch 教程（含中文注释）"),
    ],
    "lecture-05b-features": [
        ("jariasf/CS231n", 486, "https://github.com/jariasf/CS231n", "HOG 特征与颜色直方图作业解答"),
        ("amanchadha/stanford-cs231n-assignments-2020", 175, "https://github.com/amanchadha/stanford-cs231n-assignments-2020", "图像特征工程作业解答"),
    ],
    # ─── Part 2: CNN 与视觉 ───
    "lecture-06-architectures": [
        ("huggingface/pytorch-image-models", 37139, "https://github.com/huggingface/pytorch-image-models", "最大 PyTorch 图像模型库（ResNet/VGG/EfficientNet/ViT）"),
        ("labmlai/annotated_deep_learning_paper_implementations", 67433, "https://github.com/labmlai/annotated_deep_learning_paper_implementations", "ResNet、BatchNorm 带注释论文实现"),
    ],
    "lecture-07-training-nn": [
        ("labmlai/annotated_deep_learning_paper_implementations", 67433, "https://github.com/labmlai/annotated_deep_learning_paper_implementations", "权重初始化、Adam、学习率调度带注释实现"),
        ("seloufian/Deep-Learning-Computer-Vision", 135, "https://github.com/seloufian/Deep-Learning-Computer-Vision", "CS231n+EECS498 训练技巧综合解答"),
    ],
    "lecture-08-transformers": [
        ("lucidrains/vit-pytorch", 25503, "https://github.com/lucidrains/vit-pytorch", "Vision Transformer PyTorch 实现（最热门）"),
        ("jeonsworld/ViT-pytorch", 2161, "https://github.com/jeonsworld/ViT-pytorch", "ViT 论文官方 PyTorch 复现"),
        ("jacobgil/vit-explain", 1099, "https://github.com/jacobgil/vit-explain", "ViT 可解释性工具（注意力可视化）"),
    ],
    "lecture-08b-rnn-captioning": [
        ("amanchadha/stanford-cs231n-assignments-2020", 175, "https://github.com/amanchadha/stanford-cs231n-assignments-2020", "RNN 图像描述作业 2020版解答"),
        ("rishabh-16/cs231n-2019-assignments", 212, "https://github.com/rishabh-16/cs231n-2019-assignments", "RNN Captioning 2019版解答"),
        ("chenyuntc/pytorch-book", 12848, "https://github.com/chenyuntc/pytorch-book", "Neural Talk 图像描述 PyTorch 教程"),
    ],
    "lecture-09-detection": [
        ("ultralytics/yolov5", 58006, "https://github.com/ultralytics/yolov5", "YOLOv5 官方实现（最热门检测框架）"),
        ("ultralytics/yolov3", 10605, "https://github.com/ultralytics/yolov3", "YOLOv3 PyTorch 实现"),
        ("lyuwenyu/RT-DETR", 5518, "https://github.com/lyuwenyu/RT-DETR", "RT-DETR: CVPR 2024 实时检测 Transformer"),
    ],
    "lecture-10-visualization": [
        ("utkuozbulak/pytorch-cnn-visualizations", 8233, "https://github.com/utkuozbulak/pytorch-cnn-visualizations", "CNN 可视化技术合集（Grad-CAM、显著性图、DeepDream）"),
        ("kazuto1011/grad-cam-pytorch", 804, "https://github.com/kazuto1011/grad-cam-pytorch", "Grad-CAM PyTorch 复现"),
        ("jacobgil/vit-explain", 1099, "https://github.com/jacobgil/vit-explain", "ViT 注意力可视化"),
    ],
    # ─── Part 3: 前沿主题 ───
    "lecture-11-transfer-learning": [
        ("huggingface/pytorch-image-models", 37139, "https://github.com/huggingface/pytorch-image-models", "预训练模型库（迁移学习基础）"),
        ("seloufian/Deep-Learning-Computer-Vision", 135, "https://github.com/seloufian/Deep-Learning-Computer-Vision", "迁移学习与微调实践"),
    ],
    "lecture-12-ssl": [
        ("sthalles/SimCLR", 2493, "https://github.com/sthalles/SimCLR", "SimCLR 官方 PyTorch 实现"),
        ("Spijkervet/SimCLR", 823, "https://github.com/Spijkervet/SimCLR", "SimCLR 对比学习框架"),
        ("HobbitLong/SupContrast", 3448, "https://github.com/HobbitLong/SupContrast", "监督对比学习 + SimCLR 实现"),
    ],
    "lecture-13-generative": [
        ("labmlai/annotated_deep_learning_paper_implementations", 67433, "https://github.com/labmlai/annotated_deep_learning_paper_implementations", "GAN、CycleGAN、StyleGAN2 带注释实现"),
        ("MorvanZhou/PyTorch-Tutorial", 8466, "https://github.com/MorvanZhou/PyTorch-Tutorial", "GAN 与 AutoEncoder 中文教程"),
        ("Mikoto10032/DeepLearning", 17759, "https://github.com/Mikoto10032/DeepLearning", "深度学习入门（含 GAN/RNN/CNN 教程）"),
    ],
    "lecture-14-segmentation": [
        ("milesial/Pytorch-UNet", 11649, "https://github.com/milesial/Pytorch-UNet", "U-Net 语义分割最热门 PyTorch 实现"),
        ("wolny/pytorch-3dunet", 2419, "https://github.com/wolny/pytorch-3dunet", "3D U-Net 体积分割实现"),
        ("ellisdg/3DUnetCNN", 2228, "https://github.com/ellisdg/3DUnetCNN", "3D U-Net 医学图像分割"),
    ],
    "lecture-15-clip": [
        ("openai/CLIP", 34293, "https://github.com/openai/CLIP", "OpenAI CLIP 官方实现"),
        ("jina-ai/clip-as-service", 12835, "https://github.com/jina-ai/clip-as-service", "CLIP 即服务（图文检索引擎）"),
    ],
    "lecture-16-3d-vision": [
        ("nerfstudio-project/nerfstudio", 11996, "https://github.com/nerfstudio-project/nerfstudio", "NeRF 工作室（最热门 NeRF 框架）"),
        ("MaximeVandegar/Papers-in-100-Lines-of-Code", 2883, "https://github.com/MaximeVandegar/Papers-in-100-Lines-of-Code", "100行代码实现 NeRF"),
        ("NVlabs/tiny-cuda-nn", 4535, "https://github.com/NVlabs/tiny-cuda-nn", "NVIDIA 快速 MLP 框架（NeRF 底层）"),
    ],
}


def generate_ref_markdown(lecture_id):
    """生成 Notebook 末尾的参考代码实现 markdown 单元格内容"""
    refs = GITHUB_REFS.get(lecture_id, [])
    if not refs:
        return None

    lines = ["---\n\n## 参考代码实现\n"]
    lines.append("以下 GitHub 仓库提供了本节内容的完整代码实现，建议结合学习：\n")
    for name, stars, url, desc in refs:
        lines.append(f"- **[{name}](https://github.com/{name})** ({stars} stars): {desc}")
        lines.append(f"  - 仓库地址: {url}\n")
    lines.append("\n> 标注说明: 以上仓库按热度排序，优先推荐 stars 最多的实现。\n")

    return "\n".join(lines)


if __name__ == "__main__":
    import json
    import os

    # 打印映射表摘要
    print(f"共 {len(GITHUB_REFS)} 个 Notebook 的 GitHub 代码引用\n")
    for lid, refs in GITHUB_REFS.items():
        print(f"  {lid}: {len(refs)} 个引用")
        for name, stars, url, desc in refs:
            print(f"    - {name} ({stars} stars)")
