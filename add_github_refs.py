"""
给所有已生成的 Notebook 添加 GitHub 代码实现引用
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from github_refs import GITHUB_REFS, generate_ref_markdown

NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), "notebooks")

# 所有 lecture 目录到 lecture_id 的映射
LECTURE_MAP = {
    "lecture-01-intro": "lecture-01-intro",
    "lecture-02-classification": "lecture-02-classification",
    "lecture-03-optimization": "lecture-03-optimization",
    "lecture-04-backprop": "lecture-04-backprop",
    "lecture-04b-two-layer-net": "lecture-04b-two-layer-net",
    "lecture-05-cnn": "lecture-05-cnn",
    "lecture-05b-features": "lecture-05b-features",
    "lecture-06-architectures": "lecture-06-architectures",
    "lecture-07-training-nn": "lecture-07-training-nn",
    "lecture-08-transformers": "lecture-08-transformers",
    "lecture-08b-rnn-captioning": "lecture-08b-rnn-captioning",
    "lecture-09-detection": "lecture-09-detection",
    "lecture-10-visualization": "lecture-10-visualization",
    "lecture-11-transfer-learning": "lecture-11-transfer-learning",
    "lecture-12-ssl": "lecture-12-ssl",
    "lecture-13-generative": "lecture-13-generative",
    "lecture-14-segmentation": "lecture-14-segmentation",
    "lecture-15-clip": "lecture-15-clip",
    "lecture-16-3d-vision": "lecture-16-3d-vision",
}

def add_refs_to_notebook(nb_path, lecture_id):
    """在 Notebook 末尾添加 GitHub 参考代码引用 markdown 单元格"""
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 检查是否已有参考代码实现 section
    has_ref = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            if "参考代码实现" in src:
                has_ref = True
                break

    if has_ref:
        return False, "already has refs"

    ref_md = generate_ref_markdown(lecture_id)
    if not ref_md:
        return False, "no refs for this lecture"

    # 添加到末尾
    ref_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": ref_md.split('\n')
    }
    # 给每个 source 行加换行
    ref_cell["source"] = [line + '\n' for line in ref_md.split('\n')]

    nb["cells"].append(ref_cell)

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    return True, f"added ref cell ({len(ref_cell['source'])} lines)"


def main():
    added = 0
    skipped = 0
    not_found = 0

    # 遍历所有 notebook
    for root, dirs, files in os.walk(NOTEBOOKS_DIR):
        for f in files:
            if f == "practice.ipynb":
                nb_path = os.path.join(root, f)
                # 从路径提取 lecture_id
                rel = os.path.relpath(nb_path, NOTEBOOKS_DIR)
                parts = rel.split(os.sep)
                lecture_dir = parts[-2]  # e.g. lecture-01-intro

                lecture_id = LECTURE_MAP.get(lecture_dir, lecture_dir)

                ok, msg = add_refs_to_notebook(nb_path, lecture_id)
                if ok:
                    added += 1
                    print(f"  [+] {lecture_dir}: {msg}")
                else:
                    skipped += 1
                    print(f"  [=] {lecture_dir}: {msg}")

    print(f"\n总计: 添加 {added} 个, 跳过 {skipped} 个, 未找到 {not_found} 个")


if __name__ == "__main__":
    main()
