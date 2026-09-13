#!/usr/bin/env python3
"""Execute notebooks by running code cells directly and capturing outputs."""
import json
import os
import sys
import io
import base64
import re
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS_DIR = os.path.join(BASE, "notebooks")

def find_notebooks():
    nbs = []
    for root, dirs, files in os.walk(NOTEBOOKS_DIR):
        for f in files:
            if f.endswith('.ipynb'):
                nbs.append(os.path.join(root, f))
    return sorted(nbs)

def normalize_source(source):
    return ''.join(source) if isinstance(source, list) else source or ''

def fix_source(source):
    # Fix typos
    source = source.replace('traject_ories', 'trajectories')
    # Notebook font settings are overridden by global config in execute_notebook()
    # Replace plt.show() with pass
    source = source.replace('plt.show()', 'pass')
    # Replace plt.savefig() calls - we capture figures directly as base64
    source = re.sub(r"plt\.savefig\([^)]*\)", "pass", source)
    return source

def execute_notebook(path):
    print(f"\n{'='*60}")
    print(f"Executing: {os.path.relpath(path, BASE)}")

    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    nb_dir = os.path.dirname(path)
    original_cwd = os.getcwd()
    os.chdir(nb_dir)

    # Create namespace for execution
    ns = {
        '__name__': '__main__',
        '__file__': path,
        'np': np,
        'plt': plt,
        'matplotlib': matplotlib,
        'os': os,
        'sys': sys,
        'io': io,
        'base64': base64,
        're': re,
        'json': json,
    }

    # Set matplotlib font with Chinese support
    # Try multiple Chinese fonts in order of availability
    from matplotlib import font_manager
    cn_fonts = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Arial Unicode MS', 'DejaVu Sans']
    available = [f.name for f in font_manager.fontManager.ttflist]
    chosen = 'DejaVu Sans'
    for f in cn_fonts:
        if f in available:
            chosen = f
            break
    matplotlib.rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    total_images = 0
    total_texts = 0

    for idx, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'code':
            continue

        source = fix_source(normalize_source(cell.get('source', '')))
        if not source.strip():
            continue

        outputs = []

        # Capture stdout
        old_stdout = sys.stdout
        stdout_buffer = io.StringIO()
        sys.stdout = stdout_buffer

        # Clear any existing figures
        plt.close('all')

        try:
            exec(compile(source, f'<cell_{idx}>', 'exec'), ns)

            # Capture stdout
            stdout_text = stdout_buffer.getvalue()
            if stdout_text.strip():
                outputs.append({
                    'output_type': 'stream',
                    'name': 'stdout',
                    'text': stdout_text,
                })
                total_texts += 1

            # Capture figures as PNG
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                           facecolor=fig.get_facecolor(), edgecolor='none')
                buf.seek(0)
                img_data = base64.b64encode(buf.read()).decode('ascii')
                outputs.append({
                    'output_type': 'display_data',
                    'data': {
                        'image/png': img_data,
                        'text/plain': '<Figure>',
                    },
                    'metadata': {},
                })
                total_images += 1

        except Exception as e:
            tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
            # Simplify traceback for cleaner output
            tb_text = ''.join(tb_lines)
            outputs.append({
                'output_type': 'error',
                'ename': type(e).__name__,
                'evalue': str(e),
                'traceback': [tb_text],
            })
            print(f"  Cell {idx} error: {type(e).__name__}: {e}")
        finally:
            sys.stdout = old_stdout
            plt.close('all')

        cell['outputs'] = outputs
        cell['execution_count'] = idx + 1

    # Restore working directory
    os.chdir(original_cwd)

    # Save notebook with outputs
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"  Outputs: {total_images} images, {total_texts} text outputs")
    return total_images, total_texts

def main():
    notebooks = find_notebooks()
    print(f"Found {len(notebooks)} notebooks to execute")

    total_images = 0
    total_texts = 0
    for nb_path in notebooks:
        try:
            imgs, texts = execute_notebook(nb_path)
            total_images += imgs
            total_texts += texts
        except Exception as e:
            print(f"  FATAL ERROR: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Done! Total: {total_images} images, {total_texts} text outputs")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
