# Stable Diffusion Model Comparison (CE6190)

比较 Stable Diffusion v1.5 和 v2.1 的性能，包含基准测试、类别分析和消融实验。

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 测试环境
python run_all.py --test_setup

# 3. 运行实验
python run_all.py
```

---

## 项目结构

```
├── run_all.py              # 主运行脚本
├── config.py               # 配置参数
├── experiments/            # 4个实验
│   ├── exp1_baseline.py      # 实验1: 模型对比 (必做, 8-12h)
│   ├── exp2_categories.py    # 实验2: 类别分析 (必做, 30-60min)
│   ├── exp3_hyperparams.py   # 实验3: 超参数分析 (可选, 1-2h)
│   └── exp4_ablation.py      # 实验4: 消融实验 (推荐, 1h)
├── data/                   # 数据加载
├── models/                 # 模型加载
├── evaluation/             # 评估指标
├── visualization/          # 可视化
└── results/                # 实验结果
```

---

## 4个实验

### Exp1: 基准对比（必做）
在5000张COCO图像上对比两个模型的FID、CLIP Score、IS。
```bash
python run_all.py --exp1_only
# 快速测试: python run_all.py --exp1_only --num_samples 1000
```

### Exp2: 类别分析（必做）
测试5个类别（simple/scenes/multi-object/detailed/hard）各20个prompts。
```bash
python run_all.py --exp2_only
```

### Exp3: 超参数分析（可选）
分析guidance_scale和inference_steps的影响。
```bash
python run_all.py --include_exp3
```

### Exp4: 消融实验（推荐）
**真正的消融实验**：移除组件看性能下降。

3个消融：
- Text Conditioning: 完整文本 vs 空文本 vs 部分文本
- Classifier-Free Guidance: 有CFG vs 无CFG
- Model Architecture: SD v1.5 vs SD v2.1

```bash
python run_all.py --exp4_only  # 只需1小时
```

**为什么重要**: 直接对应报告Section 3.2 (Ablation Study)，学术价值高。

---

## 常用命令

```bash
# 运行所有必做实验
python run_all.py

# 运行全部实验（包含可选）
python run_all.py --include_exp3 --include_exp4

# 只运行消融实验（1小时，推荐）
python run_all.py --exp4_only

# 快速测试（减少样本）
python run_all.py --num_samples 1000

# 只生成图表
python run_all.py --plots_only
```

---

## 实验输出

```
results/
├── exp1/
│   ├── sd_v15/              # 5000张生成图
│   ├── sd_v21/              # 5000张生成图
│   └── exp1_results.json
├── exp2/
│   ├── sd_v15/              # 按类别组织
│   ├── sd_v21/
│   └── exp2_results.json
├── exp3/
│   └── exp3_results.json
├── exp4_ablation/           # 消融实验
│   ├── text_full/
│   ├── text_empty/
│   ├── cfg_with_cfg/
│   ├── model_v15/
│   └── exp4_results.json    ⭐ 最重要
└── figures/
    ├── model_comparison.png
    ├── category_comparison.png
    └── exp4_ablation/
        └── component_contributions.png  ⭐⭐⭐ 最重要的图
```

**可视化命令**:
```bash
# 自动生成所有图表（包括exp4）
python run_all.py --plots_only

# 或单独运行
python visualization/plot_results.py      # 生成exp1/2/3的图
python visualization/plot_exp4.py         # 只生成exp4的图
```

---

## 报告写作

| 报告章节 | 使用实验 | 关键文件 |
|---------|---------|---------|
| 3.1 超参数分析 | Exp 3 | exp3_results.json |
| **3.2 消融实验** | **Exp 4** | **exp4_results.json** ⭐ |
| 4.1 基准对比 | Exp 1 | exp1_results.json |
| 4.2 类别分析 | Exp 2 | exp2_results.json |

**Section 3.2 示例**:
```markdown
### 3.2 Ablation Study

#### Text Conditioning
| Variant | CLIP Score | Loss |
|---------|-----------|------|
| Full text | 0.285 | - |
| Empty text | 0.098 | -65.6% |

Finding: Text conditioning is the most critical component (65.6% contribution).

#### Component Ranking
1. Text Conditioning: 65.6% ⭐⭐⭐⭐⭐
2. Classifier-Free Guidance: 16.8% ⭐⭐⭐⭐
3. Model Architecture: 8.2% ⭐⭐⭐
```

---

## 配置

编辑 `config.py`:
```python
COCO_NUM_SAMPLES = 5000  # 改为1000加快测试
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 7.5
DEVICE = "cuda"  # 或 "cpu"
```

---

## 时间规划

**最小方案（4-5小时）**:
```bash
python run_all.py --num_samples 1000  # 3h
python run_all.py --exp4_only         # 1h
```

**完整方案（10-12小时）**:
```bash
python run_all.py --include_exp3 --include_exp4
```

---

## 故障排除

**CUDA内存不足**:
```python
# config.py
ENABLE_ATTENTION_SLICING = True
ENABLE_VAE_SLICING = True
```

**太慢**:
```bash
python run_all.py --num_samples 1000
```

**没有GPU**:
```python
# config.py
DEVICE = "cpu"  # 会非常慢
```

---

## 实验vs报告对应

- **Exp1** (基准对比) → 报告4.1节
- **Exp2** (类别分析) → 报告4.2节
- **Exp3** (超参数) → 报告3.1节
- **Exp4** (消融) → 报告3.2节 ⭐ **最重要**

**注意**: Exp3是超参数调优，Exp4是真消融实验（移除组件）。

---

## 核心要点

1. **必做**: Exp1 + Exp2
2. **推荐**: Exp4（消融实验，1小时，学术价值最高）
3. **最重要的图**: `results/figures/true_ablation/component_contributions.png`
4. **可缩短时间**: 用 `--num_samples 1000` 把Exp1从12小时减到3小时

---

Good luck! 🚀
