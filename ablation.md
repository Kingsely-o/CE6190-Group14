---
  📋 Exp4 实验设计

  🎯 输入数据

  固定的5个测试prompts（硬编码在代码中）：
  TEST_PROMPTS = [
      "a professional photograph of a cat sitting on a wooden table",
      "a beautiful sunset over the ocean with orange and pink clouds",
      "a futuristic cityscape with flying cars and neon lights",
      "a close-up portrait of a person with curly hair",
      "a bowl of fresh fruit on a kitchen counter",
  ]

  注意:
  - 不是从 COCO 数据集读取
  - 不是从 data/prompts.txt 读取
  - 就是这5个固定的prompts

  ---
  🔬 实验流程

  消融1: Text Conditioning

  | Variant              | 生成时用的prompt | 评估时用的prompt | 目的     |
  |----------------------|-------------|-------------|--------|
  | Full Text (baseline) | 完整prompt    | 完整prompt    | 基准     |
  | Empty Text           | "" (空字符串)   | 完整prompt    | 测试无文本  |
  | Partial Text         | 前3个词        | 完整prompt    | 测试部分文本 |

  例子:
  - 原prompt: "a professional photograph of a cat sitting on a wooden table"
  - 生成时用 Empty: ""
  - 生成时用 Partial: "a professional photograph"
  - 评估时都用完整prompt

  为什么这样设计？
  - 生成时改变prompt → 控制输入
  - 评估时用完整prompt → 统一标准测量 text-image alignment

  ---
  消融2: Classifier-Free Guidance (CFG)

  | Variant             | guidance_scale | prompt   | 目的     |
  |---------------------|----------------|----------|--------|
  | With CFG (baseline) | 7.5            | 完整prompt | 基准     |
  | Without CFG         | 1.0            | 完整prompt | 移除引导增强 |
  | Unconditional       | 0.0            | 完整prompt | 完全无条件  |

  ---
  消融3: Model Architecture

  | Variant            | 模型   | prompt   | 目的   |
  |--------------------|------|----------|------|
  | SD v1.5 (baseline) | v1.5 | 完整prompt | 基准   |
  | SD v2.1            | v2.1 | 完整prompt | 不同架构 |

  ---
  📊 输出指标

  对每个variant计算:
  1. CLIP Score (主要指标)
    - 衡量生成图像与完整prompt的匹配度
    - 每张图一个分数，取平均
  2. Generation Time
    - 每张图的生成时间
    - 取平均

  输出数据结构:
  {
    "variant_name": "Empty Text",
    "num_images": 5,
    "clip_scores": [0.15, 0.12, 0.09, 0.08, 0.11],
    "avg_clip_score": 0.11,
    "avg_time": 3.2,
    "quality_loss": 0.174,
    "quality_loss_pct": 61.3
  }

  ---
  🎨 生成的图像

  每个消融生成的图像数量:

  消融1 (Text):
  - Full Text: 5张
  - Empty Text: 5张
  - Partial Text: 5张
  - 小计: 15张

  消融2 (CFG):
  - With CFG: 5张
  - Without CFG: 5张
  - Unconditional: 5张
  - 小计: 15张

  消融3 (Model):
  - SD v1.5: 5张
  - SD v2.1: 5张
  - 小计: 10张

  总计: 约40张图像

  ---
  ⏱️ 时间估算

  - 每张图生成时间: ~10-15秒（50 steps）
  - 消融1: 15张 × 12秒 = ~3分钟
  - 消融2: 15张 × 12秒 = ~3分钟
  - 消融3: 10张 × 12秒 = ~2分钟
  - CLIP计算: ~2分钟
  - 总计: ~10-15分钟（加上模型加载）

  ---
  🆚 与 Exp1 的对比

  | 特征   | Exp1 (基准对比)     | Exp4 (消融实验) |
  |------|-----------------|-------------|
  | 输入数据 | COCO 5000张      | 固定5个prompts |
  | 图像数量 | 10000张 (2个模型)   | 40张 (多个配置)  |
  | 评估指标 | FID + CLIP + IS | CLIP + Time |
  | 目的   | 模型绝对性能          | 组件相对贡献      |
  | 时间   | 8-12小时          | 10-15分钟     |

  ---
  ✅ 总结

  Exp4 实验设计:
  输入: 5个固定prompts
  ↓
  对每个消融:
    1. 修改某个组件（移除/禁用/替换）
    2. 用修改后的配置生成5张图
    3. 计算CLIP Score（与完整prompt比较）
  ↓
  输出: 各组件的质量贡献百分比

  关键创新:
  - 生成时用修改的prompt/参数
  - 评估时用完整prompt
  - 这样可以测量"移除组件对text-image alignment的影响"

  ---
