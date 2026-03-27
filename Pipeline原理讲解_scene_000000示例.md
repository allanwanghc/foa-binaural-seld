# 数据集生成 Pipeline 原理讲解 — 以 scene_000000 为例

## 总览：数据流全貌

```
Medley-solos-DB (21,571 mono WAV clips, 44.1kHz)
        │
        ▼  [Step 1: 预处理]
归一化 + 重采样 → processed/{instrument}/*.wav (48kHz, -23 LUFS)
        │
        ▼  [Step 2: 场景规划]
scene_plan.json (10,000 scenes × 实验因素组合)
        │
        ├── Room RIRs (SOFA)        ◄── [Step 3: 下载 TAU RIR 数据库]
        │
        ▼  [Step 4: FOA 场景生成]
SpatialScaper: 干声 ⊛ RIR → 4ch FOA (W,Y,Z,X)
        │
        ├── HRTF (SOFA)             ◄── [已下载: KU100, CIPIC, SADIE II]
        │
        ▼  [Step 5: Binaural 渲染]
SH-domain 解码: FOA × HRTF滤波器 → 2ch Binaural (L,R)
        │
        ▼  [Step 6-7: QC + 划分]
最终数据集: 10,000 scenes × {foa_WXYZ.wav, binaural_LR.wav, labels.tsv, meta.json}
```

---

## Step 1: 音频预处理

**输入**: Medley-solos-DB 原始文件（44.1kHz, 各种响度）

**处理**:
1. **响度归一化** → ITU-R BS.1770-4 标准，目标 -23 LUFS
   - 用 `pyloudnorm` 库测量积分响度（考虑人耳 K-weighting 频率响应）
   - 线性增益调整到 -23 LUFS，确保所有 clip 的感知响度一致
   - 为什么 -23 LUFS？这是 EBU R128 广播标准，保证混合后不过载

2. **重采样** → 44.1kHz → 48kHz
   - 用 `librosa.resample`（polyphase anti-aliasing 滤波器）
   - 48kHz 是因为 HRTF (KU100, SADIE II) 和 RIR 都是 48kHz 原生采样率

3. **按乐器分类存储** → `processed/{instrument_name}/*.wav`
   - SpatialScaper 要求 foreground 音频按类名子目录组织
   - 8 类: clarinet, distorted_electric_guitar, female_singer, flute, piano, tenor_saxophone, trumpet, violin

**关键文件**: `scripts/01b_process_medleysolosdb.py`, `src/audio_utils.py`

---

## Step 2: 场景参数预采样

**目的**: 在生成前，确定性地规划全部 10,000 个场景的参数，确保实验因素均衡覆盖。

**实验因素网格**:
| 因素 | 取值 | 数量 |
|------|------|------|
| Room | bomb_shelter, gym, sc203 | 3 |
| HRTF | ku100, cipic, sadie_ii | 3 |
| 乐器 | 8 类 | 8 |
| Azimuth | -90° 到 +90°, 15°步长 | 13 |
| Elevation | -30° 到 +30°, 10°步长 | 7 |
| Sources/scene | 2 或 3 | 2 |
| Duration | 2.0 – 6.0 s | 连续 |

**采样策略**:
- **Room × HRTF**: 轮转循环（3×3=9 种组合），每种 ~1,111 场景 → 完美均衡
- **乐器**: 加权采样，优先选择出现次数少的类 → 最终每类 ~12.5%
- **同场景内**: 乐器不重复，空间位置不重复（从 13×7=91 个 az/el 组合中抽取）
- **音频文件**: 每类内无放回抽取，耗尽后重新洗牌 → 最大化多样性
- **种子固定**: `seed=42` → 完全可复现

**输出**: `output/scene_plan.json` (7.7 MB, 10,000 个场景的完整参数)

**关键文件**: `src/scene_plan.py`

---

## Step 3: Room Impulse Response (RIR)

**原理**: RIR 是声音从声源到麦克风的完整传播路径记录（直达声 + 早期反射 + 晚期混响）。将干声与 RIR 卷积，就相当于把干声"放进"那个房间。

**数据来源**: TAU-SRIR 数据库（芬兰坦佩雷大学测量的 9 个真实房间）
- 用 Eigenmike 球形麦克风阵列在不同位置测量
- 转换为 FOA 格式的 SOFA 文件（每个房间一个 .sofa）
- 每个 SOFA 文件包含数百至数千个 source→receiver 的 RIR 测量

**我们选的 3 个房间**（不同混响特性）:
| 房间 | 描述 | RIR 测量数 | 空间范围 |
|------|------|-----------|---------|
| bomb_shelter | 大型地下掩体，石墙，高混响 | 6,480 | 10m × 10m |
| gym | 大型开放体育馆，高混响 | 6,480 | 10m × 10m |
| sc203 | 小教室，地毯，低混响 | 1,592 | 10.7m × 7m |

**为什么不用 METU？** METU 房间只有 `mic` 格式（四面体麦克风），没有 `foa` 格式。我们需要 FOA。

**关键文件**: `scripts/03_download_rirs.py`, `scripts/03b_recover_rirs.py`

---

## Step 4: FOA 场景生成（核心步骤）

这是整个 pipeline 最关键的一步。它的作用：**把多个干声乐器 clip "放进"一个虚拟房间的指定位置，生成 4 通道 FOA 音频**。

### 4.1 什么是 FOA？

First-Order Ambisonics 用 4 个通道表示 3D 声场：

```
通道 0: W = 全向（omnidirectional）— 像一个普通麦克风
通道 1: Y = 左右（left-right）    — sin(az)·cos(el)
通道 2: Z = 上下（up-down）       — sin(el)
通道 3: X = 前后（front-back）    — cos(az)·cos(el)
```

这 4 个通道本质是 **球谐函数 (Spherical Harmonics)** 的 0 阶和 1 阶：
- W 是 0 阶（均匀全向）
- Y, Z, X 是 1 阶（三个正交的指向性图案）

**类比**: W 记录"总能量"，Y/Z/X 分别记录"声音偏左/右多少"、"偏上/下多少"、"偏前/后多少"。这四个信号组合起来，就能还原出任意方向来的声音。

**ACN/N3D**: 通道排序标准（Ambisonic Channel Number）和归一化标准（fully Normalized 3D）。

### 4.2 SpatialScaper 的工作流程

对每个场景（例如 bomb_shelter 房间 + 2 个源）:

```python
# 1. 加载房间的全部 RIR
all_irs, ir_sr, all_ir_xyzs = load_rir_pos("bomb_shelter_foa.sofa")
# all_irs: (6480, 4, N_samples) — 6480个位置 × 4通道FOA × RIR长度
# all_ir_xyzs: (6480, 3) — 每个RIR对应的XYZ坐标

# 2. 对每个源：
for source in scene["sources"]:
    # a) 将目标 (azimuth, elevation) 转为房间 XYZ 坐标
    xyz = azel_to_room_xyz(az=-45°, el=-20°, distance=0.96m, receiver=[0,0,0])

    # b) 找到最近的 RIR（snap to nearest measured position）
    ir_idx = argmin(||all_ir_xyzs - xyz||)
    rir = all_irs[ir_idx]  # shape: (4, N)

    # c) 卷积：dry_audio ⊛ rir → spatialized_audio
    #    对每个FOA通道分别卷积
    for ch in range(4):
        spatialized[ch] = convolve(dry_audio, rir[ch])

    # d) SNR 缩放 + 叠加到输出
    output_audio += scale(spatialized, snr=15dB)
```

**关键**: 卷积 = 将干声"放进"房间的那个位置。RIR 编码了：
- **直达声**：最先到达的部分 → 决定方向
- **早期反射**：几毫秒后的墙壁反射 → 提供空间感
- **晚期混响**：持续衰减的扩散声 → 提供房间大小感

### 4.3 为什么 elevation 会偏？

场景规划指定 `elevation = -20°`，但实际标签显示 `-12°`。原因：
- SpatialScaper 选择**最近的已测量 RIR 位置**
- TAU 数据库的测量网格是物理离散的（不是所有位置都有 RIR）
- 最近的测量点可能在 elevation -12° 而不是 -20°
- **这是正确行为** — 标签反映的是实际渲染方向，不是理想方向

### 4.4 输出

每个场景产出：
- `foa_WXYZ.wav`: 4 通道，48kHz，32-bit float
- `labels.tsv`: 100fps，每帧记录活跃源的 (class_id, source_id, azimuth, elevation)
- `meta.json`: 房间参数、接收器位置、源文件路径等

**关键文件**: `scripts/04_generate_scenes.py`

---

## Step 5: FOA → Binaural 渲染

**目的**: 把 4 通道 FOA 转换为 2 通道 Binaural（戴耳机能听到 3D 空间感的立体声）。

### 5.1 SH-domain HRTF 解码原理

核心思路：**HRTF 本身可以用球谐函数展开**。

```
HRTF 直接方法（不可行）:
  对每个方向 θ → 查表 → 取对应的 HRIR → 卷积

SH-domain 方法（我们用的）:
  1. 将 HRTF 数据集在球谐域做最小二乘拟合
  2. 得到 4 个解码滤波器/每耳 (对应 W, Y, Z, X)
  3. 直接卷积 FOA 4通道 → 2通道 binaural
```

数学上：

```
设 H_L(θ,φ,t) = 左耳 HRIR 在方向(θ,φ)的时域响应

将 H_L 展开为球谐函数:
  H_L(θ,φ,t) ≈ Σ_{l,m} h_L^{lm}(t) · Y_{lm}(θ,φ)

对于 FOA (1阶), 只需 4 个系数:
  H_L ≈ h_L^W(t)·Y_00 + h_L^Y(t)·Y_1(-1) + h_L^Z(t)·Y_10 + h_L^X(t)·Y_11

Binaural 左耳信号:
  L(t) = W(t)⊛h_L^W(t) + Y(t)⊛h_L^Y(t) + Z(t)⊛h_L^Z(t) + X(t)⊛h_L^X(t)

同理右耳 R(t)。
```

### 5.2 解码滤波器计算

```python
# Y: (M, 4) — M个测量方向的球谐值
# H: (M, N) — M个方向的HRIR (每耳)

# 最小二乘: h = (Y^T Y)^{-1} Y^T H
decode_filters = pinv(Y.T @ Y) @ Y.T @ H  # shape: (4, N)
```

这给出 4 个 FIR 滤波器/每耳。对每个 FOA 通道分别卷积再相加，就是 binaural 信号。

### 5.3 三个 HRTF 集

| HRTF | 来源 | 测量数 | 采样率 | 头型 |
|------|------|--------|--------|------|
| KU100 | TH Köln dummy head | 16,020 | 48kHz | 人工头 |
| CIPIC Subject 003 | UC Davis | 1,250 | 44.1kHz | 真人 |
| SADIE II H3 | York University | 2,818 | 48kHz | 真人 |

不同 HRTF = 不同的头型和耳廓 → 不同的空间感知 → 增加数据多样性。

**关键文件**: `src/binaural_renderer.py`, `scripts/05_render_binaural.py`

---

## Step 6: 质量控制

4 项检查：
1. **文件完整性**: FOA 4ch, Binaural 2ch, 采样率 48kHz
2. **能量检查**: RMS > -45 dBFS（非静音）
3. **时长匹配**: 音频时长 ≈ 场景计划的 duration
4. **标签一致性**: 标签中的 source 数 = meta.json 中的 num_sources

**关键文件**: `src/qc.py`, `scripts/06_quality_control.py`

---

## Step 7: 数据集划分

- **70% / 15% / 15%** → train / val / test
- **分层策略**: 按 room × hrtf_set (9种组合) 分层 → 每个子集内 9 种组合比例相同
- **完整性**: 同一场景的 FOA + Binaural + labels 始终在同一子集

**关键文件**: `scripts/07_split_dataset.py`

---

## 最终数据集结构

```
output/
├── scene_plan.json              # 10,000 场景的完整参数
├── splits.json                  # train/val/test 划分
├── qc_report.csv               # 质量控制报告
└── scenes/
    ├── scene_000000/
    │   ├── foa_WXYZ.wav         # 4ch FOA (W,Y,Z,X), 48kHz
    │   ├── binaural_LR.wav      # 2ch Binaural (L,R), 48kHz
    │   ├── labels.tsv           # 100fps SELD 标签
    │   └── meta.json            # 场景元数据
    ├── scene_000001/
    │   └── ...
    └── scene_009999/
        └── ...
```

每个场景包含 2-3 个乐器源，在 3 个房间 × 3 个 HRTF 的组合下渲染。
总计约 **~50-80 GB** 的音频数据。

---
---

# 具体例子：scene_000000 是怎么做出来的

## 第 1 步：场景规划阶段确定参数

`scene_plan.py` 用 seed=42 生成 10,000 个场景。第 0 个场景分配到：

```
scene_id: 0
room: "bomb_shelter"          ← 轮转第 1 个 (bomb_shelter → gym → sc203 → bomb_shelter → ...)
hrtf_set: "ku100"             ← 轮转第 1 个 (ku100 → cipic → sadie_ii → ...)
duration: 4.96 秒             ← random.uniform(2.0, 6.0)
num_sources: 2                ← random.randint(2, 3)
```

两个源的参数：

| | 源 0 | 源 1 |
|---|---|---|
| **乐器** | female_singer (class=2) | violin (class=7) |
| **Azimuth** | -45° (右前方) | 90° (正左方) |
| **Elevation** | -20° (略低于水平) | -20° |
| **音频文件** | `Medley-solos-DB_test-2_c83d534c...wav` | `Medley-solos-DB_test-7_68a0ea34...wav` |

乐器选择用了加权采样（优先选出现少的类），位置从 13×7=91 个 az/el 组合中随机抽 2 个不重复的。

## 第 2 步：准备干声音频

这两个 WAV 文件来自 Step 1 预处理：

```
原始 Medley-solos-DB clip (44.1kHz, 各种响度)
    ↓ pyloudnorm 测量 → 当前 LUFS
    ↓ 线性增益调整 → -23 LUFS
    ↓ librosa.resample → 48kHz
    ↓ 保存为 32-bit float WAV
处理后: 2.97秒, 48kHz, 单声道
```

## 第 3 步：加载 bomb_shelter 房间的 RIR

```python
# 从 SOFA 文件加载
bomb_shelter_foa.sofa  (1.4 GB)

内容:
  - 6,480 个测量位置（遍布 10m × 10m × 2.4m 的掩体空间）
  - 每个位置有一个 4 通道 FOA 格式的 RIR
  - 房间坐标范围: X=[-5, 5]m, Y=[-5, 5]m, Z=[-1.34, 1.06]m

接收器（虚拟麦克风）位置 = 房间中心 = [0, 0, -0.14]m
```

## 第 4 步：坐标转换 — 方位角 → 房间XYZ

我们的场景规划用 (azimuth, elevation)，但 SpatialScaper 需要房间 XYZ 坐标：

```
safe_distance = min(房间半径) × 0.8 = min(5, 5, 1.2) × 0.8 = 0.96m

源 0 (female_singer): az=-45°, el=-20°, d=0.96m
  dx = 0.96 × cos(-20°) × cos(-45°) = 0.64m
  dy = 0.96 × cos(-20°) × sin(-45°) = -0.64m    ← 负Y = 右边
  dz = 0.96 × sin(-20°) = -0.33m                 ← 负Z = 下方
  → XYZ = [0.64, -0.64, -0.47]  (加上 receiver 偏移)

源 1 (violin): az=90°, el=-20°, d=0.96m
  dx = 0.96 × cos(-20°) × cos(90°) ≈ 0           ← cos(90°) ≈ 0
  dy = 0.96 × cos(-20°) × sin(90°) = 0.90m       ← 正Y = 左边
  dz = -0.33m
  → XYZ = [0.00, 0.90, -0.47]
```

## 第 5 步：RIR 匹配 + 卷积（核心！）

这是 SpatialScaper 做的最关键的事：

```
对每个源:
  ① 在 6,480 个 RIR 测量点中找最近的
     源0 目标 [0.64, -0.64, -0.47] → 找到最近 RIR 位置
     (最近位置的 elevation 实际为 -12°，不是 -20°，因为测量网格是离散的)

  ② 取出这个位置的 RIR (4通道, 每通道是一个脉冲响应)
     rir.shape = (4, ~24000)  ← 4个FOA通道 × 约0.5秒的脉冲响应

  ③ 加载干声，归一化到峰值=1
     female_singer: shape=(142560,) = 2.97秒 @ 48kHz

  ④ 卷积 (这一步把干声"放进"了 bomb_shelter 的那个位置！)
     for ch in [W, Y, Z, X]:
         spatialized[ch] = fftconvolve(dry_audio, rir[ch])

     结果: spatialized.shape = (4, ~166000)

     为什么卷积=空间化？
     ─────────────────
     RIR 编码了: 直达声(方向) + 早期反射(空间感) + 晚期混响(房间大小)
     卷积 = 把干声的每一个采样点都替换成一次"房间响应"
     → 听起来就像这个声音真的从那个位置发出，在那个房间里回荡

  ⑤ SNR 缩放
     snr = random(10, 25) dB → 控制这个源相对于参考电平的响度
     spatialized *= scale_factor

  ⑥ 叠加到输出
     output_4ch[onset_sample : onset_sample + len(spatialized)] += spatialized
```

两个源都从 time=0 开始，叠加后得到 **4.96 秒的 4 通道 FOA mix**。

### 实际结果验证：

```
FOA 通道能量:
  W (omni):        -31.6 dBFS  ← 最强，因为 W 捕获所有方向
  Y (left-right):  -35.0 dBFS  ← 较强，因为两个源都有明显的左右偏移
  Z (up-down):     -36.3 dBFS  ← 中等，两个源都略低于水平面
  X (front-back):  -37.6 dBFS  ← 较弱，因为 violin 在 az=90°(纯左)，X分量≈0
```

**这符合预期！** violin 在正左方 (az=90°)，所以它对 Y 通道贡献大、对 X 通道贡献约为 0。female_singer 在 az=-45° 右前方，同时贡献 X 和 Y。

## 第 6 步：生成 SELD 标签

SpatialScaper 在空间化过程中记录每帧的源信息（10fps），我们上采样到 100fps：

```
原始 10fps 标签 (SpatialScaper 输出):
  frame=0: class=2(female_singer), source=0, az=-45°, el=-12°, dist=2.56m
  frame=0: class=7(violin),        source=1, az=90°,  el=-10°, dist=2.54m
  frame=1: class=2, source=0, az=-45°, el=-12°
  ...
  frame=28: class=7, source=1, az=90°, el=-10°
  (共 29 帧 × 2 源 = 58 行)

上采样到 100fps (每个 10fps 帧重复 10 次):
  frame=0~9:   同上
  frame=10~19: 同上
  ...
  frame=280~289: 最后一批
  (共 290 帧 × 2 源 = 580 行)
```

注意 elevation 的变化：
- 计划: -20° → 实际标签: **-12° (source 0), -10° (source 1)**
- 这是因为 SpatialScaper **选了最近的 RIR 测量点**，那个点的 elevation 不是正好 -20°

## 第 7 步：FOA → Binaural 渲染

场景规划指定 `hrtf_set = "ku100"`，所以用 KU100 人工头的 HRTF：

```python
# KU100 HRTF: 16,020 个方向的测量，每个方向 128 采样点的脉冲响应

# 预计算: 球谐域解码滤波器
# Y: (16020, 4) — 每个测量方向的球谐系数
# H_L: (16020, 128) — 左耳 HRIR
# h_L = pinv(Y^T @ Y) @ Y^T @ H_L → shape: (4, 128)  ← 4个解码滤波器
# h_R = 同理右耳

# 渲染:
Left  = W⊛h_L[0] + Y⊛h_L[1] + Z⊛h_L[2] + X⊛h_L[3]
Right = W⊛h_R[0] + Y⊛h_R[1] + Z⊛h_R[2] + X⊛h_R[3]

# 每个 ⊛ 是一次 fftconvolve (48000×4.96s ⊛ 128 采样 = 非常快)
```

### 实际结果：

```
Binaural:
  Left:  -31.5 dBFS
  Right: -32.0 dBFS

L/R 差异 ≈ 0.5 dB
```

右耳略弱，因为 female_singer 在右前方 (az=-45°) 直达声先到右耳（稍强），但 violin 在正左方 (az=90°) 对左耳贡献更多。两者综合后，左右能量接近但不完全相等。**这正是空间信息！** 戴耳机听的时候，你会感到 female_singer 偏右、violin 在左边。

## 第 8 步：保存

最终 `output/scenes/scene_000000/` 包含：

| 文件 | 大小 | 内容 |
|------|------|------|
| `foa_WXYZ.wav` | 1.8 MB | 4ch × 238,080 samples × 32bit float |
| `binaural_LR.wav` | 1.8 MB | 2ch × 238,080 samples × 32bit float |
| `labels.tsv` | 11 KB | 580 行 @ 100fps (2 源 × 290 帧) |
| `meta.json` | 1.1 KB | 完整元数据 |

---

## 一句话总结

```
两段干声(女声+小提琴)
  → 各自与 bomb_shelter 对应位置的 RIR 卷积 → 叠加 → 4ch FOA
  → FOA × KU100 球谐解码滤波器 → 2ch Binaural
  + 标签记录每帧每个源的方位角和仰角
```
