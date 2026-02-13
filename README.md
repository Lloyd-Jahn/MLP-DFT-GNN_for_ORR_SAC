# AI 设计单原子催化剂

基于机器学习的高性能单原子催化剂自动化设计与筛选系统
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 项目简介

本项目开发了一套完整的计算工作流程，用于高通量设计和筛选石墨烯负载的单原子催化剂（SACs）。通过结合机器学习势（MACE）进行几何优化和图神经网络（GNN）进行性能预测，实现了从结构生成到候选材料筛选的全流程自动化。

### 主要特点

- 🔬 **高通量设计**：自动生成 8000+ 种单原子催化剂结构
- 🚀 **机器学习加速**：使用 MACE-MP-0 机器学习势替代传统 DFT 优化，速度提升 100-1000 倍
- 🧠 **智能筛选**：基于 AttentiveFP 图神经网络预测催化性能
- ⚡ **HPC 友好**：支持 GPU 加速，自动断点续传
- 📊 **完整工作流**：从结构建模到性能预测的端到端解决方案

## 技术栈

| 类别 | 技术 |
|------|------|
| **结构模拟** | ASE (Atomic Simulation Environment) |
| **机器学习势** | MACE (Machine-learning Assisted Chemical Energies) |
| **深度学习** | PyTorch, PyTorch Geometric |
| **图神经网络** | AttentiveFP (Attention-based Graph Neural Network) |
| **数据处理** | Pandas, NumPy |
| **可视化** | Matplotlib, Seaborn |

## 快速开始

### 环境要求

- Python 3.8+
- CUDA（可选，用于 GPU 加速）

### 安装依赖

```bash
# 安装核心依赖
pip install ase mace-torch torch torch-geometric

# 安装数据处理和可视化工具
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 运行完整流程

```bash
# 1. 生成候选结构数据集
cd 工作1：数据集构建与结构建模
python 构建数据集.py
python 自动化建模生成xyz文件.py

# 2. 分层抽样筛选（减少计算量）
cd ../工作2：分层抽样筛选
python 多样性结构筛选.py

# 3. MACE 机器学习势优化
cd ../工作3：利用MACE机器学习势进行几何预优化
python 机器学习势_for_screened.py

# 4. 添加吸附物（O2, OOH）并优化
cd ../工作4：添加O2与OOH吸附并利用MACE优化
python 计算O2和OOH能量.py
python 添加吸附物并优化.py

# 5. 结构合理性筛选
cd ../工作5：结构合理性筛选，获取DFT计算结构
python 结构合理性筛选.py
python 提取目标xyz文件.py

# 6. 图神经网络模型训练
cd ../工作6：图神经网络模型训练
python attentivefp_train.py --data ../Final_Candidates_List.csv --xyz_dir ../xyzs

# 7. 性能预测
cd ../工作7：利用模型进行预测
python predict.py
```

## 项目结构

```
.
├── 工作1：数据集构建与结构建模/
│   ├── 构建数据集.py              # 生成金属-配位组合
│   └── 自动化建模生成xyz文件.py    # 转换为3D坐标
├── 工作2：分层抽样筛选/
│   └── 多样性结构筛选.py          # 分层抽样 15%
├── 工作3：利用MACE机器学习势进行几何预优化/
│   └── 机器学习势_for_screened.py # MACE 优化基底
├── 工作4：添加O2与OOH吸附并利用MACE优化/
│   ├── 计算O2和OOH能量.py         # 参考能量
│   └── 添加吸附物并优化.py        # 构建吸附态
├── 工作5：结构合理性筛选，获取DFT计算结构/
│   ├── 结构合理性筛选.py          # 几何和能量筛选
│   └── 提取目标xyz文件.py         # 提取候选结构
├── 工作6：图神经网络模型训练/
│   └── attentivefp_train.py       # GNN 训练
├── 工作7：利用模型进行预测/
│   └── predict.py                 # 性能预测
├── xyzs/                          # 原始结构文件
├── xyzs_optimized/                # 优化后的基底
├── Adsorbed_Structures/           # 吸附态结构
├── Final_DFT_Structures/          # 最终候选结构
├── 数据集.csv                      # 完整结构列表
├── 分层抽样筛选.csv                # 筛选后的列表
├── Final_Candidates_List.csv      # 最终候选列表
└── xyz转cif.py                    # 格式转换工具
```

## 工作流程详解

### 1️⃣ 数据集构建

- 生成 3d, 4d, 5d 过渡金属与配位元素（C, B, N, O, P, S）的所有组合
- 支持配位数 3 和 4
- 双空位结构包含三种变体：-opp, -pen, -hex
- 输出：~8000+ 种候选结构

### 2️⃣ 分层抽样

- 按金属类型进行 15% 分层抽样
- 保留同分异构体家族完整性
- 输出：~1200 个多样化结构

### 3️⃣ MACE 几何优化

- 使用 MACE-MP-0 (Medium) 机器学习势
- 收敛标准：Fmax < 0.02 eV/Å
- 最大步数：500 步
- 支持 GPU/CPU 自动切换和断点续传

### 4️⃣ 吸附物构建

- 在金属位点添加 O2 和 OOH 吸附物
- 使用 MACE 进行预优化（200 步）
- 计算吸附能和稳定性

### 5️⃣ 结构筛选

筛选标准：

- 金属-吸附物键长 ≤ 3.0 Å
- OOH 吸附能：-3.0 ~ 0.5 eV
- O2 稳定性：-3.0 ~ 0.5 eV

### 6️⃣ GNN 模型训练

- 模型：AttentiveFP（基于注意力机制的图神经网络）
- 节点特征：电负性、共价半径、价电子数、原子质量、周期、族
- 边构建：基于共价半径的动态距离阈值
- 数据标准化：StandardScaler

### 7️⃣ 性能预测

使用训练好的 GNN 模型预测新结构的催化性能

## 核心技术

### 石墨烯超胞参数

```python
晶格常数 a = 12.336457 Å
真空层厚度 c = 15.0 Å
晶格角 γ = 120°
超胞尺寸：5 × 5
周期性边界条件：[True, True, True]
```

### 结构命名规范

格式：`{Metal}_{Elements}{Counts}{Variant}`

示例：

- `Fe_N2O2-hex`：铁 + 2个氮 + 2个氧，六元环变体
- `Pt_C3`：铂 + 3个碳，单空位
- `Ni_N4`：镍 + 4个氮，双空位

### 支持的金属元素

| 系列 | 元素 |
|------|------|
| **3d** | Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn |
| **4d** | Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd |
| **5d** | La, Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg |

## 性能与优化

- ⚡ **GPU 加速**：MACE 优化支持 CUDA，计算速度提升 5-10 倍
- 💾 **断点续传**：自动跳过已完成的结构，支持中断后继续
- 📝 **日志记录**：未收敛结构自动记录到 `optimization_warnings.log`
- 🔄 **并行计算**：适配 HPC 集群，支持批量提交

## 数据格式

### XYZ 文件

```text
49                    # 原子数
Fe_N2O2-hex          # 结构名称
C  0.000  0.000  0.0  # 碳原子坐标
...
Fe 6.168  3.561  3.75 # 金属原子（最后一行）
```

### ExtXYZ 文件

优化后的结构使用 Extended XYZ 格式，包含能量和力信息：

```python
atoms = read('structure.xyz')
energy = atoms.info['energy']  # 总能量 (eV)
forces = atoms.get_forces()    # 原子受力 (eV/Å)
```

## 常见问题

### Q: MACE 优化不收敛怎么办？

A: 未收敛的结构会自动记录到 `optimization_warnings.log`。可以：

1. 增加最大步数（修改 `MAX_STEPS`）
2. 调整收敛标准（修改 `fmax`）
3. 检查初始结构是否合理

### Q: GPU 内存不足？

A: 尝试以下方法：

1. 减小 batch size
2. 使用 CPU 模式（自动检测）
3. 分批处理结构

### Q: 如何查看计算进度？

```bash
# 实时监控日志
tail -f optimization_warnings.log

# 统计已完成的文件数
ls xyzs_optimized/*.xyz | wc -l
```

## 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@software{sac_design_2025,
  title = {AI-Assisted Single-Atom Catalyst Design Framework},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/sac-design}
}
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 致谢

本项目使用了以下开源工具：

- [ASE](https://wiki.fysik.dtu.dk/ase/) - 原子模拟环境
- [MACE](https://github.com/ACEsuit/mace) - 机器学习势能面
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - 图神经网络库

---

**注意**：本项目用于学术研究目的。使用前请确保已正确安装所有依赖项。
