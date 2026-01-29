#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AttentiveFP 训练脚本
主要优势：
1. 全面的节点特征设计
2. 基于共价半径的动态边构建
3. 所有数据标准化
4. 模型超参数自动调节
5. 较强的特征工程
"""

"""
运行代码：
python attentivefp_train.py --data data.csv --xyz_dir xyzs

输出参数名称在第302、494行
"""

import os
import sys
import argparse
import json
import math
import glob
import random
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim

# PyG imports
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn.models import AttentiveFP

# plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------
# 原子属性定义
# 格式：元素: (电负性, 共价半径(pm), 价电子数, 原子质量, 周期, 族)
# -----------------------------
ATOMIC_PROPS = {
    # 配位元素
    "B": (2.04, 85, 3, 10.81, 2, 13),
    "C": (2.55, 76, 4, 12.01, 2, 14),
    "N": (3.04, 71, 5, 14.01, 2, 15),
    "O": (3.44, 66, 6, 16.00, 2, 16),
    "P": (2.19, 106, 5, 30.97, 3, 15),
    "S": (2.58, 102, 6, 32.06, 3, 16),

    # 3d 过渡金属
    "Sc": (1.36, 144, 3, 44.96, 4, 3),
    "Ti": (1.54, 136, 4, 47.87, 4, 4),
    "V":  (1.63, 125, 5, 50.94, 4, 5),
    "Cr": (1.66, 127, 6, 52.00, 4, 6),
    "Mn": (1.55, 127, 7, 54.94, 4, 7),
    "Fe": (1.83, 126, 8, 55.85, 4, 8),
    "Co": (1.88, 125, 9, 58.93, 4, 9),
    "Ni": (1.91, 124, 10, 58.69, 4, 10),
    "Cu": (1.90, 138, 11, 63.55, 4, 11),
    "Zn": (1.65, 131, 12, 65.38, 4, 12),

    # 4d 过渡金属
    "Y":  (1.22, 162, 3, 88.91, 5, 3),
    "Zr": (1.33, 148, 4, 91.22, 5, 4),
    "Nb": (1.60, 137, 5, 92.91, 5, 5),
    "Mo": (2.16, 145, 6, 95.95, 5, 6),
    "Tc": (1.90, 156, 7, 98.91, 5, 7),
    "Ru": (2.20, 126, 8, 101.07, 5, 8),
    "Rh": (2.28, 135, 9, 102.91, 5, 9),
    "Pd": (2.20, 139, 10, 106.42, 5, 10),
    "Ag": (1.93, 153, 11, 107.87, 5, 11),
    "Cd": (1.69, 148, 12, 112.41, 5, 12),

    # 5d 过渡金属
    "Hf": (1.30, 159, 4, 178.49, 6, 4),
    "Ta": (1.50, 146, 5, 180.95, 6, 5),
    "W":  (2.36, 139, 6, 183.84, 6, 6),
    "Re": (1.90, 137, 7, 186.21, 6, 7),
    "Os": (2.20, 135, 8, 190.23, 6, 8),
    "Ir": (2.20, 136, 9, 192.22, 6, 9),
    "Pt": (2.28, 136, 10, 195.08, 6, 10),
    "Au": (2.54, 144, 11, 196.97, 6, 11),
    "Hg": (2.00, 149, 12, 200.59, 6, 12),
    "Tl": (1.62, 148, 13, 204.38, 6, 13),

    # 稀土元素
    "La": (1.10, 187, 3, 138.91, 6, 3),
    "Ce": (1.12, 181, 4, 140.12, 6, 3),
    "Pr": (1.13, 182, 5, 140.91, 6, 3),
    "Nd": (1.14, 181, 6, 144.24, 6, 3),
    "Sm": (1.17, 180, 8, 150.36, 6, 3),
    "Gd": (1.20, 178, 9, 157.25, 6, 3),
}

# 配位原子
配位元素 = ["C", "N", "O", "P", "B", "S"]

# -----------------------------
# 辅助函数：解析 xyz 文件
# -----------------------------
def 解析_xyz(路径: str) -> Tuple[List[str], np.ndarray]:
    """解析 .xyz 文件，返回元素列表和坐标数组"""
    with open(路径, 'r') as f:
        raw_lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if not raw_lines:
        return [], np.zeros((0,3))
    
    atom_lines = []
    try:
        n = int(raw_lines[0].split()[0])
        if len(raw_lines) >= 2 + n:
            atom_lines = raw_lines[2:2+n]
        else:
            atom_lines = [ln for ln in raw_lines if len(ln.split()) >= 4]
    except Exception:
        atom_lines = [ln for ln in raw_lines if len(ln.split()) >= 4]

    元素列表 = []
    坐标列表 = []
    for ln in atom_lines:
        parts = ln.split()
        if len(parts) < 4:
            continue
        elem = parts[0]
        try:
            x, y, z = map(float, parts[1:4])
        except:
            continue
        元素列表.append(elem)
        坐标列表.append([x, y, z])
    
    if len(坐标列表) == 0:
        return [], np.zeros((0,3))
    return 元素列表, np.array( 坐标列表, dtype=float)

# -----------------------------
# 原子选择和特征构建
# -----------------------------
def 选取中心和配位原子(元素列表: List[str], 坐标: np.ndarray,
                   金属猜测: str = None,
                   r_max: float = 4.0) -> Tuple[List[str], np.ndarray, int]:
    """选择中心金属和配位原子"""
    n = len(元素列表)
    metal_idx = None
    
    # 优先选择从文件名中猜测的金属
    if 金属猜测:
        for i, e in enumerate(元素列表):
            if e == 金属猜测:
                metal_idx = i
                break
    
    # 否则选择第一个非配位元素的原子
    if metal_idx is None:
        for i, e in enumerate(元素列表):
            if e not in 配位元素:
                metal_idx = i
                break
    
    if metal_idx is None:
        metal_idx = 0

    metal_coord = 坐标[metal_idx]
    dists = np.linalg.norm(坐标 - metal_coord, axis=1)
    
    # 选择金属和距离在 r_max 内的所有原子
    sel_idxs = [i for i in range(n) if dists[i] <= r_max]
    
    # 确保至少选择金属原子
    if metal_idx not in sel_idxs:
        sel_idxs.append(metal_idx)
    
    sel_idxs.sort()
    new_elements = [元素列表[i] for i in sel_idxs]
    new_coords = 坐标[sel_idxs]
    new_metal_index = sel_idxs.index(metal_idx)
    
    return new_elements, new_coords, new_metal_index

def 构建图_from_xyz(文件路径: str, 元素词表: List[str],
                   r_max: float = 4.0) -> Dict[str, Any]:
    """构建分子图"""
    元素列表, 坐标矩阵 = 解析_xyz(文件路径)
    if len(元素列表) == 0:
        raise ValueError(f"无法解析 {文件路径}")

    # 从文件名猜测金属
    基名 = os.path.basename(文件路径)
    金属猜测 = None
    if "_" in 基名:
        金属猜测 = 基名.split("_")[0]
    else:
        cand = ''.join([c for c in 基名 if c.isalpha()][:2])
        if cand:
            金属猜测 = cand

    # 选择原子
    选取元素, 选取坐标, 新金属索引 = 选取中心和配位原子(元素列表, 坐标矩阵, 金属猜测, r_max)

    # 节点特征构建
    节点特征列表 = []
    for e in 选取元素:
        # one-hot 编码
        oh = [1.0 if e == ue else 0.0 for ue in 元素词表]
        
        # 原子特征
        electr, covrad_pm, val_e, mass, period, group = ATOMIC_PROPS.get(e, 
            (0.0, 100.0, 0, 0, 0, 0))
        
        covrad_A = covrad_pm / 100.0
        
        # 特征选择 + 归一化
        atomic_features = [
            electr / 4.0,  # 归一化电负性
            covrad_A / 2.0,  # 归一化共价半径
            val_e / 18.0,   # 归一化价电子数
            period / 7.0,   # 归一化周期
            group / 18.0,   # 归一化族
            float(e == 选取元素[新金属索引]),  # 是否是中心金属
        ]
        
        feat = oh + atomic_features
        节点特征列表.append(feat)
    
    x = torch.tensor(np.array(节点特征列表, dtype=float), dtype=torch.float)

    # 边构建 - 基于共价半径
    n_nodes = len(选取元素)
    edge_pairs = []
    edge_attrs = []
    
    for i in range(n_nodes):
        elem_i = 选取元素[i]
        cov_rad_i = ATOMIC_PROPS.get(elem_i, (0, 100, 0, 0, 0, 0))[1] / 100.0
        
        for j in range(i + 1, n_nodes):
            elem_j = 选取元素[j]
            cov_rad_j = ATOMIC_PROPS.get(elem_j, (0, 100, 0, 0, 0, 0))[1] / 100.0
            
            # 计算原子间距离
            dij = float(np.linalg.norm(选取坐标[i] - 选取坐标[j]))
            
            # 基于共价半径的动态阈值
            bond_threshold = 1.2 * (cov_rad_i + cov_rad_j)
            
            if dij <= bond_threshold and dij > 0.1:  # 避免自环和过近距离
                # 添加双向边
                edge_pairs.extend([[i, j], [j, i]])
                
                # 边特征
                edge_feat = [
                    dij,  # 距离
                    dij / bond_threshold,  # 归一化距离
                    float(dij <= 1.5),  # 是否是短键
                    float(dij <= 2.0),  # 是否是中等键
                ]
                edge_attrs.extend([edge_feat, edge_feat])

    # 如果没有边，创建基于距离的k近邻图
    if len(edge_pairs) == 0:
        for i in range(n_nodes):
            distances = []
            for j in range(n_nodes):
                if i != j:
                    dij = float(np.linalg.norm(选取坐标[i] - 选取坐标[j]))
                    distances.append((j, dij))
            
            # 每个节点连接最近的3个邻居
            distances.sort(key=lambda x: x[1])
            for j, dij in distances[:3]:
                edge_pairs.extend([[i, j], [j, i]])
                edge_feat = [dij, dij/2.0, 0.0, 0.0]
                edge_attrs.extend([edge_feat, edge_feat])

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.array(edge_attrs, dtype=float), dtype=torch.float)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "n_nodes": n_nodes,
        "metal_index": 新金属索引,
        "元素列表": 选取元素
    }

# -----------------------------
# 数据集构建
# -----------------------------
def 构建数据集(csv_path: str, xyz_dir: str,
             target_col: str = "d_band_center",
             r_max: float = 4.0) -> Tuple[List[Data], List[str], List[str], StandardScaler]:
    """构建数据集，返回数据列表、元素词表、缺失列表和目标值标准化器"""
    df = pd.read_csv(csv_path)
    if df.shape[0] == 0:
        raise ValueError("CSV 文件为空")

    名称列 = df.columns[0]
    xyz_map = {os.path.basename(p): p for p in glob.glob(os.path.join(xyz_dir, "*.xyz"))}

    # 收集所有元素构建词表
    有引用的_xyz = []
    for val in df[名称列].astype(str).unique():
        fname = f"{val}.xyz"
        if fname in xyz_map:
            有引用的_xyz.append(xyz_map[fname])
    
    元素集合 = set()
    for p in 有引用的_xyz:
        elems, _ = 解析_xyz(p)
        元素集合.update(elems)
    
    base_order = 配位元素.copy()
    其余 = sorted([e for e in 元素集合 if e not in base_order])
    元素词表 = base_order + 其余
    print(f"元素词表: {元素词表}")

    # 目标值标准化
    y_values = df[target_col].values.reshape(-1, 1)
    y_scaler = StandardScaler()
    y_scaler.fit(y_values)

    pyg_data_list = []
    缺失列表 = []
    失败列表 = []

    for idx, row in df.iterrows():
        名称 = str(row[名称列])
        xyz_name = f"{名称}.xyz"
        if xyz_name not in xyz_map:
            缺失列表.append(名称)
            continue
        
        路径 = xyz_map[xyz_name]
        try:
            g = 构建图_from_xyz(路径, 元素词表, r_max)
            y_val = float(row[target_col])
            # 标准化目标值
            y_val_scaled = float(y_scaler.transform([[y_val]])[0, 0])
            
            data = Data(
                x=g["x"], 
                edge_index=g["edge_index"], 
                edge_attr=g["edge_attr"], 
                y=torch.tensor([y_val_scaled], dtype=torch.float)
            )
            data.name = 名称
            data.metal_index = g["metal_index"]
            data.元素列表 = g["元素列表"]
            data.original_y = y_val  # 保存原始值用于评估
            pyg_data_list.append(data)
        except Exception as e:
            失败列表.append((路径, str(e)))
    
    print(f"构建完成：有效图数量={len(pyg_data_list)}, 缺失={len(缺失列表)}, 失败={len(失败列表)}")
    return pyg_data_list, 元素词表, 缺失列表, y_scaler

# -----------------------------
# 训练和评估函数
# -----------------------------
def 训练循环(模型, 训练加载器, 优化器, 损失函数, 设备):
    模型.train()
    总损失 = 0.0
    for 批 in 训练加载器:
        批 = 批.to(设备)
        优化器.zero_grad()
        out = 模型(批.x, 批.edge_index, 批.edge_attr, 批.batch)
        if out.dim() > 1:
            out = out.view(-1)
        loss = 损失函数(out, 批.y.view(-1))
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(模型.parameters(), max_norm=1.0)
        优化器.step()
        总损失 += loss.item() * 批.num_graphs
    平均损失 = 总损失 / len(训练加载器.dataset)
    return 平均损失

def 验证循环(模型, 验证加载器, 损失函数, 设备, y_scaler=None):
    模型.eval()
    ys = []
    preds = []
    original_ys = []
    总损失 = 0.0
    
    with torch.no_grad():
        for 批 in 验证加载器:
            批 = 批.to(设备)
            out = 模型(批.x, 批.edge_index, 批.edge_attr, 批.batch)
            if out.dim() > 1:
                out = out.view(-1)
            loss = 损失函数(out, 批.y.view(-1))
            总损失 += loss.item() * 批.num_graphs
            
            # 反标准化预测值
            if y_scaler is not None:
                out_denorm = y_scaler.inverse_transform(out.cpu().numpy().reshape(-1, 1)).flatten()
                preds.extend(out_denorm.tolist())
            else:
                preds.extend(out.cpu().numpy().tolist())
            
            # 收集原始目标值
            if hasattr(批, 'original_y'):
                original_ys.extend(批.original_y.cpu().numpy().tolist())
            else:
                original_ys.extend(批.y.view(-1).cpu().numpy().tolist())
    
    if len(original_ys) == 0:
        return None, None, None
    
    y_all = np.array(original_ys)
    p_all = np.array(preds)
    
    mse = mean_squared_error(y_all, p_all)
    mae = mean_absolute_error(y_all, p_all)
    r2 = r2_score(y_all, p_all)
    avg_loss = 总损失 / len(验证加载器.dataset) if len(验证加载器.dataset) > 0 else 0
    
    return avg_loss, mse, mae, r2, (y_all, p_all)

# -----------------------------
# 主函数
# -----------------------------
def main():
    解析器 = argparse.ArgumentParser(description="AttentiveFP 回归训练")
    解析器.add_argument('--data', type=str, required=True, help='data.csv 路径')
    解析器.add_argument('--xyz_dir', type=str, required=True, help='xyz 文件目录')
    解析器.add_argument('--out_dir', type=str, default='output', help='输出目录')
    解析器.add_argument('--epochs', type=int, default=300, help='训练轮数')
    解析器.add_argument('--batch_size', type=int, default=16, help='batch size')
    解析器.add_argument('--lr', type=float, default=5e-4, help='学习率')
    解析器.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    解析器.add_argument('--val_size', type=float, default=0.1, help='验证集比例')
    解析器.add_argument('--seed', type=int, default=42, help='随机种子')
    解析器.add_argument('--r_max', type=float, default=4.0, help='配位原子选择半径')
    
    args = 解析器.parse_args()

    # 固定随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    输出目录 = args.out_dir
    os.makedirs(输出目录, exist_ok=True)

    # 构建数据集
    print("开始构建图数据...")
    pyg_list, 元素词表, 缺失列表, y_scaler = 构建数据集(
        args.data, args.xyz_dir, "d_band_center", args.r_max
    )
    
    n_total = len(pyg_list)
    if n_total == 0:
        print("错误：没有构建到任何图。")
        sys.exit(1)

    # 数据集划分
    idxs = list(range(n_total))
    train_plus_idx, test_idx = train_test_split(idxs, test_size=args.test_size, random_state=args.seed)
    
    if args.val_size > 0 and len(train_plus_idx) > 1:
        train_idx, val_idx = train_test_split(train_plus_idx, test_size=args.val_size, random_state=args.seed)
    else:
        train_idx, val_idx = train_plus_idx, []

    train_set = [pyg_list[i] for i in train_idx]
    val_set = [pyg_list[i] for i in val_idx] if len(val_idx) > 0 else []
    test_set = [pyg_list[i] for i in test_idx]

    print(f"样本统计：总计={n_total} 训练={len(train_set)} 验证={len(val_set)} 测试={len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False) if len(val_set) > 0 else None
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # 构建模型
    in_channels = pyg_list[0].x.size(1)
    edge_dim = pyg_list[0].edge_attr.size(1) if pyg_list[0].edge_attr is not None else 0
    
    print(f"in_channels={in_channels}, edge_dim={edge_dim}, 元素词表长度={len(元素词表)}")
    
    设备 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", 设备)

    # 模型架构
    模型 = AttentiveFP(
        in_channels=in_channels,
        hidden_channels=256,  # 增加隐藏层维度
        out_channels=1,
        edge_dim=edge_dim,
        num_layers=4,  # 增加层数
        num_timesteps=2,  # 增加时间步
        dropout=0.1  # 添加dropout防止过拟合
    ).to(设备)

    # Adam优化器
    优化器 = optim.AdamW(模型.parameters(), lr=args.lr, weight_decay=1e-5)
    损失函数 = nn.MSELoss()

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(优化器, mode='min', factor=0.5, patience=10)

    # 训练循环
    最佳_val_mse = float('inf')
    历史 = {
        "train_loss": [], 
        "val_loss": [], 
        "val_mse": [], 
        "val_mae": [], 
        "val_r2": [],
        "learning_rate": []
    }
    
    保存模型路径 = os.path.join(输出目录, "best_attentivefp.pt")

    for 轮 in range(1, args.epochs + 1):
        train_loss = 训练循环(模型, train_loader, 优化器, 损失函数, 设备)
        
        if val_loader is not None and len(val_loader.dataset) > 0:
            val_loss, val_mse, val_mae, val_r2, _ = 验证循环(模型, val_loader, 损失函数, 设备, y_scaler)
        else:
            val_loss, val_mse, val_mae, val_r2, _ = 验证循环(模型, test_loader, 损失函数, 设备, y_scaler)
        
        历史["train_loss"].append(train_loss)
        历史["val_loss"].append(val_loss if val_loss is not None else float('nan'))
        历史["val_mse"].append(val_mse if val_mse is not None else float('nan'))
        历史["val_mae"].append(val_mae if val_mae is not None else float('nan'))
        历史["val_r2"].append(val_r2 if val_r2 is not None else float('nan'))
        历史["learning_rate"].append(优化器.param_groups[0]['lr'])

        # 学习率调度
        if val_mse is not None:
            scheduler.step(val_mse)

        if 轮 % 10 == 0 or 轮 == 1:
            print(f"轮次 {轮}/{args.epochs}:")
            print(f"  训练损失: {train_loss:.6f}")
            if val_mse is not None:
                print(f"  验证 MSE: {val_mse:.6f}, MAE: {val_mae:.6f}, R²: {val_r2:.4f}")
            print(f"  学习率: {优化器.param_groups[0]['lr']:.2e}")

        # 保存最佳模型
        if val_mse is not None and val_mse < 最佳_val_mse:
            最佳_val_mse = val_mse
            torch.save({
                'model_state_dict': 模型.state_dict(),
                'optimizer_state_dict': 优化器.state_dict(),
                'best_val_mse': 最佳_val_mse,
                'epoch': 轮,
                'element_vocab': 元素词表,
                'y_scaler': y_scaler
            }, 保存模型路径)
            print(f"  已保存最佳模型 (val_mse={val_mse:.6f})")

    # 加载最佳模型进行最终测试
    if os.path.exists(保存模型路径):
        checkpoint = torch.load(保存模型路径, map_location=设备, weights_only=False)
        模型.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载最佳模型 (val_mse={checkpoint['best_val_mse']:.6f})")

    # 最终评估
    test_loss, test_mse, test_mae, test_r2, (y_true, y_pred) = 验证循环(
        模型, test_loader, 损失函数, 设备, y_scaler
    )
    
    test_rmse = math.sqrt(test_mse) if test_mse is not None else None
    
    print("\n最终测试集评估:")
    print(f"MSE:  {test_mse:.6f}")
    print(f"RMSE: {test_rmse:.6f}")
    print(f"MAE:  {test_mae:.6f}")
    print(f"R²:   {test_r2:.4f}")

    # 保存结果
    with open(os.path.join(输出目录, "train_history.json"), 'w') as f:
        json.dump(历史, f, indent=2, ensure_ascii=False)

    # 绘制损失曲线
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(历史["train_loss"], label="Train loss")
    plt.plot(历史["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.title("Training and Validation Loss")

    plt.subplot(2, 2, 2)
    plt.plot(历史["val_mse"], label="Val MSE", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.title("Validation MSE")

    plt.subplot(2, 2, 3)
    plt.plot(历史["val_r2"], label="Val R²", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.legend()
    plt.grid(True)
    plt.title("Validation R²")

    plt.subplot(2, 2, 4)
    plt.semilogy(历史["learning_rate"], label="Learning rate", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.grid(True)
    plt.title("Learning Rate Schedule")

    plt.tight_layout()
    plt.savefig(os.path.join(输出目录, "training_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 绘制Parity图
    if y_true is not None and y_pred is not None:
        plt.figure(figsize=(6.5, 6))
        plt.scatter(y_true, y_pred, alpha=0.7, s=50, edgecolors='w', linewidth=0.5)
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        margin = (max_val - min_val) * 0.05
        
        plt.plot([min_val - margin, max_val + margin], 
                 [min_val - margin, max_val + margin], 
                 'r--', alpha=0.8, linewidth=2, label='Prefect prediction')
        
        plt.xlabel("True attached_energy_ev (eV)")
        plt.ylabel("Predicted attached_energy_ev (eV)")
        plt.title(f"Parity Plot (Test set, R² = {test_r2:.4f})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(输出目录, "parity_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 保存详细结果
        res_df = pd.DataFrame({
            "true": y_true.tolist(), 
            "pred": y_pred.tolist(),
            "error": (y_true - y_pred).tolist()
        })
        res_df.to_csv(os.path.join(输出目录, "detailed_results.csv"), index=False)

    # 保存元信息
    元信息 = {
        "n_total_graphs": n_total,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "元素词表": 元素词表,
        "in_channels": int(in_channels),
        "edge_dim": int(edge_dim),
        "best_val_mse": float(最佳_val_mse) if 最佳_val_mse < float('inf') else None,
        "test_mse": float(test_mse) if test_mse is not None else None,
        "test_rmse": float(test_rmse) if test_rmse is not None else None,
        "test_mae": float(test_mae) if test_mae is not None else None,
        "test_r2": float(test_r2) if test_r2 is not None else None,
        "缺失_xyz_count": len(缺失列表),
        "model_parameters": sum(p.numel() for p in 模型.parameters()),
        "training_epochs": 轮
    }
    
    with open(os.path.join(输出目录, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(元信息, f, indent=2, ensure_ascii=False)

    print(f"\n所有输出已保存到: {输出目录}")
    print("脚本完成。")

if __name__ == "__main__":
    main()