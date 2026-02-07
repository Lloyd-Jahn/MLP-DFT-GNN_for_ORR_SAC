import os
import glob
import numpy as np
import torch
import pandas as pd
from ase.io import read, write
from ase.optimize import BFGS

# 尝试导入 MACE
try:
    from mace.calculators import mace_mp
except ImportError:
    print("【严重错误】无法导入 MACE。")
    print("请终端运行: pip install mace-torch")
    exit()

# ---------------- 配置区域 ----------------
input_folder = "xyzs"              # 原始结构文件夹
output_folder = "xyzs_optimized"   # 优化后输出文件夹
list_file = "to_calc_list.csv"     # 计算列表
warning_log = "optimization_warnings.log" # 未收敛结构记录文件
MAX_STEPS = 500                    # 最大几何优化步数
# ----------------------------------------

def get_correct_5x5_cell():
    """生成 5x5 石墨烯超胞的晶格矩阵"""
    cell_a = 12.336457
    cell_c = 15.0
    gamma_rad = 120 * np.pi / 180
    vec_a = [cell_a, 0.0, 0.0]
    vec_b = [cell_a * np.cos(gamma_rad), cell_a * np.sin(gamma_rad), 0.0]
    vec_c = [0.0, 0.0, cell_c]
    return np.array([vec_a, vec_b, vec_c])

def find_metal_index(atoms):
    non_metals = ['C', 'O', 'N', 'B', 'P', 'S']
    for atom in atoms:
        if atom.symbol not in non_metals:
            return atom.index
    return -1 

def log_warning(message):
    """将警告信息写入日志文件"""
    with open(warning_log, "a") as f:
        f.write(message + "\n")

def main():
    print("="*60)
    print("🚀 启动 MACE 几何优化 (HPC版: 带步数上限与日志记录)")
    print("="*60)

    # 1. 设置设备
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    # 2. 加载 MACE
    try:
        calc = mace_mp(model="medium", device=device, default_dtype="float64")
        print(">>> MACE-MP-0 (Medium) 加载成功！")
    except Exception as e:
        print(f"ERROR: 模型加载失败: {e}")
        return

    # 3. 准备文件与筛选
    os.makedirs(output_folder, exist_ok=True)
    
    all_xyz_files = glob.glob(os.path.join(input_folder, "*.xyz"))
    if not all_xyz_files:
        print(f"在 {input_folder} 中未找到 .xyz 文件。")
        return

    # --- 读取筛选列表 ---
    try:
        if not os.path.exists(list_file):
            print(f"【警告】未找到 {list_file}，将处理所有文件！")
            target_structures = set()
            filter_mode = False
        else:
            df_list = pd.read_csv(list_file)
            target_structures = set(df_list['structure'].astype(str).values)
            filter_mode = True
            print(f">>> 已加载计算列表，包含 {len(target_structures)} 个结构")
    except Exception as e:
        print(f"【读取列表失败】: {e}")
        return

    xyz_files = []
    if filter_mode:
        for f in all_xyz_files:
            fname = os.path.basename(f)
            structure_name = os.path.splitext(fname)[0]
            if structure_name in target_structures:
                xyz_files.append(f)
        print(f">>> 筛选后，共 {len(xyz_files)} 个任务待运行。")
    else:
        xyz_files = all_xyz_files

    if not xyz_files:
        print("没有需要处理的文件。")
        return

    correct_cell = get_correct_5x5_cell()

    # 4. 循环优化
    print(f"\n开始处理... (未收敛结构将记录在 {warning_log})")
    
    for i, input_file_path in enumerate(xyz_files):
        filename = os.path.basename(input_file_path)
        output_path = os.path.join(output_folder, filename)

        # --- 断点续传 ---
        if os.path.exists(output_path):
            if i < 5 or i % 50 == 0:
                print(f"[{i+1}/{len(xyz_files)}] ⏭️  跳过已存在: {filename}")
            continue

        print(f"\n[{i+1}/{len(xyz_files)}] 正在处理: {filename}")
        
        try:
            atoms = read(input_file_path, format='xyz')
            atoms.set_cell(correct_cell)
            atoms.set_pbc([True, True, True]) 

            # --- 物理微扰 ---
            metal_idx = find_metal_index(atoms)
            original_z = 0.0
            perturbation_z = 0.3

            if metal_idx != -1:
                symbol = atoms[metal_idx].symbol
                original_z = atoms.positions[metal_idx, 2]
                atoms.positions[metal_idx, 2] += perturbation_z
                atoms.rattle(stdev=0.02, seed=42)
                print(f"  > 物理微扰: {symbol} 抬升 {perturbation_z}Å, 全局抖动 0.02Å")
            else:
                atoms.rattle(stdev=0.02, seed=42)

            atoms.calc = calc
            opt = BFGS(atoms, logfile=None)

            # --- 打印进度 ---
            def print_status():
                step = opt.get_number_of_steps()
                if step == 0 or step % 10 == 0:
                    pe = atoms.get_potential_energy()
                    forces = atoms.get_forces()
                    fmax = np.sqrt((forces**2).sum(axis=1).max())
                    print(f"    Step {step:3d}: E = {pe:.4f} eV | Fmax = {fmax:.4f}")

            opt.attach(print_status, interval=10)
            
            # --- 【HPC关键】带步数上限的运行 ---
            opt.run(fmax=0.02, steps=MAX_STEPS)
            
            # --- 【HPC关键】收敛性检查与日志 ---
            pe = atoms.get_potential_energy()
            forces = atoms.get_forces()
            final_fmax = np.sqrt((forces**2).sum(axis=1).max())
            
            if opt.get_number_of_steps() >= MAX_STEPS:
                warn_msg = f"{filename}: Not Converged (Steps={MAX_STEPS}, Fmax={final_fmax:.4f})"
                print(f"  ⚠️  [警告] {warn_msg}")
                log_warning(warn_msg)
            
            # 结果分析
            info_str = ""
            if metal_idx != -1:
                final_z = atoms.positions[metal_idx, 2]
                z_change_total = final_z - original_z
                info_str = f"| 金属 ΔZ: {z_change_total:.3f} Å"

            print(f"  > 结束. 能量: {pe:.4f} eV {info_str}")

            write(output_path, atoms, format='extxyz') 
            print(f"  > 已保存至 {output_path}")
            
        except Exception as e:
            err_msg = f"{filename}: FAILED - {e}"
            print(f"  ❌ {err_msg}")
            log_warning(err_msg)

    print("\n" + "="*60)
    print(f"全部完成！")
    print(f"未收敛或失败的结构请检查: {warning_log}")
    print("="*60)

if __name__ == "__main__":
    main()