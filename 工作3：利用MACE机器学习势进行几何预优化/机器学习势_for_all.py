import os
import glob
import numpy as np
import torch
from ase.io import read, write
from ase.optimize import BFGS, FIRE

# 尝试导入 MACE
try:
    from mace.calculators import mace_mp
except ImportError:
    print("【严重错误】无法导入 MACE。")
    print("请终端运行: pip install mace-torch")
    exit()

# ---------------- 配置区域 ----------------
input_folder = "xyzs"               # 输入文件夹
output_folder = "xyzs_optimized (All)"   # 输出文件夹
# ----------------------------------------

def get_correct_5x5_cell():
    """生成 5x5 石墨烯超胞的晶格矩阵"""
    # 基础晶胞 12.336 Å (5x5)
    cell_a = 12.336457
    cell_c = 15.0  # 真空层
    gamma_rad = 120 * np.pi / 180

    vec_a = [cell_a, 0.0, 0.0]
    vec_b = [cell_a * np.cos(gamma_rad), cell_a * np.sin(gamma_rad), 0.0]
    vec_c = [0.0, 0.0, cell_c]
    return np.array([vec_a, vec_b, vec_c])

def find_metal_index(atoms):
    """自动寻找过渡金属原子索引（排除常见非金属）"""
    # 定义非金属元素列表
    non_metals = ['C', 'H', 'O', 'N', 'B', 'P', 'S', 'F', 'Cl', 'Si']
    for atom in atoms:
        if atom.symbol not in non_metals:
            return atom.index
    return -1 

def main():
    print("="*60)
    print("🚀 启动 SOTA 几何优化程序: MACE-MP-0 (带实时迭代输出)")
    print("="*60)

    # 1. 强制设置设备为 CPU
    device = 'cpu'
    print(">>> 运行模式: CPU (Float64 高精度)")

    # 2. 加载 MACE 模型
    print("\n正在加载 MACE 模型...")
    try:
        calc = mace_mp(
            model="medium", 
            device=device, 
            default_dtype="float64" 
        )
        print(">>> MACE-MP-0 (Medium) 加载成功！")
    except Exception as e:
        print(f"ERROR: 模型加载失败: {e}")
        return

    # 3. 准备文件
    os.makedirs(output_folder, exist_ok=True)
    xyz_files = glob.glob(os.path.join(input_folder, "*.xyz"))
    if not xyz_files:
        print(f"在 {input_folder} 中未找到 .xyz 文件。")
        return
    print(f"共发现 {len(xyz_files)} 个结构待优化。")

    correct_cell = get_correct_5x5_cell()

    # 4. 循环优化
    for i, input_file_path in enumerate(xyz_files):
        filename = os.path.basename(input_file_path)
        print(f"\n" + "-"*50)
        print(f"[{i+1}/{len(xyz_files)}] 正在处理: {filename}")
        print("-"*50)
        
        try:
            atoms = read(input_file_path, format='xyz')
            
            # --- 应用晶胞 ---
            atoms.set_cell(correct_cell)
            atoms.set_pbc([True, True, True]) 

            # ==================================================
            # 【核心策略】打破对称性与死锁
            # ==================================================
            metal_idx = find_metal_index(atoms)
            perturbation_z = 0.3  # 定义抬升高度
            original_z = 0.0       # 初始化

            if metal_idx != -1:
                symbol = atoms[metal_idx].symbol
                original_z = atoms.positions[metal_idx, 2]
                
                # 动作 1: Z轴强制抬升
                atoms.positions[metal_idx, 2] += perturbation_z
                
                # 动作 2: 全局微扰
                atoms.rattle(stdev=0.02, seed=42)
                
                print(f"  > 物理微扰: {symbol} 抬升 {perturbation_z}Å, 全局抖动 0.02Å")
            else:
                print("  > 警告: 未找到金属原子，仅应用全局抖动。")
                atoms.rattle(stdev=0.02, seed=42)
            # ==================================================

            atoms.calc = calc

            # 初始化优化器
            opt = BFGS(atoms, logfile=None) # logfile=None 关闭 ASE 默认的丑陋输出

            # --- 定义迭代输出函数 ---
            def print_status():
                step = opt.get_number_of_steps()
                # 只有第1步，或者每5步打印一次
                if step == 0 or step % 5 == 0:
                    pe = atoms.get_potential_energy()
                    forces = atoms.get_forces()
                    fmax = np.sqrt((forces**2).sum(axis=1).max())
                    print(f"    Step {step:3d}: Energy = {pe:.4f} eV | Fmax = {fmax:.4f} eV/A")

            # 将函数挂载到优化器上，interval=1 表示每一步都检查 (但我们在函数内控制了打印频率)
            # 这里直接设 interval=5 更高效
            opt.attach(print_status, interval=5)
            
            # 开始运行
            opt.run(fmax=0.02)
            
            # 最后再打印一次最终状态（确保最后一步能看到）
            pe = atoms.get_potential_energy()
            forces = atoms.get_forces()
            fmax = np.sqrt((forces**2).sum(axis=1).max())
            print(f"    Step {opt.get_number_of_steps():3d}: Energy = {pe:.4f} eV | Fmax = {fmax:.4f} eV/A (FINAL)")

            # 结果分析
            info_str = ""
            if metal_idx != -1:
                final_z = atoms.positions[metal_idx, 2]
                # 计算相对于“未抬升前”的位置变化
                z_change_total = final_z - original_z
                # 计算相对于“抬升后”的位置变化 (即优化过程中掉了多少)
                z_change_relax = final_z - (original_z + perturbation_z)
                
                info_str = f"| 金属 ΔZ(总): {z_change_total:.3f} Å"

            print(f"  > 优化完成. {info_str}")

            # 保存 (extxyz 格式)
            output_path = os.path.join(output_folder, filename)
            write(output_path, atoms, format='extxyz') 
            
        except Exception as e:
            print(f"  > [失败] {e}")

    print("\n" + "="*60)
    print(f"全部完成！结果保存在: {output_folder}")
    print("="*60)

if __name__ == "__main__":
    main()