import os
import glob
import numpy as np
import torch
from ase import Atoms
from ase.io import read, write
from ase.build import add_adsorbate, molecule
from ase.optimize import BFGS
from mace.calculators import mace_mp

# ---------------- 配置区域 ----------------
input_folder = "xyz_optimzed"     # 输入: 已优化好的基底
output_dir = "Adsorbed_Structures"  # 输出: 吸附态结构
warning_log = "adsorption_warnings.log" # 未收敛记录
MAX_STEPS = 200                     # 吸附预优化最大步数
LOG_INTERVAL = 10                   # 每隔多少步打印一次进度
# ----------------------------------------

def get_correct_5x5_cell():
    """保持一致的 5x5 晶胞"""
    cell_a = 12.336457
    cell_c = 15.0
    gamma_rad = 120 * np.pi / 180
    vec_a = [cell_a, 0.0, 0.0]
    vec_b = [cell_a * np.cos(gamma_rad), cell_a * np.sin(gamma_rad), 0.0]
    vec_c = [0.0, 0.0, cell_c]
    return np.array([vec_a, vec_b, vec_c])

def find_metal_index(atoms):
    non_metals = ['C', 'H', 'O', 'N', 'B', 'P', 'S', 'F', 'Cl', 'Si']
    for atom in atoms:
        if atom.symbol not in non_metals:
            return atom.index
    return -1

def log_warning(message):
    with open(warning_log, "a") as f:
        f.write(message + "\n")

def main():
    print("="*60)
    print("🚀 启动吸附物构建 (HPC GPU版: 实时监控 + 断点续传)")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)

    # ---- 1. GPU 自动识别与加载 MACE ----
    print("\n正在检测硬件环境...")
    if torch.cuda.is_available():
        device = "cuda"
        print(">>> ✅ 检测到 GPU，将使用 CUDA 加速！")
    else:
        device = "cpu"
        print(">>> ⚠️ 未检测到 GPU，自动切换到 CPU 模式。")

    print("正在加载 MACE 模型 ...")
    try:
        calc = mace_mp(
            model="medium", 
            device=device,            # 自动应用检测到的设备
            default_dtype="float64"   # 保持高精度
        )
        print(f">>> MACE-MP-0 (Medium) 已加载到 {device.upper()}！")
    except Exception as e:
        print(f"ERROR: 模型加载失败: {e}")
        return

    # ---- 2. 准备文件 ----
    if not os.path.exists(input_folder):
        print(f"错误: 找不到输入文件夹 {input_folder}")
        return
    
    xyz_files = glob.glob(os.path.join(input_folder, "*.xyz"))
    print(f"共发现 {len(xyz_files)} 个基底结构待处理。")
    
    # ---- 3. 循环处理 ----
    for i, filepath in enumerate(xyz_files):
        filename = os.path.basename(filepath)
        base_name = os.path.splitext(filename)[0]
        
        # 进度提示
        print(f"\n" + "-"*50)
        print(f"[{i+1}/{len(xyz_files)}] 正在处理基底: {base_name}")
        print("-" * 50)

        try:
            slab = read(filepath)
            slab.set_cell(get_correct_5x5_cell())
            slab.set_pbc([True, True, True])

            metal_idx = find_metal_index(slab)
            if metal_idx == -1:
                print(f"  > 跳过: 未找到金属原子")
                log_warning(f"{filename}: Skipped - No Metal Found")
                continue
            
            metal_pos = slab.positions[metal_idx]
            site_xy = (metal_pos[0], metal_pos[1])

            # 定义吸附物
            OOH_mol = Atoms('OOH', positions=[(0, 0, 0), (0, 0, 1.3), (0.8, 0, 1.8)])
            adsorbates = {
                "O2": molecule("O2"),
                "OOH": OOH_mol
            }

            for name, mol in adsorbates.items():
                out_name = f"{base_name}_with_{name}.xyz"
                fname = os.path.join(output_dir, out_name)
                
                # ---- 【关键】断点续传检测 ----
                if os.path.exists(fname):
                    print(f"  > ⏭️  跳过已存在文件: {out_name}")
                    continue

                print(f"  > 构建吸附态: {name} ...")
                
                # 构建
                atoms = slab.copy()
                if name == "O2":
                    add_adsorbate(atoms, mol, height=2.0, position=site_xy)
                elif name == "OOH":
                    add_adsorbate(atoms, mol, height=2.1, position=site_xy, mol_index=0)
                
                # 微扰
                atoms.rattle(stdev=0.02, seed=42)
                atoms.calc = calc
                
                # ---- 优化与实时监控 ----
                opt = BFGS(atoms, logfile=None) # 关闭默认 logfile，使用自定义打印
                
                # 定义实时回调函数
                def print_status():
                    step = opt.get_number_of_steps()
                    if step == 0 or step % LOG_INTERVAL == 0:
                        pe = atoms.get_potential_energy()
                        forces = atoms.get_forces()
                        fmax = np.sqrt((forces**2).sum(axis=1).max())
                        print(f"    Step {step:3d}: E = {pe:.4f} eV | Fmax = {fmax:.4f} eV/A")

                # 挂载回调
                opt.attach(print_status, interval=1) # 这里的 interval=1 表示每步都调，我们在函数内控制打印频率
                
                # 运行优化
                opt.run(fmax=0.05, steps=MAX_STEPS)
                
                # ---- 收敛性检查 ----
                forces = atoms.get_forces()
                final_fmax = np.sqrt((forces**2).sum(axis=1).max())
                final_pe = atoms.get_potential_energy()

                if opt.get_number_of_steps() >= MAX_STEPS:
                    msg = f"{out_name}: Not Converged (Steps={MAX_STEPS}, Fmax={final_fmax:.4f})"
                    print(f"  ⚠️  [警告] {msg}")
                    log_warning(msg)
                else:
                    print(f"    Step {opt.get_number_of_steps():3d}: E = {final_pe:.4f} eV | Fmax = {final_fmax:.4f} eV/A (Done)")
                
                # 保存
                write(fname, atoms, format='extxyz')
                print(f"  > ✅ 已保存: {out_name}")

        except Exception as e:
            print(f"  ❌ [错误] 处理 {filename} 时发生异常: {e}")
            log_warning(f"{filename}: CRASHED - {e}")
            continue

    print("\n" + "="*60)
    print(f"吸附构建任务全部完成！")
    print(f"未收敛记录: {warning_log}")
    print("="*60)

if __name__ == "__main__":
    main()