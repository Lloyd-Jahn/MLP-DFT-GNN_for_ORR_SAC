import os
import sys
import numpy as np
import torch
import pandas as pd
from ase import Atoms
from ase.io import read, write
from ase.build import add_adsorbate
from ase.optimize import BFGS
from mace.calculators import mace_mp

# ================= ⚙️ 配置中心 =================
CANDIDATE_CSV = "Final_Candidates_List.csv"
INPUT_SLAB_DIR = "xyzs_optimized"
OUTPUT_DIR = "Final_DFT_Structures_Another"
ERROR_DIR = "Crash_Reports"            # 崩坏报告存放目录
MAX_STEPS = 200
ENERGY_THRESHOLD = -3000.0             # 崩坏阈值 (eV)
# ==============================================

def find_metal_index(atoms):
    """找到单原子金属的索引 (含非金属排除列表)"""
    non_metals = set(['H', 'He', 'B', 'C', 'N', 'O', 'F', 'Ne', 
                      'Si', 'P', 'S', 'Cl', 'Ar', 'Se', 'Br', 'Kr', 'I'])
    for atom in atoms:
        if atom.symbol not in non_metals:
            return atom.index
    return -1

def write_crash_report(name, energy, reason):
    """生成崩坏报告 txt"""
    os.makedirs(ERROR_DIR, exist_ok=True)
    report_path = os.path.join(ERROR_DIR, f"CRASH_{name}.txt")
    with open(report_path, "w") as f:
        f.write(f"Structure: {name}\n")
        f.write(f"Reason: {reason}\n")
        f.write(f"Last Energy: {energy:.4f} eV\n")
        f.write("Status: Discarded due to physical unreasonableness.\n")
    print(f"🚨 已生成崩坏报告: {report_path}")

def main():
    print("=" * 70)
    print("🚀  补全 O* 吸附结构优化")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 读取 CSV
    if not os.path.exists(CANDIDATE_CSV):
        print(f"❌ 找不到名单文件 {CANDIDATE_CSV}")
        return

    try:
        df = pd.read_csv(CANDIDATE_CSV)
        if "Name" in df.columns:
            names = df["Name"].tolist()
        else:
            names = df.iloc[:, 0].tolist()
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return

    # 加载 MACE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  计算设备: {device}")
    try:
        calc = mace_mp(model="medium", device=device, default_dtype="float64")
    except Exception as e:
        print(f"❌ MACE 加载失败: {e}")
        return

    print(f"📋 计划处理: {len(names)} 个结构\n")

    for i, name in enumerate(names):
        base_name = name.replace(".xyz", "")
        source_slab_path = os.path.join(INPUT_SLAB_DIR, f"{base_name}.xyz")
        
        # 检查源文件
        if not os.path.exists(source_slab_path):
            continue 

        # 检查目标文件是否已存在
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_with_O.xyz")
        if os.path.exists(output_path):
            print(f"⏩ [{i+1}/{len(names)}] {base_name}_with_O 已存在，跳过。")
            continue

        print(f"\n🔹 [{i+1}/{len(names)}] 正在处理: {base_name} ...")

        try:
            slab = read(source_slab_path)
            metal_idx = find_metal_index(slab)
            
            if metal_idx == -1:
                print(f"⚠️  未找到金属原子，跳过。")
                continue
            
            site_xy = slab.positions[metal_idx][:2]

            # === 构建 O* 结构 ===
            atoms_ads = slab.copy()
            add_adsorbate(atoms_ads, Atoms("O"), height=1.7, position=site_xy)
            atoms_ads.calc = calc
            
            # === 定义熔断器 ===
            class SafetyFuse:
                def __init__(self): 
                    self.exploded = False
                    self.last_energy = 0.0
                def check(self):
                    # 获取当前能量
                    e = atoms_ads.get_potential_energy()
                    self.last_energy = e
                    if e < ENERGY_THRESHOLD:
                        self.exploded = True
                        raise RuntimeError(f"Energy Crash ({e:.2f} eV)")

            fuse = SafetyFuse()
            
            # === 运行优化 ===
            opt = BFGS(atoms_ads, logfile='-') 
            opt.attach(fuse.check, interval=1)

            try:
                opt.run(fmax=0.02, steps=MAX_STEPS)
                
                # 优化成功，保存
                write(output_path, atoms_ads)
                print(f"✅ {base_name}_with_O 保存成功！")

            except RuntimeError as e:
                # 捕获熔断异常
                print(f"💥 {base_name} 模型崩坏！原因: {e}")
                write_crash_report(base_name, fuse.last_energy, str(e))
            
            except Exception as e:
                # 捕获其他优化异常 (如不收敛)
                print(f"❌ {base_name} 优化中断: {e}")
                # 这种情况也建议记录一下
                write_crash_report(base_name, -999, f"Optimization Error: {e}")

        except Exception as e:
            print(f"❌ 读取或构建失败: {e}")

    print("\n" + "=" * 70)
    print(f"🎉 全部任务结束！")
    print(f"📂 正常结果: {OUTPUT_DIR}")
    print(f"📂 崩坏报告: {ERROR_DIR} (如果有)")

if __name__ == "__main__":
    main()