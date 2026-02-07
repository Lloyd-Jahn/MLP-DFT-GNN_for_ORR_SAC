from ase.build import molecule
from ase import Atoms
from ase.optimize import BFGS
from mace.calculators import mace_mp
import numpy as np

# 加载模型
calc = mace_mp(model="medium", device="cpu", default_dtype="float64")

def get_optimized_energy(atoms, name="molecule"):
    """辅助函数：设置盒子、计算器并进行结构优化"""
    # 1. 设置盒子和 PBC (MACE 通常需要 PBC)
    atoms.set_cell([15, 15, 15])
    atoms.set_pbc(True)
    
    # 2. 将分子居中，避免跨越边界
    atoms.center()
    
    # 3. 绑定计算器
    atoms.calc = calc
    
    # 4. 结构优化
    print(f"--- Optimizing {name} ---")
    opt = BFGS(atoms, logfile=None) # logfile=None 不打印详细过程，想看可以设为 '-'
    opt.run(fmax=0.01) # 优化直到最大力小于 0.01 eV/A
    
    # 5. 获取最终能量
    e_pot = atoms.get_potential_energy()
    return e_pot

# 1. 计算 O2
o2 = molecule("O2")
# 可以在优化前给个初始扰动或设置初始磁矩，但在 MACE 中通常由几何决定
e_o2 = get_optimized_energy(o2, "O2")

# 2. 计算 OOH
# 给出的初始猜测
ooh = Atoms('OOH', positions=[(0, 0, 0), (0, 0, 1.3), (0.8, 0, 1.8)])
e_ooh = get_optimized_energy(ooh, "OOH")

print("="*40)
print(f"E_O2 (Optimized)  = {e_o2:.4f} eV")
print(f"E_OOH (Optimized) = {e_ooh:.4f} eV")
print("="*40)