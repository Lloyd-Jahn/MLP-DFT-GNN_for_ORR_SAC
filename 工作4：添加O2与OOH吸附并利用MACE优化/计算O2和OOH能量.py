from ase.build import molecule
from ase import Atoms
from mace.calculators import mace_mp

# 加载模型
calc = mace_mp(model="medium", device="cpu", default_dtype="float64")

# 1. 计算 O2 能量
o2 = molecule("O2")
o2.set_cell([15, 15, 15]) # 大盒子防止自相互作用
o2.set_pbc(True)
o2.calc = calc
e_o2 = o2.get_potential_energy()

# 2. 计算 OOH 能量 (手动构建，与之前一致)
ooh = Atoms('OOH', positions=[(0, 0, 0), (0, 0, 1.3), (0.8, 0, 1.8)])
ooh.set_cell([15, 15, 15])
ooh.set_pbc(True)
ooh.calc = calc
e_ooh = ooh.get_potential_energy()

print("="*40)
print(f"E_O2  = {e_o2:.4f} eV")
print(f"E_OOH = {e_ooh:.4f} eV")
print("请把这两个数值填入筛选脚本中！")
print("="*40)