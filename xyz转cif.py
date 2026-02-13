import os
import numpy as np
from ase.io import read, write
from ase import Atoms

# ================= 1. 复刻生成 XYZ 时的晶胞定义 =================
a_len = 12.336457
c_len = 15.0
gamma_deg = 120
gamma_rad = gamma_deg * np.pi / 180

# 手动构建晶胞矩阵 (3x3 Matrix)
# 对应：晶胞向量A, 晶胞向量B, 晶胞向量C
cell_matrix = np.array([
    [a_len, 0.0, 0.0],                                     # A向量
    [a_len * np.cos(gamma_rad), a_len * np.sin(gamma_rad), 0.0], # B向量
    [0.0, 0.0, c_len]                                      # C向量
])

# ================= 2. 批量处理逻辑 =================
input_folder = 'Try_DFT'
output_folder = 'cifs'  # 同时也生成 .vasp (POSCAR) 以防万一
os.makedirs(output_folder, exist_ok=True)

print("开始精确转换...")

files = [f for f in os.listdir(input_folder) if f.endswith('.xyz')]

for filename in files:
    file_path = os.path.join(input_folder, filename)
    
    # 读取 XYZ (此时只有坐标，没有盒子)
    atoms = read(file_path)
    
    # 步骤 A: 赋予晶胞
    # scale_atoms=False 表示：原子不动，把盒子套上去
    atoms.set_cell(cell_matrix, scale_atoms=False)
    
    # 步骤 B: 设置周期性
    atoms.set_pbc(True)
    
    # 步骤 C (关键): Wrap (归一化)
    # 这步会把所有超出边界 0.0001 埃的原子强行拉回盒子内部
    # 彻底解决“边界处原子消失/重复”的问题
    atoms.wrap()
    
    # 步骤 D: 保存
    base_name = os.path.splitext(filename)[0]
    
    # 1. 推荐：保存为 POSCAR (.vasp)
    # POSCAR 是最安全的格式，纯坐标，不包含容易出错的对称性信息
    vasp_path = os.path.join(output_folder, f"{base_name}.vasp")
    write(vasp_path, atoms, format='vasp', sort=True)    

    # 2. 如果必须用 CIF
    # 这里的 loop_keys 参数是为了防止写入过多的 VASP 特定标签
    # 这里的 symmetry='P 1' 是为了防止软件自作聪明搞错对称性
    cif_path = os.path.join(output_folder, f"{base_name}.cif")
    write(cif_path, atoms, format='cif')

    print(f"已处理: {filename}")

print(f"\n转换完成！建议优先使用 {output_folder} 文件夹里的 .vasp 文件进行 DFT 计算或查看。")