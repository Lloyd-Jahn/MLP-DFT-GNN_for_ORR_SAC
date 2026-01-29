import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ase.io import read
from tqdm import tqdm
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# ================= 🛡️ 配置中心 (Configuration) =================
# 1. 输入/输出路径
DIR_SLAB = "xyzs_optimized"        
DIR_ADS  = "Adsorbed_Structures"   
OUTPUT_CSV = "Final_Candidates_List.csv"
OUTPUT_REPORT = "Screening_Report.txt"
ERROR_LOG = "Error_Log.txt"

# 2. 能量基准 (eV) 
E_REF_OOH = -12.9619  
E_REF_O2  = -9.6869   

# 3. 筛选阈值 (修正版)
# 3.1 几何阈值 (放宽以减少误杀)
MAX_BOND_LENGTH = 3

# 3.2 能量阈值 (eV)
OOH_MIN = -3.0          # 收紧：-8.0 -> -3.0
OOH_MAX = 0.5           
O2_STABILITY_MAX = 0.5
O2_STABILITY_MIN = -3.0

# ================= 🛠️ 工具函数库 =================

def get_metal_info(atoms):
    non_metals = set(['H', 'He', 'B', 'C', 'N', 'O', 'F', 'Ne', 
                      'Si', 'P', 'S', 'Cl', 'Ar', 'Se', 'Br', 'Kr'])
    for atom in atoms:
        if atom.symbol not in non_metals:
            return atom.symbol, atom.index
    return "Unknown", -1

def robust_get_energy(atoms):
    try:
        if atoms.calc is not None:
            return atoms.get_potential_energy()
        elif 'energy' in atoms.info:
            return atoms.info['energy']
        else:
            return atoms.get_potential_energy()
    except:
        return None

def analyze_geometry(atoms_slab, atoms_ads, filename):
    try:
        metal_sym, metal_idx = get_metal_info(atoms_ads)
        if metal_idx == -1: return False, 99.9, "No Metal"
        
        metal_pos = atoms_ads.positions[metal_idx]
        n_slab = len(atoms_slab)
        n_total = len(atoms_ads)
        
        if n_total <= n_slab: return False, 0.0, "Atom Count Error"
            
        ads_pos = atoms_ads.positions[n_slab:]
        dists = np.linalg.norm(ads_pos - metal_pos, axis=1)
        min_dist = np.min(dists)
        
        # 严格检查：如果距离太近（比如 < 1.0），说明模型炸了，原子重叠
        if min_dist < 0.8:
            return False, min_dist, "Atom Overlap"

        is_valid = min_dist <= MAX_BOND_LENGTH
        return is_valid, min_dist, metal_sym
        
    except Exception as e:
        return False, 0.0, f"Geo Error: {str(e)}"

# ================= 🚀 主程序逻辑 =================

def main():
    print("="*80)
    print(f"{'Screening Pipeline':^80}")
    print("="*80)
    
    # 1. 文件扫描
    files_ooh = glob.glob(os.path.join(DIR_ADS, "*_with_OOH.xyz"))
    total_files = len(files_ooh)
    print(f"待处理文件总数: {total_files}")

    results = []
    
    # 详细的统计账本
    audit_log = {
        "missing_partner": 0,   # 缺少配对文件
        "read_error": 0,        # 文件损坏无法读取
        "energy_missing": 0,    # 读不到能量
        "geo_overlap": 0,       # 原子重叠（模型炸了）
        "geo_detached_ooh": 0,  # OOH 飞了
        "geo_detached_o2": 0,   # O2 飞了
        "o2_weak": 0,           # O2 吸附太弱
        "o2_strong": 0,         # O2 吸附太强
        "ooh_weak": 0,          # OOH 吸附太弱
        "ooh_strong": 0,        # OOH 吸附太强
        "valid_candidate": 0    # 最终入选
    }

    # 清空错误日志
    with open(ERROR_LOG, "w") as ferr:
        ferr.write(f"Error Log - {time.ctime()}\n================================\n")

    # 2. 核心循环
    for f_ooh in tqdm(files_ooh, desc="Screening", unit="file"):
        basename = os.path.basename(f_ooh).split("_with_")[0]
        
        try:
            # 路径构造
            f_o2 = os.path.join(DIR_ADS, f"{basename}_with_O2.xyz")
            f_slab = os.path.join(DIR_SLAB, f"{basename}.xyz")

            # [Check 1] 文件是否存在
            if not os.path.exists(f_o2) or not os.path.exists(f_slab):
                audit_log["missing_partner"] += 1
                with open(ERROR_LOG, "a") as ferr:
                    ferr.write(f"{basename}: Missing slab or O2 file\n")
                continue

            # [Check 2] 尝试读取 (ASE Read)
            # 这里最容易出问题：如果 xyz 文件是空的或截断的，这里会报错
            try:
                slab = read(f_slab)
                sys_ooh = read(f_ooh)
                sys_o2 = read(f_o2)
            except Exception as e:
                audit_log["read_error"] += 1
                with open(ERROR_LOG, "a") as ferr:
                    ferr.write(f"{basename}: ASE Read Failed - {str(e)}\n")
                continue

            # [Check 3] 几何检查
            valid_ooh, d_ooh, metal_sym = analyze_geometry(slab, sys_ooh, basename)
            if not valid_ooh:
                if d_ooh < 0.8: # 重叠
                    audit_log["geo_overlap"] += 1
                else:
                    audit_log["geo_detached_ooh"] += 1
                continue
            
            valid_o2, d_o2, _ = analyze_geometry(slab, sys_o2, basename)
            if not valid_o2:
                if d_o2 < 0.8:
                    audit_log["geo_overlap"] += 1
                else:
                    audit_log["geo_detached_o2"] += 1
                continue

            # [Check 4] 能量提取
            e_slab = robust_get_energy(slab)
            e_ooh = robust_get_energy(sys_ooh)
            e_o2 = robust_get_energy(sys_o2)

            if None in [e_slab, e_ooh, e_o2]:
                audit_log["energy_missing"] += 1
                with open(ERROR_LOG, "a") as ferr:
                    ferr.write(f"{basename}: Energy not found in atoms.info or calc\n")
                continue

            # 计算吸附能
            dE_OOH = e_ooh - e_slab - E_REF_OOH
            dE_O2 = e_o2 - e_slab - E_REF_O2

            # [Check 6] 物理筛选
            if dE_O2 < O2_STABILITY_MIN:
                audit_log["o2_strong"] += 1
            elif dE_O2 > O2_STABILITY_MAX:
                audit_log["o2_weak"] += 1
            elif dE_OOH < OOH_MIN:
                audit_log["ooh_strong"] += 1
            elif dE_OOH > OOH_MAX:
                audit_log["ooh_weak"] += 1
            else:
                # 恭喜通关
                audit_log["valid_candidate"] += 1
                results.append({
                    "Name": basename,
                    "Metal": metal_sym,
                    "dE_OOH": dE_OOH,
                    "dE_O2": dE_O2
                })

        except Exception as e:
            # 捕获所有未预料的错误
            audit_log["read_error"] += 1
            with open(ERROR_LOG, "a") as ferr:
                ferr.write(f"{basename}: UNKNOWN CRITICAL ERROR - {str(e)}\n")
            continue

    # 3. 输出报表
    df = pd.DataFrame(results)
    
    # 校验总数
    total_accounted = sum(audit_log.values())
    
    report = f"""
============================================================
🔍 DEBUGGING REPORT (Total Files: {total_files})
============================================================
1. SYSTEM ERRORS (Where data disappears):
   - Missing Files (Orphans) : {audit_log['missing_partner']}
   - Corrupted Files (Read Error) : {audit_log['read_error']}
   - Energy Missing          : {audit_log['energy_missing']}
   - Model Crashed        : {audit_log['geo_overlap']}

2. PHYSICS FAILURES (Valid file, bad property):
   - OOH Detached (> {MAX_BOND_LENGTH}A) : {audit_log['geo_detached_ooh']}
   - O2 Detached (> {MAX_BOND_LENGTH}A)  : {audit_log['geo_detached_o2']}
   - O2 Too Weak (< {O2_STABILITY_MIN}eV) : {audit_log['o2_weak']}
   - O2 Too Strong (> {O2_STABILITY_MAX}eV) : {audit_log['o2_strong']}
   - OOH Too Strong (< {OOH_MIN}eV) : {audit_log['ooh_strong']}
   - OOH Too Weak (> {OOH_MAX}eV)   : {audit_log['ooh_weak']}

3. SUCCESS:
   - VALID CANDIDATES        : {audit_log['valid_candidate']}
============================================================
TOTAL ACCOUNTED FOR: {total_accounted} / {total_files}
============================================================
Errors detailed in: {ERROR_LOG}
Candidates saved to: {OUTPUT_CSV}
"""
    print(report)
    with open(OUTPUT_REPORT, "w") as f:
        f.write(report)

    if not df.empty:
        df.to_csv(OUTPUT_CSV, index=False)
        # 简单绘图
        try:
            plt.figure(figsize=(8, 6))
            plt.scatter(df["dE_OOH"], df["dE_O2"], alpha=0.6)
            plt.xlabel("dE_OOH"); plt.ylabel("dE_O2")
            plt.title("Valid Candidates Distribution")
            plt.savefig("Analyze_Plot.pdf")
        except: pass

if __name__ == "__main__":
    main()