import os
import shutil
import pandas as pd
from tqdm import tqdm
import time

# ================= ⚙️ 配置区域 =================
# 1. 筛选结果csv文件
CANDIDATE_LIST = "Final_Candidates_List.csv"

# 2. 源文件夹
SRC_DIR_SLAB = "xyzs_optimized"        # 基底 .xyz
SRC_DIR_ADS  = "Adsorbed_Structures"   # 吸附态 .xyz

# 3. 目标文件夹
# 这个文件夹里的文件就是最终交给 VASP/CP2K 的待计算结构
TARGET_DIR = "Final_DFT_Structures"

# ===============================================

def main():
    start_time = time.time()
    print("="*60)
    print("🚀 启动 DFT 候选结构提取程序")
    print("="*60)

    # 1. 检查输入文件
    if not os.path.exists(CANDIDATE_LIST):
        print(f"❌ 错误：找不到名单文件 {CANDIDATE_LIST}")
        return

    # 2. 创建目标目录
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"📂 创建新文件夹: {TARGET_DIR}")
    else:
        print(f"📂 目标文件夹已存在: {TARGET_DIR} (将覆盖同名文件)")

    # 3. 读取名单
    try:
        df = pd.read_csv(CANDIDATE_LIST)
        candidates = df['Name'].tolist()
        print(f"📋 名单中共有 {len(candidates)} 个候选结构")
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return

    # 统计数据
    stats = {
        "success_sets": 0,
        "missing_files": 0,
        "total_files_copied": 0
    }
    
    missing_log = []

    print("\n📦 正在搬运文件...")
    
    # 4. 核心循环
    for name in tqdm(candidates, desc="Extracting", unit="set"):
        # 定义三个源文件路径
        f_slab = os.path.join(SRC_DIR_SLAB, f"{name}.xyz")
        f_ooh  = os.path.join(SRC_DIR_ADS,  f"{name}_with_OOH.xyz")
        f_o2   = os.path.join(SRC_DIR_ADS,  f"{name}_with_O2.xyz")

        # 定义三个目标文件路径
        dst_slab = os.path.join(TARGET_DIR, f"{name}.xyz")
        dst_ooh  = os.path.join(TARGET_DIR, f"{name}_with_OOH.xyz")
        dst_o2   = os.path.join(TARGET_DIR, f"{name}_with_O2.xyz")

        # 完整性检查：只有三个都在，才搬运
        if os.path.exists(f_slab) and os.path.exists(f_ooh) and os.path.exists(f_o2):
            try:
                shutil.copy2(f_slab, dst_slab)
                shutil.copy2(f_ooh, dst_ooh)
                shutil.copy2(f_o2, dst_o2)
                
                stats["success_sets"] += 1
                stats["total_files_copied"] += 3
            except Exception as e:
                print(f"❌ 复制出错 {name}: {e}")
        else:
            # 记录缺失情况
            stats["missing_files"] += 1
            missing = []
            if not os.path.exists(f_slab): missing.append("Slab")
            if not os.path.exists(f_ooh):  missing.append("OOH")
            if not os.path.exists(f_o2):   missing.append("O2")
            missing_log.append(f"{name}: Missing {', '.join(missing)}")

    # 5. 生成报告
    print("\n" + "="*60)
    print("📊 提取工作完成 summary")
    print("="*60)
    print(f"✅ 成功提取组数 : {stats['success_sets']} (共 {stats['total_files_copied']} 个文件)")
    print(f"❌ 缺失组数     : {stats['missing_files']}")
    print(f"📂 文件已保存在 : {os.path.abspath(TARGET_DIR)}")
    
    # 写入日志
    log_file = os.path.join(TARGET_DIR, "extraction_log.txt")
    with open(log_file, "w") as f:
        f.write(f"Extraction Time: {time.ctime()}\n")
        f.write(f"Total Sets: {len(candidates)}\n")
        f.write(f"Success: {stats['success_sets']}\n")
        f.write(f"Missing: {stats['missing_files']}\n")
        if missing_log:
            f.write("\n=== Missing Files Details ===\n")
            for line in missing_log:
                f.write(line + "\n")
    
    print(f"📝 详细日志已生成: {log_file}")
    print(f"⏱️ 耗时: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()