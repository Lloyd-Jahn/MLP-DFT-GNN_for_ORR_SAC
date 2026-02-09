import pandas as pd
from sklearn.model_selection import train_test_split
import re

def main():
    print("="*60)
    print("🚀 启动基于同分异构体家族的分组筛选 (Group Sampling)")
    print("="*60)

    # 1. 读取数据集
    df = pd.read_csv('数据集.csv')
    print(f"原始结构总量: {len(df)}")

    # 2. 提取“族名” (Base Structure)
    # 逻辑：去除末尾的 -hex, -pen, -opp 后缀，得到纯粹的化学配位式
    # 例如: 'Fe_N2O2-hex' -> 'Fe_N2O2'
    #       'Ag_N4'      -> 'Ag_N4' (不变)
    def get_base_name(s):
        return re.sub(r'-(hex|pen|opp)$', '', s)

    df['Base_Name'] = df['structure'].apply(get_base_name)
    df['Metal'] = df['structure'].apply(lambda x: x.split('_')[0])

    # 3. 建立“家族清单” (去重)
    # 我们只对“家族”进行抽样，而不是对单行数据抽样
    unique_groups = df[['Base_Name', 'Metal']].drop_duplicates()
    total_groups = len(unique_groups)
    print(f"共有 {total_groups} 个不同的配位家族 (Unique Base Structures)")

    # 4. 对“家族”进行分层抽样
    SAMPLE_RATIO = 0.15
    
    try:
        # 处理只有 1 个家族的金属 (防止报错)
        metal_counts = unique_groups['Metal'].value_counts()
        single_sample_metals = metal_counts[metal_counts < 2].index
        
        group_strat = unique_groups[~unique_groups['Metal'].isin(single_sample_metals)]
        group_single = unique_groups[unique_groups['Metal'].isin(single_sample_metals)]
        
        # 核心抽样步骤：针对“家族”进行
        _, sampled_groups_main = train_test_split(
            group_strat, 
            test_size=SAMPLE_RATIO, 
            stratify=group_strat['Metal'], 
            random_state=42
        )
        
        # 合并独苗金属
        sampled_groups_final = pd.concat([sampled_groups_main, group_single])
        
    except Exception as e:
        print(f"分层抽样遇到问题 ({e})，回退到随机抽样...")
        sampled_groups_final = unique_groups.sample(frac=SAMPLE_RATIO, random_state=42)

    print(f"已选中 {len(sampled_groups_final)} 个家族。")

    # 5. 家族召回 (Recall)
    # 根据选中的 Base_Name，把原始表中属于这些家族的所有异构体全捞回来
    # isin() 会自动匹配所有属于选中家族的行
    final_calc_list = df[df['Base_Name'].isin(sampled_groups_final['Base_Name'])].copy()

    # 6. 输出统计
    print("-" * 40)
    print(f"最终筛选出的结构数量: {len(final_calc_list)}")
    print(f"筛选比例 (结构数): {len(final_calc_list) / len(df):.2%}")
    print("-" * 40)
    
    # 检查多样性示例
    sample_check = final_calc_list[final_calc_list['structure'].str.contains('-')]
    if not sample_check.empty:
        example_base = sample_check.iloc[0]['Base_Name']
        siblings = final_calc_list[final_calc_list['Base_Name'] == example_base]['structure'].tolist()
        print(f"验证同分异构体召回 (以 {example_base} 为例):")
        print(f"  -> 找到了: {siblings}")
        if len(siblings) > 1:
            print("  ✅ 成功：该家族的所有异构体都已加入列表！")
        else:
            print("  ⚠️ 警告：该家族似乎只有一个成员被选中（请检查是否有异构体）")

    # 7. 保存
    # 只保留 structure 列，符合后续代码读取习惯
    final_calc_list[['structure']].to_csv('to_calc_list.csv', index=False)
    print(f"已保存至 to_calc_list.csv")

if __name__ == "__main__":
    main()