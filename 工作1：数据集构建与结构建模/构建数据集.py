from collections import Counter
from itertools import combinations_with_replacement
import pandas as pd
import sys

配位元素 = ['C', 'B', 'N', 'O', 'P', 'S']

金属列表 = [
    # 3d
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    # 4d
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    # 5d
    'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'
]

双空位变体 = ['-pen', '-hex', '-opp']

金属配位建议 = {}

def 分配配位数(金属列表, 配位数列表):
    for 金属 in 金属列表:
        金属配位建议[金属] = 配位数列表


# 3d
分配配位数(['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn'], [3, 4])

# 4d
分配配位数(['Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd'], [3, 4])

# 5d
分配配位数(['La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'], [3, 4])


def 格式化表达式(金属, 计数器):
    顺序 = ['B', 'C', 'N', 'O', 'P', 'S']
    部分 = []
    for 元素 in 顺序:
        计数 = 计数器.get(元素, 0)
        if 计数 > 0:
            部分.append(f"{元素}{计数}")
    if not 部分:
        return f"{金属}_"
    return f"{金属}_" + "".join(部分)

def 获取变体元素(计数):
    for 元素 in ['B', 'C', 'N', 'O', 'P', 'S']:
        if 计数.get(元素, 0) == 2:
            return 元素
    return None

def 生成结构表达式(金属列表, 配位元素):
    记录 = []

    for 金属 in 金属列表:
        允许的配位数 = 金属配位建议[金属]

        for 配位数 in 允许的配位数:
            组合列表 = list(combinations_with_replacement(配位元素, 配位数))

            for 组合 in 组合列表:
                计数 = Counter(组合)

                基础表达式 = 格式化表达式(金属, 计数)

                变体元素 = 获取变体元素(计数)

                if 配位数 == 4 and 变体元素 is not None:
                    for 后缀 in 双空位变体:
                        记录.append({
                            'metal': 金属,
                            'coord': 配位数,
                            'structure': 基础表达式 + 后缀,
                            'variant': 后缀,
                        })
                else:
                    记录.append({
                        'metal': 金属,
                        'coord': 配位数,
                        'structure': 基础表达式,
                        'variant': ''
                    })

    return 记录

def 主函数():
    记录 = 生成结构表达式(金属列表, 配位元素)

    数据框 = pd.DataFrame(记录)
    if 数据框.empty:
        sys.exit(1)

    数据框 = 数据框.drop_duplicates(subset=['structure']).sort_values(['metal', 'coord', 'structure']).reset_index(drop=True)

    输出路径 = '数据集.csv'
    数据框.to_csv(输出路径, index=False)

    print("生成完成：")
    print(f"  金属数量：{len(金属列表)}")
    print(f"  唯一表达式数量：{len(数据框)}")
    print(f"  CSV已保存到：{输出路径}")

if __name__ == '__main__':
    主函数()
