import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from itertools import combinations, product
import warnings

# 忽略除以零等产生的警告
warnings.filterwarnings('ignore')

# 设置中文字体（如果环境支持），否则使用默认
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path):
    """加载并预处理数据"""
    df = pd.read_csv(file_path)
    # 假设 'property' 列是目标变量 (y)，其他数值列是特征 (X)
    # 移除 ID 列 'materials'
    if 'materials' in df.columns:
        ids = df['materials']
        df = df.drop(columns=['materials'])
    else:
        ids = df.index
    
    y = df['property'].values
    X = df.drop(columns=['property'])
    feature_names = X.columns.tolist()
    X = X.values
    
    return X, y, feature_names, ids

class SISSOReproducer:
    def __init__(self, dimension=5, n_sis_select=100):
        """
        初始化 SISSO 训练器
        :param dimension: 最终模型的维度 (论文结果显示最优维度为 5)
        :param n_sis_select: SIS 阶段每一轮筛选保留的特征数量
        """
        self.dimension = dimension
        self.n_sis_select = n_sis_select
        self.selected_features = []
        self.model = None
        
    def _unary_ops(self, x):
        """一元运算符: exp, log, ^-1, ^2, ^3, ^4, sqrt, cbrt"""
        ops = {}
        # 保护 log 和 div 避免无穷大或 NaN
        with np.errstate(divide='ignore', invalid='ignore'):
            ops['exp'] = np.exp(x)
            ops['log'] = np.log(np.abs(x) + 1e-6) # 避免 log(0)
            ops['inv'] = 1 / (x + 1e-6)           # 避免 div 0
            ops['sq'] = x**2
            ops['cb'] = x**3
            ops['qt'] = x**4
            ops['sqrt'] = np.sqrt(np.abs(x))
            ops['cbrt'] = np.cbrt(x)
        return ops

    def _binary_ops(self, x1, x2):
        """二元运算符: +, -, *, /"""
        ops = {}
        with np.errstate(divide='ignore', invalid='ignore'):
            ops['add'] = x1 + x2
            ops['sub'] = x1 - x2
            ops['mult'] = x1 * x2
            ops['div'] = x1 / (x2 + 1e-6)
        return ops

    def generate_features(self, X, names):
        """
        构建特征空间 (Feature Space Construction)
        论文使用了 Φ1 (一元) 和 Φ2 (二元) 迭代
        为了演示效率，这里进行一轮二元组合扩展
        """
        print("正在构建特征空间 (Phi 1)...")
        new_features = []
        new_names = []
        
        n_features = X.shape[1]
        
        # 1. 原始特征
        for i in range(n_features):
            new_features.append(X[:, i])
            new_names.append(names[i])
            
            # 一元变换
            u_ops = self._unary_ops(X[:, i])
            for op_name, op_val in u_ops.items():
                new_features.append(op_val)
                new_names.append(f"{op_name}({names[i]})")

        # 2. 二元变换 (简化版，仅对原始特征做两两组合，防止内存爆炸)
        # 论文中会进行多层递归，这里为了 Python 运行效率做 1 层的扩充
        print(f"正在扩展二元特征 (当前基础特征数: {len(new_names)})...")
        # 仅取前 N 个原始特征进行组合，实际 SISSO 会对所有特征操作
        base_indices = range(n_features) 
        
        for i, j in combinations(base_indices, 2):
            x1, x2 = X[:, i], X[:, j]
            n1, n2 = names[i], names[j]
            
            b_ops = self._binary_ops(x1, x2)
            for op_name, op_val in b_ops.items():
                if op_name == 'add': name = f"({n1}+{n2})"
                elif op_name == 'sub': name = f"({n1}-{n2})"
                elif op_name == 'mult': name = f"({n1}*{n2})"
                elif op_name == 'div': name = f"({n1}/{n2})"
                
                # 简单的去重和无穷值检查
                if np.isfinite(op_val).all() and np.var(op_val) > 1e-10:
                    new_features.append(op_val)
                    new_names.append(name)

        return np.array(new_features).T, new_names

    def sis_screening(self, X_space, y, feature_names):
        """
        Sure Independence Screening (SIS)
        计算特征与残差的相关性，筛选 Top K
        """
        print("正在进行 SIS 特征筛选...")
        correlations = []
        # 计算每个特征与目标 y 的 Pearson 相关系数绝对值
        for i in range(X_space.shape[1]):
            f = X_space[:, i]
            # 简单的相关性 (实际 SISSO 针对残差迭代，这里简化为对 y 的相关性作为第一步)
            corr = np.abs(np.corrcoef(f, y)[0, 1])
            if np.isnan(corr): corr = 0
            correlations.append(corr)
        
        # 排序取 Top K
        top_k_idx = np.argsort(correlations)[::-1][:self.n_sis_select]
        return X_space[:, top_k_idx], [feature_names[i] for i in top_k_idx]

    def sparsifying_operator(self, X_sis, y, sis_names):
        """
        Sparsifying Operator (SO)
        这里使用逐步回归或 Lasso 的逻辑寻找最佳子集。
        标准 SISSO 使用 L0 正则化 (穷举子集或 OMP)。这里模拟 OMP 过程。
        """
        print(f"正在进行稀疏回归 (目标维度: {self.dimension})...")
        
        selected_indices = []
        current_resid = y.copy()
        
        # 正交匹配追踪 (OMP) 简化实现
        for dim in range(self.dimension):
            best_score = -np.inf
            best_idx = -1
            
            # 寻找能最大程度解释当前残差的特征
            for i in range(X_sis.shape[1]):
                if i in selected_indices: continue
                
                # 尝试添加这个特征
                candidate_indices = selected_indices + [i]
                X_curr = X_sis[:, candidate_indices]
                reg = LinearRegression().fit(X_curr, y)
                score = reg.score(X_curr, y) # R2 score
                
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            if best_idx != -1:
                selected_indices.append(best_idx)
                print(f"  -> 维度 {dim+1}: 选中特征 '{sis_names[best_idx]}' (R2: {best_score:.4f})")
            else:
                break
                
        return selected_indices

    def fit(self, X, y, names):
        # 1. 特征构建
        X_trans, names_trans = self.generate_features(X, names)
        print(f"构建了 {X_trans.shape[1]} 个特征。")
        
        # 2. SIS 筛选
        X_sis, names_sis = self.sis_screening(X_trans, y, names_trans)
        
        # 3. SO 稀疏选择
        self.selected_indices_in_sis = self.sparsifying_operator(X_sis, y, names_sis)
        
        self.final_features_names = [names_sis[i] for i in self.selected_indices_in_sis]
        self.X_final = X_sis[:, self.selected_indices_in_sis]
        
        # 4. 最终训练
        self.model = LinearRegression()
        self.model.fit(self.X_final, y)
        
    def predict(self, X, names):
        # 注意：实际预测需要复现整个特征变换过程
        # 为简化，这里直接在 fit 中返回了拟合值，或者我们假设输入就是 fit 用的数据
        # 在严格生产环境中，需要保存变换管道
        return self.model.predict(self.X_final)

# --- 主程序 ---

# 1. 读取数据
data_path = 'train.csv' 
X, y, feature_names, ids = load_data(data_path)

# 2. 划分训练集和测试集 (参考论文 60% 训练)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

print(f"原始特征数: {len(feature_names)}")
print(f"训练样本数: {len(y_train)}")

# 3. 运行 SISSO
# 注意：受限于运行环境内存，我们适当限制 SIS 筛选数量
sisso = SISSOReproducer(dimension=5, n_sis_select=500)
sisso.fit(X_train, y_train, feature_names)

# 4. 评估
y_train_pred = sisso.predict(X_train, feature_names)
r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

print("\n=== 训练结果 ===")
print(f"Training R2: {r2_train:.4f}")
print(f"Training RMSE: {rmse_train:.4f}")
print("SISSO 筛选出的最佳描述符公式:")
for i, formula in enumerate(sisso.final_features_names):
    coef = sisso.model.coef_[i]
    print(f"  Term {i+1}: ({coef:.4f}) * [{formula}]")
print(f"  Intercept: {sisso.model.intercept_:.4f}")

# 5. 可视化 (Parity Plot)
plt.figure(figsize=(8, 7))
plt.scatter(y_train, y_train_pred, alpha=0.6, c='blue', edgecolors='k', label='Training Data')

# 绘制对角线
min_val = min(y_train.min(), y_train_pred.min())
max_val = max(y_train.max(), y_train_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit')

# 添加误差带 (可选，例如 +/- 0.1)
plt.fill_between([min_val, max_val], 
                 [min_val-0.1, max_val-0.1], 
                 [min_val+0.1, max_val+0.1], 
                 color='red', alpha=0.1, label='Error Margin +/- 0.1')

plt.xlabel('DFT Calculated Property (Target)', fontsize=12)
plt.ylabel('SISSO Predicted Property', fontsize=12)
plt.title(f'SISSO Training Performance\nR2={r2_train:.3f}, RMSE={rmse_train:.3f}', fontsize=14)
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()

# 保存并展示
plt.savefig('sisso_parity_plot.png')
print("可视化图表已保存为 sisso_parity_plot.png")
plt.show()