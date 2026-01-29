#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AttentiveFP 模型预测程序

预测参数在第178行
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn.models import AttentiveFP

# 导入训练脚本中的函数
from attentivefp_train import 构建数据集, 构建图_from_xyz, ATOMIC_PROPS, 配位元素

class AttentiveFPPredictor:
    def __init__(self, 模型路径, 训练数据路径=None, xyz目录=None, 设备=None):
        """初始化预测器"""
        if 设备 is None:
            self.设备 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.设备 = 设备
            
        self.模型路径 = 模型路径
        self.训练数据路径 = 训练数据路径
        self.xyz目录 = xyz目录
        self.模型 = None
        self.y_scaler = None
        self.元素词表 = None
        
        self.加载模型()
    
    def 获取模型参数(self):
        """使用训练脚本中的确切参数"""
        # 这些参数必须与训练脚本完全一致
        return {
            'in_channels': 15,
            'hidden_channels': 256,
            'edge_dim': 4,
            'num_layers': 4,
            'num_timesteps': 2,
            'dropout': 0.1
        }
    
    def 加载模型(self):
        """加载训练好的模型和配置"""
        if not os.path.exists(self.模型路径):
            raise FileNotFoundError(f"模型文件不存在: {self.模型路径}")
        
        print("📥 加载模型...")
        checkpoint = torch.load(self.模型路径, map_location=self.设备, weights_only=False)
        
        # 获取配置
        state_dict = checkpoint['model_state_dict']
        self.元素词表 = checkpoint.get('element_vocab', [])
        self.y_scaler = checkpoint.get('y_scaler')
        
        模型参数 = self.获取模型参数()
        
        print(f"🔧 使用训练参数创建模型:")
        print(f"   输入维度: {模型参数['in_channels']}")
        print(f"   隐藏层维度: {模型参数['hidden_channels']}")
        print(f"   边特征维度: {模型参数['edge_dim']}")
        print(f"   层数: {模型参数['num_layers']}")
        print(f"   时间步: {模型参数['num_timesteps']}")
        print(f"   Dropout: {模型参数['dropout']}")
        
        # 创建与训练时完全一致的模型架构
        print("🛠️ 创建模型架构...")
        self.模型 = AttentiveFP(
            in_channels=模型参数['in_channels'],
            hidden_channels=模型参数['hidden_channels'],
            out_channels=1,
            edge_dim=模型参数['edge_dim'],
            num_layers=模型参数['num_layers'],
            num_timesteps=模型参数['num_timesteps'],
            dropout=模型参数['dropout']
        ).to(self.设备)
        
        # 加载权重
        print("📥 加载模型权重...")
        
        # 检查权重键匹配情况
        model_keys = set(self.模型.state_dict().keys())
        state_dict_keys = set(state_dict.keys())
        
        print(f"   模型参数数量: {len(model_keys)}")
        print(f"   检查点参数数量: {len(state_dict_keys)}")
        
        # 查找缺失的键
        missing_in_model = state_dict_keys - model_keys
        missing_in_checkpoint = model_keys - state_dict_keys
        
        if missing_in_model:
            print(f"   ⚠️ 检查点中有但模型中缺失的键: {missing_in_model}")
        if missing_in_checkpoint:
            print(f"   ⚠️ 模型中有但检查点中缺失的键: {missing_in_checkpoint}")
        
        # 尝试加载权重
        try:
            self.模型.load_state_dict(state_dict, strict=False)
            print("   ✅ 权重加载成功 (strict=False)")
        except Exception as e:
            print(f"   ❌ 权重加载失败: {e}")
            raise
        
        self.模型.eval()
        print("✅ 模型加载成功")
    
    def 构建预测图(self, xyz文件路径):
        """为预测构建图数据"""
        if not self.元素词表:
            raise ValueError("元素词表未初始化")
        
        g = 构建图_from_xyz(xyz文件路径, self.元素词表)
        
        data = Data(
            x=g["x"],
            edge_index=g["edge_index"],
            edge_attr=g["edge_attr"],
            y=torch.tensor([0.0])  # 占位符
        )
        data.batch = torch.zeros(len(g["x"]), dtype=torch.long)
        
        return data, g
    
    def 预测单个结构(self, xyz文件路径):
        """预测单个XYZ结构的性质"""
        try:
            if not os.path.exists(xyz文件路径):
                raise FileNotFoundError(f"XYZ文件不存在: {xyz文件路径}")
            
            data, g = self.构建预测图(xyz文件路径)
            
            # 预测
            with torch.no_grad():
                data = data.to(self.设备)
                out = self.模型(data.x, data.edge_index, data.edge_attr, data.batch)
                
                # 处理输出
                if out.dim() > 1:
                    out = out.view(-1)
                
                prediction = out.cpu().numpy()[0]
                
                # 反标准化
                if self.y_scaler is not None:
                    prediction = self.y_scaler.inverse_transform([[prediction]])[0, 0]
            
            return {
                '文件': os.path.basename(xyz文件路径),
                '预测吸附能(eV)': float(prediction),
                '原子数': len(g["元素列表"]),
                '边数量': g["edge_index"].shape[1] // 2,
                '中心金属': g["元素列表"][g["metal_index"]],
                '总原子': g["元素列表"]
            }
            
        except Exception as e:
            print(f"❌ 预测失败 {xyz文件路径}: {e}")
            return None
    
    def 预测数据集(self, csv文件路径, xyz目录):
        """预测整个数据集的性能"""
        print(f"📈 开始预测数据集...")
        print(f"   CSV文件: {csv文件路径}")
        print(f"   XYZ目录: {xyz目录}")
        
        # 构建数据集
        pyg_list, _, _, _ = 构建数据集(csv文件路径, xyz目录, "d_band_center", r_max=5.0)
        
        if len(pyg_list) == 0:
            raise ValueError("没有构建到有效的图数据")
        
        print(f"   样本数量: {len(pyg_list)}")
        
        # 创建数据加载器
        test_loader = DataLoader(pyg_list, batch_size=16, shuffle=False)
        
        # 批量预测
        all_true = []
        all_pred = []
        all_names = []
        
        self.模型.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.设备)
                out = self.模型(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                
                if out.dim() > 1:
                    out = out.view(-1)
                
                # 反标准化预测值
                if self.y_scaler is not None:
                    out_denorm = self.y_scaler.inverse_transform(out.cpu().numpy().reshape(-1, 1)).flatten()
                    all_pred.extend(out_denorm.tolist())
                else:
                    all_pred.extend(out.cpu().numpy().tolist())
                
                # 收集真实值
                if hasattr(batch, 'original_y'):
                    all_true.extend(batch.original_y.cpu().numpy().tolist())
                else:
                    all_true.extend(batch.y.view(-1).cpu().numpy().tolist())
                
                # 收集名称
                if hasattr(batch, 'name'):
                    all_names.extend(batch.name)
                else:
                    all_names.extend([f"sample_{i}" for i in range(len(batch.y))])
        
        return np.array(all_true), np.array(all_pred), all_names
    
    def 生成评估报告(self, y_true, y_pred, 输出目录):
        """生成预测评估报告和图表"""
        os.makedirs(输出目录, exist_ok=True)
        
        # 计算评估指标
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"\n📊 预测性能评估:")
        print(f"   样本数量: {len(y_true)}")
        print(f"   MSE:  {mse:.6f}")
        print(f"   RMSE: {rmse:.6f} eV")
        print(f"   MAE:  {mae:.6f} eV")
        print(f"   R²:   {r2:.4f}")
        
        # 生成Parity Plot
        self.绘制_parity_plot(y_true, y_pred, 输出目录, mse, mae, r2, rmse)
        
        # 保存详细结果
        self.保存预测结果(y_true, y_pred, 输出目录, mse, mae, r2, rmse)
        
        return mse, mae, r2, rmse
    
    def 绘制_parity_plot(self, y_true, y_pred, 输出目录, mse, mae, r2, rmse):
        """绘制Parity Plot"""
        plt.figure(figsize=(6.5, 6))
        
        # 绘制散点图
        plt.scatter(y_true, y_pred, alpha=0.7, s=50, edgecolors='w', linewidth=0.5)
        
        # 绘制对角线
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        margin = (max_val - min_val) * 0.05
        
        plt.plot([min_val - margin, max_val + margin], 
                 [min_val - margin, max_val + margin], 
                 'r--', alpha=0.8, linewidth=2, label='Prefect prediction')
        
        plt.xlabel('True attached_energy_ev (eV)', fontsize=12)
        plt.ylabel('Predicted attached_energy_ev (eV)', fontsize=12)
        plt.title(f'AttentiveFP Prediction Result\nR² = {r2:.4f}, RMSE = {rmse:.4f} eV', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 添加统计信息文本框
        textstr = f'N = {len(y_true)}\nR² = {r2:.4f}\nRMSE = {rmse:.4f} eV\nMAE = {mae:.4f} eV'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(os.path.join(输出目录, "prediction_parity_plot.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Parity plot 已保存: {输出目录}/prediction_parity_plot.png")
    
    def 保存预测结果(self, y_true, y_pred, 输出目录, mse, mae, r2, rmse):
        """保存详细的预测结果"""
        results_df = pd.DataFrame({
            'true_value': y_true,
            'pred_value': y_pred,
            'absolute_error': np.abs(y_true - y_pred),
            'relative_error_percent': np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8)) * 100
        })
        
        # 保存到CSV文件
        results_df.to_csv(os.path.join(输出目录, "prediction_results.csv"), index=False)
        
        # 保存统计信息
        统计信息 = {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2),
            '样本数量': len(y_true)
        }
        
        with open(os.path.join(输出目录, "prediction_stats.json"), 'w') as f:
            json.dump(统计信息, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 预测结果已保存")
        print(f"   平均绝对误差: {results_df['absolute_error'].mean():.4f} eV")
        print(f"   最大绝对误差: {results_df['absolute_error'].max():.4f} eV")

def main():
    """主函数 - 使用示例"""
    # 配置参数
    模型路径 = "output/best_attentivefp.pt"
    训练数据路径 = "data.csv"
    xyz目录 = "xyzs"
    输出目录 = "prediction_results"
    
    try:
        # 创建预测器
        print("🚀 初始化 AttentiveFP 预测器...")
        predictor = AttentiveFPPredictor(模型路径, 训练数据路径, xyz目录)
        
        # 预测整个数据集
        print("\n📈 开始批量预测数据集...")
        y_true, y_pred, names = predictor.预测数据集(训练数据路径, xyz目录)
        
        # 生成评估报告
        predictor.生成评估报告(y_true, y_pred, 输出目录)
        
    except Exception as e:
        print(f"❌ 预测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()