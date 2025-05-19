# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 15:39:59 2025

@author: hp
"""

# -*- coding: utf-8 -*-
"""
优化后的验证曲线分析代码
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve, train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 数据预处理
def load_data():
    data = pd.read_csv(r'E:\paper\date\zhenglishuju\170X加CONDR4验证曲线使用版1.1.csv')
    features = ['COND','SP','GR']
    
    # 确保数据类型为数值型
    X = data[features].apply(pd.to_numeric, errors='coerce').dropna()
    y = data['LITH'].iloc[X.index]  # 保持索引对齐
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def plot_validation_curve(model, X_train, y_train, param_name, param_range):
    """
    优化后的验证曲线绘制函数
    """
    train_scores, test_scores = validation_curve(
        estimator=model,
        X=X_train,
        y=y_train,
        param_name=param_name,
        param_range=param_range,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    
    # 计算统计量
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    # 可视化设置
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, 
             marker='o', markersize=7, 
             color='blue', linewidth=2, 
             label='Training Accuracy')
    plt.plot(param_range, test_mean,
             marker='s', markersize=7,
             color='green', linestyle='--',
             label='Cross-Validation Accuracy')
    
    plt.title(f'Validation Curve for {param_name}', fontsize=16)
    plt.xlabel(param_name, fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.rc('xtick', labelsize=12)    # X轴刻度
    plt.rc('ytick', labelsize=12)    # Y轴刻度
    plt.xticks(param_range)
    plt.yticks(np.arange(0.9, 1.01, 0.01))  # 纵坐标设置
    
    # 修改图例位置到右下角
    plt.legend(loc='lower right', frameon=True)  # 添加frameon让图例有背景
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 加载数据
    X_train, X_test, y_train, y_test = load_data()
    
    # 初始化模型
    rf_model = RandomForestClassifier(random_state=42)
    
    # 参数范围定义（使用整数序列）
    param_grid = {
        'n_estimators': np.arange(100, 301, 50),    # [100,150,200,250,300]
        'max_depth': np.arange(5, 16, 2),           # [5,7,9,11,13,15] 
        'min_samples_split': np.arange(2, 11, 2)    # [2,4,6,8,10]
    }
    
    # 绘制验证曲线
    for param_name, param_range in param_grid.items():
        plot_validation_curve(rf_model, X_train, y_train, 
                             param_name, param_range)
