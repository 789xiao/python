# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 16:57:07 2025

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Optimized validation curve analysis code with English labels
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve, train_test_split, StratifiedKFold
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Data preprocessing
def load_data():
    data = pd.read_csv(r'E:\paper\date\zhenglishuju\170X加CONDR4验证曲线使用版1.2XGB.csv')
    features = ['COND','SP','GR']
    
    # Ensure numeric data types
    X = data[features].apply(pd.to_numeric, errors='coerce').dropna()
    y = data['LITH'].iloc[X.index]  # Maintain index alignment
    
    # Label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

def plot_validation_curve(model, X_train, y_train, param_name, param_range):
    """
    Optimized validation curve plotting function
    """
    # Use stratified K-fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    train_scores, test_scores = validation_curve(
        estimator=model,
        X=X_train,
        y=y_train,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        error_score='raise'
    )
    
    # Calculate statistics
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Visualization settings
    plt.figure(figsize=(10, 6))
    
    # Uniform distribution for learning_rate
    if param_name == 'learning_rate':
        x_ticks = np.linspace(0, len(param_range)-1, len(param_range))
        plt.xticks(x_ticks, param_range)
    else:
        x_ticks = param_range
    
    plt.plot(x_ticks, train_mean, 
             marker='o', markersize=7, 
             color='blue', linewidth=2, 
             label='Training Accuracy')
    plt.fill_between(x_ticks, 
                     train_mean - train_std,
                     train_mean + train_std,
                     alpha=0.15, color='blue')
    
    plt.plot(x_ticks, test_mean,
             marker='s', markersize=7,
             color='green', linestyle='--',
             label='CV Accuracy')
    plt.fill_between(x_ticks,
                     test_mean - test_std,
                     test_mean + test_std,
                     alpha=0.15, color='green')
    
    plt.title(f'Validation Curve for {param_name}', fontsize=16)
    plt.xlabel(param_name, fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.rc('xtick', labelsize=12)    # X轴刻度
    plt.rc('ytick', labelsize=12)    # Y轴刻度
    plt.xticks(x_ticks)
    plt.yticks(np.arange(0.9, 1.01, 0.01))  # Y-axis 0.1 intervals
    plt.ylim(0.9, 1.01)
    
    plt.legend(loc='lower right', frameon=True)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Main program
if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test = load_data()
    except FileNotFoundError:
        raise FileNotFoundError("CSV file path error - please verify path")
    
    num_classes = len(np.unique(y_train))
    
    xgb_model = XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    param_grid = {
        'n_estimators': np.arange(100, 301, 50),
        'max_depth': np.arange(3, 10),
        'learning_rate': np.logspace(-2, -1, num=5),
        'gamma': np.linspace(0, 1, 10)
    }
    
    for param_name, param_range in param_grid.items():
        plot_validation_curve(xgb_model, X_train, y_train, param_name, param_range)

plt.rc('axes', titlesize=14)     # 坐标轴标题
plt.rc('axes', labelsize=13)     # 坐标轴标签
plt.rc('xtick', labelsize=12)    # X轴刻度
plt.rc('ytick', labelsize=12)    # Y轴刻度
plt.rc('legend', fontsize=13)    # 图例
