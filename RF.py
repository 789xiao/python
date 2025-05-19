# -*- coding: utf-8 -*-
"""
Created on Fri May  2 10:51:33 2025

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 13:04:05 2025

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 21:56:47 2025
@author: hp
"""

# -*- coding: utf-8 -*-
"""
岩性识别优化版 - 随机森林专用版
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.metrics import (accuracy_score, f1_score, classification_report, 
                             roc_curve, auc, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

# 1. 数据加载与预处理
data = pd.read_csv(r'E:\paper\date\zhenglishuju\170X加CONDR4.csv')
data.dropna(inplace=True)

features = ['COND', 'SP', 'GR']
X = data[features]
y = data['LITH']

# 标签编码处理
le = LabelEncoder() if y.dtype == object else None
if le:
    y = le.fit_transform(y)

# 2. 数据划分（训练集80%，测试集20%）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# 3. 特征归一化
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_test_df = pd.DataFrame(X_test, columns=features)  # 保存特征名称

# 4. 类别分布可视化（保持不变）
def analyze_class_distribution(y, title="类别分布"):
    unique_classes, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(10, 6))
    classes = le.classes_ if le else [f"Class {i}" for i in unique_classes]
    sns.barplot(x=classes, y=counts)
    plt.title(f"{title}\n", fontsize=14, fontweight='bold')
    plt.xlabel('类别', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    print(f"\n{title}:")
    for i, cls in enumerate(classes):
        print(f"{cls}: {counts[i]} 样本 ({counts[i]/len(y):.2%})")

analyze_class_distribution(y_train, "训练集类别分布")

# 5. 模型配置
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)

# 6. 参数优化配置（简化版）
param_space = {
    'n_estimators': Integer(100, 300),
    'max_depth': Integer(5, 15),
    'min_samples_split': Integer(2, 10),
   
}

# 7. 模型优化与训练
print("\n开始模型优化...")
bayes_optimizer = BayesSearchCV(
    estimator=model,
    search_spaces=param_space,
    scoring='f1_macro',
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    n_iter=30,
    n_jobs=-1,
    verbose=1,
    random_state=42
)
bayes_optimizer.fit(X_train, y_train)
best_model = bayes_optimizer.best_estimator_
# 新增最佳超参数输出
print("\n=== 贝叶斯优化最佳超参数 ===")
print(bayes_optimizer.best_params_)
# 8. 学习曲线分析（保持不变）
def plot_learning_curve(estimator, X, y, cv, title):
    plt.figure(figsize=(10, 6))    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='f1_macro'
    )
    
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r", label="训练集")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color="g", label="验证集")
    plt.fill_between(train_sizes, 
                     np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                     np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes,
                     np.mean(test_scores, axis=1) - np.std(test_scores, axis=1),
                     np.mean(test_scores, axis=1) + np.std(test_scores, axis=1),
                     alpha=0.1, color="g")
    
    plt.title(title)
    plt.xlabel("训练样本数")
    plt.ylabel("F1得分")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_learning_curve(best_model, X_train, y_train, 
                   StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                   "模型学习曲线")

# 9. 模型评估（保持不变）
def model_evaluation(model, X, y, dataset_name):
    y_pred = model.predict(X)
    print(f"\n{dataset_name}评估:")
    print(f"准确率: {accuracy_score(y, y_pred):.2%}")
    print(f"宏F1: {f1_score(y, y_pred, average='macro'):.2f}")
    print(classification_report(y, y_pred, target_names=le.classes_ if le else None))

print("\n==== 模型性能评估 ====")
model_evaluation(best_model, X_train, y_train, "训练集")
model_evaluation(best_model, X_test, y_test, "测试集")

# 10. 可视化分析
#混淆矩阵
def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = confusion_matrix(y_test, y_pred, normalize='true')

    plt.figure(figsize=(10, 8))
    
    # 生成复合标注矩阵，显示小数形式
    annot_matrix = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            row.append(f"{cm_norm[i,j]:.3f}")  # Changed from .1% to .3f
        annot_matrix.append(row)

    # 颜色矩阵（基于正确率）
    color_matrix = np.zeros_like(cm_norm, dtype=float)
    np.fill_diagonal(color_matrix, cm_norm.diagonal())

    # 使用浅绿色调色板
    custom_cmap = sns.light_palette("seagreen", as_cmap=True)

    ax = sns.heatmap(
        color_matrix,
        annot=annot_matrix,
        fmt='',
        cmap=custom_cmap,
        cbar=True,
        linewidths=0.5,
        linecolor='lightgray',
        xticklabels=['F', 'M', 'S'],
        yticklabels=['F', 'M', 'S'],
        annot_kws={
            "size": 14,
            "color": "black",
            "ha": "center",
            "va": "center"
        }
    )
    
    cbar = ax.collections[0].colorbar
    cbar.set_label('Classification Accuracy', rotation=270, labelpad=20)

    plt.title('Lithology Classification Confusion Matrix', fontsize=15)
    plt.xlabel('Predicted Lithology', fontsize=15)
    plt.ylabel('True Lithology', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16, rotation=0)
    plt.tight_layout()
    plt.show()
def plot_feature_importance(model):
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    
    plt.barh(range(len(features)), importances[sorted_idx], color='#2ca02c')
    plt.yticks(range(len(features)), [features[i] for i in sorted_idx])
    plt.title('特征重要性分析\n', fontsize=14, fontweight='bold')
    plt.xlabel('归一化重要性')
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(best_model, X_test, y_test)
plot_feature_importance(best_model)

# 11. ROC曲线分析
def plot_roc_curves(model, X_test, y_test, le=None):
    try:
        num_classes = len(np.unique(y_test))
        y_score = model.predict_proba(X_test)
        class_names = le.classes_ if le else [f"Class {i}" for i in range(num_classes)]
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        
        plt.figure(figsize=(10, 8))
        plt.rcParams.update({'font.sans-serif': 'SimHei', 'axes.unicode_minus': False})
        colors = plt.cm.viridis(np.linspace(0, 1, num_classes))
        
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i],
                    lw=2.5, 
                    label=f'{class_names[i]} (AUC={roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6)
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves (Test Set)\n', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"ROC曲线绘制失败: {str(e)}")
        plt.close()

print("\n==== ROC曲线可视化 ====")
plot_roc_curves(best_model, X_test, y_test, le)

# 12. SHAP分析
try:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_df)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
    plt.title("特征贡献度分析", fontsize=14)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"SHAP可视化异常: {e}")
# 14. 岩性预测可视化（完整修改版）
def plot_well_log_with_lithology(df, depth_col='DEPTH', log_curves=['GR', 'SP', 'COND'], 
                                lithology_col='LITH', predicted_lith_col=None,
                                depth_range=None, figsize=(12, 10)):
    """修改后的测井曲线可视化函数"""
    # 数据预处理
    if depth_range is not None:
        min_depth, max_depth = depth_range
        plot_df = df[(df[depth_col] >= min_depth) & (df[depth_col] <= max_depth)].copy()
    else:
        plot_df = df.copy()
    plot_df = plot_df.sort_values(by=depth_col)
    
    # 岩性颜色配置
    lithology_dict = {
        'Mudstone': {'color': '#7FC97F', 'hatch': '..'},
        'Fine sandstone': {'color': '#FDC086', 'hatch': ''},
        'Siltstone': {'color': '#BEAED4', 'hatch': '--'}
    }

    # 创建绘图轨道
    n_tracks = 2 + len(log_curves) + (1 if predicted_lith_col else 0)
    fig, ax = plt.subplots(nrows=1, ncols=n_tracks, figsize=figsize, sharey=True)
    track_idx = 0
    
    # 实际岩性道绘制逻辑修改
    def plot_lithology_track(ax, data, lith_col):
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        
        current_lith = None
        start_depth = data.iloc[0][depth_col]
        
        for idx, row in data.iterrows():
            lith = row[lith_col]
            if current_lith != lith:
                if current_lith in lithology_dict:
                    end_depth = row[depth_col]
                    ax.fill_betweenx([start_depth, end_depth], 0, 1,
                                   **lithology_dict[current_lith],
                                   edgecolor='k')
                current_lith = lith
                start_depth = row[depth_col]
        
        # 绘制最后一个层段
        if current_lith in lithology_dict:
            ax.fill_betweenx([start_depth, data[depth_col].iloc[-1]], 0, 1,
                           **lithology_dict[current_lith], edgecolor='k')

    # 实际岩性道
    ax[track_idx].set_title("Core\nlithology")
    plot_lithology_track(ax[track_idx], plot_df, lithology_col)
    track_idx += 1
    
    # 测井曲线道
    for curve in log_curves:
        if curve in plot_df.columns:
            # 确定曲线范围
            min_val = plot_df[curve].min()
            max_val = plot_df[curve].max()
            buffer = (max_val - min_val) * 0.1
            
            # 设置曲线颜色
            curve_color = 'blue'
            if curve == 'GR': curve_color = 'green'
            elif curve == 'SP': curve_color = 'blue'
            elif curve in ['COND', 'RT']: curve_color = 'red'
            
            # 绘制曲线
            ax[track_idx].plot(plot_df[curve], plot_df[depth_col], 
                             color=curve_color, linewidth=1)
            ax[track_idx].set_xlim(min_val - buffer, max_val + buffer)
            ax[track_idx].set_title(curve)
            ax[track_idx].grid(True, linestyle='--', alpha=0.5)
            ax[track_idx].spines['top'].set_visible(True)
            ax[track_idx].spines['right'].set_visible(True)
            track_idx += 1
    
    # 预测岩性道
    if predicted_lith_col:
        ax[track_idx].set_title("Predicted\nlithology")
        plot_lithology_track(ax[track_idx], plot_df, predicted_lith_col)
    
    # 通用设置
    ax[0].set_ylabel('Depth (m)')
    ax[0].invert_yaxis()
    depth_min, depth_max = plot_df[depth_col].min(), plot_df[depth_col].max()
    depth_interval = (depth_max - depth_min) / 10 if (depth_max - depth_min) > 100 else 10
    depth_ticks = np.arange(np.floor(depth_min / depth_interval) * depth_interval,
                           np.ceil(depth_max / depth_interval) * depth_interval + 1,
                           depth_interval)
    for d in depth_ticks:
        if depth_min <= d <= depth_max:
            for i in range(len(ax)):
                ax[i].axhline(y=d, color='lightgray', linestyle='-', alpha=0.5)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.05)
    plt.show()
    return fig, ax

def visualize_well_logs_with_predictions(data, model, scaler, features, 
                                        depth_col='DEPTH', lithology_col='LITH', 
                                        le=None, depth_range=None):
    """可视化函数"""
    viz_data = data.copy()
    if depth_range is not None:
        viz_data = viz_data[
            (viz_data[depth_col] >= depth_range[0]) & 
            (viz_data[depth_col] <= depth_range[1])
        ].copy()
    if viz_data.empty:
        raise ValueError(f"在{depth_range}深度范围内未找到数据")
    
    X_viz = scaler.transform(viz_data[features])
    y_pred = model.predict(X_viz)
    
    if le is not None:
        y_pred = le.inverse_transform(y_pred)
    
    viz_data['PRED_LITH'] = y_pred
    
    return plot_well_log_with_lithology(
        df=viz_data,
        depth_col=depth_col,
        log_curves=features,
        lithology_col=lithology_col,
        predicted_lith_col='PRED_LITH',
        depth_range=depth_range
    )

# 最终可视化调用
print("\n==== 测井曲线与岩性预测对比图 (1998-2055m) ====")
try:
    visualize_well_logs_with_predictions(
        data=data,
        model=best_model,
        scaler=scaler,
        features=features,
        depth_col='DEPTH',
        lithology_col='LITH',
        le=le,
        depth_range=(1998, 2055)
    )
except ValueError as e:
    print(f"可视化错误: {str(e)}")
