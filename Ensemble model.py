# -*- coding: utf-8 -*-
"""
Created on Mon May 19 17:15:00 2025

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 11:14:37 2025
@author: hp
"""

"""
岩性识别优化版 - 精简高效版（仅使用原始数据训练）
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.metrics import (accuracy_score, f1_score, classification_report, 
                             roc_curve, auc, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# 1. 数据加载与预处理
data = pd.read_csv(r'E:\paper\date\zhenglishuju\170X加CONDR4.csv')
data.dropna(inplace=True)

features = ['COND','SP','GR']
X = data[features]
y = data['LITH']

def plot_feature_scatter(data, features, target_col):
    """生成特征变量散点图矩阵"""
    plt.figure(figsize=(14, 12))
    lithology_palette = {
        'Fine sandstone': 'pink',
        'Mudstone': 'skyblue',
        'Siltstone': 'plum'
    }
    
    scatter = sns.pairplot(
        data=data,
        vars=features,
        hue=target_col,
        palette=lithology_palette,
        plot_kws={'alpha': 0.7, 's': 40, 'edgecolor': 'k'},
        diag_kind='kde'
    )
    scatter.fig.suptitle('Feature Variable Scatter Matrix\n', 
                         y=1.02, fontsize=16, fontweight='bold')
    for ax in scatter.axes.flatten():
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlabel(ax.get_xlabel(), fontsize=12)
        ax.set_ylabel(ax.get_ylabel(), fontsize=12)
    scatter._legend.set_title(target_col)
    scatter._legend.set_bbox_to_anchor((1, 0.2))
    scatter._legend.set_frame_on(True)
    plt.tight_layout()
    plt.show()

print("\n==== 特征变量分布可视化 ====")
plot_feature_scatter(data, features, 'LITH')

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
X_test_df = pd.DataFrame(X_test, columns=features)

# 4. 类分布分析
def analyze_class_distribution(y, title="Class Distribution"):
    unique_classes, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(10, 6))
    classes = ['Fine sandstone', 'Mudstone', 'Siltstone']
    sns.barplot(x=classes, y=counts)
    plt.title(f"{title}\n", fontsize=14, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Sample Count', fontsize=12)
    plt.tight_layout()
    plt.show()
    print(f"\n{title}:")
    for i, cls in enumerate(classes):
        print(f"{cls}: {counts[i]} samples ({counts[i]/len(y):.2%})")

analyze_class_distribution(y)

# 5. 基模型配置
num_classes = len(np.unique(y))

ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=num_classes,
            random_state=42,
            eval_metric='mlogloss'
        )),
        ('rf', RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42
        ))
    ],
    voting='soft',
    n_jobs=-1
)

# 6. 参数优化配置
param_space = {
    'xgb__max_depth': Integer(3, 10),
    'xgb__learning_rate': Real(0.01, 0.1, 'log-uniform'),
    'xgb__n_estimators': Integer(100, 300),
    'xgb__gamma': Real(0, 1.0, 'uniform'),
    'rf__n_estimators': Integer(100, 300),
    'rf__max_depth': Integer(5, 15),
    'rf__min_samples_split': Integer(2, 10)
}

# 7. 模型优化与训练
print("\n开始模型优化...")
bayes_optimizer = BayesSearchCV(
    estimator=ensemble,
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

print("\n=== 贝叶斯优化最佳超参数 ===")
print(bayes_optimizer.best_params_)

# 8. 学习曲线分析
def plot_learning_curve(estimator, X, y, cv, title, scoring='f1_macro'):
    plt.figure(figsize=(10, 6))    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring=scoring
    )
    metric_labels = {
        'f1_macro': 'F1 Score',
        'accuracy': 'Accuracy'
    }
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r", label="Training Set")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color="g", label="Validation Set")
    plt.fill_between(train_sizes, 
                     np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                     np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes,
                     np.mean(test_scores, axis=1) - np.std(test_scores, axis=1),
                     np.mean(test_scores, axis=1) + np.std(test_scores, axis=1),
                     alpha=0.1, color="g")
    plt.title(title)
    plt.xlabel("Number of Training Samples")
    plt.ylabel(metric_labels.get(scoring, 'Score'))
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

print("\n==== F1 Score Learning Curve ====")
plot_learning_curve(best_model, X_train, y_train, 
                   StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                   "Model F1 Score Learning Curve", scoring='f1_macro')

print("\n==== Accuracy Learning Curve ====")
plot_learning_curve(best_model, X_train, y_train, 
                   StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                   "Model Accuracy Learning Curve", scoring='accuracy')

# 9. 模型评估
def model_evaluation(model, X, y, dataset_name):
    y_pred = model.predict(X)
    print(f"\n{dataset_name}评估:")
    print(f"准确率: {accuracy_score(y, y_pred):.2%}")
    print(f"宏F1: {f1_score(y, y_pred, average='macro'):.2f}")
    print(classification_report(y, y_pred, target_names=le.classes_ if le else None))

print("\n==== 模型性能评估 ====")
model_evaluation(best_model, X_train, y_train, "训练集")
model_evaluation(best_model, X_test, y_test, "测试集")

# 10. 混淆矩阵
def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = confusion_matrix(y_test, y_pred, normalize='true')

    plt.figure(figsize=(10, 8))
    annot_matrix = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            row.append(f"{cm_norm[i,j]:.3f}")
        annot_matrix.append(row)

    color_matrix = np.zeros_like(cm_norm, dtype=float)
    np.fill_diagonal(color_matrix, cm_norm.diagonal())

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
        annot_kws={"size": 14, "color": "black", "ha": "center", "va": "center"}
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

plot_confusion_matrix(best_model, X_test, y_test)

# 11. 特征重要性
def plot_feature_importance(model):
    plt.figure(figsize=(10, 6))
    xgb_imp = model.named_estimators_['xgb'].feature_importances_
    rf_imp = model.named_estimators_['rf'].feature_importances_
    combined_imp = (xgb_imp + rf_imp) / 2
    sorted_idx = np.argsort(combined_imp)
    plt.barh(range(len(features)), combined_imp[sorted_idx], color='#2ca02c')
    plt.yticks(range(len(features)), [features[i] for i in sorted_idx])
    plt.title('Feature Importance Analysis\n', fontsize=14, fontweight='bold')
    plt.xlabel('Normalized Importance')
    plt.tight_layout()
    plt.show()

plot_feature_importance(best_model)

# 12. ROC曲线
def plot_roc_curves(model, X_test, y_test, le=None):
    try:
        num_classes = len(np.unique(y_test))
        y_score = model.predict_proba(X_test)
        class_names = le.classes_ if le else [f"Class {i}" for i in range(num_classes)]
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= num_classes
        fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, num_classes))
        for i in range(num_classes):
            plt.plot(fpr[i], tpr[i], color=colors[i], lw=1.5, linestyle='--',
                    label=f'{class_names[i]} (AUC={roc_auc[i]:.2f})')
        plt.plot(fpr["micro"], tpr["micro"],
                label=f'Micro-average (AUC={roc_auc["micro"]:.2f})',
                color='deeppink', linestyle=':', linewidth=3)
        plt.plot(fpr["macro"], tpr["macro"],
                label=f'Macro-average (AUC={roc_auc["macro"]:.2f})',
                color='navy', linestyle=':', linewidth=3)
        plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6)
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title('ROC Curve', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right', fontsize=13)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"ROC曲线绘制失败: {str(e)}")
        plt.close()

print("\n==== ROC曲线可视化 ====")
plot_roc_curves(best_model, X_test, y_test, le)

# 13. SHAP分析
try:
    explainer = shap.TreeExplainer(best_model.named_estimators_['xgb'])
    shap_values = explainer.shap_values(X_test_df)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, 
        X_test_df, 
        plot_type="bar", 
        class_names=le.classes_,
        show=False
    )
    ax = plt.gca()
    ax.legend(loc='lower right', bbox_to_anchor=(1, 0))
    plt.title("Feature Contribution Analysis", fontsize=14)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"SHAP可视化错误: {e}")

# 14. 岩性预测可视化（完整修改版）
def plot_well_log_with_lithology(df, depth_col='DEPTH', log_curves=['GR', 'SP', 'COND'], 
                                lithology_col='LITH', predicted_lith_col=None,
                                depth_range=None, figsize=(12, 10),
                                title_fontsize=14, label_fontsize=14, tick_fontsize=14):
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
    ax[track_idx].set_title("Core\nlithology", fontsize=title_fontsize)
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
            ax[track_idx].set_title(curve, fontsize=title_fontsize)
            ax[track_idx].grid(True, linestyle='--', alpha=0.5)
            ax[track_idx].spines['top'].set_visible(True)
            ax[track_idx].spines['right'].set_visible(True)
            track_idx += 1
    
    # 预测岩性道
    if predicted_lith_col:
        ax[track_idx].set_title("Predicted\nlithology", fontsize=title_fontsize)
        plot_lithology_track(ax[track_idx], plot_df, predicted_lith_col)
    
    # 通用设置
    ax[0].set_ylabel('Depth (m)', fontsize=label_fontsize)
    ax[0].invert_yaxis()
    
    # 设置刻度标签大小
    for a in ax:
        a.tick_params(labelsize=tick_fontsize)
    
    # 添加深度网格线
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
                                        le=None, depth_range=None,
                                        title_fontsize=14, label_fontsize=14, 
                                        tick_fontsize=14):
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
        depth_range=depth_range,
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize
    )

# 最终可视化调用（示例）
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
        depth_range=(1998, 2055),
        title_fontsize=14,   # 可自定义标题字号
        label_fontsize=14,   # 可自定义标签字号
        tick_fontsize=14    # 可自定义刻度字号
    )
except ValueError as e:
    print(f"可视化错误: {str(e)}")
