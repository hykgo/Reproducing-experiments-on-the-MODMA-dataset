import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class DepressionClassifier:
    def __init__(self):
        """
        初始化分类器
        """
        # 1. KNN (k=3)
        self.knn = KNeighborsClassifier(n_neighbors=3)

        # 2. AdaBoost
        self.adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)

        # 3. BF-Tree (使用决策树模拟，Best-First 搜索在 sklearn 中通常通过 max_leaf_nodes 实现)
        self.bf_tree = DecisionTreeClassifier(max_leaf_nodes=20, random_state=42)

    def evaluate_on_split(self, X_train, y_train, X_test, y_test):
        """
        在给定的训练集和测试集上评估模型性能
        :param X_train: 训练特征矩阵
        :param y_train: 训练标签向量
        :param X_test: 测试特征矩阵
        :param y_test: 测试标签向量
        :return: 包含各模型性能指标的 DataFrame
        """
        models = {
            'KNN (k=3)': self.knn,
            'AdaBoost': self.adaboost,
            'BF-Tree': self.bf_tree
        }

        results = []

        for name, model in models.items():
            # 训练模型
            model.fit(X_train, y_train)

            # 在测试集上预测
            y_pred = model.predict(X_test)

            # 计算指标
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            results.append({
                'Model': name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1
            })

        return pd.DataFrame(results)

    def get_paper_selected_features(self, df_features, ch_names=['Fp1', 'Fp2', 'T7', 'T8', 'O1', 'O2']):
        """
        根据论文逻辑筛选特征。
        """
        core_features = ['min_amp', 'max_amp', 'mean_abs_1st_diff', 'mean_abs_2nd_diff', 'mean_val', 'p2p_val']

        selected_cols = []
        for ch in ch_names:
            for feat in core_features:
                col_name = f"{ch}_{feat}"
                if col_name in df_features.columns:
                    selected_cols.append(col_name)

        return df_features[selected_cols]