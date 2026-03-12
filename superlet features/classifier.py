"""
SVM 分类器模块
严格按照论文《Towards alpha and theta band superlet features as biomarkers for depression》

分类器参数：
- 核函数：RBF（径向基函数）
- 参数 C：通过网格搜索选择（单数据集：50，跨数据集：78.84）
- Gamma：0.0059
- 交叉验证：10 折
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')


class SVMClassifier:
    """SVM 分类器"""

    def __init__(self, kernel='rbf', C=50, gamma='scale', random_state=42):
        """
        初始化 SVM 分类器

        参数：
        - kernel: 核函数类型（'rbf'）
        - C: 正则化参数
        - gamma: RBF 核参数
        - random_state: 随机种子
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state

        # 创建 SVM 模型
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=random_state)

        # 特征缩放器
        self.scaler = StandardScaler()

        # 存储训练数据统计
        self.is_fitted = False

    def fit(self, X_train, y_train):
        """
        训练 SVM 模型

        参数：
        - X_train: 训练特征 (样本数 × 特征数)
        - y_train: 训练标签 (0: HC, 1: MDD)
        """
        # 特征缩放
        X_train_scaled = self.scaler.fit_transform(X_train)

        # 训练模型
        self.model.fit(X_train_scaled, y_train)

        self.is_fitted = True

    def predict(self, X_test):
        """
        预测

        参数：
        - X_test: 测试特征 (样本数 × 特征数)

        返回：
        - y_pred: 预测标签
        """
        if not self.is_fitted:
            raise ValueError("模型未训练")

        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)

        return y_pred

    def predict_proba(self, X_test):
        """
        预测概率

        参数：
        - X_test: 测试特征

        返回：
        - proba: 预测概率
        """
        if not self.is_fitted:
            raise ValueError("模型未训练")

        X_test_scaled = self.scaler.transform(X_test)
        proba = self.model.decision_function(X_test_scaled)

        return proba


def compute_metrics(y_true, y_pred):
    """
    计算分类性能指标

    参数：
    - y_true: 真实标签
    - y_pred: 预测标签

    返回：
    - metrics: 指标字典
    """
    # 确保标签是 0 和 1
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 计算指标
    acc = (tp + tn) / (tp + tn + fp + fn)
    spc = tn / (tn + fp) if (tn + fp) > 0 else 0
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    metrics = {
        'accuracy': acc,
        'specificity': spc,
        'sensitivity': sen,
        'f1_score': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

    return metrics


def cross_validate_svm(X, y, n_splits=10, C=50, gamma='scale'):
    """
    10 折交叉验证

    参数：
    - X: 特征矩阵 (样本数 × 特征数)
    - y: 标签向量
    - n_splits: 折数（论文中为 10）
    - C: SVM 参数
    - gamma: SVM gamma 参数

    返回：
    - results: 每折的结果字典
    - overall_metrics: 整体指标
    """
    # 创建分层 K 折
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = []
    all_y_true = []
    all_y_pred = []

    fold_idx = 0
    for train_idx, test_idx in skf.split(X, y):
        fold_idx += 1

        # 分割数据
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 创建和训练分类器
        clf = SVMClassifier(kernel='rbf', C=C, gamma=gamma)
        clf.fit(X_train, y_train)

        # 预测
        y_pred = clf.predict(X_test)

        # 计算指标
        metrics = compute_metrics(y_test, y_pred)
        metrics['fold'] = fold_idx

        results.append(metrics)

        # 收集所有预测
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    # 计算整体指标
    overall_metrics = compute_metrics(all_y_true, all_y_pred)

    return results, overall_metrics


def grid_search_svm(X, y, n_splits=10):
    """
    网格搜索最优 SVM 参数

    参数：
    - X: 特征矩阵
    - y: 标签向量
    - n_splits: 交叉验证折数

    返回：
    - best_params: 最优参数
    - best_score: 最优分数
    """
    # 特征缩放
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 参数网格
    param_grid = {
        'C': [0.1, 1, 10, 50, 100],
        'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01]
    }

    # 创建 SVM 模型
    svm = SVC(kernel='rbf')

    # 网格搜索
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=skf, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_scaled, y)

    return grid_search.best_params_, grid_search.best_score_


def train_test_split_by_subject(X, y, subject_ids, train_ratio=0.8):
    """
    按受试者进行训练/测试集划分（防止数据泄露）

    参数：
    - X: 特征矩阵 (样本数 × 特征数)
    - y: 标签向量
    - subject_ids: 受试者 ID 向量
    - train_ratio: 训练集比例

    返回：
    - X_train, X_test, y_train, y_test: 划分后的数据
    """
    # 获取唯一的受试者 ID
    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)

    # 计算训练集受试者数
    n_train_subjects = int(n_subjects * train_ratio)

    # 随机选择训练集受试者
    np.random.seed(42)
    train_subjects = np.random.choice(unique_subjects, n_train_subjects, replace=False)

    # 划分数据
    train_mask = np.isin(subject_ids, train_subjects)
    test_mask = ~train_mask

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    return X_train, X_test, y_train, y_test


# 测试代码
if __name__ == "__main__":
    # 生成测试数据
    np.random.seed(42)
    n_samples = 200
    n_features = 152  # 19 通道 × 4 频段 × 2 特征

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)

    print("测试 SVM 分类器...")

    # 10 折交叉验证
    print("\n10 折交叉验证:")
    results, overall_metrics = cross_validate_svm(X, y, n_splits=10, C=50)

    print(f"总体准确率: {overall_metrics['accuracy']:.4f}")
    print(f"总体特异性: {overall_metrics['specificity']:.4f}")
    print(f"总体敏感性: {overall_metrics['sensitivity']:.4f}")
    print(f"总体 F1 分数: {overall_metrics['f1_score']:.4f}")

    # 显示每折结果
    print("\n每折结果:")
    for result in results:
        print(f"  Fold {result['fold']}: Acc={result['accuracy']:.4f}, "
              f"Spc={result['specificity']:.4f}, Sen={result['sensitivity']:.4f}")