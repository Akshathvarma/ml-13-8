# lab6.py — Decision Tree A1–A7
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

DATA_PATH = r"C:\Users\Shravya\Desktop\CSE23138_ml5\ecg_eeg_features.csv.xlsx"

#  Load dataset 
def load_dataset(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, engine="openpyxl")
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        # try excel, fallback to csv
        try:
            return pd.read_excel(path, engine="openpyxl")
        except Exception:
            return pd.read_csv(path)

data = load_dataset(DATA_PATH)
print(f"[INFO] Loaded dataset: {DATA_PATH}")
print(f"[INFO] Shape: {data.shape}")

# Identify target & features 
if "signal_type" in data.columns:
    y = data["signal_type"].astype(str)
    X = data.drop(columns=["signal_type"]).copy()
else:
    # fallback: last column as target
    print("[WARN] 'signal_type' not found, using last column as target")
    y = data.iloc[:, -1].astype(str)
    X = data.iloc[:, :-1].copy()

# Fill missing values
for c in X.columns:
    if pd.api.types.is_numeric_dtype(X[c]):
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

# A1. Entropy
def entropy(labels) -> float:
    if len(labels) == 0:
        return 0.0
    codes, _ = pd.factorize(labels)
    counts = np.bincount(codes)
    probs = counts[counts > 0] / counts.sum()
    return float(-(probs * np.log2(probs)).sum())

print("A1) Entropy of target:", entropy(y))

# A2. Gini index
def gini_index(labels) -> float:
    if len(labels) == 0:
        return 0.0
    codes, _ = pd.factorize(labels)
    counts = np.bincount(codes)
    probs = counts[counts > 0] / counts.sum()
    return float(1.0 - (probs**2).sum())

print("A2) Gini index of target:", gini_index(y))

# A4. Binning (for continuous features)
def binning(series: pd.Series, bins: int = 4, method: str = "width"):
    if not pd.api.types.is_numeric_dtype(series):
        return pd.Series(pd.Categorical(series).codes, index=series.index)

    if method == "width":
        binned = pd.cut(series, bins=bins, labels=False, include_lowest=True, duplicates="drop")
    elif method == "frequency":
        try:
            binned = pd.qcut(series, q=bins, labels=False, duplicates="drop")
        except ValueError:
            binned = pd.cut(series, bins=bins, labels=False, include_lowest=True, duplicates="drop")
    else:
        raise ValueError("method must be 'width' or 'frequency'")

    if binned.isna().any():
        fill_val = (int(binned.max()) + 1) if binned.notna().any() else 0
        binned = binned.fillna(fill_val)
    return binned.astype(int)

# A3. Root node via Information Gain
def information_gain(feature_col, labels) -> float:
    feature_col = pd.Series(feature_col).reset_index(drop=True)
    labels = pd.Series(labels).reset_index(drop=True)
    total_H = entropy(labels)
    values, counts = np.unique(feature_col, return_counts=True)
    N = counts.sum()
    weighted_H = 0.0
    for v, cnt in zip(values, counts):
        weighted_H += (cnt / N) * entropy(labels[feature_col == v])
    return float(total_H - weighted_H)

def choose_root_node(Xdf: pd.DataFrame, labels: pd.Series, bins: int = 4, bin_method: str = "width"):
    gains = {}
    for col in Xdf.columns:
        col_data = Xdf[col]
        if pd.api.types.is_numeric_dtype(col_data):
            col_data = binning(col_data, bins=bins, method=bin_method)
        else:
            col_data = pd.Series(pd.Categorical(col_data).codes, index=col_data.index)
        gains[col] = information_gain(col_data, labels)
    root_feature = max(gains, key=gains.get)
    return root_feature, gains

root, gains = choose_root_node(X, y)
print("A3) Root node via Information Gain:", root)

# A5. Train Decision Tree
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
clf.fit(X_train, y_train)
print("A5) Decision Tree accuracy (test):", clf.score(X_test, y_test))

# A6. Visualize Decision Tree
plt.figure(figsize=(18, 9))
plot_tree(
    clf,
    feature_names=X.columns.tolist(),
    class_names=[str(c) for c in clf.classes_],
    filled=True,
    rounded=True
)
plt.title("Decision Tree (entropy, max_depth=4)")
plt.tight_layout()
plt.show()

# A7. Decision Boundary (2 features)
# pick mean_val and std_dev, else fallback
two_feats = [c for c in ["mean_val", "std_dev"] if c in X.columns]
if len(two_feats) < 2:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    two_feats = num_cols[:2]

fx, fy = two_feats
X2 = X[[fx, fy]].values
y2 = y.values

Xtr2, Xte2, ytr2, yte2 = train_test_split(X2, y2, test_size=0.30, random_state=42, stratify=y2)
clf2 = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
clf2.fit(Xtr2, ytr2)
print(f"A7) 2D DT accuracy (test) using '{fx}' & '{fy}':", clf2.score(Xte2, yte2))

# Decision surface
pad = 1.0
x_min, x_max = X2[:, 0].min() - pad, X2[:, 0].max() + pad
y_min, y_max = X2[:, 1].min() - pad, X2[:, 1].max() + pad
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)

grid = np.c_[xx.ravel(), yy.ravel()]
Z_labels = clf2.predict(grid)
label_to_int = {lab: i for i, lab in enumerate(clf2.classes_)}
Z = np.vectorize(label_to_int.get)(Z_labels).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.35)
plt.scatter(X2[:, 0], X2[:, 1], c=pd.factorize(y2)[0], s=15, edgecolors="k")
plt.xlabel(fx)
plt.ylabel(fy)
plt.title(f"Decision Boundary ({fx} vs {fy}) - DecisionTree")
plt.tight_layout()
plt.show()