# Import libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
from sklearn.mixture import GaussianMixture
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from snf import compute
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns


warnings.filterwarnings("ignore")

# Load datasets
rna = pd.read_csv('data/rna.tsv', sep='\t', index_col=0)
mir = pd.read_csv('data/mir.tsv', sep='\t', index_col=0)
meth = pd.read_csv('data/meth.tsv', sep='\t', index_col=0)
survival = pd.read_csv('data/survival.tsv', sep='\t')


# Match samples
common_samples = set(rna.index) & set(mir.index) & set(meth.index)
print(len(common_samples))


#rna, mir, meth = rna[common_samples].T, mir[common_samples].T, meth[common_samples].T
survival = survival.set_index('Samples').loc[list(common_samples)]

def rank_normalize(df, top_n=100):
    top_features = df.var(axis=1).nlargest(top_n).index
    ranked = df.loc[top_features].rank(axis=1, pct=True).T  # shape: samples × features
    return ranked


rna_norm = rank_normalize(rna.T)
mir_norm = rank_normalize(mir.T)
meth_norm = rank_normalize(meth.T)

print("RNA shape:", rna_norm.shape)
print("miRNA shape:", mir_norm.shape)
print("Methylation shape:", meth_norm.shape)

# PCA to reduce to top 100 components
pca = PCA(n_components=100)
rna_pca = pca.fit_transform(rna_norm)
mir_pca = pca.fit_transform(mir_norm)
meth_pca = pca.fit_transform(meth_norm)

data_list = [rna_pca, mir_pca, meth_pca]
affinity_networks = compute.make_affinity(data_list, metric='euclidean', K=20, mu=0.5)
fused_matrix = compute.snf(affinity_networks, K=20)

# Convert to DataFrame
fused_df = pd.DataFrame(fused_matrix, index=survival.index, columns=survival.index)

significant_features = []
count=0
for col in fused_df.columns:
    temp_df = pd.DataFrame({
        'feature': fused_df[col],
        'time': survival['days'],
        'event': survival['event']
    })

    # Skip if NaN or no variance
    if temp_df['feature'].isna().any() or temp_df['feature'].std() == 0:
        continue

    try:
        cph = CoxPHFitter()
        cph.fit(temp_df, duration_col='time', event_col='event')
        if cph.summary.loc['feature', 'p'] < 0.01:
            significant_features.append(col)
    except Exception as e:
        # Optional: print(f"Skipping {col} due to error: {e}")
        count+=1
        continue

print(count)
# Final survival-associated feature matrix
X_selected = fused_df[significant_features]


print("X_selected shape:", X_selected.shape)
print("Selected features:", X_selected.columns.tolist())


gm = GaussianMixture(n_components=2, random_state=42)
clusters = gm.fit_predict(X_selected)

print(np.unique(clusters, return_counts=True))

# Add clusters to survival
survival['cluster'] = clusters

X_train, X_test, y_train, y_test = train_test_split(X_selected, clusters, test_size=0.2, random_state=42, stratify=None)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print("Precision:", precision_score(y_test, preds))
print("Recall:", recall_score(y_test, preds))
print("F1 Score:", f1_score(y_test, preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

# If rf is your trained model
importances = rf.feature_importances_
feat_names = X_selected.columns

sns.barplot(x=importances, y=feat_names)
plt.title("Feature Importance (Random Forest)")
plt.show()

# Concordance Index
c_index = concordance_index(survival.loc[X_test.index, 'days'],
                            -preds,  # higher risk → lower predicted label
                            survival.loc[X_test.index, 'event'])
print(f"C-index: {c_index:.4f}")

# Kaplan-Meier Plot
kmf = KaplanMeierFitter()
plt.figure(figsize=(8, 6))

for label in np.unique(preds):
    mask = preds == label
    kmf.fit(survival.loc[X_test.index[mask], 'days'],
            survival.loc[X_test.index[mask], 'event'],
            label=f'Subtype {label+1}')
    kmf.plot_survival_function()

plt.title("Kaplan-Meier Survival by Predicted Subtypes")
plt.xlabel("Days")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.show()