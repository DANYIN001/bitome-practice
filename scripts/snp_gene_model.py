# %% --- IMPORT REQUIRED PACKAGES ---

# built-in modules
from pathlib import Path

# third-party modules
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.utils import resample
from xgboost import XGBClassifier

# built-in modules
from bitome.core import Bitome

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']

#%% --- LOAD BITOME AND TRAINING DATA ---

FIG_PATH = Path('figures', 'figure_4')

test_bitome = Bitome.init_from_file(Path('matrix_data', 'bitome.pkl'))

train_matrix = np.load(Path('matrix_data', 'TRAIN_snp_gene.npy'))
feature_names = test_bitome.matrix_row_labels + ['transcription_direction']
feature_names_seq_only = feature_names[:4]
feature_names_no_seq = feature_names[8:]

# assumes that the training matrix in the .npy file has ALL SNP (1) labels FIRST, THEN the non-SNP labels (0)
n_snps = int(train_matrix[:, -1].sum())
n_non_snps = train_matrix.shape[0] - n_snps

snp_matrix = train_matrix[:n_snps, :]
non_snp_matrix = train_matrix[n_snps:, :]

#%% --- PERFORM MODEL SELECTION ---

models_to_try = {
    'LR': LogisticRegression(
        penalty='l1',
        solver='liblinear'
    ),
    'SVM': LinearSVC(
        penalty='l1',
        dual=False
    ),
    'RF': RandomForestClassifier(),
    'XGBoost': XGBClassifier()
}

accuracy_df = pd.DataFrame(columns=list(models_to_try.keys()))
accuracy_df_seq_only = pd.DataFrame(columns=list(models_to_try.keys()))
accuracy_df_no_seq = pd.DataFrame(columns=list(models_to_try.keys()))
accuracy_df_shuffled = pd.DataFrame(columns=list(models_to_try.keys()))
accuracy_df_list = [accuracy_df, accuracy_df_seq_only, accuracy_df_no_seq, accuracy_df_shuffled]

n_exploratory = 10
for i in range(n_exploratory):

    down_sample_snp_matrix = resample(snp_matrix, n_samples=n_non_snps, replace=False)
    down_sample_matrix = np.concatenate([down_sample_snp_matrix, non_snp_matrix], axis=0)
    np.random.shuffle(down_sample_matrix)
    X, y = down_sample_matrix[:, :-1], down_sample_matrix[:, -1]

    # prepare subsets of the features, and shuffled targets
    X_seq_only = X[:, :4]
    X_no_seq = X[:, 8:]
    y_shuffled = np.random.permutation(y)

    # set up the training matrices we want to use
    X_list = [X, X_seq_only, X_no_seq, X]
    y_list = [y, y, y, y_shuffled]

    for model_name, model in models_to_try.items():
        print(f'{i}: {model_name}')

        for X_current, y_current, result_df in zip(X_list, y_list, accuracy_df_list):

            cv_scores = cross_validate(
                model,
                X_current,
                y=y_current,
                cv=5,
                scoring='accuracy',
                verbose=3,
                n_jobs=4
            )
            result_df.loc[i, model_name] = np.mean(cv_scores['test_score'])

accuracy_df.to_csv(Path('matrix_data', 'model_select_accuracy.csv'))
accuracy_df_seq_only.to_csv(Path('matrix_data', 'model_select_accuracy_seq_only.csv'))
accuracy_df_no_seq.to_csv(Path('matrix_data', 'model_select_accuracy_no_seq.csv'))
accuracy_df_shuffled.to_csv(Path('matrix_data', 'model_select_accuracy_shuffled.csv'))

#%% --- Reload Model Selection Data If Necessary ---

RELOAD_DATA = True
if RELOAD_DATA:
    accuracy_df = pd.read_csv(Path('matrix_data', 'model_select_accuracy.csv'), index_col=0)
    accuracy_df_seq_only = pd.read_csv(Path('matrix_data', 'model_select_accuracy_seq_only.csv'), index_col=0)
    accuracy_df_no_seq = pd.read_csv(Path('matrix_data', 'model_select_accuracy_no_seq.csv'), index_col=0)
    accuracy_df_shuffled = pd.read_csv(Path('matrix_data', 'model_select_accuracy_shuffled.csv'), index_col=0)

#%% ---

_, ax = plt.subplots()
sns.swarmplot(data=accuracy_df, color='tab:blue')
sns.swarmplot(data=accuracy_df_shuffled, color='tab:gray')
ax.tick_params(axis='both', labelsize=22)
plt.ylabel('Accuracy', fontsize=24)
legend_elems = [
    Patch(facecolor='tab:blue', edgecolor='tab:blue', label='bitome'),
    Patch(facecolor='tab:gray', edgecolor='tab:gray', label='shuffled')
]
plt.legend(handles=legend_elems, prop={'size': 18}, loc='upper left')
plt.savefig(Path(FIG_PATH, 'model_selection.svg'))
plt.show()

# create a single dataframe with the Random Forest values for the different data subsets
feature_v_sequence_df = pd.DataFrame(data={
    'bitome': accuracy_df['RF'],
    'no seq': accuracy_df_no_seq['RF'],
    'seq only': accuracy_df_seq_only['RF']
})

_, ax = plt.subplots()
sns.barplot(data=feature_v_sequence_df)
plt.ylim(0.6, 0.75)
ax.tick_params(axis='both', labelsize=22)
plt.ylabel('Accuracy', fontsize=24)
plt.savefig(Path(FIG_PATH, 'bitome_v_sequence.svg'))
plt.show()

#%% --- PERFORM HYPERPARAMETER OPTIMIZATION ---

random_forest_hyperopt = RandomForestClassifier()
param_distributions = {
    'n_estimators': np.arange(150, 250),
    'max_depth': np.arange(5, 10),
    'min_samples_split': np.arange(0.001, 0.02, 0.001),
    'min_samples_leaf': np.arange(5, 15)
}
random_search_hyperopt = RandomizedSearchCV(
    random_forest_hyperopt,
    param_distributions,
    n_iter=25,
    scoring='accuracy',
    n_jobs=6,
    cv=5,
    verbose=1,
    return_train_score=True
)

down_sample_snp_matrix_hyperopt = resample(snp_matrix, n_samples=n_non_snps, replace=False)
down_sample_matrix_hyperopt = np.concatenate([down_sample_snp_matrix_hyperopt, non_snp_matrix], axis=0)
np.random.shuffle(down_sample_matrix_hyperopt)
X_hyperopt, y_hyperopt = down_sample_matrix_hyperopt[:, :-1], down_sample_matrix_hyperopt[:, -1]

random_search_hyperopt.fit(X_hyperopt, y_hyperopt)
hyperopt_results_df = pd.DataFrame(random_search_hyperopt.cv_results_).sort_values(
    by='mean_test_score',
    ascending=False
)

#%% --- MODEL ASSESSMENT ---

down_sample_snp_matrix_final = resample(snp_matrix, n_samples=n_non_snps, replace=False)
down_sample_matrix_final = np.concatenate([down_sample_snp_matrix_final, non_snp_matrix], axis=0)
np.random.shuffle(down_sample_matrix_final)
X_final, y_final = down_sample_matrix_final[:, :-1], down_sample_matrix_final[:, -1]

# re-train a fresh random forest on the full training data using the hyperparameters from the above cell
random_forest_final = RandomForestClassifier(
    n_estimators=200,
    max_depth=7,
    min_samples_split=0.01,
    min_samples_leaf=10
)
random_forest_final.fit(X_final, y_final)

# make sure we have a balanced lockbox (didn't make sure of this before)
lockbox_matrix = np.load(Path('matrix_data', 'LOCK_snp_gene.npy'))
# assumes that the training matrix in the .npy file has ALL SNP (1) labels FIRST, THEN the non-SNP labels (0)
n_snps_lockbox = int(lockbox_matrix[:, -1].sum())
n_non_snps_lockbox = lockbox_matrix.shape[0] - n_snps_lockbox

snp_lockbox_matrix = lockbox_matrix[:n_snps_lockbox, :]
non_snp_lockbox_matrix = lockbox_matrix[n_snps_lockbox:, :]

down_sample_snp_lockbox_matrix = resample(snp_lockbox_matrix, n_samples=n_non_snps_lockbox, replace=False)
balanced_lockbox_matrix = np.concatenate([down_sample_snp_lockbox_matrix, non_snp_lockbox_matrix], axis=0)
np.random.shuffle(balanced_lockbox_matrix)
X_lockbox, y_lockbox = balanced_lockbox_matrix[:, :-1], balanced_lockbox_matrix[:, -1]

final_score = random_forest_final.score(X_lockbox, y_lockbox)
print(f'Final Model Accuracy: {final_score:.3f}')

np.save(Path('matrix_data', 'X_final.npy'), X_final)
np.save(Path('matrix_data', 'y_final.npy'), y_final)
np.save(Path('matrix_data', 'X_lockbox.npy'), X_lockbox)
np.save(Path('matrix_data', 'y_lockbox.npy'), y_lockbox)
with open('random_forest_final.pickle', 'wb') as handle:
    pickle.dump(random_forest_final, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%% --- LOADING MODELS FOR VISUALIZATION ---

X_final = np.load(Path('matrix_data', 'X_final.npy'))
y_final = np.load(Path('matrix_data', 'y_final.npy'))
X_lockbox = np.load(Path('matrix_data', 'X_lockbox.npy'))
y_lockbox = np.load(Path('matrix_data', 'y_lockbox.npy'))
with open('random_forest_final.pickle', 'rb') as handle:
    random_forest_final = pickle.load(handle)

y_pred = random_forest_final.predict(X_lockbox)
confusion_mat = confusion_matrix(y_lockbox, y_pred, normalize='true')

_, ax = plt.subplots()
sns.heatmap(
    confusion_mat,
    cmap="PuBu",
    annot=True,
    annot_kws={'fontsize': 22},
    cbar=False,
    square=True,
    xticklabels=['No SNP', 'SNP'],
    yticklabels=['No SNP', 'SNP']
)
ax.tick_params(axis='both', labelsize=22)
ax.tick_params(axis='y', rotation=0)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['top'].set_visible(True)
ax.spines['bottom'].set_visible(True)
plt.xlabel('Predicted Class', fontsize=24)
plt.ylabel('True Class', fontsize=24)
plt.savefig(Path(FIG_PATH, 'confusion_matrix.svg'))
plt.show()

_, ax = plt.subplots()
sorted_feature_names, sorted_feature_importances = zip(*sorted(
    zip(feature_names, random_forest_final.feature_importances_),
    key=lambda tup: tup[1],
    reverse=True
))
sns.distplot(sorted_feature_importances, kde=False, hist_kws={'log': True})
plt.xticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])
ax.axvline(x=0.01, ymax=0.05, color='r', linewidth=2)
ax.tick_params(axis='both', labelsize=20)
plt.xlabel('Feature Importance', fontsize=22),
plt.ylabel('Count', fontsize=22)
plt.savefig(Path(FIG_PATH, 'feature_importances.png'), bbox_inches='tight')
plt.show()

important_names_df = pd.DataFrame(data={
    'feature': sorted_feature_names[:13],
    'importance': np.around(sorted_feature_importances[:13], decimals=3)}
)
important_names_df.to_csv(Path(FIG_PATH, 'important_features.csv'))
