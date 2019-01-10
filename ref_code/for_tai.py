import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import ShuffleSplit, cross_validate
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix

# manually created data structure to record pdb cross-references for msa PF16592
pdb_refs = {
    'CAS9_STAAU': {
        '224-428': {
            '5AXW': {
                'A': '224-428'
            },
            '5CZZ': {
                'A': '224-428'
            }
        }
    },
    'CAS9_STRP1': {
        '181-710': {
            '4UN3': {
                'B': '181-710'
            },
            '4ZT9': {
                'A': '181-710'
            },
            '5FW1': {
                'B': '181-710'
            },
        },
        '181-712': {
            '4CMP': {
                'A': '181-712',
                'B': '181-712'
            },
            '4CMQ': {
                'A': '181-712',
                'B': '181-712'
            },
            '4OO8': {
                'A': '181-712',
                'D': '181-712'
            },
            '4UN4': {
                'B': '181-712'
            },
            '4UN5': {
                'B': '181-712'
            },
            '4ZT0': {
                'A': '181-712',
                'C': '181-712'
            },
            '4ZT9': {
                'C': '181-712'
            },
            '5B2T': {
                'B': '181-712'
            },
            '5F9R': {
                'B': '181-712'
            },
            '5FQ5': {
                'B': '181-712'
            },
            '5FW2': {
                'B': '181-712'
            },
            '5FW3': {
                'B': '181-712'
            },
        },
        '184-712': {
            '5B2R': {
                'B': '184-712'
            },
            '5B2S': {
                'B': '184-712'
            },
        }
    }
}

# a code snippet for interpreting the structure of pdb_refs
# for uniprot_id, ref in pdb_refs.items():
#     for uniprot_start_end, pdb in ref.items():
#         for pdb_id, chains in pdb.items():
#             for chain, pdb_start_end in chains.items():
#                 print(uniprot_id, uniprot_start_end, pdb_id, chain,
#                       pdb_start_end)

# get msa from fasta file
msa = pd.DataFrame([[seq.id] + list(str(seq.seq))
                    for seq in SeqIO.parse('PF16592_full.txt', 'fasta')])
msa.set_index(0, inplace=True)

# filter msa, uses same logic as dca.m
msa_filtered = msa.values.copy()
msa_filtered_keep = (msa_filtered[0] != '.') & (np.asarray(
    [x.upper() == x for x in msa_filtered[0]]))
msa_filtered_idx = np.where(msa_filtered_keep)[0]
msa_filtered = msa_filtered[:, msa_filtered_keep]
n_samples, n_features = msa_filtered.shape

# unique AAs per position
classes = [np.unique(col) for col in msa_filtered.T]
n_classes = [c.shape[0] for c in classes]

# plt.imshow(msa == '-')
# plt.show()

# list of unique amino acids
aa = np.unique(msa_filtered.flatten())
n_aa = len(aa)

# letter to number mappings, one for each position
letter_to_number = [
    dict(zip(c, np.arange(len(c), dtype=int))) for c in classes
]

# integer representation of filtered msa
y = np.zeros(msa_filtered.shape, dtype=int)
for i in range(msa_filtered.shape[0]):
    for j in range(msa_filtered.shape[1]):
        y[i, j] = letter_to_number[j][msa_filtered[i, j]]

# one-hot representation of filtered, integer msa
offset = np.insert(np.cumsum(n_classes), 0, 0)
data = np.ones(n_features * n_samples)
rows = np.repeat(range(n_samples), n_features)
cols = (y + offset[:-1]).flatten()
X = coo_matrix((data, (rows, cols)), shape=(n_samples, offset[-1])).tocsr()

# column frequencies
p = np.asarray(X.mean(0)).squeeze()

# for i, j in zip(offset[:-1], offset[1:]):
#     print(p[i:j].sum()) # should equal 1

# multinomial logistic regression classifier with l1-regularization
clf = LogisticRegression(
    multi_class='multinomial',
    solver='saga',
    penalty='l1',
    random_state=0,
    C=0.5,
    n_jobs=-1)

# coefficient matrix
w = np.zeros((offset[-1], offset[-1]))

# cross validation settings, set to False for faster fitting
cross_validation = True
scores = []
scoring = ['accuracy']

# for each column (sequence position)
# note: this serial implementation can be parallelized
for j in range(n_features):

    # split into inputs and target
    j1, j2 = offset[j], offset[j + 1]
    not_j = np.delete(range(offset[-1]), range(j1, j2))
    Xi, yi = X[:, not_j], y[:, j]

    # 4-fold cross-validation
    if cross_validation:
        cv = ShuffleSplit(n_splits=4, test_size=.25, random_state=0)
        scores.append(
            cross_validate(
                clf, Xi, yi, cv=cv, scoring=scoring, return_train_score=True))
        print(j, 'unique AAs in this pos:', n_classes[j], 'train acc:',
              scores[-1]['train_accuracy'].mean(), 'test acc:',
              scores[-1]['test_accuracy'].mean())

    # do the fitting
    clf = clf.fit(Xi, yi)

    # record fitting results in coeff matrix
    if np.unique(y).shape[0] > 1:
        w[:j1, j1:j2] = clf.coef_[:, :j1].T
        w[j2:, j1:j2] = clf.coef_[:, j1:].T
    else:
        w[:j1, j1] = clf.coef_[0, :j1].T
        w[j2:, j1] = clf.coef_[0, j1:].T
        w[:, j1 + 1] = -w[:, j1]

np.save('w.npy', w)
