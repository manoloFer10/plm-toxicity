import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tqdm import tqdm



def layerwise_linear_probe(acts, labels, positions="mean"):
    '''
    Expects acts to be on cpu.
    '''
    N, P, L, d = acts.shape
    # collapse the positions dimension ------------------------------------
    if positions == "mean":
        embeds = acts.mean(1)                # (N, L, d)
    elif positions == "flatten":             # concat all positions
        embeds = acts.reshape(N, L, P*d)     # (N, L, P·d)
    else:                                    # explicit list/array of idx
        embeds = acts[:, positions, :].reshape(N, L, -1)

    acc, auc, f1, clfs = [], [], [], []
    for l in tqdm(range(L), total = L, desc='Linear Probing of layers'):
        X = embeds[:, l, :]                  # (N, F)
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X) #Scale data

        # training and test splits
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, labels, test_size=0.3, random_state=42, stratify=labels
        )

        clf = LogisticRegressionCV(
            Cs=10, cv=5, max_iter=1000, scoring="roc_auc", n_jobs=-1
        ).fit(X_tr, y_tr)

        y_hat = clf.predict(X_te)

        #report accuracy on test splits
        acc.append( accuracy_score(y_te, y_hat) )
        auc.append( roc_auc_score(y_te, clf.predict_proba(X_te)[:,1]) )
        f1.append( f1_score(y_true=y_te, y_pred=y_hat))
        #clfs.append(clf)

    metrics={
        'accuracy': np.array(acc),
        'auc': np.array(auc),
        'f1': np.array(f1)
    }
    
    return metrics #, clfs #return the models if so.

def fisher_ratio(acts, labels, positions="mean", eps=1e-6):
    if positions == "mean":
        embeds = acts.mean(1)          # (N, L, d)
    else:                              # e.g. [-1] or "flatten"
        embeds = acts.reshape(acts.shape[0], acts.shape[2], -1)

    fisher = []
    for l in tqdm(range(embeds.shape[1]), total = embeds.shape[1], desc='Fisher Ratio of layers'):
        Xl = embeds[:, l, :]           # (N, F)
        tox, nontox = Xl[labels==0], Xl[labels==1]

        mu_t, mu_nt = tox.mean(0), nontox.mean(0)
        diff = (mu_t - mu_nt).pow(2).sum()

        Sw = tox.var(0) + nontox.var(0) + eps   # pooled within-class var
        fisher.append( (diff / Sw.sum()).item() )
    return np.array(fisher)

def cosine_rdm(X):                       # X: (samples, features)
    # 1 - cosine similarity matrix
    X = torch.nn.functional.normalize(torch.from_numpy(X), dim=1)
    return 1 - X @ X.T                   # (samples, samples)

def layerwise_rsa(acts, labels, positions="mean"):
    if positions == "mean":
        embeds = acts.mean(1).cpu().numpy()
    else:
        embeds = acts.reshape(acts.shape[0], acts.shape[2], -1).cpu().numpy()

    rsa_scores = []
    for l in tqdm(range(embeds.shape[1]), total = embeds.shape[1], desc='RSA of layers'):
        X = embeds[:, l, :]
        rdm_t  = cosine_rdm(X[labels==0])
        rdm_nt = cosine_rdm(X[labels==1])
        # Pearson corr between upper triangles
        mask = np.triu_indices_from(rdm_t, k=1)
        corr = np.corrcoef(rdm_t[mask], rdm_nt[mask])[0,1]
        rsa_scores.append(corr)
    return np.array(rsa_scores)   # 1 ≈ identical geometry, 0 ≈ unrelated