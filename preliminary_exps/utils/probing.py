import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm



def layerwise_linear_probe(acts, labels, positions="mean"):
    """
    Leak-safe layerwise probe using Pipeline[StandardScaler -> LogisticRegression]
    tuned with GridSearchCV over C and class_weight. Uses StratifiedKFold.
    Returns metrics per layer. Expects acts to be on CPU.
    """
    N, P, L, d = acts.shape

    # Collapse positions
    if positions == "mean":
        embeds = acts.mean(1)                # (N, L, d)
    elif positions == "flatten":
        embeds = acts.reshape(N, L, P * d)   # (N, L, P·d)
    else:
        embeds = acts[:, positions, :].reshape(N, L, -1)

    y = labels.cpu().numpy()
    min_class = np.bincount(y).min()
    inner_cv = max(2, min(5, min_class))     # avoid 1-class folds
    skf = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=42)

    # Pipeline keeps scaling inside each CV fold (no leakage)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="liblinear", max_iter=2000))
    ])

    # Tune both C and class_weight
    param_grid = {
        "clf__C": np.logspace(-3, 3, 10),
        "clf__class_weight": [None, "balanced"],
    }

    acc, auc, f1, bal_acc, ap = [], [], [], [], []

    for l in tqdm(range(L), total=L, desc='Linear Probing of layers', leave=True):
        X = embeds[:, l, :].cpu().numpy()

        # Hold-out split (kept clean). Stratified to maintain ratios.
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )

        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=skf,
            scoring="roc_auc",     
            n_jobs=-1,
            refit=True
        )
        search.fit(X_tr, y_tr)
        best_model = search.best_estimator_

        y_proba = best_model.predict_proba(X_te)[:, 1]
        y_hat = best_model.predict(X_te)

        # Metrics
        acc.append(accuracy_score(y_te, y_hat))
        #bal_acc.append(balanced_accuracy_score(y_te, y_hat))

        if np.unique(y_te).size < 2:
            auc.append(np.nan)
            #ap.append(np.nan)
        else:
            auc.append(roc_auc_score(y_te, y_proba))
           # ap.append(average_precision_score(y_te, y_proba))

        f1.append(f1_score(y_true=y_te, y_pred=y_hat))

    metrics = {
        "accuracy": np.array(acc),
        #"balanced_accuracy": np.array(bal_acc),
        "auc": np.array(auc),
        #"average_precision": np.array(ap),
        "f1": np.array(f1),
    }
    return metrics #, best_models #return the models if so.

def fisher_ratio(acts, labels, positions="mean", eps=1e-6):
    if positions == "mean":
        embeds = acts.mean(1)          # (N, L, d)
    else:                              # e.g. [-1] or "flatten"
        embeds = acts.reshape(acts.shape[0], acts.shape[2], -1)

    tox_mask    = (labels == 1)
    nontox_mask = (labels == 0)

    fisher = []
    for l in tqdm(range(embeds.shape[1]), total = embeds.shape[1], desc='Fisher Ratio of layers', leave=True):
        Xl = embeds[:, l, :]           # (N, F)
        tox, nontox = Xl[tox_mask], Xl[nontox_mask]

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

    tox = (labels.cpu().numpy() == 1)
    nt  = (labels.cpu().numpy() == 0)

    def _safe_norm_rows(X, eps=1e-12):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.clip(n, eps, None)

    rsa = []
    for l in tqdm(range(embeds.shape[1]), total=embeds.shape[1], desc='RSA of layers', leave=True):
        Xt, Xn = embeds[tox, l, :], embeds[nt, l, :]
        if Xt.shape[0] < 2 or Xn.shape[0] < 2:
            rsa.append(np.nan); continue
        Xt, Xn = _safe_norm_rows(Xt), _safe_norm_rows(Xn)
        rdm_t = 1 - Xt @ Xt.T
        rdm_n = 1 - Xn @ Xn.T
        iu = np.triu_indices_from(rdm_t, k=1)
        vt, vn = rdm_t[iu], rdm_n[iu]
        if (np.isnan(vt).any() or np.isnan(vn).any() or
            np.std(vt) == 0 or np.std(vn) == 0):
            rsa.append(np.nan); continue
        rsa.append(np.corrcoef(vt, vn)[0, 1])
    return np.array(rsa)   # 1 ≈ identical geometry, 0 ≈ unrelated