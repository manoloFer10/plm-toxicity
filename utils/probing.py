import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, make_scorer, precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm



def _safe_auc_for_cv(y_true, y_score):
    if np.unique(y_true).size < 2:
        return 0.5  # neutral baseline to avoid NaNs & warnings
    return roc_auc_score(y_true, y_score)

safe_auc_scorer = make_scorer(_safe_auc_for_cv, needs_proba=True, greater_is_better=True)


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

     # Custom scorer: signature (estimator, X, y)
    def safe_auc_scorer(estimator, X, y):
        y_proba = estimator.predict_proba(X)[:, 1]
        if np.unique(y).size < 2:
            return 0.5  # neutral baseline when validation fold is single-class
        return roc_auc_score(y, y_proba)

    acc, auc, f1, bal_acc, ap = [], [], [], [], []

    for l in tqdm(range(L), total=L, desc='Linear Probing of layers', leave=True):
        X = embeds[:, l, :].cpu().numpy()

        # Hold-out split. Stratified to maintain ratios.
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # decide inner CV based on y_tr 
        binc = np.bincount(y_tr)
        min_class_tr = binc.min() if binc.size > 1 else 0

        if min_class_tr < 2:
            # Too few minority examples to run stratified CV safely
            best_model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    solver="liblinear", max_iter=2000, class_weight="balanced", C=1.0
                ))
            ])
            best_model.fit(X_tr, y_tr)
        else:
            n_splits = max(2, min(5, int(min_class_tr)))
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            search = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                cv=skf,
                scoring=safe_auc_scorer,        
                n_jobs=-1,
                refit=True
            )
            search.fit(X_tr, y_tr)
            best_model = search.best_estimator_

        oof = cross_val_predict(best_model, X_tr, y_tr, cv=skf, method="predict_proba")[:,1]
        prec, rec, thr = precision_recall_curve(y_tr, oof)
        f1s = 2*prec*rec/(prec+rec+1e-12)
        best_thr = thr[f1s[:-1].argmax()] if len(thr) else 0.5  # choose on train only

        y_proba = best_model.predict_proba(X_te)[:,1]
        y_hat   = (y_proba >= best_thr).astype(int)

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

    y = labels.cpu().numpy()
    tox_mask = (y == 1)
    nt_mask  = (y == 0)

    def _safe_norm_rows(X, eps=1e-12):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.clip(n, eps, None)

    rng = np.random.default_rng(0)
    L = embeds.shape[1]
    rsa_scores = []

    for l in tqdm(range(L), total=L, desc='RSA of layers', leave=True):
        Xt = embeds[tox_mask, l, :]   # (n_t, F)
        Xn = embeds[nt_mask,  l, :]   # (n_n, F)
        n_t, n_n = Xt.shape[0], Xn.shape[0]

        m = min(n_t, n_n)

        boot_corrs = []
        for b in range(max(1, 3)):
            # reproducible but layer- and boot-variant sampling
            idx_t = rng.choice(n_t, size=m, replace=False)
            idx_n = rng.choice(n_n, size=m, replace=False)

            Xt_s = _safe_norm_rows(Xt[idx_t])
            Xn_s = _safe_norm_rows(Xn[idx_n])

            rdm_t = 1.0 - Xt_s @ Xt_s.T      # (m, m)
            rdm_n = 1.0 - Xn_s @ Xn_s.T      # (m, m)

            iu = np.triu_indices(m, k=1)
            vt, vn = rdm_t[iu], rdm_n[iu]

            # guard degeneracy
            if (np.isnan(vt).any() or np.isnan(vn).any() or
                np.std(vt) == 0.0 or np.std(vn) == 0.0):
                continue

            boot_corrs.append(np.corrcoef(vt, vn)[0, 1])

        rsa_scores.append(np.nanmean(boot_corrs) if boot_corrs else np.nan)

    return np.array(rsa_scores)   # 1 ≈ identical geometry, 0 ≈ unrelated