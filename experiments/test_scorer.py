from pathlib import Path
import csv, math
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from utils.scoring import ToxDL2Scorer   
from datasets import load_dataset
from tqdm import tqdm

def get_dataset(path):
    data = load_dataset(path)['train']

    data = data.filter(lambda example: example['Length']>50) # get all samples higher than the model capability

    tox = data.filter(lambda example: example['Toxin']).to_pandas().sample(n=50, random_state=42)
    non_tox = data.filter(lambda example: not example['Toxin']).to_pandas().sample(n=50, random_state=42)

    print('Loaded data successfully')

    return pd.concat([tox, non_tox])

def eval_sliding(dataset, out_path: Path, plddt_gate: float = 70.0):
    scorer = ToxDL2Scorer()
    preds, gts, out_rows = [], [], []
    for _, ex in tqdm(dataset.iterrows(), total= len(dataset), desc='Testing Scorer'):
        res = scorer.score(ex["Sequence"])  # dict: tox_prob, non_tox_prob, best_window, mean_plddt
        tox_p = res["tox_prob"]
        non_tox_p = res["non_tox_prob"]
        plddt = res["mean_plddt"]
        # if plddt < plddt_gate: continue
        preds.append(tox_p)
        gts.append(ex["Toxin"]) # label
        out_rows.append({
            "id": ex["Unnamed: 0"],
            "len": ex["Length"],
            "species": ex['species'],
            "tox_prob": tox_p,
            "non_tox_prob": non_tox_p,
            "pred_label": int(tox_p >= 0.5),
            "true_label": ex["Toxin"],
            "best_window_start": res["best_window"][0],
            "best_window_end": res["best_window"][1],
            "mean_plddt": plddt
        })

    # métricas con hard labels (umbral 0.5) + AUC si posible
    y_true = np.array(gts, dtype=int)
    y_pred = (np.array(preds) >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    f1  = float(f1_score(y_true, y_pred))
    prec= float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, np.array(preds)))
    

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out_rows).assign(
        acc=acc,
        f1=f1,
        precision=prec,
        recall=rec,
        roc_auc=auc,
    )

    df.to_csv(out_path, index=False)

    print(f"N={len(y_true)}  ACC={acc:.3f}  F1={f1:.3f}  PREC={prec:.3f}  REC={rec:.3f}  AUC={auc:.3f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="Data dir (HF)")
    ap.add_argument("--out", required=True, help="Ruta CSV de salida")
    ap.add_argument("--plddt_gate", type=float, default=70.0)
    args = ap.parse_args()

    dataset = get_dataset(args.data_path)
    if not dataset:
        raise SystemExit("No hay secuencias >50 aa en el CSV.")
    eval_sliding(dataset, Path(args.out), plddt_gate=args.plddt_gate)