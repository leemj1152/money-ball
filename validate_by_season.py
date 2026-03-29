import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from data_fetch import build_game_level_pitching_features, fetch_schedule
from rolling_features_1 import build_training_set_rolling
from statcast_features import build_game_level_statcast_features


class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def parse_args():
    ap = argparse.ArgumentParser(description="Multi-season validation for rolling MLB features")
    ap.add_argument("--start-year", type=int, default=2022, help="Start season year")
    ap.add_argument("--end-year", type=int, default=2024, help="End season year")
    ap.add_argument("--lookback", type=int, default=10, help="Lookback window for rolling features")
    ap.add_argument("--train-end-month", type=int, default=4, help="Training cut-off month")
    ap.add_argument("--train-end-day", type=int, default=30, help="Training cut-off day")
    ap.add_argument("--epochs", type=int, default=20, help="Training epochs for the validation MLP")
    return ap.parse_args()


def _prepare_xy(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_cols = [c for c in x_train.columns if c != "gamePk"]
    train_num = x_train[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    val_num = x_val[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    keep_cols = train_num.columns[train_num.notna().sum() > 0].tolist()
    train_num = train_num[keep_cols]
    val_num = val_num[keep_cols]

    medians = train_num.median(numeric_only=True)
    train_num = train_num.fillna(medians)
    val_num = val_num.fillna(medians)

    zero_var = train_num.columns[train_num.nunique(dropna=False) <= 1].tolist()
    if zero_var:
        train_num = train_num.drop(columns=zero_var)
        val_num = val_num.drop(columns=zero_var, errors="ignore")

    return train_num.values.astype(np.float32), val_num.values.astype(np.float32), train_num.columns.tolist()


def _best_accuracy_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.35, 0.65, 31)
    best_acc = -1.0
    best_th = 0.5
    for th in thresholds:
        pred = (y_prob >= th).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc = acc
            best_th = float(th)
    return best_acc, best_th


def train_and_validate_by_season(
    start_year: int,
    end_year: int,
    lookback: int,
    train_end_month: int,
    train_end_day: int,
    epochs: int,
):
    np.random.seed(42)
    torch.manual_seed(42)
    seasons = []
    for year in range(start_year, end_year + 1):
        seasons.append(
            {
                "year": year,
                "start": f"{year}-03-20",
                "train_end": f"{year}-{train_end_month:02d}-{train_end_day:02d}",
                "val_end": f"{year}-09-30",
            }
        )

    all_results = []

    for season_cfg in seasons:
        print(f"\n{'=' * 60}")
        print(f"[Season {season_cfg['year']}]")
        print(f"{'=' * 60}")

        df_games = fetch_schedule(season_cfg["start"], season_cfg["val_end"])
        print(f"[Data] schedule {season_cfg['start']} ~ {season_cfg['val_end']}, rows={len(df_games)}")
        print("  Building starter/bullpen context from boxscores...")
        df_pitch_context = build_game_level_pitching_features(df_games)
        print(f"  Pitch context rows={len(df_pitch_context)}")
        print("  Building Statcast context...")
        df_statcast_context = build_game_level_statcast_features(df_games, lookback=lookback)
        print(f"  Statcast context rows={len(df_statcast_context)}")
        df_game_context = df_pitch_context.merge(df_statcast_context, on="gamePk", how="outer")
        print(f"  Combined context rows={len(df_game_context)}")

        x_train, y_train, _ = build_training_set_rolling(
            df_games,
            season_cfg["train_end"],
            lookback=lookback,
            df_game_context=df_game_context,
        )
        if x_train.empty:
            print("  No training data for this season.")
            continue
        print(f"  Train rows before split prep={len(x_train)}")

        x_all, y_all, merged_all = build_training_set_rolling(
            df_games,
            season_cfg["val_end"],
            lookback=lookback,
            df_game_context=df_game_context,
        )
        if x_all.empty:
            print("  No validation data for this season.")
            continue
        if "date" not in merged_all.columns:
            raise RuntimeError("merged_all must include date column for validation split")

        valid_mask = pd.to_datetime(merged_all["date"]) > pd.to_datetime(season_cfg["train_end"])
        x_val = x_all.loc[valid_mask].reset_index(drop=True)
        y_val = y_all.loc[valid_mask].reset_index(drop=True)
        print(f"  Validation rows before prep={len(x_val)}")

        if x_val.empty:
            print("  Validation set is empty after date split.")
            continue

        xmat_train, xmat_val, used_cols = _prepare_xy(x_train, x_val)
        if xmat_train.size == 0 or xmat_val.size == 0:
            print("  No usable features after preprocessing.")
            continue
        print(f"  Features used={len(used_cols)}")

        scaler = StandardScaler()
        xs_train = scaler.fit_transform(xmat_train)
        xs_val = scaler.transform(xmat_val)

        xt_train = torch.tensor(xs_train, dtype=torch.float32)
        yt_train = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)

        torch.manual_seed(42)
        model = MLP(in_dim=xt_train.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        bce = nn.BCELoss()

        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            pred = model(xt_train)
            loss = bce(pred, yt_train)
            loss.backward()
            opt.step()

        model.eval()
        xt_val = torch.tensor(xs_val, dtype=torch.float32)
        with torch.no_grad():
            pred_val = model(xt_val).numpy().ravel()

        y_val_np = y_val.values.astype(int)
        val_auc = roc_auc_score(y_val_np, pred_val)
        val_ll = log_loss(y_val_np, pred_val, labels=[0, 1])
        val_brier = brier_score_loss(y_val_np, pred_val)
        acc_05 = accuracy_score(y_val_np, (pred_val >= 0.5).astype(int))
        best_acc, best_th = _best_accuracy_threshold(y_val_np, pred_val)
        home_baseline = float((y_val_np == 1).mean())

        print(
            "  "
            f"AUC={val_auc:.4f}, LogLoss={val_ll:.4f}, Brier={val_brier:.4f}, "
            f"Acc@0.5={acc_05:.4f}, BestAcc={best_acc:.4f} @ th={best_th:.3f}, "
            f"HomeBaseline={home_baseline:.4f}"
        )
        all_results.append(
            {
                "year": season_cfg["year"],
                "train_rows": len(x_train),
                "val_rows": len(x_val),
                "features": len(used_cols),
                "auc": val_auc,
                "log_loss": val_ll,
                "brier": val_brier,
                "acc_05": acc_05,
                "best_acc": best_acc,
                "best_th": best_th,
                "home_baseline": home_baseline,
            }
        )

    if not all_results:
        print("No season results to summarize.")
        return

    avg_auc = np.mean([r["auc"] for r in all_results])
    avg_ll = np.mean([r["log_loss"] for r in all_results])
    avg_brier = np.mean([r["brier"] for r in all_results])
    avg_acc = np.mean([r["acc_05"] for r in all_results])
    avg_best_acc = np.mean([r["best_acc"] for r in all_results])

    print(f"\n{'=' * 60}")
    print("[Average across seasons]")
    print(f"{'=' * 60}")
    for res in all_results:
        print(
            f"{res['year']} -> "
            f"AUC={res['auc']:.4f}, Acc@0.5={res['acc_05']:.4f}, "
            f"BestAcc={res['best_acc']:.4f}, Baseline={res['home_baseline']:.4f}, "
            f"val_rows={res['val_rows']}, feats={res['features']}"
        )
    print(
        f"\nAvg AUC={avg_auc:.4f}, Avg LogLoss={avg_ll:.4f}, "
        f"Avg Brier={avg_brier:.4f}, Avg Acc@0.5={avg_acc:.4f}, Avg BestAcc={avg_best_acc:.4f}"
    )


if __name__ == "__main__":
    args = parse_args()
    train_and_validate_by_season(
        start_year=args.start_year,
        end_year=args.end_year,
        lookback=args.lookback,
        train_end_month=args.train_end_month,
        train_end_day=args.train_end_day,
        epochs=args.epochs,
    )
