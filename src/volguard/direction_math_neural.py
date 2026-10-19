"""Small-budget, clean-room numeric LSTM and CRAFT-inspired Korean panel experiments.

Not an exact reproduction: 24 centroids, width 32, one temporal layer, four
epochs, annual refits on five years. No text modality is invented.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits
from torch import nn

from .direction_math_study import KEYS, RANGE
from .platform_study import local_settings
from .util import atomic_json, digest, utcnow

torch.set_num_threads(2)
COLS = [
    "return_1",
    "volume_ratio",
    "range_1",
    "gap",
    "market_return",
    "market_breadth",
    "neighbor_return",
    "residual1",
]


class NumericLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(len(COLS), 32, batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, x, mask=None):
        b, length, n, f = x.shape
        h, _ = self.rnn(x.permute(0, 2, 1, 3).reshape(b * n, length, f))
        return self.head(h[:, -1]).reshape(b, n), None


class CraftAdapted(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(20, 32)
        self.position = nn.Parameter(torch.zeros(1, 10, 32))
        layer = nn.TransformerEncoderLayer(32, 2, dim_feedforward=64, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.stock = nn.MultiheadAttention(32, 2, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(32)
        self.head = nn.Linear(32, 1)
        self.tokens = nn.Linear(32, 24)

    def forward(self, x, mask=None):
        b, length, n, f = x.shape
        h = self.proj(x.permute(0, 2, 1, 3).reshape(b * n, length, f)) + self.position[:, :length]
        h = self.encoder(
            h, mask=torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
        )
        tokens = self.tokens(h).reshape(b, n, length, 24).permute(0, 2, 1, 3)
        last = h[:, -1].reshape(b, n, 32)
        attended, _ = self.stock(last, last, last, key_padding_mask=mask, need_weights=False)
        return self.head(self.norm(last + attended)).squeeze(-1), tokens


def contexts(out):
    frame = pd.read_parquet(out / "context.parquet")
    frame = frame[frame.date.ge("2014-01-01")].sort_values(["date", "symbol"])
    dates = np.sort(frame.date.unique())
    symbols = sorted(frame.symbol.unique())
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    dense = frame.set_index(["date", "symbol"]).reindex(idx)
    numeric = dense[COLS].to_numpy().reshape(len(dates), len(symbols), len(COLS)).astype("float32")
    lag = (
        dense[[f"lag_{i}" for i in range(9, -1, -1)]]
        .to_numpy()
        .reshape(len(dates), len(symbols), 10)
        .astype("float32")
    )
    r = numeric[:, :, 0]
    weights = {}
    previous = None
    # Past-only rolling SVD relationship embedding; no bfill before listing.
    with threadpool_limits(limits=2):
        for t, d in enumerate(pd.to_datetime(dates)):
            period = str(d.to_period("M"))
            if period == previous or t < 60:
                continue
            history = r[t - 60 : t]
            valid = np.isfinite(history).all(axis=0) & (np.nanstd(history, axis=0) > 1e-8)
            ids = np.flatnonzero(valid)
            w = np.zeros((len(symbols), len(symbols)), dtype="float32")
            if len(ids) > 2:
                corr = np.corrcoef(history[:, ids].T)
                _, vectors = np.linalg.eigh(corr)
                e = vectors[:, -min(10, len(ids)) :]
                distance = ((e[:, None, :] - e[None, :, :]) ** 2).sum(axis=-1)
                small = np.exp(-distance)
                np.fill_diagonal(small, 0)
                small /= small.sum(axis=1, keepdims=True)
                w[np.ix_(ids, ids)] = small
            weights[period] = w
            previous = period
    return frame, pd.to_datetime(dates), symbols, numeric, lag, weights


def run(out):
    frame, dates, symbols, numeric, lag, weights = contexts(out)
    data = pd.read_parquet(out / "features.parquet")
    data = data[data[RANGE].notna().all(axis=1)]
    date_map = {d: i for i, d in enumerate(dates)}
    symbol_map = {s: i for i, s in enumerate(symbols)}
    data["ti"] = data.date.map(date_map)
    data["si"] = data.symbol.map(symbol_map)
    y = np.zeros(numeric.shape[:2], dtype="float32")
    eligible = np.zeros(y.shape, dtype=bool)
    y[data.ti, data.si] = data.label
    eligible[data.ti, data.si] = True
    all_results = []
    for year in [2020, 2021, 2022, 2023]:
        test = data[data.target_date.dt.year.eq(year)].copy()
        cutoff = test.date.min()
        issue = (cutoff + pd.Timedelta(hours=18, minutes=30)).tz_localize("Asia/Seoul")
        train = data[
            (data.target_date <= cutoff)
            & (data.target_date > cutoff - pd.DateOffset(years=5))
            & (pd.to_datetime(data.target_available_at, utc=True) <= issue)
        ]
        train_mask = np.zeros(y.shape, dtype=bool)
        train_mask[train.ti, train.si] = True
        fit_days = np.unique(train.ti)
        fit_days = fit_days[fit_days >= 9]
        test_days = np.unique(test.ti)
        end = max(test_days) + 1
        for kind in ["lstm_numeric", "craft_adapted"]:
            dest = out / f"neural-{kind}-{year}.parquet"
            if dest.exists():
                meta = json.loads((out / f"neural-{kind}-{year}.json").read_text())
                if meta["hash"] != digest(dest.read_bytes()):
                    raise ValueError("Neural checkpoint hash")
                all_results.append(pd.read_parquet(dest))
                continue
            torch.manual_seed(1707)
            rng = np.random.default_rng(1707)
            if kind == "lstm_numeric":
                fitx = numeric[fit_days].reshape(-1, len(COLS))
                mu = np.nanmean(fitx, axis=0)
                sd = np.nanstd(fitx, axis=0)
                x = np.nan_to_num(np.clip((numeric[:end] - mu) / np.maximum(sd, 1e-6), -10, 10)).astype(
                    "float32"
                )
                model = NumericLSTM()
                token_ids = None
                preprocessing = dict(mean=mu.tolist(), std=sd.tolist())
            else:
                train_lag = lag[train.ti, train.si]
                good = np.isfinite(train_lag).all(axis=1)
                with threadpool_limits(limits=2):
                    km = MiniBatchKMeans(n_clusters=24, random_state=1707, n_init=3, batch_size=2048).fit(
                        train_lag[good] * 100
                    )
                    token_ids = km.predict(np.nan_to_num(lag[:end]).reshape(-1, 10) * 100).reshape(
                        end, len(symbols)
                    )
                local = km.cluster_centers_[token_ids].astype("float32")
                global_ctx = np.zeros_like(local)
                for t in range(end):
                    w = weights.get(str(dates[t].to_period("M")))
                    if w is not None:
                        global_ctx[t] = w @ local[t]
                x = np.concatenate([local, global_ctx], axis=-1)
                model = CraftAdapted()
                preprocessing = dict(centroids=km.cluster_centers_.tolist())
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            xt = torch.from_numpy(x)
            yt = torch.from_numpy(y)
            mt = torch.from_numpy(train_mask)
            # Only known current inputs mask attention, not next-day label eligibility.
            missing = torch.from_numpy(~np.isfinite(lag[:end]).all(axis=-1))
            losses = []
            for epoch in range(4):
                model.train()
                order = rng.permutation(fit_days)
                tot = 0.0
                count = 0
                for start in range(0, len(order), 8):
                    ds = order[start : start + 8]
                    seq = ds[:, None] + np.arange(-9, 1)[None, :]
                    mask = mt[ds]
                    valid_attention = missing[ds].clone()
                    valid_attention[valid_attention.all(dim=1), 0] = False
                    logits, tokens = model(xt[seq], valid_attention)
                    w = np.exp2(-(cutoff - dates[ds]).days.to_numpy() / 730)
                    weight = torch.tensor(w, dtype=torch.float32)[:, None].expand_as(logits)
                    loss = (
                        nn.functional.binary_cross_entropy_with_logits(logits, yt[ds], reduction="none")[mask]
                        * weight[mask]
                    ).mean()
                    if tokens is not None:
                        target = torch.from_numpy(token_ids[seq[:, 1:]]).long()
                        tokenmask = ~missing[seq[:, 1:]]
                        auxiliary = nn.functional.cross_entropy(
                            tokens[:, :-1].reshape(-1, 24), target.reshape(-1), reduction="none"
                        ).reshape(target.shape)
                        if tokenmask.any():
                            loss = loss + 0.1 * auxiliary[tokenmask].mean()
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite neural loss")
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    tot += float(loss.detach())
                    count += 1
                losses.append(tot / count)
                print(kind, year, "epoch", epoch + 1, "loss", round(losses[-1], 5), flush=True)
            model.eval()
            probs = {}
            with torch.no_grad():
                for start in range(0, len(test_days), 8):
                    ds = test_days[start : start + 8]
                    seq = ds[:, None] + np.arange(-9, 1)[None, :]
                    mask = missing[ds].clone()
                    mask[mask.all(dim=1), 0] = False
                    logits, _ = model(xt[seq], mask)
                    for d, p in zip(ds, torch.sigmoid(logits).numpy()):
                        probs[d] = p
            result = test[KEYS + ["markov_high", "market_momentum"]].copy()
            result["raw_p"] = [float(probs[t][s]) for t, s in zip(test.ti, test.si)]
            result["prior"] = float(train.label.mean())
            result["period"] = str(year)
            result["fit_end"] = train.target_date.max()
            result["model"] = kind
            result.to_parquet(dest, index=False)
            torch.save(
                dict(model=model.state_dict(), preprocessing=preprocessing, research_only=True),
                out / f"{kind}-{year}.pt",
            )
            atomic_json(
                out / f"neural-{kind}-{year}.json",
                dict(
                    created_at=utcnow(),
                    hash=digest(dest.read_bytes()),
                    epochs=4,
                    train_n=len(train),
                    train_end=str(train.target_date.max()),
                    loss=losses,
                    scope="reduced-budget Korean adaptation, not original-paper reproduction",
                ),
            )
            all_results.append(result)
    pd.concat(all_results, ignore_index=True).to_parquet(out / "neural-predictions.parquet", index=False)
    print("NEURAL_DONE", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    run(args.out or local_settings().data_dir / "runs/direction-math-20261017")
