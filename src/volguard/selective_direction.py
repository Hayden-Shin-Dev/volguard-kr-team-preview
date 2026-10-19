"""Selective two-sided direction research: actual model refits, OOF calibration, frozen gates."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from .direction_math_features import BASE
from .direction_math_study import KEYS, config, factory, training_rows, safe_json
from .direction_integration import calibrate
from .new_information_sources import output
from .platform_study import local_settings
from .probability_evidence import metrics
from .util import atomic_json, digest, utcnow

MODELS = ["logistic", "lightgbm", "catboost", "regime_mixture", "ensemble"]
THRESHOLDS = [.55, .60, .65, .70]


def estimator(name):
    if name == "logistic":
        return factory(config(BASE))
    if name == "catboost":
        model = CatBoostClassifier(iterations=240, depth=5, learning_rate=.035, l2_leaf_reg=15,
                                   thread_count=3, random_seed=1710, verbose=False, allow_writing_files=False)
    else:
        model = LGBMClassifier(n_estimators=240, num_leaves=15, learning_rate=.035,
                               min_child_samples=400, reg_lambda=15, n_jobs=3,
                               random_state=1710, verbosity=-1)
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), model)


def fit_predict(train, test, name):
    weights = np.exp2(-(test.date.min()-train.target_date).dt.days.to_numpy()/730)
    model = estimator(name)
    model.fit(train[BASE], train.label, **{list(model.named_steps)[-1]+"__sample_weight": weights})
    pooled = model.predict_proba(test[BASE])[:, 1]
    experts = {}
    if name == "regime_mixture":
        # Gate uses only observed trailing market momentum, never future labels.
        for state in [False, True]:
            tr_mask = train.market_momentum.ge(0).eq(state)
            te_mask = test.market_momentum.ge(0).eq(state)
            if tr_mask.sum() < 5000 or not te_mask.any():
                continue
            expert = estimator("lightgbm")
            expert.fit(train.loc[tr_mask, BASE], train.loc[tr_mask, "label"],
                       **{list(expert.named_steps)[-1]+"__sample_weight": weights[tr_mask]})
            local = expert.predict_proba(test.loc[te_mask, BASE])[:, 1]
            pooled[te_mask] = .5*pooled[te_mask]+.5*local
            experts[state] = expert
    return np.clip(pooled, 1e-6, 1-1e-6), dict(pooled=model, experts=experts, cols=BASE, research_only=True)


def side_evidence(frame, side, threshold, draws=1000):
    confidence = frame.p_up if side == "up" else 1-frame.p_up
    mask = (confidence >= threshold) & (confidence > .5)
    g = frame[mask].copy()
    result = dict(side=side, threshold=threshold, n=len(g), dates=int(g.target_date.nunique()),
                  coverage=len(g)/len(frame), supported=False)
    if g.empty:
        return result
    g["confidence"] = confidence[mask]
    g["hit"] = g.label if side == "up" else 1-g.label
    g["prior_hit"] = (g.prior > .5).eq(g.label).astype(float)
    g["baseline_hit"] = (g.baseline_p > .5).eq(g.label).astype(float)
    g["excess"] = g.hit-g.baseline_hit
    g["gap"] = g.hit-g.confidence
    g["one"] = 1
    dates = sorted(frame.target_date.unique())
    daily = g.groupby("target_date")[["one", "hit", "gap", "excess"]].sum().reindex(dates, fill_value=0).to_numpy()
    block = min(20, len(dates))
    rng = np.random.default_rng(1710)
    starts = rng.integers(0,len(dates)-block+1,(draws,int(np.ceil(len(dates)/block))))
    idx = (starts[:,:,None]+np.arange(block)).reshape(draws,-1)[:,:len(dates)]
    sums = daily[idx].sum(axis=1)
    valid = sums[:,0] > 0
    samples = sums[valid,1:]/sums[valid,0,None]
    intervals = np.quantile(samples,[.025,.975],axis=0).T.tolist()
    support = len(g) >= 300 and g.target_date.nunique() >= 60 and valid.mean() >= .95
    result.update(average_confidence=float(g.confidence.mean()), accuracy=float(g.hit.mean()),
                  accuracy_ci95=intervals[0], gap=float(g.gap.mean()), gap_ci95=intervals[1],
                  baseline_accuracy=float(g.baseline_hit.mean()), prior_accuracy=float(g.prior_hit.mean()),
                  excess_ci95=intervals[2],
                  supported=bool(support and intervals[0][0]>.5 and intervals[1][0]>=-.05 and intervals[1][1]<=.05),
                  examples=[dict(symbol=str(r.symbol), date=str(r.date.date()),target_date=str(r.target_date.date()),
                                 p_up=float(r.p_up),p_down=float(1-r.p_up),
                                 actual="상승" if r.label else "하락",return_pct=float(r.direction_return*100),
                                 range_low=float(r.range_forecast_low),range_high=float(r.range_forecast_high),
                                 actual_range=float(r.y))
                            for r in g.sort_values(["target_date","symbol"]).tail(6).itertuples()])
    return result


def describe(f):
    return dict(**metrics(f, f.p_up), up=side_evidence(f,"up",.5), down=side_evidence(f,"down",.5),
                signals=[side_evidence(f,s,t) for s in ["up","down"] for t in THRESHOLDS],
                spread=dict(min=float(f.p_up.min()),max=float(f.p_up.max()),
                            q05=float(f.p_up.quantile(.05)),q95=float(f.p_up.quantile(.95)),
                            confident60=float((np.maximum(f.p_up,1-f.p_up)>=.6).mean())))


def run():
    root = local_settings().data_dir
    folder = root/"runs/selective-direction-20261017"
    folder.mkdir(exist_ok=True)
    source = output()/"global-features.parquet"
    protocol = dict(source_hash=digest(source.read_bytes()), code_hash=digest(Path(__file__).read_bytes()),
                    features=BASE, models=MODELS, calibrations=["raw","sigmoid"], thresholds=THRESHOLDS,
                    folds="quarterly 2020-2023; 5y rolling; half-life 730d; previous 1y OOF calibration",
                    selection="2021-22 only: require whole Brier/logloss no worse than logistic raw; maximize supported >=60% side coverage, ties logloss; otherwise min logloss",
                    signal_gate=">=300 rows, >=60 dates, 20-day joint date blocks, 95% actual hit lower >50%, entire gap CI within +/-5pp",
                    evaluation="2023 reused development; no independent test; never widen probabilities", research_only=True)
    path = folder/"protocol.json"
    if path.exists() and json.loads(path.read_text("utf8"))["protocol"] != protocol:
        raise ValueError("Protocol changed")
    if not path.exists():
        atomic_json(path,dict(at=utcnow(),protocol=protocol))
    data = pd.read_parquet(source)
    if data.target_date.max() >= pd.Timestamp("2024-01-01"):
        raise ValueError("Locked periods forbidden")
    raw = {}
    for name in MODELS[:-1]:
        parts=[]
        for period,test in data[data.target_date.dt.year.between(2020,2023)].groupby(data.target_date.dt.to_period("Q"),sort=True):
            dest=folder/f"raw-{name}-{period}.parquet"
            if dest.exists():
                meta=json.loads(dest.with_suffix(".json").read_text("utf8"))
                if digest(dest.read_bytes())!=meta["hash"]:
                    raise ValueError("Prediction checkpoint changed")
                parts.append(pd.read_parquet(dest)); continue
            train=training_rows(data,test,config(BASE))
            with threadpool_limits(limits=3):
                p,model=fit_predict(train,test,name)
            f=test[KEYS].copy()
            f["raw_p"],f["prior"],f["period"],f["fit_end"]=p,float(train.label.mean()),str(period),train.target_date.max()
            f.to_parquet(dest,index=False)
            atomic_json(dest.with_suffix(".json"),dict(hash=digest(dest.read_bytes()),train_n=len(train),fit_end=str(train.target_date.max())))
            parts.append(f)
            if str(period)=="2023Q4": joblib.dump(model,folder/f"{name}.joblib")
            print(name,str(period),len(train),len(test),flush=True)
        raw[name]=pd.concat(parts,ignore_index=True).sort_values(["target_date","symbol"]).reset_index(drop=True)
    ensemble=raw["lightgbm"].copy()
    for f in raw.values():
        pd.testing.assert_frame_equal(f[["symbol","target_date","label"]],ensemble[["symbol","target_date","label"]])
    ensemble["raw_p"]=np.mean([raw[n].raw_p.to_numpy() for n in ["lightgbm","catboost","regime_mixture"]],axis=0)
    raw["ensemble"]=ensemble
    predictions={}
    for name,f in raw.items():
        for method in ["raw","sigmoid"]:
            predictions[name+"__"+method]=calibrate(f,method).sort_values(["target_date","symbol"]).reset_index(drop=True)
    base=predictions["logistic__raw"]
    for f in predictions.values():
        pd.testing.assert_frame_equal(f[["symbol","target_date","label"]],base[["symbol","target_date","label"]])
        f["baseline_p"]=base.p_up.to_numpy()
    development={n:describe(f[f.target_date.dt.year<2023]) for n,f in predictions.items()}
    reference=development["logistic__raw"]
    valid=[n for n,r in development.items() if r["brier"]<=reference["brier"] and r["log_loss"]<=reference["log_loss"]]
    def signal_coverage(n):
        # Each side chooses its lowest supported threshold >=60%, on development only.
        r=development[n]
        return sum(max([s["coverage"] for s in r["signals"] if s["side"]==side and s["threshold"]>=.6 and s["supported"]] or [0]) for side in ["up","down"])
    selected=sorted(valid,key=lambda n:(-signal_coverage(n),development[n]["log_loss"]))[0]
    sides={s:next((t for t in THRESHOLDS if t>=.6 and next(r for r in development[selected]["signals"] if r["side"]==s and r["threshold"]==t)["supported"]),None) for s in ["up","down"]}
    selection=dict(selected=selected,thresholds=sides,development=development)
    sp=folder/"selection.json"
    if sp.exists():
        old=json.loads(sp.read_text("utf8"))
        if old["selection"]!=safe_json(selection): raise ValueError("Frozen selection changed")
    else: atomic_json(sp,dict(at=utcnow(),selection=safe_json(selection)))
    results={n:describe(f[f.target_date.dt.year==2023]) for n,f in predictions.items()}
    final=predictions[selected]
    final=final[final.target_date.dt.year==2023]
    chosen_signals={s:side_evidence(final,s,t) if t is not None else dict(side=s,status="no_development_signal",supported=False) for s,t in sides.items()}
    report=dict(status="complete",at=utcnow(),selected=selected,thresholds=sides,results=results,
                development=development,selected_signals=chosen_signals,protocol=protocol,
                evidence_id=folder.name,operational_enabled=False,
                limitations=["상승과 하락은 하나의 일관된 분포로 합계 100%이며 보합 제외 조건부입니다.",
                             "60% 이상 강한 신호는 개발 기간에서 방향별로 선택합니다. 확인 기간 결과로 임계값을 바꾸지 않습니다.",
                             "2023년 재사용 개발 평가이며 신규 독립 시험이 아닙니다. 높은 확률 자체를 성능으로 간주하지 않습니다.",
                             "같은 날짜의 종목을 묶은 20거래일 블록 신뢰구간입니다. 여러 후보·임계값의 점별 구간은 동시 보장이 아닙니다.",
                             "기존 변동폭 ML과 운영 방향 확률 제한을 유지합니다. 사례의 변동폭 구간은 방향 비조건부입니다."])
    pd.concat([f.assign(candidate=n) for n,f in predictions.items()],ignore_index=True).to_parquet(folder/"predictions.parquet",index=False)
    atomic_json(folder/"report.json",safe_json(report))
    atomic_json(root/"reports/selective-direction.json",safe_json(report))
    print(json.dumps(dict(selected=selected,thresholds=sides,metrics=results[selected],selected_signals=chosen_signals)),flush=True)


if __name__=="__main__":
    run()
