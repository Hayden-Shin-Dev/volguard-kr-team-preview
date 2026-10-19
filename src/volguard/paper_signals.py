"""Paper-inspired causal session/residual signals; unavailable SUE/OFI never synthesized."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .direction_math_features import BASE
from .direction_math_study import KEYS, config, training_rows, safe_json
from .direction_integration import calibrate
from .new_information_sources import output
from .platform_study import local_settings
from .selective_direction import estimator, describe, side_evidence, THRESHOLDS
from .util import atomic_json, digest, utcnow

SESSION = ["overnight_log","intraday_log","overnight_mean5","intraday_mean5",
           "overnight_mean20","intraday_mean20","session_disagreement","overnight_to_intraday",
           "intraday_to_overnight"]
RESIDUAL = ["lagged_beta","abnormal_return","abnormal_mean5","abnormal_z"]
INTERACTIONS = ["session_residual_interaction","gap_market_interaction","day_market_interaction"]


def signal_features(context):
    parts=[]
    for _,raw in context.groupby("symbol",sort=True):
        g=raw.sort_values("date").copy()
        night=np.log1p(g.gap)
        day=np.log1p(g.body_pct/100)
        ret=np.log1p(g.return_1)
        valid=np.isfinite(night+day+ret) & np.isclose(night+day,ret,atol=1e-8,rtol=1e-6)
        g["decomposition_valid"]=valid
        g["overnight_log"]=night.where(valid)
        g["intraday_log"]=day.where(valid)
        for n in [5,20]:
            g[f"overnight_mean{n}"]=g.overnight_log.rolling(n,min_periods=n).mean()
            g[f"intraday_mean{n}"]=g.intraday_log.rolling(n,min_periods=n).mean()
        g["session_disagreement"]=(g.overnight_log*g.intraday_log<0).astype(float).where(valid)
        g["overnight_to_intraday"]=g.overnight_log.shift(1).rolling(60,min_periods=40).corr(g.intraday_log)
        g["intraday_to_overnight"]=g.intraday_log.shift(1).rolling(60,min_periods=40).corr(g.overnight_log)
        # Market is explicitly a fixed-universe equal-weight return proxy, not a sector index.
        market=g.market_return
        beta=(g.return_1.rolling(60,min_periods=40).cov(market)/market.rolling(60,min_periods=40).var().replace(0,np.nan)).shift(1)
        g["lagged_beta"]=beta
        residual=g.return_1-beta*market
        g["abnormal_return"]=residual
        g["abnormal_mean5"]=residual.rolling(5,min_periods=5).mean()
        g["abnormal_z"]=residual/residual.shift(1).rolling(60,min_periods=40).std().replace(0,np.nan)
        g["session_residual_interaction"]=g.session_disagreement*g.abnormal_z
        g["gap_market_interaction"]=g.overnight_log*g.market_momentum
        g["day_market_interaction"]=g.intraday_log*g.market_momentum
        consecutive=g.target_date.eq(g.date.shift(-1))
        g["next_night"]=g.overnight_log.shift(-1).where(consecutive)
        g["next_day"]=g.intraday_log.shift(-1).where(consecutive)
        # Auxiliary joint label. The final target remains next adjusted close direction.
        good=g.next_night.notna() & g.next_day.notna() & np.isclose(np.expm1(g.next_night+g.next_day),g.direction_return,atol=1e-8,rtol=1e-6)
        g["joint_label"]=(4*g.next_night.gt(0).astype(int)+2*g.next_day.gt(0).astype(int)+g.label).where(good)
        parts.append(g)
    return pd.concat(parts,ignore_index=True).replace([np.inf,-np.inf],np.nan)


def validate_earnings_inputs(frame):
    required={"symbol","fiscal_period","metric","currency","basis","actual","consensus",
              "announcement_at","consensus_available_at","actual_available_at","source_id"}
    if not required.issubset(frame):
        raise ValueError("Actual historical consensus, matching basis and source timestamps required")
    if frame[list(required)].isna().any().any():
        raise ValueError("Missing earnings evidence")
    announcement=pd.to_datetime(frame.announcement_at,utc=True)
    if not (pd.to_datetime(frame.consensus_available_at,utc=True)<announcement).all():
        raise ValueError("Consensus must predate announcement")
    if not (pd.to_datetime(frame.actual_available_at,utc=True)>=announcement).all():
        raise ValueError("Actual publication timestamp invalid")
    return frame.copy()


def run():
    root=local_settings().data_dir
    out=root/"runs/paper-signals-20261017"
    out.mkdir(exist_ok=True)
    context_path=root/"runs/direction-math-20261017/context.parquet"
    matched_path=output()/"global-features.parquet"
    groups={"paper_baseline":BASE,"session_split":BASE+SESSION,"market_residual":BASE+RESIDUAL,
            "conditional_signals":BASE+SESSION+RESIDUAL+INTERACTIONS,
            "joint_session":BASE+SESSION+RESIDUAL+INTERACTIONS}
    manifest=dict(created_at=utcnow(),context_hash=digest(context_path.read_bytes()),
                  match_hash=digest(matched_path.read_bytes()),code_hash=digest(Path(__file__).read_bytes()),
                  groups=groups,selection="2021-22: Brier/logloss no worse than baseline; supported >=60% side coverage first, then logloss",
                  protocol="same quarterly LightGBM; prior year OOF sigmoid; 2023 reused confirmation; fixed 55/60/65/70 thresholds",
                  availability="historical official daily bars assumed known at 18:00 KST; actual release/revision timestamps unverified; research only")
    path=out/"protocol.json"
    if path.exists():
        old=json.loads(path.read_text("utf8"))
        if {k:v for k,v in old.items() if k!="created_at"}!={k:v for k,v in manifest.items() if k!="created_at"}: raise ValueError("Changed paper protocol")
    else: atomic_json(path,manifest)
    cached=out/"features.parquet"
    if cached.exists():
        data=pd.read_parquet(cached)
        audit=json.loads((out/"data-audit.json").read_text("utf8"))
        assert digest(cached.read_bytes())==audit["feature_hash"]
    else:
        context=pd.read_parquet(context_path)
        enhanced=signal_features(context)
        matched=pd.read_parquet(matched_path,columns=["symbol","date","target_date"])
        f=enhanced.merge(matched,on=["symbol","date","target_date"],validate="one_to_one")
        # Same evaluation rows for all candidates, including the auxiliary joint-label model.
        valid=f.joint_label.notna() & f[SESSION+RESIDUAL+INTERACTIONS].notna().all(axis=1)
        data=f[valid].copy()
        data.to_parquet(cached,index=False)
        audit=dict(rows_before=len(f),rows_after=len(data),excluded=len(f)-len(data),
                   confirmation_before=int(f.target_date.dt.year.eq(2023).sum()),confirmation_after=int(data.target_date.dt.year.eq(2023).sum()),
                   identity_valid=int(enhanced.decomposition_valid.sum()),context_rows=len(context),
                   feature_hash=digest(cached.read_bytes()),all_pit_flags_false=bool(~context.pit_verified.any()),
                   available_signals=["session_split","market_residual","conditional_interactions"],
                   unavailable={"consensus_surprise":"No announcement-preceding consensus snapshots; accounting growth is not SUE",
                                "closing_order_imbalance":"No historical KRX closing trade-side/order-book stream for 2021-23",
                                "sector_residual":"No verified historical sector membership and sector close series"})
        atomic_json(out/"data-audit.json",audit)
    predictions={}
    for name,cols in groups.items():
        parts=[]
        for period,test in data[data.target_date.dt.year.between(2020,2023)].groupby(data.target_date.dt.to_period("Q"),sort=True):
            dest=out/f"raw-{name}-{period}.parquet"
            if dest.exists():
                meta=json.loads(dest.with_suffix(".json").read_text("utf8"))
                assert digest(dest.read_bytes())==meta["hash"]
                parts.append(pd.read_parquet(dest)); continue
            train=training_rows(data,test,config(cols))
            weights=np.exp2(-(test.date.min()-train.target_date).dt.days.to_numpy()/730)
            model=estimator("lightgbm")
            target=train.joint_label.astype(int) if name=="joint_session" else train.label
            with threadpool_limits(limits=3):
                model.fit(train[cols],target,**{list(model.named_steps)[-1]+"__sample_weight":weights})
                probs=model.predict_proba(test[cols])
            if name=="joint_session":
                classes=model.classes_.astype(int)
                p=probs[:,classes%2==1].sum(axis=1)
            else: p=probs[:,1]
            f=test[KEYS].copy()
            f["raw_p"],f["prior"],f["period"],f["fit_end"]=np.clip(p,1e-6,1-1e-6),float(train.label.mean()),str(period),train.target_date.max()
            f.to_parquet(dest,index=False)
            atomic_json(dest.with_suffix(".json"),dict(hash=digest(dest.read_bytes()),fit_end=str(train.target_date.max()),train_n=len(train)))
            parts.append(f)
            if str(period)=="2023Q4": joblib.dump(dict(model=model,cols=cols,research_only=True),out/f"{name}.joblib")
            print(name,str(period),len(train),len(test),flush=True)
        raw=pd.concat(parts,ignore_index=True)
        for cal in ["raw","sigmoid"]:
            predictions[name+"__"+cal]=calibrate(raw,cal).sort_values(["target_date","symbol"]).reset_index(drop=True)
    baseline=predictions["paper_baseline__raw"]
    for f in predictions.values():
        pd.testing.assert_frame_equal(f[["symbol","target_date","label"]],baseline[["symbol","target_date","label"]])
        f["baseline_p"]=baseline.p_up.to_numpy()
    dev={n:describe(f[f.target_date.dt.year<2023]) for n,f in predictions.items()}
    ref=dev["paper_baseline__raw"]
    valid=[n for n,r in dev.items() if r["brier"]<=ref["brier"] and r["log_loss"]<=ref["log_loss"]]
    def coverage(n):
        return sum(max([r["coverage"] for r in dev[n]["signals"] if r["side"]==s and r["threshold"]>=.6 and r["supported"]] or [0]) for s in ["up","down"])
    chosen=sorted(valid,key=lambda n:(-coverage(n),dev[n]["log_loss"]))[0]
    thresholds={s:next((t for t in THRESHOLDS if t>=.6 and next(r for r in dev[chosen]["signals"] if r["side"]==s and r["threshold"]==t)["supported"]),None) for s in ["up","down"]}
    selection=dict(selected=chosen,thresholds=thresholds,development=dev)
    sp=out/"selection.json"
    if sp.exists():
        assert json.loads(sp.read_text("utf8"))["selection"]==safe_json(selection)
    else: atomic_json(sp,dict(at=utcnow(),selection=safe_json(selection)))
    results={n:describe(f[f.target_date.dt.year==2023]) for n,f in predictions.items()}
    final=predictions[chosen]; final=final[final.target_date.dt.year==2023]
    selected_signals={s:side_evidence(final,s,t) if t else dict(side=s,supported=False,status="no_development_signal") for s,t in thresholds.items()}
    report=dict(status="complete",at=utcnow(),selected=chosen,thresholds=thresholds,results=results,development=dev,
                selected_signals=selected_signals,evidence_id=out.name,protocol=manifest,audit=audit,operational_enabled=False,
                limitations=["기존 OHLC를 재구성한 신호이며 새 외부 정보를 확보했다고 주장하지 않습니다.",
                             "장중·야간 합과 일간 수익률의 항등식이 맞는 거래일만 사용했고 모든 후보의 평가 표본을 일치시켰습니다.",
                             "시장 잔차는 고정 유니버스 동일가중 시장 대용치 기준이며 실제 업종 잔차가 아닙니다.",
                             "SUE와 종가 주문 불균형은 필요한 과거 원자료가 없어 입력하지 않았습니다.",
                             "과거 자료의 18시 이용 가능 시각은 연구 가정입니다. 당시 공표·개정 이력이 입증된 것은 아닙니다.",
                             "상승·하락 합계 100%는 보합 제외 조건부 확률입니다. 2023년은 재사용 개발 확인입니다.",
                             "8개 결합 상태의 확률을 학습한 모델은 종가 상승 상태들의 확률을 합산합니다. 두 구간의 확률을 독립으로 곱하지 않습니다."])
    pd.concat([f.assign(candidate=n) for n,f in predictions.items()],ignore_index=True).to_parquet(out/"predictions.parquet",index=False)
    atomic_json(out/"report.json",safe_json(report)); atomic_json(root/"reports/paper-signals.json",safe_json(report))
    print(json.dumps(dict(selected=chosen,thresholds=thresholds,metrics=results[chosen],audit=audit)),flush=True)


if __name__=="__main__": run()
