import { useEffect, useState } from "react";
import { api, number, type Data } from "./api";

const pct=(x:number|undefined)=>x==null?"—":`${number(x*100,2)}%`;
const groups:Record<string,string>={baseline:"기존 피처",rsi:"RSI·진입·이탈",macd:"MACD·교차",moving_average:"이동평균 교차",bands:"볼린저밴드",stochastic:"스토캐스틱",trend_atr:"ADX·DMI·ATR",volume:"OBV·거래량 돌파",candles:"캔들 패턴",interactions:"지표·국면 결합",all_standard:"전체 표준 지표",all_strict:"전체 엄격 기준",base:"기존 피처",technical:"기술적 지표 포함",old2015_weighted5:"2015 제한·최근 5년 가중",expanding:"전체 누적",rolling3:"최근 3년",rolling5:"최근 5년",rolling10:"최근 10년",weighted_all:"전체 이력·최근 가중",regime_all:"전체 이력·국면별",raw:"보정 전",sigmoid:"시간순 확률 보정",logistic_baseline:"로지스틱 기준",lightgbm_base:"LightGBM 기존 피처",lightgbm_technical:"LightGBM 기술적 지표",catboost_base:"CatBoost 기존 피처",catboost_technical:"CatBoost 기술적 지표"};
const label=(key:string)=>key.split("__").map(x=>groups[x]||x).join(" · ");
const events:Record<string,string>={rsi_exit_low:"RSI 과매도 이탈",macd_cross_up:"MACD 상향 교차",macd_cross_down:"MACD 하향 교차",bb_lower_reentry:"볼린저 하단 재진입",bb_upper_reentry:"볼린저 상단 재진입",golden_cross:"골든크로스",dead_cross:"데드크로스",volume_breakout:"거래량 동반 상승 돌파",volume_breakdown:"거래량 동반 하락 돌파"};
const contexts:Record<string,string>={any:"추가 조건 없음",market_up:"시장 상승 추세",market_down:"시장 하락 추세",high_vol:"고변동성 국면",low_vol:"저변동성 국면",volume_surge:"거래량 2배 이상",trend_aligned:"ADX 추세와 방향 일치",other_indicator:"스토캐스틱 교차 동시 발생"};

export function TechnicalResearch(){
  const [mode,setMode]=useState("indicators");
  const [data,setData]=useState<Data|null>(null);
  const [error,setError]=useState("");
  const [signal,setSignal]=useState("rsi_exit_low");
  const [context,setContext]=useState("any");
  useEffect(()=>{let active=true;setData(null);setError("");api(`/technical-research/${mode}`).then(d=>{if(active)setData(d);}).catch(e=>{if(active)setError(e.message);});return()=>{active=false;};},[mode]);
  const result=data?.results?.[data.selected];
  const proof=mode==="conditions"?data?.conditions?.[`${signal}__${context}`]:data?.signals?.standard?.[signal];
  const difference=result?.comparison||data?.comparisons?.[data?.selected];
  return <section className="probability-evidence">
    <span className="eyebrow">기술적 분석 · 실제 KRX 자료로 검증</span>
    <h3>익숙한 투자 신호, 다음 날에도 유효했을까요?</h3>
    <div className="probability-bins" aria-label="기술적 분석 연구 종류">
      {[["indicators","지표별 비교"],["history","학습 이력 비교"],["conditions","사건·시장 조건"]].map(([key,title])=><button key={key} aria-pressed={mode===key} onClick={()=>{if(key!==mode){setData(null);setMode(key);setSignal("rsi_exit_low");}}}>{title}</button>)}
    </div>
    {!result?<p>{error||"연구 결과 생성 중입니다. 운영 모델은 유지됩니다."}</p>:<>
      <p><strong>2023년 재사용 진단 · 독립 검증 아님.</strong> 개발 기간에서 조건과 모델을 선택했습니다. 운영 방향 모델은 변경하지 않았습니다.</p>
      {mode==="history"&&<p>원자료 시작 {data?.audit.context_start} · 기술적 지표 공통 학습 시작 {data?.audit.technical_start} · 2015년 이전 {number(data?.audit.pre2015)}건 추가 사용 가능. 이 비교의 기준도 변동폭 OOF 존재 조건을 해제한 방향 전용 자료를 사용합니다.</p>}
      {mode==="conditions"&&<p>시장 추세는 현재 고정 종목군의 시장 대용 지표입니다. 과거 업종 구성·업종 수익률은 검증되지 않아 제외했습니다.</p>}
      <div className="probability-summary"><article><span>개발에서 선택한 방법</span><strong>{label(data!.selected)}</strong><p>확인 적중률 {pct(result.accuracy)} · Brier {number(result.brier,6)} · Log Loss {number(result.log_loss,6)}</p></article>
      <article><span>동일 표본 기준 대비 적중률 차이</span><strong>{number(difference?.accuracy.delta*100,2)}%p</strong><p>95% 구간 {number(difference?.accuracy.ci95[0]*100,2)}~{number(difference?.accuracy.ci95[1]*100,2)}%p · {number(result.n)}건</p></article></div>
      {mode==="history"&&<div className="probability-summary">{["up","down"].map(side=>{const s=result.strong.find((v:Data)=>v.side===side&&v.threshold===.6);const m=data!.strong_subset_metrics?.[side];return <article key={side}><span>{side==="up"?"상승":"하락"} 확률 60% 이상 · 선택 전략</span><strong>평균 예측 {pct(s.average_confidence)} → 실제 {pct(s.accuracy)}</strong><p>{number(s.n)}건 · 전체의 {pct(s.coverage)} · 같은 사례 기준 {pct(s.baseline_accuracy)}</p>{m&&<p>Brier {number(m.brier,6)} / 기준 {number(m.baseline_brier,6)}</p>}<p>{s.supported?"확인 표본의 보정·표본 기준 충족. 기준모델보다 우수하다는 의미는 아닙니다.":"보정·표본 근거 미확보"} 독립 검증 대기.</p></article>;})}</div>}
      {mode!=="history"&&<>
        <label>기술적 신호 <select aria-label="기술적 신호" value={signal} onChange={e=>setSignal(e.target.value)}>
          {mode==="conditions"?Object.entries(events).map(([k,v])=><option value={k} key={k}>{v}</option>):Object.entries(data!.signals.standard).map(([k,v])=><option value={k} key={k}>{(v as Data).description}</option>)}
        </select></label>
        {mode==="conditions"&&<label> 함께 관측된 조건 <select aria-label="함께 관측된 조건" value={context} onChange={e=>setContext(e.target.value)}>{Object.entries(contexts).map(([k,v])=><option value={k} key={k}>{v}</option>)}</select></label>}
        {proof&&<div aria-live="polite"><p>{number(proof.n)}건 · {proof.dates}개 거래일 · 전체 종목·거래일의 {pct(proof.coverage)}</p>
          {proof.n>0?<><div className="probability-summary"><article><span>이 신호 이후 실제 다음 날</span><strong>상승 {pct(proof.up_frequency)} · 하락 {pct(proof.down_frequency)}</strong><p>관행적 해석 방향의 적중률 {pct(proof.interpretation_accuracy)}<br/>95% 구간 {pct(proof.ci95.interpretation[0])}~{pct(proof.ci95.interpretation[1])}</p></article>
          <article><span>이전 자료에서 추정한 확률</span><strong>상승 {pct(proof.p_up)} · 하락 {pct(proof.p_down)}</strong><p>확률 기준 적중률 {pct(proof.probability_accuracy??proof.accuracy)} · 동일 사례 기존 모델 {pct(proof.baseline_accuracy)}</p></article></div>
          <p>Brier {number(proof.brier,6)} / 기준 {number(proof.baseline_brier,6)} · Log Loss {number(proof.log_loss,6)} / 기준 {number(proof.baseline_log_loss,6)}.</p>
          <p>확률 기준 적중률 개선 구간 {number(proof.ci95.accuracy_delta[0]*100,2)}~{number(proof.ci95.accuracy_delta[1]*100,2)}%p.</p></>:<p>이 조건에 해당하는 표본이 없습니다.</p>}
          <strong>{(mode==="conditions"?data!.selected_conditions.includes(`${signal}__${context}`):data!.selected_signals[signal]==="standard")?"개발 선정 조건 충족 · 독립 검증 대기":"개발 선정 근거 미확보 · 운영 미적용"}</strong>
          <p>관측 빈도를 사후에 예측확률로 바꾸지 않습니다. 조건별 개발 선정 여부와 별개로 독립 검증은 남아 있습니다.</p>
          {mode==="conditions"&&data!.conditional_models&&<details><summary>같은 사건에서 LightGBM·CatBoost는 어땠나요?</summary><div className="direction-table"><table><thead><tr><th>방법</th><th>적중률</th><th>Brier</th><th>Log Loss</th><th>기준 대비 적중률 차이 95% 구간</th></tr></thead><tbody>
          {Object.entries(data!.conditional_models[`${signal}__${context}`]).map(([k,v])=>{const x=v as Data;return <tr key={k}><td>{label(k)}</td><td>{pct(x.accuracy)}</td><td>{number(x.brier,6)}</td><td>{number(x.log_loss,6)}</td><td>{x.n?`${number(x.comparison_ci95.accuracy[0]*100,2)}~${number(x.comparison_ci95.accuracy[1]*100,2)}%p`:"표본 없음"}</td></tr>;})}
          </tbody></table></div><p>이미 고정한 예측을 같은 사건 표본에서 비교했습니다. 이 결과로 조건이나 모델을 다시 선택하지 않습니다.</p></details>}
        </div>}
      </>}
      <details><summary>전체 방법 비교·확률 보정</summary><div className="direction-table"><table><thead><tr><th>방법</th><th>적중률</th><th>Brier</th><th>Log Loss</th><th>보정 오차</th><th>방향·변동폭 동시 적중</th></tr></thead><tbody>
      {Object.entries(data!.results).map(([k,v])=>{const r=v as Data;return <tr key={k}><td>{label(k)}</td><td>{pct(r.accuracy)}</td><td>{number(r.brier,6)}</td><td>{number(r.log_loss,6)}</td><td>{pct(r.ece)}</td><td>{pct(r.joint_accuracy)}</td></tr>;})}
      </tbody></table></div></details>
      {mode==="history"&&<details><summary>학습 전략별 60% 이상 강한 신호 비교</summary>
        <p>상승·하락을 따로 확인합니다. 아래 문턱은 비교용 고정 기준이며, 독립 검증이나 운영 승격을 뜻하지 않습니다.</p>
        <div className="direction-table"><table><thead><tr><th>방법·방향</th><th>표본·제공 비율</th><th>평균 확신</th><th>실제 적중률·95% 구간</th></tr></thead><tbody>
        {Object.entries(data!.results).flatMap(([k,value])=>(value as Data).strong.filter((s:Data)=>s.threshold===.6).map((s:Data)=><tr key={`${k}-${s.side}`}><td>{label(k)} · {s.side==="up"?"상승":"하락"}</td><td>{number(s.n)}건 · {pct(s.coverage)}</td><td>{pct(s.average_confidence)}</td><td>{s.n?`${pct(s.accuracy)} (${pct(s.accuracy_ci95[0])}~${pct(s.accuracy_ci95[1])})`:"표본 없음"}</td></tr>))}
        </tbody></table></div>
      </details>}
      <details><summary>자료와 선택의 한계</summary>{data!.limitations.map((s:string)=><p key={s}>{s}</p>)}<p>신뢰구간은 동일 거래일의 종목들을 묶은 20거래일 블록으로 계산했습니다. 충분한 표본과 개발 연도별 안정성을 확인하며 높은 적중률만으로 채택하지 않습니다.</p></details>
    </>}
  </section>;
}
