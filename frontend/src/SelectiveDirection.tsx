import { useEffect, useState } from "react";
import { api, number, type Data } from "./api";

const names: Record<string,string> = {logistic:"로지스틱 기준",lightgbm:"비선형 LightGBM",catboost:"비선형 CatBoost",regime_mixture:"시장 국면별 전문가 결합",ensemble:"세 비선형 모델 결합"};
Object.assign(names,{paper_baseline:"동일 표본 기준",session_split:"야간·장중 분리",market_residual:"시장 대비 잔차",conditional_signals:"조건부 신호 결합",joint_session:"야간·장중 결합 상태 모델"});
const name = (key:string) => `${names[key.split("__")[0]]} · ${key.endsWith("__raw") ? "보정 전" : "시간순 확률 보정"}`;
const pct = (value:number | undefined) => value == null ? "—" : `${number(value*100,2)}%`;

export function SelectiveDirection({endpoint="/selective-direction"}:{endpoint?:string}) {
  const [data,setData] = useState<Data|null>(null);
  const [error,setError] = useState("");
  const [model,setModel] = useState("");
  const [threshold,setThreshold] = useState(.6);
  useEffect(() => {
    let active=true;
    api(endpoint).then(d => { if(active){setData(d);setModel(d.selected || "");} }).catch(e=>{if(active)setError(e.message);});
    return ()=>{active=false;};
  },[endpoint]);
  if(!data || data.status!=="complete") return <p>{error || "상승·하락 신호 학습 결과 확인 중"}</p>;
  const result=data.results[model];
  if(!result) return null;
  return <section className="probability-evidence">
    <span className="eyebrow">상승과 하락 · 강한 신호의 실제 근거</span>
    <h3>강한 확률을 낼 때, 실제로 더 잘 구분했을까요?</h3>
    <p>{data.audit ? "야간·장중 분리, 시장 잔차, 조건부 신호와 결합 상태 모델을 실제 학습했습니다." : "시장 국면별 전문가 모델과 비선형 모델을 다시 학습했습니다."} 상승·하락별로
      확률의 크기, 실제 적중률, 신호가 나온 비율을 함께 검증합니다. 보합 제외 · 2023년 재사용 개발 평가입니다.</p>
    {data.audit && <p>실적 발표 전 컨센서스와 과거 종가 호가·체결 방향 원자료는 미확보여서 해당 신호는 넣지 않았습니다.
      당시 공표·개정 이력도 미입증이므로, 이번 결과는 연구용입니다.</p>}
    {data.matched_existing && <p>동일 표본의 기존 로지스틱 적중률 {pct(data.matched_existing.accuracy)}, Brier {number(data.matched_existing.brier,6)}.
      새 선택 모델과의 적중률 차이 95% 구간 {number(data.matched_comparison.accuracy.ci95[0]*100,2)}~{number(data.matched_comparison.accuracy.ci95[1]*100,2)}%p.</p>}
    <div className="probability-summary">
      <div><span>2021~2022년에서 고정한 선택</span><strong>{name(data.selected)}</strong></div>
      <div><span>개발 기간에서 확보한 강한 신호 기준</span><strong>상승 {data.thresholds.up ? pct(data.thresholds.up) : "미확보"} · 하락 {data.thresholds.down ? pct(data.thresholds.down) : "미확보"}</strong></div>
    </div>
    <p>아래 선택은 연구 결과를 비교하기 위한 기능입니다. 실시간 모델 선택이나 자동 승격이 아닙니다.</p>
    <label>연구 모델 <select value={model} onChange={e=>setModel(e.target.value)}>
      {Object.keys(data.results).map(k=><option value={k} key={k}>{name(k)}</option>)}
    </select></label>
    <div className="probability-summary">
      <div><span>출력된 상승확률의 중앙 90% 범위</span><strong>{pct(result.spread.q05)}~{pct(result.spread.q95)}</strong></div>
      <div><span>상승 또는 하락 60% 이상 신호 비율</span><strong>{pct(result.spread.confident60)}</strong></div>
    </div>
    <div className="probability-bins" aria-label="방향 신호 임계값">
      {[.55,.6,.65,.7].map(t=><button key={t} aria-pressed={t===threshold} onClick={()=>setThreshold(t)}>{number(t*100)}% 이상{t===.55 ? " · 약한 신호 비교" : ""}</button>)}
    </div>
    <div className="probability-summary">
      {["up","down"].map(side=>{
        const signal=result.signals.find((s:Data)=>s.side===side && s.threshold===threshold);
        return <article className="probability-proof" key={side} aria-live="polite">
          <h4>{side==="up"?"상승":"하락"} 확률 {number(threshold*100)}% 이상</h4>
          <strong>{signal.n ? `평균 예측 ${pct(signal.average_confidence)} → 실제 ${pct(signal.accuracy)}` : "해당 확률을 낸 사례 없음"}</strong>
          <p>{number(signal.n)}건 · {signal.dates}거래일 · 전체의 {pct(signal.coverage)}</p>
          {signal.n>0 && <>
            <p>실제 적중률 95% 구간 {pct(signal.accuracy_ci95[0])}~{pct(signal.accuracy_ci95[1])}</p>
            <p>확률 오차 95% 구간 {number(signal.gap_ci95[0]*100,2)}~{number(signal.gap_ci95[1]*100,2)}%p</p>
            <p>동일 사례의 기존 모델 적중률 {pct(signal.baseline_accuracy)} · 과거 상승 빈도 기준 {pct(signal.prior_accuracy)}</p>
          </>}
          <strong>{signal.supported ? "이 확인 표본의 연구 기준 충족" : "강한 신호의 신뢰 근거 미확보"}</strong>
          <p>표본 300건·60거래일, 적중률 구간 하한 50% 초과, 확률 오차 구간 전체 ±5%p 이내를 함께 확인합니다.
            개발에서 선택하지 않은 임계값의 결과는 참고 비교이며 채택 근거가 아닙니다.</p>
          {signal.n>0 && <details><summary>{side==="up"?"상승":"하락"} 신호의 실제 사례와 변동폭</summary>
            <p>날짜순 마지막 6건입니다. 정답 여부로 선별하지 않습니다. 변동폭 구간은 기존 모델의 방향 비조건부 예측입니다.</p>
            <div className="direction-table"><table><thead><tr><th>종목·대상일</th><th>상승 / 하락</th><th>실제</th><th>고저 변동폭 예측 / 실제</th></tr></thead><tbody>
              {signal.examples.map((e:Data)=><tr key={`${e.symbol}-${e.target_date}`}><td>{e.symbol}<br/>{e.target_date}</td><td>{pct(e.p_up)} / {pct(e.p_down)}</td><td>{e.actual} {number(e.return_pct,2)}%</td><td>{number(e.range_low,2)}~{number(e.range_high,2)}% / {number(e.actual_range,2)}%</td></tr>)}
            </tbody></table></div>
          </details>}
        </article>;
      })}
    </div>
    <details><summary>전체 모델 성능과 양방향 비교</summary>
      <div className="direction-table"><table><thead><tr><th>모델</th><th>전체 적중률</th><th>상승 선택 적중률</th><th>하락 선택 적중률</th><th>Brier</th><th>Log Loss</th><th>60% 신호 비율</th></tr></thead>
        <tbody>{Object.entries(data.results).map(([k,value])=>{const r=value as Data;return <tr key={k}><td>{name(k)}</td><td>{pct(r.accuracy)}</td><td>{pct(r.up.accuracy)}</td><td>{pct(r.down.accuracy)}</td><td>{number(r.brier,6)}</td><td>{number(r.log_loss,6)}</td><td>{pct(r.spread.confident60)}</td></tr>;})}</tbody>
      </table></div>
    </details>
    <details><summary>선택·학습·검증의 근거</summary>
      <p>5년 이동 학습 창, 분기별 재학습, 최근 자료 가중치, 직전 1년의 시간순 외표본 확률 보정을 사용했습니다.
        {data.audit ? "특징군별 비교와 결합 상태 모델을 같은 표본에서 평가했습니다." : "전문가 결합은 예측 시점에 이미 관측된 시장 모멘텀으로 나눈 모델과 전체 모델을 절반씩 결합합니다."}</p>
      <p>개발에서 Brier와 Log Loss가 기준보다 나쁘지 않은 후보 중 검증된 60% 이상 신호 비율을 우선했습니다.
        강한 신호가 없으면 이를 그대로 미확보로 기록합니다.</p>
      {data.limitations.map((s:string)=><p key={s}>{s}</p>)}
      {data.audit && <p>가격·시점 데이터 재점검: {number(data.audit.rows_before)}건 중 {number(data.audit.excluded)}건 제외.
        2023년 비교 {number(data.audit.confirmation_after)}건. 실적 컨센서스·종가 주문 불균형·과거 업종 소속 원자료는 미확보입니다.</p>}
      {data.audit && <p>연구 근거: <a href="https://ideas.repec.org/a/eee/pacfin/v53y2019icp379-398.html" target="_blank" rel="noreferrer">한국 실적 발표 후 반응</a> · <a href="https://www.sciencedirect.com/science/article/pii/S1386418199000129" target="_blank" rel="noreferrer">종가 주문 불균형</a> · <a href="https://www.sciencedirect.com/science/article/pii/S0304405X19300650" target="_blank" rel="noreferrer">야간·장중 수익률</a>.
        해당 논문은 이 모델의 다음 날 방향 성능을 입증하는 자료가 아닙니다.</p>}
      <p>근거 ID: {data.evidence_id}</p>
    </details>
  </section>;
}
