import { useEffect, useState } from "react";
import { api, number, type Data } from "./api";

const names: Record<string,string> = {market_down:"시장 급락",market_up:"시장 급등",abnormal_down:"시장 대비 비정상 하락",abnormal_up:"시장 대비 비정상 상승",volume_down:"거래량 급증 · 하락",volume_up:"거래량 급증 · 상승",gap_down:"하락 갭",gap_up:"상승 갭"};
const pct = (x:number) => `${number(x*100,2)}%`;
export function EventDirection(){
  const [data,setData]=useState<Data|null>(null);
  const [error,setError]=useState("");
  const [event,setEvent]=useState("abnormal_up");
  useEffect(()=>{let active=true;api("/event-direction").then(d=>{if(active)setData(d);}).catch(e=>{if(active)setError(e.message);});return()=>{active=false;};},[]);
  if(!data?.diagnostic) return <p>{error || "사건별 검증 결과 확인 중"}</p>;
  const r=data.diagnostic[event];
  return <section className="probability-evidence">
    <span className="eyebrow">사건이 있을 때만 · 선택적 방향 연구</span>
    <h3>특별한 움직임 이후, 다음 날 방향은 달랐을까요?</h3>
    <p>모델을 추가하지 않고 관측 가능한 8개 사건을 비교했습니다. 개발에서 선정된 강한 신호 {data.selected.length}개 · 운영 적용 없음.</p>
    <p><strong>독립 검증 대기.</strong> 아래는 이미 사용한 2023년 자료의 탐색 결과입니다. 새로운 시험 성적이 아닙니다.</p>
    <label>금융 사건 <select aria-label="금융 사건" value={event} onChange={e=>setEvent(e.target.value)}>{Object.entries(names).map(([k,v])=><option value={k} key={k}>{v}</option>)}</select></label>
    <p>{r.definition}</p>
    <div className="probability-summary" aria-live="polite">
      <article><span>이전 자료에서 고정한 확률</span><strong>상승 {pct(r.p_up)} · 하락 {pct(r.p_down)}</strong><p>실제 상승 {pct(r.up_frequency)} · 하락 {pct(r.down_frequency)}</p></article>
      <article><span>더 높은 확률 방향의 적중률</span><strong>{pct(r.accuracy)}</strong><p>같은 사건의 기존 모델 {pct(r.baseline_accuracy)}<br/>95% 구간 {pct(r.ci95.hit[0])}~{pct(r.ci95.hit[1])}</p></article>
    </div>
    <p>{number(r.n)}개 종목·거래일 / {r.dates}개 거래일. 전체 종목·거래일 중 {pct(r.row_coverage)}, 사건이 하나라도 발생한 거래일 {pct(r.day_coverage)}.</p>
    <p>기존 모델 대비 적중률 차이의 95% 구간 {number(r.ci95.accuracy_delta[0]*100,2)}~{number(r.ci95.accuracy_delta[1]*100,2)}%p.
      같은 날짜의 여러 종목을 묶어 계산했습니다.</p>
    <div className="direction-table"><table><thead><tr><th>확률 평가</th><th>사건 확률</th><th>동일 사건 기존 모델</th></tr></thead><tbody>
      <tr><td>Brier · 낮을수록 좋음</td><td>{number(r.brier,6)}</td><td>{number(r.baseline_brier,6)}</td></tr>
      <tr><td>Log Loss · 낮을수록 좋음</td><td>{number(r.log_loss,6)}</td><td>{number(r.baseline_log_loss,6)}</td></tr>
      <tr><td>방향·변동폭 동시 적중 · 참고</td><td>{pct(r.joint_accuracy)}</td><td>{pct(r.baseline_joint_accuracy)}</td></tr>
    </tbody></table></div>
    <details><summary>사건 정의·선정 기준·독립 검증의 제한</summary>
      <p>2015~2020년 확률 추정 → 2021~2022년 후보 선정 → 2015~2022년 확률 재추정 후 동결 → 2023년 재사용 확인. 8개 사건과 3개 개선 지표의 다중 비교를 고려합니다.</p>
      <p>60% 이상 확신, 300건·60거래일, 기준 대비 적중률·Brier·Log Loss 개선과 확률 보정 조건을 개발에서 함께 요구했습니다. 미관측 기간의 실제 사전 발행 자료는 아직 0건이며 자동 수집·운영 승격을 의미하지 않습니다.</p>
      {data.limitations.map((s:string)=><p key={s}>{s}</p>)}
    </details>
  </section>;
}
