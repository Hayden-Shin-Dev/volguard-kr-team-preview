import { useEffect, useState } from "react";
import { api, number, type Data } from "./api";

const pct = (v: number | null | undefined) => v == null ? "—" : `${number(v*100, 2)}%`;
const names: Record<string,string> = {initial:"초기 모델 · 확률 보정",technical_expanding:"현재 기술적 지표 연구 모델"};

export function HistoricalReliability() {
  const [data,setData] = useState<Data|null>(null);
  const [side,setSide] = useState("up");
  const [error,setError] = useState("");
  useEffect(()=>{let active=true;api("/direction-historical-reliability").then(d=>{if(active)setData(d);}).catch(()=>{if(active)setError("과거 비교 자료를 불러오지 못했습니다.");});return()=>{active=false;};},[]);
  if (!data?.models) return error ? <p role="alert">{error}</p> : null;
  return <section className="probability-evidence">
    <span className="eyebrow">2024년 회고 비교 · 고정 모델 · 동일 종목과 거래일</span>
    <h3>초기 모델과 연구 모델, 어느 확률이 더 맞았나요?</h3>
    <p>{data.audit.start}–{data.audit.end} · {number(data.audit.matched_n)}건 · {data.audit.symbols}종목 · {data.audit.dates}거래일. 초기 모델의 보정에 쓰인 2023년은 비교에서 제외했습니다. 이미 사용한 과거 기간이며 새로운 독립 검증은 아닙니다.</p>
    <div className="probability-summary">{Object.entries(data.models).map(([key,value])=>{const r=value as Data;return <article key={key}><span>{names[key]}</span><strong>Brier {number(r.brier,6)}</strong><p>Log Loss {number(r.log_loss,6)} · 구간 보정 오차 {pct(r.ece)}<br/>60% 이상 신호 비율 {pct(r.strong60)} · 70% 이상 {pct(r.strong70)}</p></article>;})}</div>
    <p>{data.conclusion==="initial_better"?"같은 표본의 Brier와 Log Loss는 초기 모델이 유의하게 낮습니다.":data.conclusion==="research_better"?"같은 표본의 Brier와 Log Loss는 연구 모델이 유의하게 낮습니다.":"Brier와 Log Loss에서 어느 한 모델의 우위를 확정하기 어렵습니다."} 특정 60%·70% 구간의 신뢰성은 아래에서 별도로 확인합니다.</p>
    <div className="probability-bins" aria-label="과거 확률 비교 방향">{[["up","과거 상승 확률"],["down","과거 하락 확률"]].map(([k,t])=><button key={k} aria-pressed={side===k} onClick={()=>setSide(k)}>{t}</button>)}</div>
    <div className="direction-table"><table><thead><tr><th>모델</th><th>확률 구간</th><th>예측 평균</th><th>실제 빈도</th><th>표본·거래일</th><th>실제 빈도 95% 구간</th><th>연구 판정</th></tr></thead><tbody>
      {[.5,.6,.7].flatMap(lo=>Object.entries(data.models).map(([key,value])=>{const r=(value as Data).bins.find((b:Data)=>b.side===side&&b.low===lo);return <tr key={`${key}-${lo}`}><td>{names[key]}</td><td>{lo*100}–{Math.round((lo+.1)*100)}% 미만</td><td>{pct(r.mean_probability)}</td><td>{pct(r.actual_frequency)}</td><td>{number(r.n)}건 · {r.dates}일</td><td>{r.frequency_ci95&&r.status!=="insufficient"?`${pct(r.frequency_ci95[0])}–${pct(r.frequency_ci95[1])}`:"표본 부족"}</td><td>{r.status==="insufficient"?"표본 부족":r.status==="compatible_retrospective"?"오차 ±5%p 내 정합":"정합성 미확인"}</td></tr>;}))}
    </tbody></table></div>
    <p>50%는 50–60% 미만, 60%는 60–70% 미만, 70%는 70–80% 미만입니다. 각 구간의 예측 평균과 실제 빈도를 비교합니다. 모델마다 해당 구간에 들어가는 사례는 다릅니다.</p>
    <details><summary>비교 오차와 해석 범위</summary>{Object.entries(data.comparison).map(([key,value])=>{const r=value as Data;return <p key={key}>연구−초기 {key}: {number(r.delta,6)} · 95% 구간 {number(r.ci95[0],6)}–{number(r.ci95[1],6)}. 양수이면 초기 모델의 오차가 더 작습니다.</p>;})}<p>20거래일 블록으로 같은 날짜의 종목을 함께 재표집했습니다. 500표본·60거래일 이상이며 확률 오차의 95% 구간 전체가 ±5%p 안에 있을 때만 회고 정합으로 표시합니다.</p>{data.limitations.map((s:string)=><p key={s}>{s}</p>)}</details>
  </section>;
}
