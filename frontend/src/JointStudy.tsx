import { useEffect, useState } from "react";
import { api, number, type Data } from "./api";
const pct=(x:number)=>`${number(x*100,2)}%`;
export function JointStudy(){
  const [data,setData]=useState<Data|null>(null);
  useEffect(()=>{let active=true;api('/direction-joint-study').then(d=>{if(active)setData(d);}).catch(()=>{});return()=>{active=false;};},[]);
  if(!data?.results)return null;
  const d=data.diagnostic;
  return <section className="probability-evidence"><span className="eyebrow">시장 반등과 종목 선택을 분리한 실제 실험</span><h3>81.74%의 상승률은 어디에서 나왔나요?</h3>
    <div className="probability-summary"><article><span>기존 상승 60–70% 예측 · 241건</span><strong>{pct(d.pooled.actual)}</strong><p>날짜에 같은 비중을 주면 {pct(d.equal_date_signal_up)}<br/>최다 발생일을 제외하면 {pct(d.without_largest.actual)}</p></article><article><span>같은 날짜의 시장 비교</span><strong>종목 상승 빈도 {pct(d.pooled.same_date_breadth)}</strong><p>코스피200 상승일 비율 {pct(d.pooled.same_date_market_up)}<br/>신호 건수로 가중한 비교 · 현재 구성 중 평가 가능한 종목</p></article></div>
    <p>2024년 8월 6일에 151건이 몰렸습니다. 전날 코스피200 −9.12% 이후 +2.97% 반등했고, 신호 종목의 89.40%가 상승했습니다. 같은 날 전체 평가 종목의 상승률은 90.45%여서 시장 반등의 영향이 컸습니다.</p>
    <h4>시장 수익률 + 종목 초과수익률 결합 모델</h4><p>2020년까지 학습 → 2021년 같은 날짜·종목의 두 예측 오차를 짝지어 결합분포 구성 → 2022년 확률 보정 → 2023년 선택 → 2024년 회고 평가. 상승확률을 곱하거나 더하지 않고, 두 수익률의 합이 양수인 비율을 계산했습니다.</p>
    <div className="direction-table"><table><thead><tr><th>방법</th><th>적중률</th><th>Brier ↓</th><th>Log Loss ↓</th><th>AUC</th><th>보정 오차</th><th>상승 60%대 예측 → 실제</th></tr></thead><tbody>{Object.entries(data.results).map(([k,v])=>{const r=v as Data;return <tr key={k}><td>{k==='initial_refit'?'초기 방식 · 같은 기간 재학습':'시장·종목 결합'}</td><td>{pct(r.accuracy)}</td><td>{number(r.brier,6)}</td><td>{number(r.log_loss,6)}</td><td>{number(r.roc_auc,4)}</td><td>{number(r.ece*100,2)}%p</td><td>{pct(r.up60_70.predicted)} → {pct(r.up60_70.actual)} · {r.up60_70.n}건</td></tr>;})}</tbody></table></div>
    <p>이번 결합 구조의 개선은 확인되지 않았습니다. 개발에서 정한 상승·하락 추세, 비급변일, 고변동성 조건 중 표본·확률 정합·동일 날짜 대비 우위를 함께 충족한 조건은 없었습니다. 기존 모델과 하락 60% 연구 신호를 유지합니다. 2024년은 재사용 회고 평가이며 새로운 독립 검증이 아닙니다.</p>
    <details><summary>27거래일 성공·실패와 같은 날짜 기준모델</summary><div className="direction-table"><table><thead><tr><th>정답일</th><th>코스피200</th><th>신호 수</th><th>성공 / 실패</th><th>예측 평균</th><th>전체 상승률</th><th>동일 신호 초기 적중률</th></tr></thead><tbody>{d.dates.map((x:Data)=><tr key={x.target_date}><td>{x.target_date}</td><td>{pct(x.kospi200_return)}</td><td>{x.signal_n}</td><td>{x.successes} / {x.failures}</td><td>{pct(x.mean_prediction)}</td><td>{pct(x.all_up_frequency)}</td><td>{pct(x.baseline_same_signals_accuracy)}</td></tr>)}</tbody></table></div><p>같은 날짜 전체 상승 빈도 대비 차이 {number(d.pooled.breadth_lift*100,2)}%p · 날짜 블록 95% 구간 [{d.pooled.breadth_lift_ci95.map((x:number)=>number(x*100,2)).join(', ')}]%p. 코스피200 상승일 비율 대비 차이의 구간은 [{d.pooled.market_lift_ci95.map((x:number)=>number(x*100,2)).join(', ')}]%p입니다. 탐색적 비교이며 27거래일의 정합·안정성을 입증하지 않습니다.</p></details>
  </section>;
}
