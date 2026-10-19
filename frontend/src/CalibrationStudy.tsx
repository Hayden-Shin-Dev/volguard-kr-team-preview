import { useEffect, useState } from "react";
import { api, number, type Data } from "./api";

const pct = (v: number | null) => v == null ? "—" : `${number(v * 100, 2)}%`;
export function CalibrationStudy() {
  const [data, setData] = useState<Data | null>(null);
  const [side, setSide] = useState("up");
  useEffect(() => { let active = true; api("/direction-calibration-study").then(d => { if (active) setData(d); }).catch(() => {}); return () => { active = false; }; }, []);
  if (!data?.results) return null;
  const name = (k: string) => k === data.baseline ? "초기 방식 · 동일 조건 재학습" : "최근 자료 가중 · 시장별 확률 보정";
  const better = Object.values(data.comparison).every(v => (v as Data).ci95[1] < 0);
  return <section className="probability-evidence">
    <span className="eyebrow">실제 재학습 · 학습과 확률 보정 분리</span>
    <h3>강한 확률도 실제 빈도와 맞아야 합니다</h3>
    <p>2010–2021 학습 → 2022 확률 보정 → 2023 후보 선택 → 2024 회고 평가. 3개 모델을 학습하고 5가지 보정 방법을 비교했습니다. 최근 가중 후보는 2017–2021 자료를 사용합니다. 평가 자료는 동일한 193종목·244거래일·42,458건입니다.</p>
    <div className="probability-summary">{Object.entries(data.results).map(([k,v]) => { const r=v as Data; return <article key={k}><span>{name(k)}</span><strong>Brier {number(r.brier,6)}</strong><p>Log Loss {number(r.log_loss,6)} · 확률 보정 오차 {pct(r.ece)}<br/>방향 적중률 {pct(r.accuracy)} · 60% 이상 신호 비율 {pct(r.strong60)}</p></article>; })}</div>
    <p>{better ? "선택 후보의 확률 오차 감소가 회고 평가에서 확인됐습니다." : "새 후보의 확률 오차 개선은 확인되지 않았습니다. 운영 모델은 교체하지 않았습니다."} 개발 기간에서도 표본과 오차 범위를 모두 통과한 상승 60% 이상 구간은 발견되지 않았습니다.</p>
    <div className="probability-bins" aria-label="재학습 확률 비교 방향">{[["up","재학습 상승 확률"],["down","재학습 하락 확률"]].map(([k,t]) => <button key={k} aria-pressed={side===k} onClick={() => setSide(k)}>{t}</button>)}</div>
    <div className="direction-table"><table><thead><tr><th>모델</th><th>확률 구간</th><th>예측 평균</th><th>실제 빈도</th><th>표본·거래일</th><th>신호 비율</th><th>판정</th></tr></thead><tbody>{[.5,.6,.7].flatMap(lo => Object.entries(data.results).map(([k,v]) => { const b=(v as Data).bins.find((x:Data) => x.side===side && x.low===lo); return <tr key={`${k}-${lo}`}><td>{name(k)}</td><td>{lo*100}–{Math.round((lo+.1)*100)}% 미만</td><td>{pct(b.mean_probability)}</td><td>{pct(b.actual_frequency)}</td><td>{number(b.n)}건 · {b.dates}일</td><td>{pct(b.coverage)}</td><td>{b.status==="insufficient"?"표본 부족":b.status==="compatible_retrospective"?"오차 ±5%p 내 정합":"정합성 미확인"}</td></tr>; }))}</tbody></table></div>
    <p>상승 60%대 후보의 실제 상승 빈도는 높았지만 27거래일에 집중돼 있습니다. 높은 적중률을 곧바로 신뢰할 수 있는 확률로 해석하지 않습니다. 70%대 역시 표본이 부족합니다.</p>
    <details><summary>신뢰구간·모델 선택·해석 범위</summary>{Object.entries(data.comparison).map(([k,v]) => {const r=v as Data;return <p key={k}>후보−초기 {k}: {number(r.delta,6)} · 95% 구간 [{number(r.ci95[0],6)}, {number(r.ci95[1],6)}]</p>;})}<p>2023년 Brier와 Log Loss가 초기 방식보다 나쁘지 않은 후보 중 선택했습니다. 2024년 결과를 보고 재선택하지 않았습니다. 같은 날짜의 종목을 함께 묶은 20거래일 블록을 2,000회 재표집했습니다. 정합 판정은 최소 500건·60거래일과 확률 오차 95% 구간 전체 ±5%p 조건을 적용합니다.</p>{data.limitations.map((s:string) => <p key={s}>{s}</p>)}</details>
  </section>;
}
