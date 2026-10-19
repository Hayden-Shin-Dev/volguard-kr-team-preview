import { useEffect, useState } from "react";
import { api, number, type Data } from "./api";

const pct = (x: number) => `${number(x * 100, 2)}%`;
const label: Record<string, string> = {
  supported_research: "연구 기준 내 보정 근거", insufficient: "표본 부족", not_supported: "허용 오차 내 근거 부족",
};

export function ProbabilityEvidence() {
  const [data, setData] = useState<Data | null>(null);
  const [error, setError] = useState("");
  const [bin, setBin] = useState("P3");
  const [raw, setRaw] = useState(false);
  useEffect(() => {
    let active = true;
    api("/probability-evidence").then(d => { if (active) setData(d); })
      .catch(e => { if (active) setError(e.message); });
    return () => { active = false; };
  }, []);
  if (!data || data.status !== "complete") return <p>{error || "확률 검증 근거 확인 중"}</p>;
  const bins: Data[] = raw ? data.raw_bins : data.bins;
  const chosen = bins.find(b => b.id === bin) || bins[0];
  const points = bins.filter(b => b.n);
  return <section className="probability-evidence">
    <span className="eyebrow">확률과 근거 · {data.evidence_id}</span>
    <h3>60%라고 말한 예측은, 실제로 얼마나 올랐을까요?</h3>
    <p>{data.evaluation}. {data.scope}. 현재 시세의 상승 원인을 설명하는 자료가 아니라,
      과거에 발행 가능했던 확률의 검증 근거입니다.</p>
    <div className="probability-summary">
      <div><span>기준 → 보정 Brier · 낮을수록 좋음</span><strong>{number(data.baseline.brier, 6)} → {number(data.calibrated.brier, 6)}</strong></div>
      <div><span>기준 → 보정 Log Loss</span><strong>{number(data.baseline.log_loss, 6)} → {number(data.calibrated.log_loss, 6)}</strong></div>
    </div>
    <p>보정 방식은 2022년에서 선택한 {data.selected}입니다. 2023년 확률 오차는 소폭 악화됐습니다.
      보정 근거가 있는 구간과 모델 개선 여부는 별개이며, 실시간 모델로 승격하지 않았습니다.</p>
    <label>비교할 확률 <select value={raw ? "raw" : "calibrated"} onChange={e => setRaw(e.target.value === "raw")}>
      <option value="calibrated">시간순 보정 후</option><option value="raw">보정 전 기준 모델</option>
    </select></label>
    <svg viewBox="0 0 600 270" role="img" aria-label="예측 상승확률과 실제 상승률 보정 그래프" className="probability-chart">
      <path d="M50 220 L550 220 M50 220 L50 20" fill="none" stroke="#c5c3d4" />
      <path d="M50 220 L550 20" stroke="#a8a3c4" strokeDasharray="5 5" />
      <text x="180" y="258">평균 예측 상승확률 0 → 100%</text>
      <text x="55" y="16">실제 상승 비율 · 세로선: 95% 구간</text>
      {points.map(b => <g key={b.id}>
        <line x1={50+b.mean_probability*500} x2={50+b.mean_probability*500}
          y1={220-b.rate_ci95[0]*200} y2={220-b.rate_ci95[1]*200} stroke="#7462d1" strokeWidth="2" />
        <circle cx={50+b.mean_probability*500} cy={220-b.actual_up*200} r={b.id===bin ? 7 : 4} fill="#6851c9">
          <title>{pct(b.mean_probability)} 예측 · 실제 {pct(b.actual_up)} · {b.n}건</title>
        </circle>
      </g>)}
    </svg>
    <div className="probability-bins" aria-label="확률 구간 선택">
      {bins.map(b => <button key={b.id} aria-pressed={bin===b.id} onClick={() => setBin(b.id)}>
        {number(b.low*100)}~{number(b.high*100)}%<small>{number(b.n)}건</small>
      </button>)}
    </div>
    <article aria-live="polite" className="probability-proof">
      <h4>{number(chosen.low*100)}~{number(chosen.high*100)}% 구간 · {label[chosen.status]}</h4>
      {chosen.n ? <>
        <div className="probability-summary">
          <div><span>평균 예측 · 보합 제외</span><strong>상승 {pct(chosen.mean_probability)} / 하락 {pct(1-chosen.mean_probability)}</strong></div>
          <div><span>실제 관측</span><strong>상승 {pct(chosen.actual_up)} / 하락 {pct(chosen.actual_down)}</strong></div>
        </div>
        <p>근거: {number(chosen.n)}건 · {chosen.dates}거래일 · 전체의 {pct(chosen.coverage)}.
          실제 상승 비율의 95% 구간 {pct(chosen.rate_ci95[0])}~{pct(chosen.rate_ci95[1])}.</p>
        <p>실제 비율 − 예측 확률의 오차 구간: {number(chosen.gap_ci95[0]*100,2)}~{number(chosen.gap_ci95[1]*100,2)}%p.
          넓은 구간에 예측값이 포함되는 것만으로 통과시키지 않습니다.</p>
        <details><summary>근거 사례 보기 · 날짜순 마지막 8건, 성공·실패 모두 포함</summary>
          <div className="direction-table"><table><thead><tr><th>종목</th><th>정보 기준일</th><th>예측 대상일</th><th>상승확률</th><th>실제 결과</th></tr></thead>
            <tbody>{chosen.examples.map((e:Data) => <tr key={`${e.symbol}-${e.target_date}`}><td>{e.symbol}</td><td>{e.feature_date}</td><td>{e.target_date}</td><td>{pct(e.probability)}</td><td>{e.actual} · {number(e.return_pct,2)}%</td></tr>)}</tbody>
          </table></div>
        </details>
      </> : <p>이 확률을 출력한 표본이 없습니다. 70% 확률이 검증됐다고 표시하지 않습니다.</p>}
    </article>
    <details><summary>평가 방법·근거 식별자·사용 제한</summary>
      <p>2021년 보정 학습 → 2022년 방식 선택 → 2021~2022년 재보정 → 2023년 확인.
        같은 날짜의 종목을 묶은 20거래일 블록 부트스트랩 2,000회.</p>
      {data.limitations.map((s:string) => <p key={s}>{s}</p>)}
      <p>입력 해시: {data.input_hash} · 예측 해시: {data.predictions_hash}</p>
    </details>
  </section>;
}
