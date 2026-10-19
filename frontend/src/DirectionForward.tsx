import { useEffect, useState } from "react";
import { api, number, type Data } from "./api";

const pct = (v: number | null | undefined) => v == null ? "—" : `${number(v * 100, 1)}%`;

export function DirectionForward() {
  const [data, setData] = useState<Data | null>(null);
  const [model, setModel] = useState("technical_expanding");
  const [side, setSide] = useState("up");
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    const load = () => api("/direction-forward").then(d => { if (active) { setData(d); setError(""); } })
      .catch(() => { if (active) setError("실측 검증 기록을 불러오지 못했습니다. 자동으로 다시 연결합니다."); });
    void load(); const timer = setInterval(load, 15000);
    return () => { active = false; clearInterval(timer); };
  }, []);
  const selected = data?.models?.find((m: Data) => m.id === model);
  const strong = selected?.strong?.find((r: Data) => r.side === "down");
  return <section className="probability-evidence">
    <span className="eyebrow">실제 미래 데이터 · 연구 전용</span>
    <h3>말했던 확률만큼, 실제로 일어났나요?</h3>
    <p>오늘 KRX 정규장 종가 → 다음 거래일 KRX 종가. 장 마감 후 저장한 확률만 집계합니다. 보합은 별도 기록하며 상승·하락 확률은 보합 제외 조건부 확률입니다.</p>
    {error && <p role="alert">{error}</p>}
    {!data && !error && <p>실측 검증 기록을 불러오는 중입니다.</p>}
    {data && <>
      <label>비교할 연구 모델 <select aria-label="전향 검증 모델" value={model} onChange={e => setModel(e.target.value)}>
        {data.models?.map((m: Data) => <option key={m.id} value={m.id}>{m.name}</option>)}
      </select></label>
      {selected && <>
        <div className="probability-summary">
          <article><span>미리 기록한 예측</span><strong>{selected.issued}건</strong><p>채점 {selected.scored}건 · 보합 {selected.flat}건 · 정답 대기 {selected.pending}건</p></article>
          <article><span>하락 60% 이상 · 연구 신호</span><strong>{strong?.n ?? 0}개 검증 표본</strong><p>예측 평균 {pct(strong?.mean_probability)} → 실제 하락 {pct(strong?.observed_frequency)}<br/>신호 발생 비율 {pct(strong?.coverage)} · 운영 채택 아님</p></article>
        </div>
        {selected.issued === 0 && <p>아직 사전에 발행한 미래 예측이 없습니다. 다음 거래일 마감 후 자동 기록을 시작하며, 과거 데이터 실행 점검은 이 표본에 넣지 않습니다.</p>}
        <div className="probability-bins" aria-label="검증 방향">
          {[['up', '상승 확률'], ['down', '하락 확률']].map(([key, title]) => <button key={key} aria-pressed={side === key} onClick={() => setSide(key)}>{title}</button>)}
        </div>
        <div className="direction-table"><table><thead><tr><th>예측 구간</th><th>예측 평균</th><th>실제 {side === "up" ? "상승" : "하락"} 빈도</th><th>표본·거래일</th><th>보합·대기</th><th>검증 상태</th></tr></thead><tbody>
          {selected.bins.filter((r: Data) => r.side === side).map((r: Data) => <tr key={r.low}>
            <td>{pct(r.low)}–{pct(r.high)}{r.high < 1 ? " 미만" : " 이하"}</td><td>{pct(r.mean_probability)}</td><td>{pct(r.observed_frequency)}</td>
            <td>{r.n}건 · {r.dates}일</td><td>{r.flat} · {r.pending}</td><td>{r.status === "compatible_research" ? "연구 기준 내 정합" : r.status === "not_confirmed" ? "확률 정합성 미확인" : "표본 축적 중"}</td>
          </tr>)}
        </tbody></table></div>
        <p>{data.qualification}</p>
        <details><summary>자동 배치·모델 비교 근거</summary>
          <p>18:30 실행 · 자료 미도착 시 19:00·20:00 재시도. 이미 기록한 확률은 변경하지 않습니다. 고정 모델을 비교하며 자동 재학습·운영 승격은 하지 않습니다.</p>
          <p>현재 모델 학습 이력: {selected.fitted_history}. 오래된 학습 자료의 공개·수정 이력과 현재 구성 종목 생존편향은 남아 있습니다.</p>
          {data.batch?.reason && <p>{data.batch.reason}</p>}
          {data.batch?.results?.filter((r: Data) => r.reason).map((r: Data) => <p key={`${r.symbol}-${r.model}`}>{r.symbol}: {r.reason}</p>)}
          {data.scoring?.pending?.map((r: Data, i: number) => <p key={i}>{r.symbol} · {r.target_date}: {r.reason}</p>)}
          {data.comparison?.length ? <table><thead><tr><th>같은 미래 표본 비교</th><th>표본</th><th>Brier</th><th>Log Loss</th></tr></thead><tbody>{data.comparison.map((r: Data) => <tr key={r.model}><td>{data.models.find((m: Data) => m.id === r.model)?.name}</td><td>{r.n}</td><td>{number(r.brier, 5)}</td><td>{number(r.log_loss, 5)}</td></tr>)}</tbody></table> : <p>모델들이 공통으로 채점된 미래 표본을 기다리고 있습니다.</p>}
        </details>
      </>}
    </>}
  </section>;
}
