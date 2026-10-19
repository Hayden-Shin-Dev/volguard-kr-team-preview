import { useEffect, useState } from "react";
import { api, number, type Data } from "./api";

const directionName: Data = {
  up: "상승",
  down: "하락",
  tie: "동률 · 예측 보류",
};
export function DirectionComparison({
  symbol = "005930",
  compact = false,
}: {
  symbol?: string;
  compact?: boolean;
}) {
  const [data, setData] = useState<Data | null>(null),
    [model, setModel] = useState(""),
    [history, setHistory] = useState<Data | null>(null),
    [day, setDay] = useState(""),
    [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    api("/direction-comparison")
      .then((r) => {
        if (active) {
          setData(r);
          setModel(r.selected || "");
        }
      })
      .catch((e) => {
        if (active) setError(e.message);
      });
    return () => {
      active = false;
    };
  }, []);
  useEffect(() => {
    if (!model) return;
    let active = true;
    setHistory(null);
    setError("");
    api(
      `/stocks/${symbol}/direction-history?model=${encodeURIComponent(model)}`,
    )
      .then((r) => {
        if (active) {
          setHistory(r);
          setDay(r.rows.at(-1)?.target_date || "");
        }
      })
      .catch((e) => {
        if (active) setError(e.message);
      });
    return () => {
      active = false;
    };
  }, [symbol, model]);
  if (!data) return <p>{error || "방향 모델 비교를 불러오고 있습니다."}</p>;
  if (!data.results.length)
    return (
      <p>여러 방향 모델을 학습·평가 중입니다. 결과를 꾸며 표시하지 않습니다.</p>
    );
  const chosen = data.results.find((r: Data) => r.model === data.selected);
  const selected = history?.rows.find((r: Data) => r.target_date === day);
  const body = (
    <div className="direction-comparison">
      <span className="eyebrow">
        상승·하락 확률 → 높은 쪽을 선택 → 실제 방향과 채점
      </span>
      <h3>다음 날 방향을 몇 번 맞혔나요?</h3>
      <p>
        상승 확률이 더 높으면 상승, 하락 확률이 더 높으면 하락으로 예측합니다.
        기준일 종가 대비 다음 거래일 종가 방향이 같으면 정답입니다.
      </p>
      <div className="development-metrics">
        <article>
          <small>사전 선택 모델 · {chosen.name}</small>
          <strong>
            {number(chosen.confirmation.accuracy * 100, 2)}
            <em>%</em>
          </strong>
          <span>
            2023년 {number(chosen.confirmation.correct)} /{" "}
            {number(chosen.confirmation.predicted_n)}건 정답
          </span>
        </article>
        <article>
          <small>상승으로 예측했을 때</small>
          <strong>
            {number(chosen.confirmation.up.accuracy * 100, 2)}
            <em>%</em>
          </strong>
          <span>
            {number(chosen.confirmation.up.correct)} /{" "}
            {number(chosen.confirmation.up.n)}건 정답
          </span>
        </article>
        <article>
          <small>하락으로 예측했을 때</small>
          <strong>
            {number(chosen.confirmation.down.accuracy * 100, 2)}
            <em>%</em>
          </strong>
          <span>
            {number(chosen.confirmation.down.correct)} /{" "}
            {number(chosen.confirmation.down.n)}건 정답
          </span>
        </article>
      </div>
      <p>
        2021~2022년 적중률로 모델을 먼저 선택한 뒤 2023년 결과를 확인했습니다.
        모두 재사용 개발 구간이며 새 최종 시험은 아닙니다. 보합 제외, 정확히
        50:50이면 예측 보류입니다.
      </p>
      <details open={!compact}>
        <summary>서로 다른 모델 7종 + 앙상블 비교</summary>
        <div className="direction-table">
          <table>
            <thead>
              <tr>
                <th>모델</th>
                <th>2021~2023 전체</th>
                <th>2023 적중률</th>
                <th>상승 선택 적중률</th>
                <th>하락 선택 적중률</th>
                <th>맞힌 수 / 평가 수</th>
              </tr>
            </thead>
            <tbody>
              {data.results.map((r: Data) => (
                <tr
                  key={r.model}
                  className={r.model === data.selected ? "selected-result" : ""}
                >
                  <td>
                    {r.name}
                    {r.model === data.selected ? " · 사전 선택" : ""}
                  </td>
                  <td>{number(r.overall.accuracy * 100, 2)}%</td>
                  <td>{number(r.confirmation.accuracy * 100, 2)}%</td>
                  <td>
                    {r.confirmation.up.accuracy == null
                      ? "상승 선택 없음"
                      : `${number(r.confirmation.up.accuracy * 100, 2)}%`}
                  </td>
                  <td>
                    {r.confirmation.down.accuracy == null
                      ? "하락 선택 없음"
                      : `${number(r.confirmation.down.accuracy * 100, 2)}%`}
                  </td>
                  <td>
                    {number(r.confirmation.correct)} /{" "}
                    {number(r.confirmation.predicted_n)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p>
          비교 기준: 매일 상승이라고 했을 때{" "}
          {number(chosen.always_up_accuracy * 100, 2)}%, 매일 하락이라고 했을 때{" "}
          {number(chosen.always_down_accuracy * 100, 2)}%. 한 방향만 많이 선택해
          얻은 적중률도 이 기준과 비교합니다.
        </p>
      </details>
      <h4>실제 과거 거래일의 확률과 정답 확인 · {symbol}</h4>
      <div className="replay-picker">
        <label>
          방향 모델{" "}
          <select
            aria-label="비교할 방향 모델"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          >
            {Object.entries(data.names).map(([k, v]) => (
              <option key={k} value={k}>
                {String(v)}
              </option>
            ))}
          </select>
        </label>
        <label>
          예측 대상일{" "}
          <select
            aria-label="방향 평가 거래일"
            value={day}
            onChange={(e) => setDay(e.target.value)}
          >
            {[...(history?.rows || [])].reverse().map((r: Data) => (
              <option key={r.target_date} value={r.target_date}>
                {r.target_date}
              </option>
            ))}
          </select>
        </label>
      </div>
      {error && <p role="alert">{error}</p>}
      {selected && (
        <div className="direction-example">
          <p>
            {selected.date}까지의 자료 → <b>{selected.target_date}</b> 종가 방향
            예측 · 과거 검증 기록
          </p>
          <div
            className="direction-probabilities"
            aria-label="과거 모델의 상승 하락 확률"
          >
            <div className={selected.prediction === "up" ? "picked" : ""}>
              <small>상승 확률</small>
              <strong>{number(selected.p_up * 100, 2)}%</strong>
            </div>
            <div className={selected.prediction === "down" ? "picked" : ""}>
              <small>하락 확률</small>
              <strong>{number(selected.p_down * 100, 2)}%</strong>
            </div>
          </div>
          <p>
            높은 확률 선택: <b>{directionName[selected.prediction]}</b> → 실제:{" "}
            <b>{directionName[selected.actual]}</b> (
            {number(selected.actual_return * 100, 2)}%) →{" "}
            <b>
              {selected.prediction === "tie"
                ? "채점 제외"
                : selected.correct
                  ? "정답"
                  : "오답"}
            </b>
          </p>
          <small>
            위 확률은 해당 과거 날짜의 실제 모델 출력입니다. 현재 종목의 다음
            거래일 확률로 사용하지 않습니다.
          </small>
        </div>
      )}
      {history && !history.rows.length && (
        <p>이 종목은 필요한 가격 이력이 부족해 해당 평가 기록이 없습니다.</p>
      )}
      <details>
        <summary>방향 전용 데이터 전처리와 확률 품질 지표</summary>
        {data.dataset?.folds?.length > 0 && (
          <>
            <p>
              정제 후 사용 가능한 표본 {number(data.dataset.eligible_rows)}개 ·
              방향용 입력 변수 {data.dataset.feature_count}개. 한 표본은 한
              종목의 한 거래일입니다.
            </p>
            <div className="direction-table">
              <table>
                <thead>
                  <tr>
                    <th>평가 연도</th>
                    <th>학습 표본</th>
                    <th>확률 보정 표본</th>
                    <th>평가 표본</th>
                  </tr>
                </thead>
                <tbody>
                  {data.dataset.folds.map((f: Data) => (
                    <tr key={f.year}>
                      <td>{f.year}</td>
                      <td>
                        {number(f.train.n)} · {f.train.start}~{f.train.end}
                      </td>
                      <td>{number(f.calibration.n)}</td>
                      <td>{number(f.evaluation.n)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
        <p>
          KRX 원본 일봉에서 방향용 정답과 피처를 별도로 생성합니다. 가격 조정을
          확인한 수익률로 상승·하락을 만들고, 거래정지·미확인 기업행동·이력
          부족은 제외합니다. 학습 자료로만 표준화하고 이전 연도로만 확률을
          보정합니다.
        </p>
        <p>
          현재 실험은 수익률·RSI·거래량 변화·시장 대비 상대 움직임 등을
          사용합니다. 기존 변동폭 모델의 예측값을 입력한 결합 모델은 아닙니다.
          보합 {number(data.excluded_flat)}개는 원본 방향 전처리에서 별도
          제외했습니다.
        </p>
        <p>
          선택 모델의 2023년 AUC {number(chosen.probability_metrics.roc_auc, 4)}
          , Brier {number(chosen.probability_metrics.brier, 5)}. 적중률과 별도로
          확률 수치의 신뢰성을 검사합니다. 운영 확률은 아직 승격하지 않았습니다.
        </p>
      </details>
    </div>
  );
  return compact ? (
    <details className="direction-workbench">
      <summary>상승·하락 ML · 모델별 적중률과 과거 확률 확인</summary>
      {body}
    </details>
  ) : (
    body
  );
}
