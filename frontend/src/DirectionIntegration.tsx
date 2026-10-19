import { useEffect, useState } from "react";
import { api, number, type Data } from "./api";

const names: Data = {
  annual_base: "연 1회 · 기본 입력",
  monthly_unweighted: "매월 · 동일 가중치",
  monthly_base: "매월 · 최근 자료 가중",
  monthly_range: "매월 · 변동폭 OOF 추가",
  monthly_rich: "매월 · 상대 움직임 확장",
  monthly_rich_short: "매월 · 최근 3년",
  lgb7: "LightGBM · 잎 7개",
  lgb31: "LightGBM · 잎 31개",
  cat4: "CatBoost · 깊이 4",
  cat6: "CatBoost · 깊이 6",
  ensemble: "세 모델 앙상블",
  external_base: "기본 입력",
  external_range: "변동폭 OOF 추가",
  external_rich: "가격·시장 변수 확장",
  external_full: "수급·공시 추가",
};
const calibrationNames: Data = {
  raw: "보정 전",
  sigmoid: "로지스틱 보정",
  isotonic: "단조 보정",
};
const label = (key: string) => {
  const [a, b] = key.split("__");
  return `${names[a] || a} · ${calibrationNames[b] || b}`;
};
function ScoreTable({ rows }: { rows: Data[] }) {
  return (
    <div className="direction-table">
      <table>
        <thead>
          <tr>
            <th>방식</th>
            <th>방향 적중률</th>
            <th>Brier ↓</th>
            <th>Log Loss ↓</th>
            <th>보정 오차 ↓</th>
            <th>동시 적중률</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name}>
              <td>{r.name}</td>
              <td>{number(r.accuracy * 100, 2)}%</td>
              <td>{number(r.brier, 5)}</td>
              <td>{number(r.log_loss, 5)}</td>
              <td>{number(r.ece * 100, 2)}%p</td>
              <td>{number(r.joint_accuracy * 100, 2)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function DirectionIntegration({
  symbol = "005930",
  compact = false,
}: {
  symbol?: string;
  compact?: boolean;
}) {
  const [data, setData] = useState<Data | null>(null),
    [history, setHistory] = useState<Data | null>(null),
    [day, setDay] = useState(""),
    [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    api("/direction-integration")
      .then((r) => {
        if (active) setData(r);
      })
      .catch((e) => {
        if (active) setError(e.message);
      });
    return () => {
      active = false;
    };
  }, []);
  useEffect(() => {
    let active = true;
    setHistory(null);
    api(`/stocks/${symbol}/direction-integration-history`)
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
  }, [symbol]);
  if (!data || data.status === "pending")
    return (
      <p>
        {error || "변동폭·방향 결합 실험의 실제 학습 결과를 준비하고 있습니다."}
      </p>
    );
  const c = data.comparison,
    record = history?.rows.find((r: Data) => r.target_date === day);
  const content = (
    <div className="direction-comparison">
      <span className="eyebrow">변동폭 예측을 방향 모델의 입력으로</span>
      <h3>결합했을 때, 실제로 더 잘 맞았나요?</h3>
      {error && <p role="alert">{error}</p>}
      <p>
        2021~2022년으로 방식을 선택한 뒤 같은 2023년 {number(c.candidate.n, 0)}
        개 종목·거래일을 비교했습니다. 재사용 개발 평가이며 새 최종시험은
        아닙니다.
      </p>
      <p>
        보합을 제외한 상승·하락 분류입니다. 확률이 정확히 같으면 방향 선택을
        보류하며, 동시 적중률에는 성공으로 포함하지 않습니다.
      </p>
      <p>
        <strong>
          {data.development_pass
            ? "개발 개선 관문 통과"
            : "통합 모델 채택 보류"}
        </strong>{" "}
        ·{" "}
        {data.operational_probability_enabled
          ? "운영 적용"
          : "현재 운영 예측은 유지합니다."}
      </p>
      <ScoreTable
        rows={[
          {
            name: "기존 방향 ML · 동일 평가 행",
            ...data.versus_original.reference,
          },
          { name: "비교 기준 · " + label(data.reference), ...c.reference },
          { name: "선택 후보 · " + label(data.selected), ...c.candidate },
        ]}
      />
      {!data.development_pass && (
        <p>
          채택 보류 이유: 적중률 차이의 95% 구간은{" "}
          {number(c.ci.accuracy[0] * 100, 2)}~
          {number(c.ci.accuracy[1] * 100, 2)}%p입니다. 확률 오차(Brier)는 비교
          기준 대비 {c.deltas.brier > 0 ? "증가" : "감소"}했고, 전체 개선 조건을
          충족하지 못했습니다.
        </p>
      )}
      <p>
        동시 적중 = 높은 확률로 선택한 방향이 맞고, 실제 고저 변동폭도 예측구간
        안에 든 경우. 두 적중률을 곱해 추정하지 않고 거래일별로 직접
        채점했습니다.
      </p>
      <details>
        <summary>
          변동폭 예측값 4개만 추가한 비교 · 같은 모델과 학습 조건
        </summary>
        {Object.entries(data.range_ablations).map(([method, v]) => {
          const x = v as Data;
          return (
            <div key={method}>
              <h4>
                {method === "raw"
                  ? "보정 전"
                  : method === "sigmoid"
                    ? "로지스틱 확률 보정"
                    : "단조 확률 보정"}
              </h4>
              <ScoreTable
                rows={[
                  { name: "기본 방향 입력", ...x.reference },
                  { name: "하한·중앙값·상한·폭 추가", ...x.candidate },
                ]}
              />
            </div>
          );
        })}
      </details>
      <details>
        <summary>전체 학습·보정 후보 비교 ({data.results.length}개)</summary>
        <ScoreTable
          rows={data.results.map((r: Data) => ({
            name: label(r.candidate),
            ...r.confirmation,
          }))}
        />
      </details>
      <details>
        <summary>삼성전자·SK하이닉스·현대차 수급·공시 추가 실험</summary>
        <p>
          세 종목에 한정한 별도 비교입니다. 과거 정정 전 자료는 입증되지 않아
          탐색 결과로만 사용합니다. 기록일 다음 날부터 반영했습니다.
        </p>
        <ScoreTable
          rows={data.external_results.map((r: Data) => ({
            name: label(r.candidate),
            ...r.confirmation,
          }))}
        />
      </details>
      <details>
        <summary>시간순 OOF와 채택 조건 확인</summary>
        <p>
          {number(data.oof.rows, 0)}개 변동폭 OOF를 생성했습니다. 연도별 모델은
          평가 2년 전까지 학습하고 직전 연도로 보정했으며, 각 날짜까지 확정된
          오차만 반영했습니다. 이후 방향 모델의 확률 보정에도 이전 OOF 출력만
          사용했습니다.
        </p>
        <p>
          선택 후보의 적중률 차이 95% 구간: {number(c.ci.accuracy[0] * 100, 2)}{" "}
          ~ {number(c.ci.accuracy[1] * 100, 2)}%p. 동시 적중률 차이:{" "}
          {number(c.ci.joint[0] * 100, 2)} ~ {number(c.ci.joint[1] * 100, 2)}%p.
        </p>
        <p>
          적중률·동시 적중률 증가, Brier·Log Loss 감소의 날짜 블록 신뢰구간과
          확률 보정 품질을 함께 검사합니다. 구조는 현재 고정한 모델의 회고
          재학습이며 과거 실제 발행 기록은 아닙니다.
        </p>
      </details>
      {!!history?.rows.length && (
        <div className="direction-example">
          <h4>같은 날의 방향과 변동폭 함께 채점 · {symbol}</h4>
          <label>
            과거 대상일{" "}
            <select
              aria-label="결합 모델 평가 거래일"
              value={day}
              onChange={(e) => setDay(e.target.value)}
            >
              {history.rows
                .slice()
                .reverse()
                .map((r: Data) => (
                  <option key={r.target_date}>{r.target_date}</option>
                ))}
            </select>
          </label>
          {record && (
            <>
              <div className="direction-probabilities">
                <div className={record.p_up > 0.5 ? "picked" : ""}>
                  <small>상승 확률</small>
                  <strong>{number(record.p_up * 100, 2)}%</strong>
                </div>
                <div className={record.p_down > 0.5 ? "picked" : ""}>
                  <small>하락 확률</small>
                  <strong>{number(record.p_down * 100, 2)}%</strong>
                </div>
              </div>
              <p>
                실제 방향 {record.actual_up ? "상승" : "하락"} (
                {number(record.actual_return * 100, 2)}%) · 방향{" "}
                {record.correct ? "정답" : "오답"}
              </p>
              <p>
                예측 변동폭 {number(record.low, 2)}~{number(record.high, 2)}% ·
                실제 {number(record.actual_range, 2)}% · 구간{" "}
                {record.inside ? "포함" : "이탈"}
              </p>
              <strong>
                동시 채점:{" "}
                {record.joint ? "두 조건 모두 적중" : "동시 적중 실패"}
              </strong>
              <p>
                과거 모델의 실제 출력입니다. 현재 다음 거래일에 대한 확률이
                아닙니다.
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
  return compact ? (
    <details>
      <summary>변동폭·방향 결합 ML · 실제 비교 결과</summary>
      {content}
    </details>
  ) : (
    content
  );
}
