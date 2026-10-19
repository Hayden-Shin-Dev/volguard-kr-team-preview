import { useEffect, useState } from "react";
import { api, number, type Data } from "./api";

const groups: Data = {
  persistence: "자기상관·지속성",
  trend: "모멘텀·평균회귀",
  conditional_vol: "GARCH·EGARCH",
  residual: "시장 잔차·베타",
  relations: "종목 간 관계·공통 요인",
  tails: "왜도·첨도·꼬리 위험",
  liquidity: "유동성·거래 흐름",
  nonlinear: "국면·엔트로피·변화점",
  qlib_subset: "Qlib 기반 가격 통계",
};
const modelNames: Data = {
  baseline: "동일 조건 기준 모델",
  all_math: "금융 통계 전체",
  monthly: "월별 재학습",
  expanding: "누적 학습",
  rolling_uniform: "최근 5년 · 동일 가중",
  rolling3: "최근 3년",
  recent1y: "최근 자료 가중 강화",
  regime: "국면별 모델",
  stock: "종목별 모델",
  balanced: "클래스 균형 조정",
  selection30: "학습 내 특징 선택",
  lightgbm: "LightGBM",
  catboost: "CatBoost",
  lstm_numeric: "수치형 LSTM",
  craft_adapted: "CRAFT 축소 적용",
  ensemble: "균등 앙상블",
  with_range: "변동폭 예측 입력 추가",
};
const calibrations: Data = {
  raw: "보정 전",
  sigmoid: "로지스틱 보정",
  isotonic: "단조 보정",
};
export function mathName(key: string) {
  const [name, cal] = key.split("__");
  const title = name.startsWith("add_")
    ? `${groups[name.slice(4)]} 추가`
    : name.startsWith("drop_")
      ? `${groups[name.slice(5)]} 제외`
      : modelNames[name] || name;
  return `${title}${cal ? " · " + calibrations[cal] : ""}`;
}
function Metrics({ rows }: { rows: Data[] }) {
  return (
    <div className="direction-table">
      <table>
        <thead>
          <tr>
            <th>모델</th>
            <th>방향 적중률</th>
            <th>균형 적중률</th>
            <th>AUC</th>
            <th>MCC</th>
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
              <td>{number(r.balanced_accuracy * 100, 2)}%</td>
              <td>{number(r.roc_auc, 4)}</td>
              <td>{number(r.mcc, 4)}</td>
              <td>{number(r.brier, 6)}</td>
              <td>{number(r.log_loss, 6)}</td>
              <td>{number(r.ece * 100, 2)}%p</td>
              <td>{number(r.joint_accuracy * 100, 2)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
export function DirectionMath() {
  const [data, setData] = useState<Data | null>(null),
    [error, setError] = useState(""),
    [symbol, setSymbol] = useState("005930");
  useEffect(() => {
    let active = true;
    api("/direction-math")
      .then((d) => {
        if (active) setData(d);
      })
      .catch((e) => {
        if (active) setError(e.message);
      });
    return () => {
      active = false;
    };
  }, []);
  if (error)
    return <p role="alert">금융 통계 연구 결과를 읽지 못했습니다: {error}</p>;
  if (!data || data.status === "pending") return null;
  const d = data,
    delta = d.comparison.accuracy,
    high = d.selected_confidence;
  return (
    <section className="direction-comparison">
      <span className="eyebrow">금융 수학 · 정보 가치 검증</span>
      <h3>새로운 정보가 방향 예측을 바꿨을까요?</h3>
      <p>
        2021~2022년으로 특징·학습 방식·확신도 기준을 정한 뒤, 같은 2023년{" "}
        {number(d.rows)}개 종목·거래일을 비교했습니다.{" "}
        <strong>재사용 개발 평가이며 독립 시험은 아닙니다.</strong>
      </p>
      <p className="status-note">
        {mathName(d.selected)} ·{" "}
        {d.development_gate
          ? "개발 비교 조건 통과 · 독립 검증 필요"
          : "개발 비교 조건 미충족"}{" "}
        · 현재 운영 방향 확률에는 적용하지 않았습니다.
      </p>
      <Metrics
        rows={[
          { ...d.baseline, name: "동일 조건 기준 모델" },
          { ...d.standalone, name: "선택 방향 모델" },
          { ...d.combined, name: "변동폭 입력 추가 후보" },
        ]}
      />
      <p>
        방향 적중률 차이 {number(delta.delta * 100, 3)}%p · 날짜 블록 기준 95%
        신뢰구간 {number(delta.ci95[0] * 100, 3)}~
        {number(delta.ci95[1] * 100, 3)}%p. 구간에 0이 있으면 개선을 확정할 수
        없습니다.
      </p>
      <details>
        <summary>어떤 금융 통계 정보가 기여했나요?</summary>
        <p>
          특징군을 하나씩 추가하거나 전체 입력에서 제거했습니다. 아래 수치는
          동일 평가 행의 비교이며, 많은 실험 중 가장 좋은 결과만 고른 성적이
          아닙니다.
        </p>
        <div className="direction-table">
          <table>
            <thead>
              <tr>
                <th>특징군</th>
                <th>추가 시 적중률 차이</th>
                <th>추가 시 Brier 차이 ↓</th>
                <th>제거 시 적중률 차이</th>
              </tr>
            </thead>
            <tbody>
              {(d.ablations || []).map((r: Data) => (
                <tr key={r.group}>
                  <td>{groups[r.group]}</td>
                  <td>{number(r.add.accuracy.delta * 100, 3)}%p</td>
                  <td>{number(r.add.brier.delta, 6)}</td>
                  <td>{number(r.drop.accuracy.delta * 100, 3)}%p</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <details>
        <summary>확신도가 높은 날만 보면 · 적중률과 제공 비율</summary>
        <p>
          개발 기간에 고정한 기준은 {number(d.threshold * 100)}%입니다. 평가
          결과를 보고 기준을 바꾸지 않았습니다. 해당 기준의 적중률{" "}
          {number(high.accuracy == null ? null : high.accuracy * 100, 2)}%, 제공
          비율 {number(high.coverage * 100, 2)}%, 표본 {number(high.n)}건입니다.
        </p>
        <div className="direction-table">
          <table>
            <thead>
              <tr>
                <th>확신도 기준</th>
                <th>표본</th>
                <th>제공 비율</th>
                <th>실제 방향 적중률</th>
                <th>날짜 블록 95% 범위</th>
              </tr>
            </thead>
            <tbody>
              {d.confidence.map((r: Data) => (
                <tr key={r.threshold}>
                  <td>{number(r.threshold * 100)}% 이상</td>
                  <td>{number(r.n)}</td>
                  <td>{number(r.coverage * 100, 2)}%</td>
                  <td>
                    {number(r.accuracy == null ? null : r.accuracy * 100, 2)}%
                  </td>
                  <td>
                    {r.accuracy_block_ci95
                      ? `${number(r.accuracy_block_ci95[0] * 100, 2)}~${number(r.accuracy_block_ci95[1] * 100, 2)}%`
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p>
          보합 제외 조건부 확률입니다. 높은 확률로 표시한 일부 표본의 적중률은
          전체 거래일의 성능과 다릅니다. 가격 경로 자체의 발생 확률은 검증하지
          않았습니다.
        </p>
        <h4>예측한 상승 확률과 실제 상승 비율</h4>
        <div className="direction-table">
          <table>
            <thead>
              <tr>
                <th>확률 구간</th>
                <th>표본</th>
                <th>평균 상승 예측</th>
                <th>실제 상승 비율</th>
              </tr>
            </thead>
            <tbody>
              {d.standalone.calibration.map((r: Data) => (
                <tr key={r.low}>
                  <td>
                    {number(r.low * 100)}~{number((r.low + 0.1) * 100)}%
                  </td>
                  <td>{number(r.n)}</td>
                  <td>{number(r.predicted * 100, 2)}%</td>
                  <td>{number(r.observed * 100, 2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <details>
        <summary>종목·국면별 안정성과 성공·실패 사례</summary>
        <label>
          연구 종목{" "}
          <select
            aria-label="금융 통계 연구 종목"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
          >
            {Object.keys(d.by_symbol)
              .sort()
              .map((s) => (
                <option key={s} value={s}>
                  {(
                    {
                      "005930": "삼성전자",
                      "000660": "SK하이닉스",
                      "005380": "현대차",
                    } as Data
                  )[s] || s}
                </option>
              ))}
          </select>
        </label>
        <Metrics
          rows={[
            { ...d.by_symbol[symbol], name: symbol },
            ...Object.entries(d.by_regime).map(([k, v]) => ({
              ...(v as Data),
              name:
                k === "high_vol"
                  ? "고변동성 국면"
                  : k === "low_vol"
                    ? "저변동성 국면"
                    : "국면 자료 부족",
            })),
          ]}
        />
        <p>
          국면은 당시 관측 자료로 필터링했습니다. 업종의 과거 구성 자료가 없어
          실제 업종 분석으로 표시하지 않습니다.
        </p>
        {d.examples
          .filter((r: Data) => r.symbol === symbol)
          .map((r: Data) => (
            <p key={r.date}>
              {r.date} · 상승 {number(r.p_up * 100, 1)}% · 실제 수익률{" "}
              {number(r.actual_return * 100, 2)}% ·{" "}
              {r.correct ? "방향 적중" : "방향 실패"} · 변동폭{" "}
              {r.range_inside ? "구간 안" : "구간 밖"}
            </p>
          ))}
      </details>
      <details>
        <summary>연구 구조·학습 전략 전체 비교</summary>
        <Metrics
          rows={d.results.map((r: Data) => ({ ...r, name: mathName(r.name) }))}
        />
        <p>
          CRAFT는 한국 데이터용 축소 구조이며, Qlib은 선택한 가격 통계 특징을
          구현했습니다. StockIT와 FusionLSTM-CNF의 뉴스 모달리티는 실제 과거
          자료 부족으로 재현하지 않았습니다. 원 논문의 성능을 이 표에 가져오지
          않았습니다.
        </p>
      </details>
      <details>
        <summary>변동폭 결합과 독립 검증의 한계</summary>
        <p>
          기존 nested7_tail50 유지 · 이 평가 행의 포함률{" "}
          {number(d.standalone.range_coverage * 100, 2)}% · 평균 구간 폭{" "}
          {number(d.standalone.range_width, 3)}%p · Interval Score{" "}
          {number(d.standalone.interval_score, 4)}.
        </p>
        <p>
          변동폭 추가 비교의 동일 구조 기준: {mathName(d.range_comparator)}.
          적중률 차이 {number(d.range_comparison.accuracy.delta * 100, 3)}%p.
          수학적으로 복잡한 피처를 추가했다는 사실만으로 개선을 의미하지
          않습니다.
        </p>
        <p>
          새 독립 평가: 아직 없음. 기존에 열어본 기간을 새 시험으로 재명명하지
          않았습니다. 미래에 사전 발행한 방향 예측으로 검증하기 전에는 운영
          성능을 입증했다고 표시하지 않습니다.
        </p>
      </details>
    </section>
  );
}
