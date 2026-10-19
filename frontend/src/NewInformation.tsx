import { useEffect, useState } from "react";
import { api, number, type Data } from "./api";

const names: Data = {
  baseline: "기존 입력 · 동일 학습 기준",
  nasdaq_vintage: "나스닥 · 당시 공개 자료",
  overseas_latest_delayed: "해외시장·미국 금리 · 지연 적용",
  fx_rates_latest_delayed: "환율·국내 금리 · 지연 적용",
  market_program: "시장 프로그램매매",
  market_investor: "시장 투자자 수급",
  sector_investor: "업종 간 수급 분포",
  execution_strength_core: "실제 체결강도",
  dart_events_core: "DART 공시 이벤트",
  earnings_core: "DART 보고 실적",
  previous_stock_flow_core: "기존 종목 수급 · 대조군",
  all_new_information: "위 정보 전체 결합",
};
function ResultTable({ study }: { study: Data }) {
  return (
    <>
      <p>
        {number(study.rows)}개 종목·거래일 · {number(study.symbols)}종목 · 개발
        기간에서 선택: {names[study.selected]}
      </p>
      <div className="direction-table">
        <table>
          <thead>
            <tr>
              <th>추가 입력</th>
              <th>방향 적중률</th>
              <th>기준 대비</th>
              <th>차이의 95% 구간</th>
              <th>Brier ↓</th>
              <th>Log Loss ↓</th>
              <th>새 입력 완비율</th>
            </tr>
          </thead>
          <tbody>
            {study.results.map((r: Data) => (
              <tr key={r.name}>
                <td>{names[r.name] || r.name}</td>
                <td>{number(r.accuracy * 100, 2)}%</td>
                <td>
                  {r.comparison
                    ? `${number(r.comparison.accuracy.delta * 100, 3)}%p`
                    : "기준"}
                </td>
                <td>
                  {r.comparison
                    ? `${number(r.comparison.accuracy.ci95[0] * 100, 3)}~${number(r.comparison.accuracy.ci95[1] * 100, 3)}%p`
                    : "—"}
                </td>
                <td>{number(r.brier, 6)}</td>
                <td>{number(r.log_loss, 6)}</td>
                <td>
                  {r.observed_coverage == null
                    ? "—"
                    : `${number(r.observed_coverage * 100, 1)}%`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p>
        같은 종목·거래일과 고정된 로지스틱 회귀로 입력 정보만 비교했습니다.
        결측은 학습 자료로만 보완하며, 새 입력이 모두 있는 표본의 별도 결과도
        보고서에 보존합니다.
      </p>
    </>
  );
}
export function NewInformation() {
  const [data, setData] = useState<Data | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    api("/new-information")
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
    return <p role="alert">새 금융정보 연구를 읽지 못했습니다: {error}</p>;
  if (!data || data.status === "pending") return null;
  return (
    <section className="direction-comparison">
      <span className="eyebrow">새 금융정보 · 실제 수집과 검증</span>
      <h3>새로운 정보가 더 정확한 방향을 알려줄까요?</h3>
      <p>
        가격 변환이나 모델 추가 없이 수급·해외시장·체결강도·공시를 실제로
        수집했습니다.{" "}
        <strong>
          2023년 재사용 개발 확인 결과이며, 운영 모델은 변경하지 않았습니다.
        </strong>
      </p>
      <ResultTable study={data.global} />
      <details>
        <summary>삼성전자·SK하이닉스·현대차 · 체결강도와 기업 이벤트</summary>
        <ResultTable study={data.core} />
        <p>
          이 비교는 세 종목만으로 학습·평가했습니다. 전체 유니버스의 약 53%와
          직접 비교하지 않습니다. 실적 성장률은 시장 예상 대비 실적 서프라이즈가
          아닙니다.
        </p>
      </details>
      <details>
        <summary>실제 확보 범위와 시점 정합성</summary>
        <div className="direction-table">
          <table>
            <thead>
              <tr>
                <th>자료</th>
                <th>확보 범위</th>
                <th>사용 제한</th>
              </tr>
            </thead>
            <tbody>
              {(data.sources || []).map((s: Data) => (
                <tr key={s.name}>
                  <td>{s.name}</td>
                  <td>{s.range}</td>
                  <td>{s.limitation}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p>
          나스닥은 과거 시점별 저장 자료의 공개 날짜를 확인합니다. 나머지 현재
          조회값의 수정 이력을 확인하지 못한 항목은 지연을 두어 탐색하되, 엄격한
          과거 시점 검증을 통과했다고 표시하지 않습니다.
        </p>
      </details>
      <details>
        <summary>성능 해석과 남은 정보 공백</summary>
        <p>
          모든 예측의 기준은 거래일 18:30 한국시간입니다. 그 뒤의 미국 장 마감,
          다음 날 장중 체결, 이후 공시·정정 자료를 앞당겨 입력하지 않습니다.
          발표 시각이 없는 일별 자료는 당일 입력에서 제외합니다.
        </p>
        <p>
          과거 호가장, 장중 데이터의 충분한 공통 기간, 발표 당시의 실적
          예상치·뉴스 원문은 아직 확보되지 않았습니다. 제한된 표본의 좋은 수치를
          전체 방향 예측력으로 일반화하지 않습니다.
        </p>
      </details>
    </section>
  );
}
