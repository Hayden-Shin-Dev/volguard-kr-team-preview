import { useEffect, useState } from "react";
import { api, number, type Data } from "./api";
import { DirectionComparison } from "./DirectionComparison";
import { DirectionIntegration } from "./DirectionIntegration";
import { DirectionMath } from "./DirectionMath";
import { NewInformation } from "./NewInformation";
import { ProbabilityEvidence } from "./ProbabilityEvidence";
import { SelectiveDirection } from "./SelectiveDirection";
import { EventDirection } from "./EventDirection";
import { TechnicalResearch } from "./TechnicalResearch";
import { DirectionForward } from "./DirectionForward";
import { HistoricalReliability } from "./HistoricalReliability";
import { CalibrationStudy } from "./CalibrationStudy";
import { JointStudy } from "./JointStudy";
import { ProbabilityQuality } from "./ProbabilityQuality";

const candidateNames: Data = {
  original_logistic: "기존 방향 모델",
  relative_logistic: "시장·상대 수익률 추가",
  recent_logistic: "최근 자료 가중 회귀",
  relative_tree7: "시장·상대 수익률 트리",
  recent_tree7: "최근 자료 가중 트리",
  recent_tree15: "최근 자료 가중 확장 트리",
};

function DevelopmentProgress({ data }: { data: Data }) {
  const d = data.direction_development,
    r = data.range_adaptation,
    q = data.agent_quality;
  if (!d?.results) return null;
  const best = d.results.find((x: Data) => x.candidate === d.selected);
  return (
    <div className="development-progress">
      <div className="development-intro">
        <span className="eyebrow">개선 기록 · 개발 구간</span>
        <h3>무엇이 좋아졌고, 무엇이 남았는지</h3>
        <p>
          새 실험은 2023년까지의 개발 자료만 사용했습니다. 기존 2024년 방향
          평가와 운영 예측을 보존했습니다.
        </p>
      </div>
      <div className="development-metrics">
        <article>
          <small>방향 확률 오차 감소</small>
          <strong>
            {number(d.brier_improvement_pct, 2)}
            <em>%</em>
          </strong>
          <span>기존 방향 모델 대비 · 운영 확률은 검증 대기</span>
        </article>
        <article>
          <small>방향 분류 AUC</small>
          <strong>{number(best?.roc_auc, 4)}</strong>
          <span>
            기존 {number(d.baseline?.roc_auc, 4)} · 2021~2023 개발 비교
          </span>
        </article>
        <article>
          <small>AI 출력 검증 제외</small>
          <strong>
            {q?.current?.rejected ?? "—"}
            <em>건</em>
          </strong>
          <span>
            직전 기준 실행 {q?.previous?.rejected ?? "—"}건 · 개별 실행 결과
          </span>
        </article>
      </div>
      <details>
        <summary>방향 모델 6개 후보 비교</summary>
        <p>
          낮은 Brier가 더 좋습니다. 최고 후보를 선택한 개발 결과이므로 새 시험
          성적을 의미하지 않습니다.
        </p>
        <table>
          <thead>
            <tr>
              <th>후보</th>
              <th>AUC</th>
              <th>Brier</th>
              <th>Log Loss</th>
              <th>개발 관문</th>
            </tr>
          </thead>
          <tbody>
            {d.results.map((x: Data) => (
              <tr
                key={x.candidate}
                className={x.candidate === d.selected ? "selected-result" : ""}
              >
                <td>{candidateNames[x.candidate]}</td>
                <td>{number(x.roc_auc, 4)}</td>
                <td>{number(x.brier, 6)}</td>
                <td>{number(x.log_loss, 6)}</td>
                <td>{x.development_gate_pass ? "통과" : "추가 개선 필요"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p>
          상승·하락이 있는 날만 평가합니다. 최근 자료를 더 강조하거나 모델을
          복잡하게 만드는 것만으로 성능이 좋아지지는 않았습니다.
        </p>
        <h4>선택 후보의 확률 구간별 표본</h4>
        <table>
          <thead>
            <tr>
              <th>평균 예측 확률</th>
              <th>실제 상승 비율</th>
              <th>표본 수</th>
            </tr>
          </thead>
          <tbody>
            {best?.calibration_bins?.map((b: Data, i: number) => (
              <tr key={i}>
                <td>{number(b.predicted * 100, 2)}%</td>
                <td>{number(b.observed * 100, 2)}%</td>
                <td>{number(b.n)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
      {r?.candidates && (
        <details>
          <summary>변동폭 · 과거 오차로 하단과 상단을 따로 보정한 실험</summary>
          <p>
            종목별 직전 오차 120개가 모인 뒤 보정합니다.{" "}
            {r.improving_candidates.length
              ? "개발 기준을 만족한 후보가 있습니다."
              : "폭 축소와 점수 개선을 함께 검증한 후보가 없어 기존 모델을 유지합니다."}
          </p>
          <table>
            <thead>
              <tr>
                <th>참조 거래일</th>
                <th>보정 비중</th>
                <th>포함률</th>
                <th>구간 폭</th>
                <th>Interval Score</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td colSpan={2}>기존 개선 모델</td>
                <td>{number(r.baseline.coverage * 100, 2)}%</td>
                <td>{number(r.baseline.width, 4)}</td>
                <td>{number(r.baseline.score, 4)}</td>
              </tr>
              {r.candidates.map((x: Data) => (
                <tr key={`${x.window}-${x.strength}`}>
                  <td>{x.window}일</td>
                  <td>{x.strength * 100}%</td>
                  <td>{number(x.coverage * 100, 2)}%</td>
                  <td>{number(x.width, 4)}</td>
                  <td>{number(x.score, 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
      {q?.current && (
        <details>
          <summary>AI 조사 · 원문 선택과 오류 수정 기록</summary>
          <p>
            원문 발췌를 직접 연결하고 수치·단위·기업을 대조합니다. 검증에 실패한
            답변은 한 번 수정할 수 있으며, 계속 실패하면 제외합니다.
          </p>
          <p>
            현재 실행: 근거 주장 {q.current.claims}개 · 교차 검토{" "}
            {q.current.reviews}개 · 반론 답변 {q.current.responses}개 · 수정{" "}
            {q.current.repairs ?? 0}회 · 최종 제외 {q.current.rejected}개 · 실행
            실패 {q.current.failed_stages}개.
          </p>
          <p>
            실제 API 요청 {q.current.api_calls ?? "—"}회. 한 조사당 기본 8회,
            검증 수정 최대 3회로 제한합니다. 한 번의 비교이므로 일반적인
            성공률로 해석하지 않습니다.
          </p>
          <p>
            원문 대조는 의미·인과관계의 타당성을 보장하지 않습니다. 공시 수집
            범위와 자료 부족은 별도로 표시합니다.
          </p>
        </details>
      )}
    </div>
  );
}
export function ResearchValidation() {
  const [data, setData] = useState<Data | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    api("/platform-validation")
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);
  if (!data) return <p>{error || "추가 실험 결과 확인 중"}</p>;
  const d = data.direction,
    m = d.metrics?.selected_calibrated,
    cal = m?.calibration;
  return (
    <section className="analysis-flow">
      <header className="flow-heading">
        <div>
          <span className="eyebrow">독립적인 두 예측 대상</span>
          <h3>변동폭 개선 실험과 방향 확률 검증</h3>
        </div>
      </header>
      <ProbabilityQuality />
      <details><summary>전향 예측 발행·채점 상세</summary><DirectionForward /></details>
      <details><summary>시장·종목 분리 실험과 날짜 집중 원인</summary><JointStudy /></details>
      <details><summary>이전 확률 보정 재학습 결과</summary><CalibrationStudy /></details>
      <details><summary>이전 고정 모델의 과거 확률 비교</summary><HistoricalReliability /></details>
      <TechnicalResearch />
      <details><summary>이전 단순 사건 조건부 연구</summary><EventDirection /></details>
      <details><summary>이전 일봉 신호 연구</summary><SelectiveDirection endpoint="/paper-signals" /></details>
      <details><summary>시장 국면별 모델 개편 비교</summary><SelectiveDirection /></details>
      <details><summary>이전 모델의 확률 보정 근거</summary><ProbabilityEvidence /></details>
      <details><summary>새 금융정보 추가 실험</summary><NewInformation /></details>
      <details>
        <summary>이전 금융 통계 특징군 연구</summary>
        <DirectionMath />
      </details>
      <details>
        <summary>이전 변동폭 결합 연구 기록</summary>
        <DirectionIntegration />
      </details>
      <details>
        <summary>이전 7종 방향 모델 비교 기록</summary>
        <DirectionComparison />
      </details>
      <details>
        <summary>이전 확률 오차 중심 실험 기록</summary>
        <DevelopmentProgress data={data} />
      </details>
      <div className="flow-detail">
        <h3>변동폭 · 기존 개선 모델 유지</h3>
        <p>
          기존 개발 검증 {data.range.baseline?.n?.toLocaleString()}건. 추가 축소
          실험에서 구간 폭과 Interval Score를 동시에 개선한 후보가 없습니다.
        </p>
        <table>
          <thead>
            <tr>
              <th>폭 배율</th>
              <th>포함률</th>
              <th>평균 구간 폭</th>
              <th>Interval Score</th>
            </tr>
          </thead>
          <tbody>
            {data.range.candidates?.map((r: Data) => (
              <tr key={r.factor}>
                <td>{r.factor}×</td>
                <td>{number(100 * r.coverage, 2)}%</td>
                <td>{number(r.width, 4)}%p</td>
                <td>{number(r.score, 4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <h3>방향 · 일반 화면의 확률 표시 제한</h3>
        <p>
          {d.reason}. {d.evaluation_period}. 보합을 제외한 상승·하락 분류입니다.
        </p>
        {m && (
          <>
            <table>
              <thead>
                <tr>
                  <th>방법</th>
                  <th>정확도</th>
                  <th>균형 정확도</th>
                  <th>AUC</th>
                  <th>PR-AUC</th>
                  <th>Brier</th>
                  <th>Log Loss</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(d.metrics).map(([k, v]) => {
                  const r = v as Data;
                  return (
                    <tr key={k}>
                      <td>
                        {
                          (
                            {
                              selected_calibrated: "선택 모델 + 확률 보정",
                              training_prior: "학습 상승 빈도",
                              fixed_50: "50% 고정",
                              logistic: "로지스틱 회귀",
                            } as Data
                          )[k]
                        }
                      </td>
                      <td>{number(r.accuracy * 100, 2)}%</td>
                      <td>{number(r.balanced_accuracy * 100, 2)}%</td>
                      <td>{number(r.roc_auc, 4)}</td>
                      <td>{number(r.pr_auc, 4)}</td>
                      <td>{number(r.brier, 5)}</td>
                      <td>{number(r.log_loss, 5)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p>
              평가 {number(m.n)}건 · 학습 2010~2021 → 선택 2022 → 보정 2023 →
              최종 회고 평가 2024. 기존 변동폭 잠금 시험 2025~2026은 사용하지
              않았습니다.
            </p>
            <h4>예측한 확률과 실제 상승 비율</h4>
            <svg
              viewBox="0 0 360 250"
              width="360"
              style={{ maxWidth: "100%" }}
              role="img"
              aria-label="방향 확률 보정 곡선"
            >
              <path
                d="M35 210L325 210M35 210L35 15"
                fill="none"
                stroke="#c9bfd7"
              />
              <path d="M35 210L325 15" stroke="#c9bfd7" strokeDasharray="4 4" />
              {cal.predicted.map((p: number, i: number) => (
                <circle
                  key={i}
                  cx={35 + p * 290}
                  cy={210 - cal.observed[i] * 195}
                  r="4"
                  fill="#8967b4"
                />
              ))}
              <text x="130" y="242" fontSize="12" fill="#8d8097">
                예측 확률 0 → 1
              </text>
              <text x="40" y="12" fontSize="11" fill="#8d8097">
                실제 상승 비율
              </text>
            </svg>
            <p>
              점은 확률 구간별 관측 평균입니다. 극단 구간은 표본이 적을 수
              있으며, 그래프만으로 신뢰성을 보장하지 않습니다.
            </p>
          </>
        )}
        <details>
          <summary>검증 한계</summary>
          <p>
            현재 구성 종목의 생존편향, 과거 공표 시각 가정, 당시 개정 이력
            미확보가 있습니다. 업종 이력은 검증된 분류 자료가 없어 평가를
            제한했습니다. 과거 일봉 재생과 가상 장중 경로는 검증된 장중 예측
            모델을 의미하지 않습니다.
          </p>
        </details>
      </div>
    </section>
  );
}
