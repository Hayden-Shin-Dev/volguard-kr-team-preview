import { Activity, Clock3 } from "lucide-react";
import { number, type Data } from "./api";
import { gaugeLabels } from "./domain.mjs";
export function RiskGauge({
  stock,
  prediction,
}: {
  stock: Data;
  prediction: Data | null;
}) {
  const g = stock.gauge || {},
    issued = stock.gauge_prediction,
    bar = stock.display_bar || stock.chart_candles?.at(-1);
  const active =
    stock.market_open &&
    g.status === "ready" &&
    Number.isFinite(g.consumption_pct);
  const observed = stock.market_open
    ? g.range_pct
    : bar?.low > 0
      ? 100 * (bar.high / bar.low - 1)
      : null;
  const day = stock.market_open ? stock.quote?.trade_date : bar?.time;
  const scale = active ? Math.max(issued.high * 1.25, observed * 1.1) : 1;
  return (
    <section
      className={`risk-comparison ${active && g.consumption_pct > 100 ? "breach" : ""}`}
      aria-label="오늘 변동폭과 예측 비교"
    >
      <div className="risk-heading">
        <Activity size={22} />
        <div>
          <h3>오늘 움직임, 예상보다 큰가요?</h3>
          <p>장중 고가와 저가의 차이를 오늘용 예측 상한과 비교합니다.</p>
        </div>
        <span className="risk-state">
          {active
            ? "실시간 비교"
            : stock.market_open
              ? "비교 대기"
              : "정규장 종료"}
        </span>
      </div>
      {active ? (
        <>
          <div className="risk-live-values">
            <div>
              <small>{day} 실제 변동폭</small>
              <b>{number(observed, 2)}%</b>
            </div>
            <span>÷</span>
            <div>
              <small>오늘 예측 상한</small>
              <b>{number(issued.high, 2)}%</b>
            </div>
            <span>=</span>
            <div>
              <small>상한 대비 도달률</small>
              <b>{number(g.consumption_pct, 1)}%</b>
            </div>
          </div>
          <div className="risk-axis">
            <div
              className="risk-expected"
              style={{
                left: `${(100 * issued.low) / scale}%`,
                width: `${(100 * (issued.high - issued.low)) / scale}%`,
              }}
            />
            <div
              className="risk-observed"
              style={{ width: `${(100 * observed) / scale}%` }}
            />
            <span
              className="risk-limit"
              style={{ left: `${(100 * issued.high) / scale}%` }}
            >
              예측 상한
            </span>
          </div>
          <p className="risk-explanation">
            {g.consumption_pct > 100
              ? "예측 상한을 넘었습니다."
              : g.consumption_pct >= 80
                ? "예측 상한의 80% 이상에 도달했습니다."
                : "아직 예측 상한보다 작은 움직임입니다."}{" "}
            80%부터 주의, 100% 초과 시 알림을 기록합니다. 남은 상승 여력이나
            가격 방향을 뜻하지 않습니다.
          </p>
        </>
      ) : (
        <>
          <div className="risk-wait-grid">
            <div>
              <small>
                {day || "오늘"} ·{" "}
                {stock.market_open ? "장중 관측" : "종료된 장의 실제 변동폭"}
              </small>
              <b>
                {number(observed, 2)}
                <em>%</em>
              </b>
              <span>(이날의 고가 ÷ 저가 − 1) × 100</span>
            </div>
            <div className="risk-next">
              <Clock3 size={20} />
              <div>
                <strong>
                  {prediction
                    ? `${prediction.target_date} 예측 ${number(prediction.low, 2)}~${number(prediction.high, 2)}%`
                    : "대상 거래일 예측 확인 중"}
                </strong>
                <p>
                  {!stock.market_open
                    ? prediction?.target_date === day
                      ? "해당 정규장이 종료됐습니다. 확정 일봉 채점 결과는 실제 발행 기록에서 확인할 수 있습니다."
                      : prediction
                        ? `위 ${day || "종료된 장"} 관측값과 날짜가 달라 도달률을 계산하지 않습니다. 대상 정규장이 시작되면 실제 움직임과 자동으로 비교합니다.`
                        : "해당 날짜에 사전 발행된 예측이 없어 비교하지 않습니다. 다음 예측 발행을 기다리고 있습니다."
                    : (gaugeLabels[g.status as keyof typeof gaugeLabels] ||
                        "시세 상태 확인 중") +
                      ". 연결·시세·당일 발행 예측이 확인되면 비교를 시작합니다."}
                </p>
              </div>
            </div>
          </div>
        </>
      )}
      <details className="risk-help">
        <summary>도달률은 어떻게 읽나요?</summary>
        <p>
          예측 상한이 4%이고 오늘 실제 고저 변동폭이 2%라면 도달률은 50%입니다.
          이는 계산 설명용 예시입니다. 장 초반 실제 변동폭이 예측 하한보다
          작다고 경보를 내지 않습니다.
        </p>
      </details>
    </section>
  );
}
