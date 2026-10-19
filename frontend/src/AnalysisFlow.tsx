import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  FlaskConical,
  Pause,
  Play,
  RotateCcw,
  Sparkles,
  X,
} from "lucide-react";
import { api, number, type Data } from "./api";
import "./analysis-flow.css";
import { NextDayCanvas } from "./NextDayCanvas";
import { DirectionComparison } from "./DirectionComparison";
import { DirectionIntegration } from "./DirectionIntegration";

function PathCanvas({
  before,
  future,
  step,
  prediction,
  virtual,
  actual,
}: {
  before: Data[];
  future: Data[];
  step: number;
  prediction: Data;
  virtual: boolean;
  actual?: Data | null;
}) {
  const past = before.slice(virtual ? -4 : -14),
    all = [
      ...past,
      ...future.slice(0, virtual ? future.length : step),
      ...(actual ? [actual] : []),
    ];
  if (!all.length) return null;
  const low = Math.min(...all.map((r) => r.low)) * 0.995,
    high = Math.max(...all.map((r) => r.high)) * 1.005;
  const w = 1000,
    top = 44,
    bottom = 298,
    left = 18,
    right = 82,
    count = past.length + future.length + 2;
  const dx = (w - left - right) / count,
    y = (v: number) =>
      bottom - ((v - low) / (high - low || 1)) * (bottom - top);
  const x = (i: number) => left + dx * (i + 0.5),
    boundary = x(past.length) - dx / 2;
  const candle = (r: Data, i: number, ghost: boolean) => {
    const color = ghost ? "#8a6bc7" : r.close >= r.open ? "#d65358" : "#5576dd";
    return (
      <g
        key={`${ghost ? "v" : "r"}-${i}`}
        className={ghost ? "path-candle" : ""}
      >
        <line x1={x(i)} x2={x(i)} y1={y(r.high)} y2={y(r.low)} stroke={color} />
        <rect
          x={x(i) - dx * 0.29}
          width={dx * 0.58}
          y={y(Math.max(r.open, r.close))}
          height={Math.max(1.5, Math.abs(y(r.open) - y(r.close)))}
          fill={ghost ? "#e3d9f4" : color}
          stroke={ghost ? color : "none"}
          rx=".5"
        />
      </g>
    );
  };
  return (
    <svg
      className="path-canvas"
      viewBox="0 0 1000 390"
      role="img"
      aria-label={
        virtual ? "과거 실제 일봉과 보라색 가상 경로" : "과거 실제 일봉 재생"
      }
    >
      <defs>
        <linearGradient id="future-wash" x1="0" x2="1">
          <stop stopColor="#f1ecfa" />
          <stop offset="1" stopColor="#fcfaff" />
        </linearGradient>
      </defs>
      <rect
        x={boundary}
        y="12"
        width={w - right - boundary}
        height="310"
        fill="url(#future-wash)"
        rx="12"
      />
      {[0, 0.25, 0.5, 0.75, 1].map((t) => {
        const v = low + (high - low) * t;
        return (
          <g key={t}>
            <line
              x1={left}
              x2={w - right}
              y1={y(v)}
              y2={y(v)}
              stroke="#ecebf1"
            />
            <text x={w - right + 12} y={y(v) + 4} fill="#898595" fontSize="11">
              {number(v)}
            </text>
          </g>
        );
      })}
      <line
        x1={boundary}
        x2={boundary}
        y1="20"
        y2="312"
        stroke="#bcb0d4"
        strokeDasharray="4 5"
      />
      <text x="24" y="28" className="svg-caption">
        실제 확정 일봉
      </text>
      <text x={boundary + 16} y="28" className="svg-caption">
        {virtual
          ? "가정한 다음 거래일 · 발생 확률 미부여"
          : "선택일 이후 실제 일봉"}
      </text>
      {past.map((r, i) => candle(r, i, false))}
      {future.slice(0, step).map((r, i) => candle(r, past.length + i, virtual))}
      <text x="24" y="318" className="svg-caption">
        {past[0]?.time}
      </text>
      <text x={boundary + 12} y="318" className="svg-caption">
        {virtual ? "가상 구간 1 → 24" : future[0]?.time}
      </text>
      <rect x="18" y="336" width="900" height="42" fill="#f8f5fc" rx="8" />
      <text x="32" y="361" className="svg-caption">
        ML 고저 변동폭 90% 구간
      </text>
      <line
        x1="266"
        x2="600"
        y1="357"
        y2="357"
        stroke="#b49adb"
        strokeWidth="7"
        strokeLinecap="round"
      />
      <circle
        cx={
          266 +
          (334 * (prediction.median - prediction.low)) /
            (prediction.high - prediction.low || 1)
        }
        cy="357"
        r="6"
        fill="#7251b4"
      />
      <text x="620" y="361" fill="#574175" fontSize="12">
        {number(prediction.low, 2)} ~ {number(prediction.high, 2)}% · 중앙{" "}
        {number(prediction.median, 2)}%
      </text>
      {actual && (
        <g>
          <line
            x1="900"
            x2="900"
            y1={y(actual.high)}
            y2={y(actual.low)}
            stroke="#168d82"
            strokeWidth="2"
          />
          <rect
            x="894"
            width="12"
            y={y(Math.max(actual.open, actual.close))}
            height={Math.max(2, Math.abs(y(actual.open) - y(actual.close)))}
            fill="#40aa9a"
          />
          <text x="818" y="42" className="svg-caption">
            실제 장중 캔들
          </text>
          <text x="32" y="385" className="svg-caption">
            실제 수신 고저 변동폭{" "}
            {number(100 * (actual.high / actual.low - 1), 2)}% · 가상 경로와
            별도 관측
          </text>
        </g>
      )}
    </svg>
  );
}

export function AnalysisFlow({
  stock,
  analysis,
  onAnalysis,
  onActive,
  onResearch,
  onPortfolio,
  live,
}: {
  stock: Data;
  analysis: Data | null;
  onAnalysis: (a: Data | null) => void;
  onActive: (v: boolean) => void;
  onResearch: () => void;
  onPortfolio: () => void;
  live: Data | null;
}) {
  const [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [mode, setMode] = useState("live"),
    [archive, setArchive] = useState<Data | null>(null),
    [basis, setBasis] = useState("development"),
    [day, setDay] = useState(""),
    [record, setRecord] = useState<Data | null>(null),
    [step, setStep] = useState(0),
    [playing, setPlaying] = useState(false),
    [speed, setSpeed] = useState(1);
  const generation = useRef(0);
  useEffect(() => {
    generation.current++;
    setBusy(false);
    setMode("live");
    setArchive(null);
    setRecord(null);
    setPlaying(false);
    setStep(0);
    setError("");
  }, [stock.symbol]);
  const a = analysis?.symbol === stock.symbol ? analysis : null;
  const simulation = mode === "live" ? a?.scenario : null;
  const future: Data[] =
    mode === "replay" ? record?.actual_after || [] : simulation?.candles || [];
  const steps = mode === "replay" ? future.length : 24;
  useEffect(() => {
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    setStep(reduced && mode === "live" ? 24 : 0);
    setPlaying(!reduced);
  }, [a?.id, record?.target_date]);
  useEffect(() => {
    if (!playing || step >= steps) {
      if (step >= steps) setPlaying(false);
      return;
    }
    const id = setTimeout(
      () => setStep((s) => s + 1),
      (mode === "replay" ? 650 : 80) / speed,
    );
    return () => clearTimeout(id);
  }, [playing, step, speed, steps, mode]);
  async function create(
    direction = "up",
    magnitude = "median",
    parent?: string,
  ) {
    const request = ++generation.current;
    setBusy(true);
    setError("");
    setMode("live");
    try {
      const value = await api("/analyses", {
        method: "POST",
        body: JSON.stringify({
          symbol: stock.symbol,
          direction,
          magnitude,
          parent_id: parent || null,
        }),
      });
      if (request === generation.current) onAnalysis(value);
    } catch (e) {
      if (request === generation.current) setError((e as Error).message);
    } finally {
      if (request === generation.current) setBusy(false);
    }
  }
  async function history() {
    setMode("replay");
    setPlaying(false);
    setError("");
    if (archive) return;
    const request = ++generation.current;
    setBusy(true);
    try {
      const r = await api(`/stocks/${stock.symbol}/prediction-history`);
      if (request === generation.current) {
        setArchive(r);
        setDay(r.development.rows.at(-1)?.time || "");
      }
    } catch (e) {
      if (request === generation.current) setError((e as Error).message);
    } finally {
      if (request === generation.current) setBusy(false);
    }
  }
  async function loadReplay() {
    setBusy(true);
    setError("");
    const request = ++generation.current;
    try {
      const r = await api(`/stocks/${stock.symbol}/replay/${day}`);
      if (request === generation.current) setRecord(r);
    } catch (e) {
      if (request === generation.current) setError((e as Error).message);
    } finally {
      if (request === generation.current) setBusy(false);
    }
  }
  const pred = mode === "replay" ? record?.prediction : a?.prediction;
  const before =
    mode === "replay"
      ? record?.before || []
      : (a?.market.chart_candles || a?.market.candles || []).filter(
          (r: Data) => r.time <= a?.anchor_date,
        );
  const active = !!(mode === "replay" ? record : a);
  useEffect(() => {
    onActive(active);
  }, [active]);
  const risk = a?.portfolio.result;
  return (
    <section
      className={`analysis-flow ${active ? "expanded" : ""}`}
      aria-label="통합 미래 시나리오 분석"
    >
      <header className="flow-heading">
        <div>
          <span className="eyebrow">변동폭 예측과 방향 연구를 한눈에</span>
          <h3>
            {mode === "replay"
              ? "그날의 예측, 이후의 실제 움직임"
              : "내일의 방향과 변동폭"}
          </h3>
        </div>
        <div className="flow-actions">
          <button className="primary" disabled={busy} onClick={() => create()}>
            <Sparkles size={16} />
            {busy
              ? "자료 연결 중"
              : a
                ? "새 기준으로 분석"
                : "방향·변동폭 분석"}
          </button>
          <button onClick={history} disabled={busy}>
            <FlaskConical size={16} />
            과거 거래일 재생
          </button>
          {active && (
            <button
              aria-label="시나리오 닫기"
              onClick={() => {
                setRecord(null);
                onAnalysis(null);
                setPlaying(false);
                setMode("live");
              }}
            >
              <X size={16} />
            </button>
          )}
        </div>
      </header>
      {error && (
        <p role="alert" className="inline-warning">
          {error}
        </p>
      )}
      {mode === "replay" && (
        <div className="replay-picker">
          <label>
            기록 구분{" "}
            <select
              value={basis}
              onChange={(e) => {
                setBasis(e.target.value);
                setRecord(null);
                setDay(archive?.[e.target.value]?.rows.at(-1)?.time || "");
              }}
            >
              <option value="development">개발 검증 · 회고</option>
              <option value="forward">실제 사전 발행</option>
            </select>
          </label>
          <label>
            예측 대상일{" "}
            <select
              aria-label="재생 거래일"
              value={day}
              onChange={(e) => setDay(e.target.value)}
            >
              {[...(archive?.[basis]?.rows || [])].reverse().map((r: Data) => (
                <option key={r.time}>{r.time}</option>
              ))}
            </select>
          </label>
          <button disabled={!day || busy} onClick={loadReplay}>
            그날부터 재생 <Play size={14} />
          </button>
          <span>
            {archive?.[basis]?.rows.length === 0
              ? "아직 채점 가능한 사전 발행 기록이 없습니다."
              : "미래 실제값은 재생과 함께 한 개씩 공개합니다."}
          </span>
        </div>
      )}
      {active && (
        <>
          {simulation && (
            <div className="scenario-choices">
              <div>
                <small>포트폴리오에 적용할 방향</small>
                {[
                  ["up", "상승"],
                  ["down", "하락"],
                ].map(([v, n]) => (
                  <button
                    disabled={busy}
                    className={simulation.direction === v ? "selected" : ""}
                    key={v}
                    onClick={() => create(v, simulation.magnitude, a!.id)}
                  >
                    {n}{a?.direction.probability != null && <strong className="choice-probability"> {number(100 * (v === "up" ? a.direction.probability : 1-a.direction.probability), 1)}%</strong>}
                  </button>
                ))}
              </div>
              <span>{a!.target_date}</span>
            </div>
          )}
          <div
            className={`path-viewport ${mode === "live" ? "tomorrow-viewport" : ""}`}
            tabIndex={0}
            aria-label={mode === "live" ? "상승과 하락 변동폭 비교" : "과거 실제 일봉 재생"}
          >
            {mode === "live" ? (
              <NextDayCanvas
                before={before}
                analysis={a!}
                progress={step / 24}
                actual={live?.time === a?.target_date ? live : null}
              />
            ) : (
              <PathCanvas
                before={before}
                future={future}
                step={step}
                prediction={pred!}
                virtual={mode === "live"}
                actual={live?.time === a?.target_date ? live : null}
              />
            )}
          </div>
          <small className={`mobile-chart-hint ${mode === "live" ? "hidden" : ""}`}>
            차트를 좌우로 움직여 전체 경로를 확인하세요.
          </small>
          <div className="playback-bar">
            <button
              aria-label={playing ? "재생 일시정지" : "재생 시작"}
              onClick={() => {
                if (step >= steps) setStep(0);
                setPlaying(!playing);
              }}
            >
              {playing ? <Pause size={17} /> : <Play size={17} />}
            </button>
            <button
              aria-label="처음부터 재생"
              onClick={() => {
                setStep(0);
                setPlaying(true);
              }}
            >
              <RotateCcw size={16} />
            </button>
            <input
              aria-label="재생 위치"
              type="range"
              min="0"
              max={steps}
              value={step}
              onChange={(e) => {
                setStep(+e.target.value);
                setPlaying(false);
              }}
            />
            <span>
              {mode === "replay"
                ? `${step}/${steps} 거래일`
                : `변동폭 표시 ${Math.round((step / 24) * 100)}%`}
            </span>
            <select
              aria-label="재생 속도"
              value={speed}
              onChange={(e) => setSpeed(+e.target.value)}
            >
              {[0.5, 1, 2, 4].map((v) => (
                <option key={v} value={v}>
                  {v}×
                </option>
              ))}
            </select>
          </div>
          {mode === "replay" ? (
            <div className="replay-outcome">
              <strong>{record!.label}</strong>
              <p>{record!.note}</p>
              {step > 0 && (
                <p>
                  선택일 실제 변동폭 <b>{number(record!.score.actual, 2)}%</b> ·{" "}
                  {record!.score.inside ? "예측 구간 안" : "예측 구간 밖"} ·
                  Interval Score {number(record!.score.score, 3)}
                </p>
              )}
            </div>
          ) : (
            <>
              <div className="scenario-note">
                <span>포트폴리오 가정</span>
                <p>
                  아래 손익은 별도의 가상 시나리오 계산입니다. 방향 선택은 이 계산에만 적용되며, 위 ML 예측값과 확률을 바꾸지 않습니다.
                </p>
              </div>
              <div className="flow-impact">
                <div>
                  <small>등록한 포트폴리오</small>
                  <b>{number(a!.impact.total_krw)}원</b>
                </div>
                <div>
                  <small>선택 시나리오의 평가손익</small>
                  <b
                    className={a!.impact.pnl_krw < 0 ? "negative" : "positive"}
                  >
                    {number(a!.impact.pnl_krw)}원
                  </b>
                </div>
                <div>
                  <small>기존 1일 95% VaR</small>
                  <b>
                    {risk && ["passed", "cash_only"].includes(risk.status)
                      ? `${number(risk.var_krw)}원`
                      : "현재 검증된 결과 없음"}
                  </b>
                </div>
                <button onClick={onPortfolio}>
                  보유 위험 확인 <ArrowRight size={16} />
                </button>
              </div>
              <details className="flow-detail">
                <summary>방향 확률 검증 · 시나리오 가정 · 종목별 손익</summary>
                <p>
                  {a!.direction.reason} 모델 버전 {a!.direction.version?.slice(0, 12)}
                </p>
                <p>
                  {a!.direction.limitations?.join(" / ")}
                </p>
                <DirectionIntegration symbol={stock.symbol} compact />
                <DirectionComparison symbol={stock.symbol} compact />
                <p>{simulation.validation}</p>
                {!a!.portfolio.saved.holdings[stock.symbol] && (
                  <p>
                    현재 구성에 선택한 종목이 없어 이 시나리오의 직접 평가손익은
                    0원입니다.
                  </p>
                )}
                <p>{a!.impact.note}</p>
                {a!.impact.contributions.map((r: Data) => (
                  <p key={r.symbol}>
                    {r.symbol} · 비중 {number(r.weight * 100, 1)}% · 시나리오
                    기여 {number(r.pnl_krw)}원
                  </p>
                ))}
                {!a!.impact.contributions.length && (
                  <p>
                    입력한 주식 보유가 없습니다. 예시 포트폴리오를 임의로 만들지
                    않습니다.
                  </p>
                )}
                <p>
                  분석 기준 {a!.created_at} · 스냅샷 {a!.id.slice(0, 8)}
                </p>
              </details>
              <div className="flow-next">
                <p>
                  <b>이 가정을 네 명의 분석가에게 전달합니다.</b>
                  <span>
                    같은 예측·시나리오·위험 결과를 기준으로 근거와 반론을
                    확인합니다.
                  </span>
                </p>
                <button className="primary" onClick={onResearch}>
                  이 분석으로 AI 교차 조사 <ArrowRight size={16} />
                </button>
              </div>
            </>
          )}
        </>
      )}
    </section>
  );
}

export function AnalysisContext({
  analysis,
  onMarket,
}: {
  analysis: Data;
  onMarket: () => void;
}) {
  return (
    <div className="analysis-context">
      <span className="context-dot" />
      <strong>
        {analysis.market.name || analysis.symbol} · {analysis.target_date} 분석
      </strong>
      <span>
        {
          ({ up: "상승", flat: "횡보", down: "하락" } as Data)[
            analysis.scenario.direction
          ]
        }{" "}
        가정 · 동일 스냅샷 유지
      </span>
      <button onClick={onMarket}>
        차트로 돌아가기 <ArrowRight size={14} />
      </button>
    </div>
  );
}
