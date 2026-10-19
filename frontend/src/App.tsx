import { useEffect, useState } from "react";
import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  Bell,
  Check,
  ChevronRight,
  Compass,
  LoaderCircle,
  Plus,
  Search,
  ShieldCheck,
  Sparkles,
  Wallet,
  Star,
  X,
} from "lucide-react";
import { api, session, number, dateTime, type Data } from "./api";
import { CandleChart } from "./Chart";
import { RiskGauge } from "./RiskGauge";
import { AnalysisFlow, AnalysisContext } from "./AnalysisFlow";
import { ResearchValidation } from "./ResearchValidation";
import { Intelligence } from "./Intelligence";
import { liveCandle } from "./domain.mjs";

function Mark() {
  return (
    <svg viewBox="0 0 36 36" aria-label="VolGuard">
      <path
        d="M6 9L17 29L30 7M12 9L18 20L25 7"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
      />
      <circle cx="29" cy="27" r="2" fill="currentColor" />
    </svg>
  );
}
function Spark({ values, up }: { values: number[]; up: boolean }) {
  if (values.length < 2) return null;
  const low = Math.min(...values),
    range = Math.max(...values) - low || 1;
  return (
    <svg
      className={"spark " + (up ? "positive" : "negative")}
      viewBox="0 0 70 25"
      aria-hidden="true"
    >
      <path
        d={values
          .map(
            (v, i) =>
              `${i ? "L" : "M"}${(i / (values.length - 1)) * 70} ${23 - ((v - low) / range) * 21}`,
          )
          .join(" ")}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.2"
      />
    </svg>
  );
}
function ModelStatus({ data }: { data: Data | undefined }) {
  if (!data?.candidate) return null;
  return (
    <section className="model-status">
      <header>
        <strong>
          개선 ML ·{" "}
          {data.development_pass ? "개발 검증 통과" : "개발 검증 확인 필요"}
        </strong>
        <span>{data.evaluation_period}</span>
      </header>
      <div>
        <span>
          HAR 대비 구간 폭 <b>−{number(data.width_reduction * 100, 2)}%</b>
        </span>
        <span>
          Interval Score <b>−{number(data.score_reduction * 100, 2)}%</b>
        </span>
        <span>
          포함률 <b>{number(data.coverage * 100, 2)}%</b>
        </span>
      </div>
      <p>
        {data.candidate} · {data.status}. 위 수치는 개발 평가의 전체 행 평균
        비교입니다.
      </p>
      <details>
        <summary>평가 결과의 의미</summary>
        <p>
          기존 ML의 별도 잠금 시험 개선율은{" "}
          {number(data.original_improvement * 100, 2)}%로 5% 기준 미달입니다.
          개선 ML은 개발 검증을 통과해 기본 예측에 적용했습니다. 실제 미래
          성능은 별도로 관찰합니다. 서로 다른 모델·평가 기간의 결과입니다.
        </p>
      </details>
    </section>
  );
}

function Portfolio({
  universe,
  error,
}: {
  universe: Data[];
  error: (s: string) => void;
}) {
  const [rows, setRows] = useState<{ symbol: string; value: string }[]>([]),
    [cash, setCash] = useState("0"),
    [result, setResult] = useState<Data | null>(null),
    [busy, setBusy] = useState(false),
    [loaded, setLoaded] = useState(false);
  useEffect(() => {
    api("/portfolio")
      .then((d) => {
        setRows(
          Object.entries(d.saved.holdings).map(([symbol, value]) => ({
            symbol,
            value: String(value),
          })),
        );
        setCash(String(d.saved.cash));
        setResult(d.result);
        setLoaded(true);
      })
      .catch((e) => error(e.message));
  }, []);
  async function calculate() {
    if (
      rows.some((r) => !r.symbol) ||
      new Set(rows.map((r) => r.symbol)).size !== rows.length
    ) {
      error("보유 종목을 선택하고 중복을 제거해주세요.");
      return;
    }
    setBusy(true);
    try {
      setResult(
        await api("/portfolio", {
          method: "POST",
          body: JSON.stringify({
            holdings: Object.fromEntries(
              rows.map((r) => [r.symbol, Number(r.value)]),
            ),
            cash: Number(cash),
          }),
        }),
      );
    } catch (e) {
      error((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const total = rows.reduce(
      (s, r) => s + (Number(r.value) || 0),
      Number(cash) || 0,
    ),
    changed =
      !!result &&
      (JSON.stringify(
        Object.fromEntries(rows.map((r) => [r.symbol, Number(r.value)]).sort()),
      ) !==
        JSON.stringify(
          Object.fromEntries(Object.entries(result.holdings).sort()),
        ) ||
        Number(cash) !== result.cash);
  const usable = result && !changed,
    shown = usable && ["passed", "cash_only"].includes(result.status);
  return (
    <div className="portfolio-page page-enter">
      <header className="page-heading">
        <div>
          <div className="eyebrow">
            PORTFOLIO LAB <span>손실의 크기를 이해하는 공간</span>
          </div>
          <h1>
            내 자산이 마주할 <br />
            <em>다음 하루의 위험.</em>
          </h1>
          <p>
            보유 금액과 현금을 입력하면 같은 날짜의 수익률로 함께 평가합니다.
          </p>
        </div>
      </header>
      <div className="portfolio-layout">
        <section className="holding-editor">
          <div className="section-title">
            <h2>보유 구성</h2>
            <span>롱온리 · 원화</span>
          </div>
          {!loaded ? (
            <LoaderCircle className="spin" />
          ) : (
            rows.map((r, i) => (
              <div className="holding-row" key={i}>
                <select
                  aria-label={`보유 종목 ${i + 1}`}
                  value={r.symbol}
                  onChange={(e) =>
                    setRows(
                      rows.map((v, j) =>
                        j === i ? { ...v, symbol: e.target.value } : v,
                      ),
                    )
                  }
                >
                  <option value="">종목 선택</option>
                  {universe.map((s) => (
                    <option key={s.symbol} value={s.symbol}>
                      {s.name}
                    </option>
                  ))}
                </select>
                <label>
                  <span className="sr-only">평가금액</span>
                  <input
                    type="number"
                    min="0"
                    step="10000"
                    value={r.value}
                    onChange={(e) =>
                      setRows(
                        rows.map((v, j) =>
                          j === i ? { ...v, value: e.target.value } : v,
                        ),
                      )
                    }
                  />
                  <span>원</span>
                </label>
                <button
                  aria-label="보유 종목 제거"
                  onClick={() => setRows(rows.filter((_, j) => j !== i))}
                >
                  <X size={15} />
                </button>
              </div>
            ))
          )}
          <button
            className="text-action"
            onClick={() => setRows([...rows, { symbol: "", value: "" }])}
          >
            <Plus size={15} /> 보유 종목 추가
          </button>
          <div className="cash-row">
            <label htmlFor="cash">현금</label>
            <input
              id="cash"
              type="number"
              min="0"
              value={cash}
              onChange={(e) => setCash(e.target.value)}
            />
            <span>원</span>
          </div>
          <div className="portfolio-total">
            <span>총 평가금액</span>
            <b>
              {number(total)}
              <small>원</small>
            </b>
          </div>
          <button
            className="primary"
            disabled={busy || !loaded || total <= 0}
            onClick={calculate}
          >
            {busy ? (
              <LoaderCircle size={16} className="spin" />
            ) : (
              <ArrowRight size={16} />
            )}{" "}
            구성 저장 · 위험 분석
          </button>
          <p className="fineprint">
            사용자가 입력한 구성만 사용합니다. 계좌를 조회하거나 주문하지
            않습니다.
          </p>
        </section>
        <section className="risk-result">
          <div className="eyebrow">하루 손실 위험 · 95% VaR</div>
          <div className="var-value">
            {shown ? number(result.var_krw) : "—"}
            <small>{shown ? "원" : "검증 대기"}</small>
          </div>
          <p>
            {changed
              ? "구성이 변경되었습니다. 다시 분석하면 결과가 갱신됩니다."
              : result?.reason ||
                (!result
                  ? "보유 구성을 저장하면 검증을 시작합니다."
                  : shown
                    ? "정해진 백테스트 관문을 통과한 추정치입니다. 최대 손실 한도가 아닙니다."
                    : "이 구성은 검증 관문을 통과하지 않아 수치를 표시하지 않습니다.")}
          </p>
          <div className="risk-contour" aria-hidden="true">
            <svg viewBox="0 0 600 130">
              {Array.from({ length: 8 }, (_, i) => (
                <path
                  key={i}
                  d={`M0 ${120 - i * 2} C190 ${120 - i * 5} 190 ${20 + i * 4} 300 ${15 + i * 7} S410 ${120 - i * 4} 600 ${120 - i * 2}`}
                  fill="none"
                  stroke="currentColor"
                  strokeOpacity={0.1 + i * 0.08}
                />
              ))}
            </svg>
          </div>
          {usable && (
            <div className="risk-stats">
              <div>
                <span>외표본 평가</span>
                <b>{number(result.n)}일</b>
              </div>
              <div>
                <span>실제 초과율</span>
                <b>
                  {number(
                    result.exception_rate == null
                      ? null
                      : result.exception_rate * 100,
                    2,
                  )}
                  %
                </b>
              </div>
              <div>
                <span>검증 결과</span>
                <b>
                  {result.status === "cash_only"
                    ? "주식 노출 없음"
                    : shown
                      ? "관문 통과"
                      : "사용 제한"}
                </b>
              </div>
            </div>
          )}
          <p className="fineprint">
            현재 비중을 과거에 적용한 가상 포트폴리오 검증 · 변동성 보정 역사적
            방식 · 종목별 VaR 합산 아님
          </p>
        </section>
      </div>
      <section className="stress-section">
        <div className="section-title">
          <h2>만약 시장에 충격이 온다면</h2>
          <span>발생 확률을 부여하지 않는 시나리오</span>
        </div>
        {usable ? (
          result.stress?.map((s: Data, i: number) => (
            <div className="stress-row" key={i}>
              <span className="stress-index">0{i + 1}</span>
              <div>
                <h3>{s.scenario}</h3>
                <p>
                  {s.status === "hypothetical"
                    ? "모든 주식에 같은 하락률 적용"
                    : "과거 위기를 현재 구성에 적용"}
                </p>
              </div>
              <div className="stress-bar">
                <i
                  style={{
                    width: `${Math.min(100, (Math.abs(s.pnl_krw || 0) / Math.max(total * 0.5, 1)) * 100)}%`,
                  }}
                />
              </div>
              <b>
                {s.pnl_krw === null ? "자료 불완전" : `${number(s.pnl_krw)}원`}
              </b>
            </div>
          ))
        ) : (
          <p className="empty-copy">
            현재 구성의 위험 분석을 실행하면 스트레스 손익이 표시됩니다.
          </p>
        )}
      </section>
    </div>
  );
}

function Validation({ error }: { error: (s: string) => void }) {
  const [data, setData] = useState<Data | null>(null);
  useEffect(() => {
    api("/validation")
      .then(setData)
      .catch((e) => error(e.message));
  }, []);
  const ops = data?.operations || {};
  return (
    <div className="validation-page page-enter">
      <header className="page-heading">
        <div>
          <div className="eyebrow">TRUST & OBSERVABILITY</div>
          <h1>
            숫자 뒤의 <br />
            <em>검증을 공개합니다.</em>
          </h1>
          <p>과거 평가, 개발 실험, 실제 운영 관측은 서로 다른 근거입니다.</p>
        </div>
        <ShieldCheck className="heading-icon" size={74} strokeWidth={0.6} />
      </header>
      <section className="validation-band">
        <span className="eyebrow">실제 운영 인수</span>
        <h2>{ops.status === "passed" ? "운영 관문 통과" : "관찰 중"}</h2>
        <p>
          20거래일의 실제 장중 관측과 배치·화면 지연 기준을 모두 확인해야
          완료됩니다.
        </p>
        <div className="observation-days">
          {Array.from({ length: 20 }, (_, i) => (
            <span
              key={i}
              className={i < (ops.intraday_observed_days || 0) ? "filled" : ""}
            >
              {i + 1}
            </span>
          ))}
        </div>
        <div className="risk-stats">
          <div>
            <span>장중 관찰</span>
            <b>{number(ops.intraday_observed_days)} / 20일</b>
          </div>
          <div>
            <span>필수 자료 확보일</span>
            <b>{number(ops.eligible_days)}일</b>
          </div>
          <div>
            <span>배치 성공률</span>
            <b>
              {ops.batch_success_rate === null
                ? "측정 대기"
                : number(ops.batch_success_rate * 100, 1) + "%"}
            </b>
          </div>
          <div>
            <span>활성 화면 지연 p95</span>
            <b>
              {ops.render_delay_p95_seconds == null
                ? "측정 대기"
                : number(ops.render_delay_p95_seconds, 2) + "초"}
            </b>
          </div>
        </div>
      </section>
      <ModelStatus data={data?.model_comparison} />
      <details className="data-section">
        <summary>이전 모델의 실험 기록</summary>
        <section className="data-section">
          <div className="section-title">
            <h2>보존된 HAR · 기존 ML 평가</h2>
            <span>
              {data?.test?.test_start} ~ {data?.test?.test_end} · 회고 평가
            </span>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>모델</th>
                  <th>실제 포함률</th>
                  <th>평균 구간 폭</th>
                  <th>구간 점수</th>
                  <th>운영 승격 관문</th>
                </tr>
              </thead>
              <tbody>
                {["har", "ml"].map((method) => {
                  const m = data?.test?.metrics?.[method];
                  return (
                    <tr key={method}>
                      <td>{method === "ml" ? "기존 ML" : "HAR"}</td>
                      <td>
                        {number(
                          m?.coverage == null ? null : m.coverage * 100,
                          2,
                        )}
                        %
                      </td>
                      <td>{number(m?.width, 4)}%p</td>
                      <td>{number(m?.score, 4)}</td>
                      <td>
                        {data?.test?.[method + "_pass"] ? "통과" : "미통과"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="fineprint">
            구간 점수는 폭과 이탈 크기를 함께 평가하며 낮을수록 좋습니다. 이후
            개발 실험과 구분한 기존 평가이며, 실제 미래 성능이 아닙니다.
          </p>
        </section>
      </details>
      <div className="principles">
        <section>
          <span>01</span>
          <h3>개선 ML을 기본으로 사용</h3>
          <p>
            개발 검증을 통과한 nested7_tail50을 사용합니다. 계산 장애 시에만
            대체 모델을 사용하며 그 사실을 알립니다.
          </p>
        </section>
        <section>
          <span>02</span>
          <h3>90%는 목표 포함률</h3>
          <p>
            다음 거래일의 실제 고저 변동폭이 구간 안에 포함되는 비율을
            평가합니다. 개별 거래일에 90%의 확실성을 보장하지 않습니다.
          </p>
        </section>
        <section>
          <span>03</span>
          <h3>근거가 부족하면 수치 제한</h3>
          <p>
            오래된 시세, 자료 누락, 검증 미통과는 각각 표시합니다. 이미 발행한
            예측을 장중에 바꾸거나 실패한 예측을 지우지 않습니다.
          </p>
        </section>
      </div>
      <p className="fineprint">
        운영 보고서 갱신 {dateTime(ops.generated_at)} · 활성 화면 표본{" "}
        {number(ops.render_sample_count)}개 · 단위 테스트는 실제 운영 증거를
        대체하지 않습니다.
      </p>
    </div>
  );
}

export default function App() {
  const [analysis, setAnalysis] = useState<Data | null>(null),
    [flowActive, setFlowActive] = useState(false);
  const [view, setView] = useState("market"),
    [overview, setOverview] = useState<Data | null>(null),
    [sess, setSession] = useState<Data>({}),
    [symbol, setSymbol] = useState("005930"),
    [stock, setStock] = useState<Data | null>(null),
    [loading, setLoading] = useState(true),
    [failure, setFailure] = useState(""),
    [toast, setToast] = useState(""),
    [search, setSearch] = useState(""),
    [analyzing, setAnalyzing] = useState(false),
    [preview, setPreview] = useState<Data | null>(null),
    [forecastBusy, setForecastBusy] = useState(false),
    [forecastError, setForecastError] = useState(""),
    [scenario, setScenario] = useState<number | null>(null),
    [direction, setDirection] = useState("up"),
    [live, setLive] = useState<Data | null>(null),
    [connected, setConnected] = useState(false),
    [alerts, setAlerts] = useState(false),
    [quoteNow, setQuoteNow] = useState(Date.now());
  const error = (s: string) => setToast(s);
  function rememberAnalysis(value: Data | null) {
    setAnalysis(value);
    if (value) sessionStorage.setItem(`analysis-${symbol}`, value.id);
    else sessionStorage.removeItem(`analysis-${symbol}`);
  }
  useEffect(() => {
    let disposed = false;
    setAnalysis(null);
    setFlowActive(false);
    const key = sessionStorage.getItem(`analysis-${symbol}`);
    if (key)
      api(`/analyses/${key}`)
        .then((a) => {
          if (!disposed) setAnalysis(a);
        })
        .catch(() => {});
    return () => {
      disposed = true;
    };
  }, [symbol]);
  async function refresh() {
    const [s, o] = await Promise.all([session(), api("/overview")]);
    setSession(s);
    setOverview(o);
    if (!o.universe.some((r: Data) => r.symbol === symbol))
      setSymbol(o.watchlist[0]?.symbol || o.universe[0]?.symbol);
  }
  useEffect(() => {
    refresh().catch((e) => setFailure(e.message));
  }, []);
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "instant" });
  }, [view, symbol]);
  useEffect(() => {
    let disposed = false;
    const timer = setInterval(() => {
      if (document.visibilityState !== "visible") return;
      api(`/stocks/${symbol}`)
        .then((d) => {
          if (!disposed) setStock(d);
        })
        .catch(() => {});
      api("/overview")
        .then((d) => {
          if (!disposed) setOverview(d);
        })
        .catch(() => {});
    }, 60000);
    return () => {
      disposed = true;
      clearInterval(timer);
    };
  }, [symbol]);
  useEffect(() => {
    if (!sess.csrf) return;
    let disposed = false;
    let inflight = false;
    const update = () => {
      setQuoteNow(Date.now());
      if (document.visibilityState !== "visible" || inflight) return;
      inflight = true;
      api(`/stocks/${symbol}/refresh`, { method: "POST" })
        .then((d) => {
          if (!disposed) setStock((s) => (s?.symbol === d.symbol ? d : s));
        })
        .catch(() => {})
        .finally(() => {
          inflight = false;
        });
    };
    document.addEventListener("visibilitychange", update);
    update();
    const timer = setInterval(update, 5000);
    return () => {
      disposed = true;
      clearInterval(timer);
      document.removeEventListener("visibilitychange", update);
    };
  }, [symbol, sess.csrf]);
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(""), 7000);
    return () => clearTimeout(t);
  }, [toast]);
  useEffect(() => {
    let disposed = false;
    setLoading(true);
    setFailure("");
    setStock(null);
    setAnalyzing(false);
    setPreview(null);
    setForecastError("");
    setScenario(null);
    setLive(null);
    api(`/stocks/${symbol}`)
      .then((d) => {
        if (!disposed) {
          setStock(d);
          setLive(d.live_candle);
        }
      })
      .catch((e) => {
        if (!disposed) setFailure(e.message);
      })
      .finally(() => {
        if (!disposed) setLoading(false);
      });
    return () => {
      disposed = true;
    };
  }, [symbol]);
  useEffect(() => {
    if (!stock) return;
    let disposed = false;
    const es = new EventSource(`/api/stream/${symbol}`);
    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);
    es.onmessage = (e) => {
      const p = JSON.parse(e.data);
      if (p.alerts) setOverview((o) => (o ? { ...o, alerts: p.alerts } : o));
      setStock((s) =>
        s
          ? {
              ...s,
              quote: p.quote,
              gauge: p.gauge,
              market_open: p.market_open,
              prediction: p.prediction,
            }
          : s,
      );
      setLive(liveCandle(p));
      if (
        p.render_ticket &&
        view === "market" &&
        document.visibilityState === "visible"
      )
        requestAnimationFrame(() =>
          requestAnimationFrame(() => {
            if (!disposed && document.visibilityState === "visible")
              api("/render", {
                method: "POST",
                body: JSON.stringify({
                  ticket: p.render_ticket,
                  visible: true,
                }),
              }).catch(() => {});
          }),
        );
    };
    return () => {
      disposed = true;
      es.close();
      setConnected(false);
    };
  }, [symbol, !!stock, view]);
  const selectedName =
    stock?.name ||
    overview?.universe.find((r: Data) => r.symbol === symbol)?.name ||
    symbol;
  const shownBars = stock?.chart_candles || stock?.candles || [];
  const latestBar = live || shownBars.at(-1);
  const currentQuote = stock?.display_current;
  const quoteValues = currentQuote || latestBar;
  const price = quoteValues?.close,
    prev =
      currentQuote?.previous_close ||
      (!live ? stock?.display_bar?.previous_close : null) ||
      (live && live.time !== shownBars.at(-1)?.time
        ? shownBars.at(-1)?.close
        : shownBars.at(-2)?.close),
    change = price && prev ? 100 * (price / prev - 1) : null;
  const prediction =
    stock?.prediction?.status === "ready" &&
    stock.prediction.target_date === stock.forecast_target
      ? stock.prediction
      : preview?.symbol === symbol &&
          preview.target_date === stock?.forecast_target &&
          preview.anchor_date === stock?.chart_candles?.at(-1)?.time
        ? preview
        : null;
  async function showForecast() {
    if (forecastBusy) return;
    if (analyzing) {
      reset();
      return;
    }
    setAnalyzing(true);
    setScenario(null);
    setForecastError("");
    if (prediction) return;
    setForecastBusy(true);
    try {
      setPreview(await api(`/stocks/${symbol}/forecast`, { method: "POST" }));
    } catch (e) {
      setForecastError((e as Error).message);
    } finally {
      setForecastBusy(false);
    }
  }
  async function toggleWatch(s: string) {
    if (!overview) return;
    const values = overview.watchlist.map((v: Data) => v.symbol);
    const next = values.includes(s)
      ? values.filter((v: string) => v !== s)
      : [...values, s];
    if (next.length > 30) {
      error("관심종목은 최대 30개까지 등록할 수 있습니다.");
      return;
    }
    try {
      await api("/watchlist", {
        method: "PUT",
        body: JSON.stringify({ symbols: next }),
      });
      setOverview(await api("/overview"));
    } catch (e) {
      error((e as Error).message);
    }
  }
  function reset() {
    setAnalyzing(false);
    setScenario(null);
  }
  const watch = search
    ? overview?.universe
        .filter(
          (s: Data) =>
            s.name.toLowerCase().includes(search.toLowerCase()) ||
            s.symbol.includes(search),
        )
        .map((s: Data) => ({
          ...s,
          ...overview.watchlist.find((w: Data) => w.symbol === s.symbol),
        }))
    : overview?.watchlist;
  return (
    <div className={"app-shell market-terminal studio " + view}>
      <nav className="nav-rail" aria-label="주요 메뉴">
        <button
          className="brand-mark"
          aria-label="VolGuard 홈"
          onClick={() => setView("market")}
        >
          <Mark />
        </button>
        <div className="rail-links">
          {[
            { id: "market", icon: Compass, label: "시장 탐색" },
            { id: "intelligence", icon: Sparkles, label: "AI 심층 조사" },
            { id: "portfolio", icon: Wallet, label: "포트폴리오" },
            { id: "validation", icon: ShieldCheck, label: "검증 현황" },
          ].map(({ id, icon: Icon, label }) => (
            <button
              key={id}
              className={view === id ? "active" : ""}
              aria-current={view === id ? "page" : undefined}
              aria-label={label}
              title={label}
              onClick={() => setView(id)}
            >
              <Icon size={21} strokeWidth={1.4} />
              <span>{label}</span>
            </button>
          ))}
        </div>
        <span className="rail-bottom">
          KR
          <br />
          <i className={"status-dot " + (connected ? "live" : "")} />
        </span>
      </nav>
      <aside className="watch-panel">
        <div className="wordmark">
          VOLGUARD<span>KR</span>
        </div>
        <div className="watch-heading">
          <span>나의 관심종목</span>
          <small>{overview?.watchlist.length || 0} / 30</small>
        </div>
        <label className="search-box">
          <Search size={15} />
          <input
            aria-label="종목 검색"
            placeholder="종목 또는 코드 검색"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          {search && (
            <button aria-label="검색 지우기" onClick={() => setSearch("")}>
              <X size={13} />
            </button>
          )}
        </label>
        <div className="watch-list">
          {watch?.map((s: Data) => {
            const saved = overview?.watchlist.some(
              (w: Data) => w.symbol === s.symbol,
            );
            return (
              <div
                key={s.symbol}
                className={
                  "watch-item " + (symbol === s.symbol ? "selected" : "")
                }
              >
                <button
                  className="watch-select"
                  onClick={() => {
                    setSymbol(s.symbol);
                    setView("market");
                  }}
                >
                  <div>
                    <b>{s.name}</b>
                    <span>{s.symbol}</span>
                  </div>
                  {s.spark ? (
                    <Spark values={s.spark} up={s.change >= 0} />
                  ) : (
                    <span className="caption">KRX</span>
                  )}
                  <div className="watch-price">
                    <b>{number(s.close)}</b>
                    <span className={s.change >= 0 ? "positive" : "negative"}>
                      {s.change > 0 ? "+" : ""}
                      {number(s.change, 2)}
                      {s.change != null ? "%" : ""}
                    </span>
                  </div>
                </button>
                {search && (
                  <button
                    className="watch-add"
                    aria-label={`${s.name} ${saved ? "관심 해제" : "관심 등록"}`}
                    onClick={() => toggleWatch(s.symbol)}
                  >
                    {saved ? <Check size={14} /> : <Plus size={14} />}
                  </button>
                )}
              </div>
            );
          })}
          {watch?.length === 0 && (
            <p className="empty-copy">검색 결과가 없습니다.</p>
          )}
        </div>
        <div className="watch-footer">
          <span className="eyebrow">데이터 기준</span>
          <p>
            마지막 확정 일봉 · {stock?.as_of || "확인 중"}
            <br />
            정규장 실시간은 수신 시 별도 표시
          </p>
          <div className="small-brand">
            <Activity size={14} />
            <span>위험을 이해하는 새로운 관점</span>
          </div>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <div className="breadcrumb">
            작업 공간 <ChevronRight size={12} />
            <b>
              {view === "market"
                ? "시장 탐색"
                : view === "intelligence"
                  ? "AI 심층 조사"
                  : view === "portfolio"
                    ? "포트폴리오"
                    : "검증 현황"}
            </b>
          </div>
          <div className="topbar-right">
            <span>
              <i
                className={"status-dot " + (stock?.market_open ? "live" : "")}
              />
              {stock?.market_open ? "KRX 정규장" : "KRX 장외"}
            </span>
            <span className="connection-label">
              {connected ? "로컬 서비스 연결" : "연결 확인 중"}
            </span>
            <button
              className="notification-button"
              aria-label="위험 알림함"
              onClick={() => setAlerts(!alerts)}
            >
              <Bell size={17} />
              {!!overview?.alerts.length && <i />}
            </button>
          </div>
        </header>
        {alerts && (
          <div className="alerts-popover">
            <div className="section-title">
              <h3>위험 알림함</h3>
              <button aria-label="알림함 닫기" onClick={() => setAlerts(false)}>
                <X size={16} />
              </button>
            </div>
            {overview?.alerts.length ? (
              overview.alerts.map((a: Data, i: number) => (
                <p key={i}>
                  {overview.universe.find((s: Data) => s.symbol === a.symbol)
                    ?.name || a.symbol}{" "}
                  · 상한 소진율 {a.threshold}%
                  <small>{dateTime(a.created_at)}</small>
                </p>
              ))
            ) : (
              <p>
                기록된 위험 알림이 없습니다.
                <br />
                <small>시세·예측이 유효한 정규장에만 판정합니다.</small>
              </p>
            )}
          </div>
        )}
        {analysis && view !== "market" && (
          <AnalysisContext
            analysis={analysis}
            onMarket={() => setView("market")}
          />
        )}
        {view === "market" ? (
          <div className="market-page page-enter">
            <div className="stock-heading">
              <div>
                <div className="eyebrow">
                  국내 주식 <span>{symbol} · 코스피</span>
                </div>
                <h1>
                  {selectedName}
                  <span className="exchange-tag">KRX</span>
                  <button
                    className="favorite-toggle"
                    aria-label={
                      overview?.watchlist.some((w: Data) => w.symbol === symbol)
                        ? "관심종목 해제"
                        : "관심종목 등록"
                    }
                    onClick={() => toggleWatch(symbol)}
                  >
                    <Star
                      size={17}
                      fill={
                        overview?.watchlist.some(
                          (w: Data) => w.symbol === symbol,
                        )
                          ? "currentColor"
                          : "none"
                      }
                      strokeWidth={1}
                    />
                  </button>
                </h1>
                <div className="price-line">
                  <strong>{number(price)}</strong>
                  <span>원</span>
                  <b
                    className={
                      change != null && change >= 0 ? "positive" : "negative"
                    }
                  >
                    {change != null && change > 0 ? "+" : ""}
                    {number(change, 2)}%
                  </b>
                </div>
              </div>
              <div className="stock-actions">
                <button
                  className="subtle"
                  onClick={() => {
                    setView("intelligence");
                  }}
                >
                  <Sparkles size={16} /> AI 심층 조사 <ArrowUpRight size={14} />
                </button>
                <span>
                  {currentQuote ? "통합 현재가 · SOR" : "확인된 KRX 일봉"} ·{" "}
                  {currentQuote?.trade_clock
                    ? `마지막 체결 ${currentQuote.trade_clock}`
                    : latestBar?.time || "확인 중"}
                </span>
              </div>
              <div className="quote-details">
                <p className="price-provenance">
                  {currentQuote
                    ? `${currentQuote.fresh && quoteNow - Date.parse(currentQuote.retrieved_at) <= 15000 ? "자동 갱신 중" : "자동 재연결 중"} · ${new Date(currentQuote.retrieved_at).toLocaleTimeString("ko-KR", { hour12: false })} 조회`
                    : "최신 시세 연결 확인 중"}
                  <br />장 마감 후에도 5초마다 조회합니다. 거래가 없으면 가격은
                  유지됩니다.
                </p>
                <h3>시세 정보</h3>
                <dl>
                  {[
                    ["기준가", prev],
                    ["시가", quoteValues?.open],
                    ["고가", quoteValues?.high],
                    ["저가", quoteValues?.low],
                    ["거래량", quoteValues?.volume],
                  ].map(([label, value]) => (
                    <div key={String(label)}>
                      <dt>{label}</dt>
                      <dd>{number(value as number)}</dd>
                    </div>
                  ))}
                </dl>
                <p className="price-provenance">
                  통합 현재가와 KRX 일봉 차트의 고저·거래량은 다를 수 있습니다.
                  체결 시각은 제공사의 시각 값이며 체결일은 확인되지 않습니다.
                  위험 게이지는 KRX 정규장만 사용합니다.
                </p>
                <label className="symbol-picker">
                  종목 선택
                  <select
                    aria-label="종목 선택"
                    value={symbol}
                    onChange={(e) => setSymbol(e.target.value)}
                  >
                    {overview?.universe.map((s: Data) => (
                      <option key={s.symbol} value={s.symbol}>
                        {s.name} · {s.symbol}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="quick-watch">
                  {overview?.watchlist.map((s: Data) => (
                    <button
                      className={s.symbol === symbol ? "selected" : ""}
                      key={s.symbol}
                      onClick={() => setSymbol(s.symbol)}
                    >
                      {s.name}
                    </button>
                  ))}
                </div>
              </div>
            </div>
            {failure ? (
              <div className="inline-warning">
                {failure}
                <button onClick={() => location.reload()}>다시 연결</button>
              </div>
            ) : loading ? (
              <div className="main-loading">
                <LoaderCircle className="spin" />
                <span>실제 거래일 데이터를 불러오고 있습니다</span>
              </div>
            ) : (
              stock && (
                <>
                  <div className="market-workspace">
                    <AnalysisFlow
                      stock={stock}
                      analysis={analysis}
                      onAnalysis={rememberAnalysis}
                      onActive={setFlowActive}
                      onResearch={() => setView("intelligence")}
                      onPortfolio={() => setView("portfolio")}
                      live={live}
                    />
                    {!flowActive && (
                      <CandleChart
                        showForecastControl={false}
                        stock={stock}
                        prediction={prediction}
                        analyzing={analyzing}
                        live={live}
                        scenario={scenario}
                        direction={direction}
                        onDirection={setDirection}
                        onReset={reset}
                        onAnalyze={showForecast}
                        onScenario={setScenario}
                        busy={forecastBusy}
                        forecastError={forecastError}
                      />
                    )}
                  </div>
                  <RiskGauge stock={stock} prediction={prediction} />
                  <div className="market-bottom">
                    <div>
                      <span className="eyebrow">관측에서 이해로</span>
                      <h3>
                        위험 신호의 배경을,
                        <br />네 개의 관점으로.
                      </h3>
                    </div>
                    <p>
                      가격·공시·재무·검증 결과를 모으고
                      <br />
                      주장과 반대 근거를 함께 연결합니다.
                    </p>
                    <button
                      className="circle-action"
                      aria-label="AI 조사 공간 열기"
                      onClick={() => setView("intelligence")}
                    >
                      <ArrowUpRight size={25} strokeWidth={1} />
                    </button>
                    <span className="bottom-word">
                      EVIDENCE
                      <br />
                      BEFORE INSIGHT.
                    </span>
                  </div>
                </>
              )
            )}
          </div>
        ) : view === "intelligence" ? (
          <Intelligence
            analysisId={analysis?.id}
            symbol={symbol}
            name={selectedName}
            session={sess}
            universe={overview?.universe || []}
            error={error}
          />
        ) : view === "portfolio" ? (
          <>
            {analysis && (
              <section className="analysis-flow">
                <div className="flow-heading">
                  <h3>선택한 시나리오의 보유 영향</h3>
                  <strong>{number(analysis.impact.pnl_krw)}원</strong>
                </div>
                <div className="flow-detail">
                  <p>{analysis.impact.note}</p>
                  {analysis.impact.contributions.map((r: Data) => (
                    <p key={r.symbol}>
                      {overview?.universe.find(
                        (s: Data) => s.symbol === r.symbol,
                      )?.name || r.symbol}{" "}
                      · 비중 {number(r.weight * 100, 1)}% · 손익 기여{" "}
                      {number(r.pnl_krw)}원
                    </p>
                  ))}
                  <p>
                    아래 보유 구성을 변경하면 새 차트 분석을 시작해 스냅샷에
                    반영하세요.
                  </p>
                </div>
              </section>
            )}
            <Portfolio universe={overview?.universe || []} error={error} />
          </>
        ) : (
          <>
            <ResearchValidation />
            <Validation error={error} />
          </>
        )}
        <footer className="app-footer">
          <span>VolGuard KR · 개인용 위험 분석</span>
          <span>매수·매도 추천이 아닌, 판단을 위한 근거</span>
        </footer>
      </main>
      {toast && (
        <div className="toast" role="alert">
          <span>{toast}</span>
          <button aria-label="메시지 닫기" onClick={() => setToast("")}>
            <X size={16} />
          </button>
        </div>
      )}
    </div>
  );
}
