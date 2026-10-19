import { useEffect, useRef, useState } from "react";
import {
  createChart,
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  createSeriesMarkers,
  ColorType,
  CrosshairMode,
  type IChartApi,
  type ISeriesApi,
} from "lightweight-charts";
import {
  Maximize2,
  Minus,
  Plus,
  Sparkles,
  LoaderCircle,
  History,
  RotateCcw,
} from "lucide-react";
import { api, number, type Data } from "./api";
import { aggregateCandles } from "./domain.mjs";

type Props = {
  stock: Data;
  prediction: Data | null;
  analyzing: boolean;
  live: Data | null;
  scenario: number | null;
  direction: string;
  onDirection: (v: string) => void;
  onReset: () => void;
  onAnalyze: () => void;
  onScenario: (v: number | null) => void;
  busy: boolean;
  forecastError: string;
  showForecastControl?: boolean;
};
export function CandleChart({
  stock,
  prediction,
  analyzing,
  live,
  onReset,
  onAnalyze,
  busy,
  forecastError,
  showForecastControl = true,
}: Props) {
  const host = useRef<HTMLDivElement>(null),
    chart = useRef<IChartApi | null>(null);
  const candles = useRef<ISeriesApi<"Candlestick"> | null>(null),
    volume = useRef<ISeriesApi<"Histogram"> | null>(null);
  const [unit, setUnit] = useState("day"),
    [period, setPeriod] = useState(3),
    [past, setPast] = useState(false),
    [archive, setArchive] = useState<Data | null>(null),
    [basis, setBasis] = useState("development"),
    [loading, setLoading] = useState(false),
    [error, setError] = useState(""),
    [hover, setHover] = useState<Data | null>(null),
    [replay, setReplay] = useState(0);
  const liveMarkers = useRef<{ setMarkers: (rows: any[]) => void } | null>(
    null,
  );
  const generation = useRef(0),
    actualSeries = useRef<ISeriesApi<"Line"> | null>(null);
  useEffect(() => {
    generation.current++;
    setPast(false);
    setArchive(null);
    setError("");
    setLoading(false);
    setHover(null);
  }, [stock.symbol]);
  const selected = archive?.[basis],
    records: Data[] = past ? selected?.rows || [] : [];
  const all: Data[] = past
    ? selected?.candles || []
    : stock.chart_candles || stock.candles || [];
  const merged =
    !past && live && (!all.length || live.time >= all.at(-1)!.time)
      ? [...all.filter((r) => r.time !== live.time), live]
      : all;
  const ranges = analyzing || past;
  const data: Data[] = aggregateCandles(merged, ranges ? "day" : unit);
  const sample = hover?.bar || data.at(-1);
  async function toggleHistory() {
    if (past) {
      setPast(false);
      return;
    }
    onReset();
    setPast(true);
    if (archive) return;
    const request = ++generation.current;
    setLoading(true);
    setError("");
    try {
      const result = await api(`/stocks/${stock.symbol}/prediction-history`);
      if (generation.current === request) setArchive(result);
    } catch (e) {
      if (generation.current === request) setError(String(e));
    } finally {
      if (generation.current === request) setLoading(false);
    }
  }
  useEffect(() => {
    if (!host.current) return;
    const c = createChart(host.current, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: "#ffffff" },
        textColor: "#828392",
        fontFamily: '"Noto Sans KR Variable",sans-serif',
        fontSize: 11,
        attributionLogo: true,
      },
      grid: {
        vertLines: { color: "#f0eff5" },
        horzLines: { color: "#f3f1f7" },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: {
        borderVisible: false,
        scaleMargins: { top: 0.1, bottom: 0.08 },
      },
      timeScale: { borderVisible: false, rightOffset: 5, barSpacing: 14 },
      localization: { locale: "ko-KR" },
    });
    const s = c.addSeries(CandlestickSeries, {
      upColor: "#d84e49",
      downColor: "#5776f6",
      borderVisible: false,
      wickUpColor: "#d84e49",
      wickDownColor: "#5776f6",
      priceFormat: { type: "price", precision: 0, minMove: 1 },
    });
    s.setData(data as any);
    candles.current = s;
    let raf = 0,
      zoomRaf = 0;
    if (ranges) {
      const fmt = {
        type: "custom" as const,
        minMove: 0.01,
        formatter: (v: number) => `${v.toFixed(1)}%`,
      };
      const band = c.addSeries(
        CandlestickSeries,
        {
          upColor: "#c8baf080",
          downColor: "#c8baf080",
          borderVisible: true,
          borderUpColor: "#9c87dc",
          borderDownColor: "#9c87dc",
          wickVisible: false,
          priceFormat: fmt,
          lastValueVisible: false,
          priceLineVisible: false,
        },
        1,
      );
      const actual = c.addSeries(
        LineSeries,
        {
          color: "#178e87",
          lineWidth: 2,
          pointMarkersVisible: true,
          pointMarkersRadius: 3,
          priceFormat: fmt,
          lastValueVisible: false,
          priceLineVisible: false,
        },
        1,
      );
      const median = c.addSeries(
        LineSeries,
        {
          color: "#7962c7",
          lineWidth: 1,
          lineStyle: 2,
          pointMarkersVisible: true,
          pointMarkersRadius: 2,
          priceFormat: fmt,
          lastValueVisible: false,
          priceLineVisible: false,
        },
        1,
      );
      const actuals = past
        ? records.map((r) => ({ time: r.time, value: r.actual }))
        : data.map((r) => ({
            time: r.time,
            value: 100 * (r.high / r.low - 1),
          }));
      actual.setData(actuals as any);
      actualSeries.current = actual;
      if (!past) liveMarkers.current = createSeriesMarkers(actual, []);
      band.applyOptions({
        autoscaleInfoProvider: (
          original: () => {
            priceRange: { minValue: number; maxValue: number };
          } | null,
        ) => {
          const info = original();
          return info
            ? {
                ...info,
                priceRange: {
                  minValue: 0,
                  maxValue:
                    (!past && prediction
                      ? prediction.high
                      : info.priceRange.maxValue) * 1.12,
                },
              }
            : null;
        },
      });
      band.setData(
        records.map((r) => ({
          time: r.time,
          open: r.low,
          close: r.high,
          low: r.low,
          high: r.high,
        })) as any,
      );
      median.setData(
        records.map((r) => ({ time: r.time, value: r.median })) as any,
      );
      if (past)
        createSeriesMarkers(
          actual,
          records
            .filter((r) => !r.inside)
            .map((r) => ({
              time: r.time,
              position: "inBar",
              shape: "circle",
              color: "#dc674c",
              size: 1,
            })) as any,
        );
      if (!past && prediction) {
        const p = prediction,
          t = p.target_date;
        const existing = actuals.find((r) => r.time === t);
        const forecastPoint = {
          time: t,
          open: p.low,
          close: p.high,
          low: p.low,
          high: p.high,
        };
        band.setData([forecastPoint] as any);
        median.setData([{ time: t, value: p.median }] as any);
        createSeriesMarkers(band, [
          {
            time: t,
            position: "aboveBar",
            shape: "arrowDown",
            color: "#7255c1",
            text: `${t.slice(5)} 예측`,
            size: 1,
          },
        ] as any);
        for (const [value, label] of [
          [p.low, "하한"],
          [p.median, "중앙"],
          [p.high, "상한"],
        ] as const)
          band.createPriceLine({
            price: value,
            color: "#b9a8e4",
            lineWidth: 1,
            lineStyle: 2,
            axisLabelVisible: true,
            title: label,
          });
        // Only the actual observed range belongs on the % pane; never turn it into a price target.
        if (existing && existing.value > p.high)
          liveMarkers.current?.setMarkers([
            {
              time: t,
              position: "aboveBar",
              shape: "circle",
              color: "#dc674c",
              text: "예측 상한 초과",
            },
          ]);
        if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
          const start = performance.now();
          const tick = (now: number) => {
            const f = 1 - Math.pow(1 - Math.min(1, (now - start) / 1100), 3);
            band.update({
              ...forecastPoint,
              open: p.median + (p.low - p.median) * f,
              low: p.median + (p.low - p.median) * f,
              close: p.median + (p.high - p.median) * f,
              high: p.median + (p.high - p.median) * f,
            } as any);
            if (f < 1) raf = requestAnimationFrame(tick);
          };
          raf = requestAnimationFrame(tick);
        }
      }
      band
        .priceScale()
        .applyOptions({ scaleMargins: { top: 0.19, bottom: 0.12 } });
      c.panes()[1].setHeight(220);
    }
    const v = c.addSeries(
      HistogramSeries,
      {
        priceFormat: { type: "volume" },
        lastValueVisible: false,
        priceLineVisible: false,
      },
      ranges ? 2 : 1,
    );
    v.setData(
      data.map((r) => ({
        time: r.time,
        value: r.volume,
        color: r.close >= r.open ? "#df49457a" : "#4c7bff7a",
      })) as any,
    );
    volume.current = v;
    v.priceScale().applyOptions({ scaleMargins: { top: 0.1, bottom: 0 } });
    c.panes()[0].setStretchFactor(ranges ? 5 : 8);
    if (ranges) c.panes()[1].setStretchFactor(4);
    c.panes()[ranges ? 2 : 1].setStretchFactor(1);
    c.subscribeCrosshairMove((p) => {
      const b = p.seriesData.get(s);
      setHover(
        p.time
          ? {
              time: String(p.time),
              bar:
                b && "open" in b
                  ? { ...b, volume: (p.seriesData.get(v) as any)?.value }
                  : null,
              record: records.find((r) => r.time === p.time),
            }
          : null,
      );
    });
    chart.current = c;
    const cutoff = new Date((data.at(-1)?.time || "2026-01-01") + "T00:00:00Z");
    cutoff.setUTCMonth(cutoff.getUTCMonth() - period);
    const from = past
      ? Math.max(0, data.length - 45)
      : ranges
        ? Math.max(0, data.length - 24)
        : period
          ? Math.max(
              0,
              data.findIndex(
                (r) => r.time >= cutoff.toISOString().slice(0, 10),
              ),
            )
          : 0;
    if (data.length) {
      const to = data.length + (ranges && !past ? 5 : 2);
      c.timeScale().setVisibleLogicalRange({ from, to });
      if (
        ranges &&
        !past &&
        !matchMedia("(prefers-reduced-motion: reduce)").matches
      ) {
        const began = performance.now(),
          origin = Math.max(0, data.length - 45);
        const focus = (now: number) => {
          const t = Math.min(1, (now - began) / 750),
            ease = 1 - Math.pow(1 - t, 4);
          c.timeScale().setVisibleLogicalRange({
            from: origin + (from - origin) * ease,
            to,
          });
          if (t < 1) zoomRaf = requestAnimationFrame(focus);
        };
        zoomRaf = requestAnimationFrame(focus);
      }
    }
    return () => {
      cancelAnimationFrame(raf);
      cancelAnimationFrame(zoomRaf);
      c.remove();
      chart.current = null;
      actualSeries.current = null;
      liveMarkers.current = null;
    };
  }, [
    stock.symbol,
    ranges,
    past,
    basis,
    archive,
    prediction?.target_date,
    replay,
    unit,
    period,
  ]);
  // Keep incoming candles live without rebuilding the graph or resetting the user's zoom.
  useEffect(() => {
    if (past) return;
    const current = data.find((r) => r.time === prediction?.target_date);
    liveMarkers.current?.setMarkers(
      current &&
        prediction &&
        100 * (current.high / current.low - 1) > prediction.high
        ? [
            {
              time: current.time,
              position: "aboveBar",
              shape: "circle",
              color: "#dc674c",
              text: "예측 상한 초과",
            },
          ]
        : [],
    );
    actualSeries.current?.setData(
      data.map((r) => ({
        time: r.time,
        value: 100 * (r.high / r.low - 1),
      })) as any,
    );
    candles.current?.setData(data as any);
    volume.current?.setData(
      data.map((r) => ({
        time: r.time,
        value: r.volume,
        color: r.close >= r.open ? "#df49457a" : "#4c7bff7a",
      })) as any,
    );
  }, [live, stock.display_bar?.retrieved_at, stock.as_of, past]);
  function zoom(f: number) {
    const c = chart.current,
      r = c?.timeScale().getVisibleLogicalRange();
    if (c && r)
      c.timeScale().setVisibleLogicalRange({
        from: r.to - (r.to - r.from) * f,
        to: r.to,
      });
  }
  const detail = hover?.record || records.at(-1);
  return (
    <section
      className={`chart-stage integrated-chart ${ranges ? "with-ranges" : ""} ${past ? "show-history" : ""} ${past && !records.length ? "empty-history" : ""}`}
      aria-label="종목 일봉과 변동폭 차트"
    >
      <div className="chart-toolbar">
        <div className="segmented bar-units">
          {[
            ["day", "일"],
            ["week", "주"],
            ["month", "월"],
            ["year", "년"],
          ].map(([key, label]) => (
            <button
              key={key}
              className={(ranges ? "day" : unit) === key ? "selected" : ""}
              onClick={() => {
                setPast(false);
                onReset();
                setUnit(key);
              }}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="segmented periods">
          {[1, 3, 6, 0].map((p) => (
            <button
              key={p}
              className={period === p ? "selected" : ""}
              onClick={() => setPeriod(p)}
            >
              {p ? p + "개월" : "전체"}
            </button>
          ))}
        </div>
        {showForecastControl && <button
          className={`forecast-trigger ${analyzing && !past ? "selected" : ""}`}
          onClick={() => {
            setPast(false);
            onAnalyze();
          }}
          disabled={busy}
        >
          {busy ? (
            <LoaderCircle size={15} className="spin" />
          ) : (
            <Sparkles size={15} />
          )}{" "}
          {analyzing && !past ? "예측 닫기" : "다음 거래일 변동폭"}
        </button>}
        <button
          className={`forecast-trigger ${past ? "selected" : ""}`}
          onClick={toggleHistory}
        >
          <History size={15} />
          {past ? "현재 차트로" : "과거 예측 비교"}
        </button>
        <div className="chart-controls">
          <button aria-label="차트 확대" onClick={() => zoom(0.7)}>
            <Plus size={15} />
          </button>
          <button aria-label="차트 축소" onClick={() => zoom(1.4)}>
            <Minus size={15} />
          </button>
          <button
            aria-label="차트 전체 보기"
            onClick={() => chart.current?.timeScale().fitContent()}
          >
            <Maximize2 size={15} />
          </button>
        </div>
      </div>
      {past && (
        <div className="history-context">
          <div className="segmented">
            <button
              className={basis === "development" ? "selected" : ""}
              onClick={() => setBasis("development")}
            >
              개발 검증 기록
            </button>
            <button
              className={basis === "forward" ? "selected" : ""}
              onClick={() => setBasis("forward")}
            >
              실제 발행 기록
            </button>
          </div>
          <span>
            {selected?.label}
            <br />
            {records[0]?.time} ~ {records.at(-1)?.time}
          </span>
        </div>
      )}
      {ranges && (
        <div className="range-caption" aria-live="polite">
          <div>
            <span className="eyebrow">
              {past
                ? "예측과 실제, 같은 날짜에서 비교"
                : "다음 거래일, 어느 정도 움직일까"}
            </span>
            <h3>
              {past
                ? records.length
                  ? `${selected?.n || 0}일 중 ${records.filter((r) => r.inside).length}일이 구간 안에`
                  : "실제 결과를 기다리고 있습니다"
                : `${prediction?.target_date || "계산 중"} 변동폭 예측`}
            </h3>
          </div>
          <div className="range-numbers">
            {past && !records.length ? (
              <span>장 마감 후 확정 데이터로 채점합니다.</span>
            ) : past ? (
              <>
                <b>
                  {number(
                    selected?.coverage == null ? null : 100 * selected.coverage,
                    1,
                  )}
                  %<small>최근 120개 이내 기록의 포함률</small>
                </b>
                <b>
                  {number(selected?.score, 2)}
                  <small>Interval Score · 낮을수록 좋음</small>
                </b>
              </>
            ) : prediction ? (
              <>
                <b>
                  {number(prediction.low, 2)}
                  <small>하한 %</small>
                </b>
                <b>
                  {number(prediction.median, 2)}
                  <small>중앙값 %</small>
                </b>
                <b>
                  {number(prediction.high, 2)}
                  <small>상한 %</small>
                </b>
                <button
                  aria-label="예측 애니메이션 다시 재생"
                  onClick={() => setReplay(replay + 1)}
                >
                  <RotateCcw size={17} />
                </button>
              </>
            ) : (
              <span>{busy ? "확정 일봉으로 계산 중" : forecastError}</span>
            )}
          </div>
        </div>
      )}
      {prediction?.fallback && !past && analyzing && (
        <p className="range-readout">
          대체 계산을 표시합니다. {prediction.fallback.reason}
        </p>
      )}
      <div className="ohlcv-strip">
        <time>{sample?.time || hover?.time}</time>
        {[
          ["시", sample?.open],
          ["고", sample?.high],
          ["저", sample?.low],
          ["종", sample?.close],
          ["거래량", sample?.volume],
        ].map(([k, v]) => (
          <span key={k as string}>
            {k} <b>{number(v as number)}</b>
          </span>
        ))}
      </div>
      {ranges && (
        <div className="range-legend">
          <span>
            <i className="band-key" />
            보라 막대: 예측 하한~상한
          </span>
          <span>
            <i className="actual-key" />
            초록 선: 실제 고저 변동폭
          </span>
          <span>
            <i className="miss-key" />
            주황 점: 구간 밖
          </span>
          <small>
            위: 가격(원) · 가운데: 변동폭(%) · 아래: 거래량 — 날짜축 공유
          </small>
        </div>
      )}
      <div className="range-chart-canvas">
        <div className="candle-host" ref={host} />
        {(loading || error || (past && !records.length)) && (
          <div className="range-empty">
            {loading
              ? "보존된 예측과 실제 결과를 불러오는 중"
              : error ||
                "아직 채점된 실제 발행 예측이 없습니다. 대상 거래일 장 마감 후 결과가 쌓입니다."}
          </div>
        )}
      </div>
      {ranges && (
        <div className="range-readout" aria-live="polite">
          {past && !detail ? (
            <span>
              운영 결과가 쌓이면 이 위치에서 날짜별 예측과 실제를 비교할 수
              있습니다.
            </span>
          ) : past && detail ? (
            <>
              <strong>{detail.time}</strong>
              <span>
                예측 {number(detail.low, 2)}~{number(detail.high, 2)}%
              </span>
              <span>실제 {number(detail.actual, 2)}%</span>
              <b className={detail.inside ? "inside" : "outside"}>
                {detail.inside ? "구간 안" : "구간 밖"}
              </b>
              <small>차트 위를 움직여 날짜별 결과를 확인하세요.</small>
            </>
          ) : (
            <>
              <strong>보라 막대가 다음 거래일의 예측입니다.</strong>
              <span>
                고저 변동폭의 90% 목표 구간이며 주가의 상·하단 가격 예측이
                아닙니다.
              </span>
            </>
          )}
        </div>
      )}
      {past && records.length > 0 && (
        <details className="range-table">
          <summary>날짜별 숫자 확인 · 최근 {records.length}개 기록</summary>
          <div>
            <table>
              <thead>
                <tr>
                  {["날짜", "하한", "중앙값", "상한", "실제", "결과"].map(
                    (t) => (
                      <th key={t}>{t}</th>
                    ),
                  )}
                </tr>
              </thead>
              <tbody>
                {[...records].reverse().map((r) => (
                  <tr key={r.time}>
                    <td>{r.time}</td>
                    {[r.low, r.median, r.high, r.actual].map((v, i) => (
                      <td key={i}>{number(v, 2)}%</td>
                    ))}
                    <td>{r.inside ? "구간 안" : "구간 밖"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
      <div className="chart-foot">
        <span>
          {past
            ? "nested7_tail50 · 보존된 평가 결과, 재학습·소급 발행 없음"
            : stock.price_basis}
        </span>
        <a href="https://www.tradingview.com/" target="_blank" rel="noreferrer">
          차트 · TradingView
        </a>
      </div>
    </section>
  );
}
