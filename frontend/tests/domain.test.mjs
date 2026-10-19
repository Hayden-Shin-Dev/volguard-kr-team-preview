import test from "node:test";
import assert from "node:assert/strict";
import { scenarioCandle, liveCandle, safeUrl } from "../src/domain.mjs";
test("scenario preserves high/low percentage independently of direction", () => {
  for (const range of [0, 1.2, 3.4, 30])
    for (const direction of ["up", "down"]) {
      const c = scenarioCandle(70000, range, direction);
      assert.ok(Math.abs(100 * (c.high / c.low - 1) - range) < 1e-10);
      assert.ok(
        c.low <= c.open &&
          c.high >= c.open &&
          c.close >= c.low &&
          c.close <= c.high,
      );
    }
  assert.equal(scenarioCandle(0, 2), null);
});
test("live candle needs session, date, completeness and independently fresh connection/price", () => {
  const today = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Seoul",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date());
  const p = {
    market_open: true,
    quote: {
      complete: true,
      exchange: "KRX",
      session: "regular",
      trade_date: today,
      open: 100,
      high: 105,
      low: 98,
      price: 102,
      volume: 1000,
    },
    gauge: {
      price_freshness: { status: "fresh" },
      connection_freshness: { status: "fresh" },
    },
  };
  assert.equal(liveCandle(p).close, 102);
  assert.equal(liveCandle({ ...p, market_open: false }), null);
  assert.equal(
    liveCandle({ ...p, quote: { ...p.quote, trade_date: "2020-01-01" } }),
    null,
  );
  assert.equal(
    liveCandle({ ...p, quote: { ...p.quote, exchange: "NXT" } }),
    null,
  );
  assert.equal(
    liveCandle({ ...p, quote: { ...p.quote, open: undefined } }),
    null,
  );
  assert.equal(
    liveCandle({
      ...p,
      gauge: { ...p.gauge, price_freshness: { status: "stale" } },
    }),
    null,
  );
});
test("source links never execute generated code", () => {
  assert.equal(safeUrl("javascript:alert(1)"), undefined);
  assert.equal(safeUrl("http://untrusted.invalid"), undefined);
  assert.equal(safeUrl("https://dart.fss.or.kr/"), "https://dart.fss.or.kr/");
});
import { aggregateCandles } from "../src/domain.mjs";
test("weekly and monthly candles aggregate only observed OHLCV in chronological order", () => {
  const rows = [
    {
      time: "2026-10-15",
      open: 100,
      high: 110,
      low: 98,
      close: 105,
      volume: 5,
    },
    {
      time: "2026-10-16",
      open: 105,
      high: 112,
      low: 103,
      close: 108,
      volume: 7,
    },
    {
      time: "2026-10-19",
      open: 108,
      high: 115,
      low: 100,
      close: 101,
      volume: 2,
    },
  ];
  assert.deepEqual(aggregateCandles(rows, "week"), [
    {
      time: "2026-10-12",
      open: 100,
      high: 112,
      low: 98,
      close: 108,
      volume: 12,
    },
    {
      time: "2026-10-19",
      open: 108,
      high: 115,
      low: 100,
      close: 101,
      volume: 2,
    },
  ]);
  assert.equal(aggregateCandles(rows, "month")[0].volume, 14);
  assert.equal(aggregateCandles(rows, "month")[0].close, 101);
  assert.equal(rows[0].volume, 5);
});
