export function scenarioCandle(anchor, range, direction = "up") {
  if (!(anchor > 0) || !Number.isFinite(range) || range < 0) return null;
  const ratio = 1 + range / 100;
  const low = anchor / Math.sqrt(ratio),
    high = anchor * Math.sqrt(ratio);
  return {
    open: anchor,
    low,
    high,
    close: anchor + (direction === "up" ? 1 : -1) * (high - low) * 0.32,
  };
}
export function liveCandle(payload) {
  const q = payload?.quote,
    g = payload?.gauge;
  const today = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Seoul",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date());
  if (
    !payload?.market_open ||
    !q?.complete ||
    q.exchange !== "KRX" ||
    q.session !== "regular" ||
    q.trade_date !== today ||
    g?.price_freshness?.status !== "fresh" ||
    g?.connection_freshness?.status !== "fresh"
  )
    return null;
  if (
    ![q.open, q.high, q.low, q.price, q.volume].every(Number.isFinite) ||
    q.low <= 0 ||
    q.high < Math.max(q.open, q.price) ||
    q.low > Math.min(q.open, q.price)
  )
    return null;
  return {
    time: q.trade_date,
    open: q.open,
    high: q.high,
    low: q.low,
    close: q.price,
    volume: q.volume,
  };
}
export function safeUrl(value) {
  try {
    const u = new URL(value);
    return u.protocol === "https:" ? u.href : undefined;
  } catch {
    return undefined;
  }
}
export function aggregateCandles(rows, unit = "day") {
  if (unit === "day") return rows;
  const groups = new Map();
  for (const row of rows) {
    const date = new Date(row.time + "T00:00:00Z");
    if (unit === "week")
      date.setUTCDate(date.getUTCDate() - ((date.getUTCDay() + 6) % 7));
    else if (unit === "month") date.setUTCDate(1);
    else {
      date.setUTCMonth(0, 1);
    }
    const key = date.toISOString().slice(0, 10);
    const previous = groups.get(key);
    if (!previous) groups.set(key, { ...row, time: key });
    else {
      previous.high = Math.max(previous.high, row.high);
      previous.low = Math.min(previous.low, row.low);
      previous.close = row.close;
      previous.volume += row.volume;
    }
  }
  return [...groups.values()];
}
export const gaugeLabels = {
  ready: "실시간 관측 중",
  no_quote: "수신된 시세 없음",
  wrong_session: "정규장 기준 확인 필요",
  previous_day: "이전 거래일 시세",
  market_closed: "정규장 종료",
  stale: "시세 확인 지연",
  incomplete: "장중 데이터 불완전",
  no_prediction: "당일 예측 미발행",
  restricted: "모델 사용 제한",
  zero_or_invalid_bound: "구간 상한 확인 필요",
  invalid_price: "가격 검증 필요",
  invalid_timestamp: "시세 시각 확인 필요",
};
