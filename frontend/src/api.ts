export type Data = Record<string, any>;
let csrf = "";
export async function api<T = Data>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const send = () =>
    fetch("/api" + path, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        "X-VG-CSRF": csrf,
        ...options.headers,
      },
    });
  let response = await send();
  // A local service restart rotates its CSRF token. Refresh once so polling recovers.
  if (response.status === 403 && options.method && path !== "/session") {
    const renewed = await fetch("/api/session");
    if (renewed.ok) {
      const value = await renewed.json();
      csrf = value.csrf;
      response = await send();
    }
  }
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "입력 또는 서버 상태를 확인해주세요.",
    );
  return data;
}
export async function session() {
  const data = await api("/session");
  csrf = data.csrf;
  return data;
}
export const number = (value: any, digits = 0) =>
  typeof value === "number" && Number.isFinite(value)
    ? (value === 0 ? 0 : value).toLocaleString("ko-KR", {
        maximumFractionDigits: digits,
        minimumFractionDigits: digits,
      })
    : "—";
export function financialAmount(value: unknown, currency: string) {
  const numeric = Number(String(value ?? "").replaceAll(",", ""));
  if (value == null || value === "" || !Number.isFinite(numeric)) return "—";
  if (currency !== "KRW") return `${number(numeric)} ${currency || ""}`;
  if (Math.abs(numeric) >= 1e12) return `${number(numeric / 1e12, 2)}조원`;
  if (Math.abs(numeric) >= 1e8) return `${number(numeric / 1e8, 2)}억원`;
  return `${number(numeric)}원`;
}
export const dateTime = (value: any) =>
  value
    ? new Date(value).toLocaleString("ko-KR", {
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
      })
    : "—";
