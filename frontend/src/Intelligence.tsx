import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  ArrowDownToLine,
  ArrowRight,
  Check,
  ExternalLink,
  LoaderCircle,
  Play,
  X,
} from "lucide-react";
import { api, dateTime, number, financialAmount, type Data } from "./api";
import marketPortrait from "./assets/agents/market.png";
import companyPortrait from "./assets/agents/company.png";
import riskPortrait from "./assets/agents/risk.png";
import criticPortrait from "./assets/agents/critic.png";
import { safeUrl, gaugeLabels } from "./domain.mjs";

const roles = ["market", "company", "risk", "critic"];
const statementNames: Data = {
  BS: "재무상태표",
  IS: "손익계산서",
  CIS: "포괄손익계산서",
  CF: "현금흐름표",
  SCE: "자본변동표",
};
const names: Data = {
  market: "시장 분석가",
  company: "기업 분석가",
  risk: "리스크 분석가",
  critic: "반론 검토자",
};
const tools: Data = {
  market: "가격 · 거래량 · 시장 지수",
  company: "DART 공시 · 재무계정",
  risk: "예측 · VaR · 스트레스",
  critic: "주장 · 원문 · 반대 근거",
};
const labels: Data = {
  queued: "대기",
  running: "조사 중",
  completed: "검증 완료",
  partial: "일부 제한",
  failed: "실행 실패",
  interrupted: "실행 중단",
  limited: "자료·출력 제한",
};
const stance: Data = {
  risk: "위험 근거",
  counter: "반대 근거",
  uncertainty: "확인 필요",
};
const verdict: Data = {
  supported: "근거 지지",
  challenge: "반론",
  insufficient: "근거 부족",
};
const portraits: Data = {
  market: marketPortrait,
  company: companyPortrait,
  risk: riskPortrait,
  critic: criticPortrait,
};
export function Avatar({
  role,
  active = false,
}: {
  role: string;
  active?: boolean;
}) {
  return (
    <img
      className={"avatar portrait " + (active ? "active" : "")}
      src={portraits[role]}
      alt={`${names[role]} 캐릭터`}
    />
  );
}

function Evidence({ source, onClose }: { source: Data; onClose: () => void }) {
  useEffect(() => {
    const close = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [onClose]);
  const url = safeUrl(source.url);
  return createPortal(
    <div className="evidence-drawer" role="dialog" aria-label="원문 근거">
      <div className="drawer-heading">
        <span className="eyebrow">근거 {source.id}</span>
        <button autoFocus aria-label="근거 닫기" onClick={onClose}>
          <X size={18} />
        </button>
      </div>
      <h3>{source.title}</h3>
      <p className="caption">
        수집 {dateTime(source.retrieved_at)}
        {source.receipt_date && ` · 공시 ${source.receipt_date}`}
      </p>
      {source.selectedQuotes?.length ? (
        <>
          <h4>이 주장에 사용된 원문 발췌</h4>
          {source.selectedQuotes.map((q: Data, i: number) => (
            <blockquote className="verified-excerpt" key={i}>
              {q.quote}
            </blockquote>
          ))}
          <details>
            <summary>수집한 원문 전체 펼치기</summary>
            <pre>{source.text}</pre>
          </details>
        </>
      ) : (
        <pre>{source.text}</pre>
      )}
      {url && (
        <a className="text-action" href={url} target="_blank" rel="noreferrer">
          원문에서 확인 <ExternalLink size={14} />
        </a>
      )}
      <p className="fineprint">
        인용 일치와 수치 검사는 주장 전체의 의미적 정확성이나 인과관계를
        보장하지 않습니다.
      </p>
    </div>,
    document.body,
  );
}

export function Intelligence({
  analysisId,
  symbol,
  name,
  session,
  universe,
  error,
}: {
  analysisId?: string;
  symbol: string;
  name: string;
  session: Data;
  universe: Data[];
  error: (s: string) => void;
}) {
  const [run, setRun] = useState<Data | null>(null),
    [history, setHistory] = useState<Data[]>([]),
    [legacy, setLegacy] = useState<Data[]>([]),
    [busy, setBusy] = useState(false),
    [source, setSource] = useState<Data | null>(null),
    [focused, setFocused] = useState(""),
    [panel, setPanel] = useState("summary"),
    [mobileDetail, setMobileDetail] = useState(false),
    [now, setNow] = useState(Date.now()),
    [streamError, setStreamError] = useState(false);
  const conversation = useRef<HTMLDivElement>(null);
  const follow = useRef(true);
  const running = run && ["queued", "running"].includes(run.status);
  function refreshHistory() {
    api<Data[]>("/investigations")
      .then(setHistory)
      .catch((e) => error(e.message));
  }
  useEffect(() => {
    refreshHistory();
    api<Data[]>("/reports/legacy")
      .then(setLegacy)
      .catch(() => {});
    const key = sessionStorage.getItem("volguard-investigation");
    if (key)
      api(`/investigations/${key}`)
        .then(setRun)
        .catch(() => sessionStorage.removeItem("volguard-investigation"));
  }, []);
  useEffect(() => {
    if (run?.id) sessionStorage.setItem("volguard-investigation", run.id);
  }, [run?.id]);
  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [running]);
  useEffect(() => {
    if (!run?.id || !running) return;
    let disposed = false;
    const es = new EventSource(
      `/api/investigations/${run.id}/events?after=${run.events?.at(-1)?.seq || 0}`,
    );
    es.onopen = () => setStreamError(false);
    es.onmessage = () => {
      api(`/investigations/${run.id}`)
        .then((r) => {
          if (!disposed) {
            setRun(r);
            if (!["queued", "running"].includes(r.status)) {
              es.close();
              refreshHistory();
            }
          }
        })
        .catch((e) => error(e.message));
    };
    es.onerror = () => {
      setStreamError(true);
      api(`/investigations/${run.id}`)
        .then((r) => {
          if (!disposed) setRun(r);
        })
        .catch(() => {});
    };
    return () => {
      disposed = true;
      es.close();
    };
  }, [run?.id, !!running]);
  async function start() {
    setBusy(true);
    try {
      const r = await api("/investigations", {
        method: "POST",
        body: JSON.stringify({
          symbols: [symbol],
          request_id: crypto.randomUUID().replaceAll("-", ""),
          analysis_id: analysisId || null,
        }),
      });
      setRun(await api(`/investigations/${r.id}`));
      refreshHistory();
    } catch (e) {
      error((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const stages = run?.stages || {},
    evidence = stages.evidence || {},
    report = stages.report;
  const outputs: Data[] = Object.entries(stages)
    .filter(([k]) => k.includes(":"))
    .map(([, v]) => v as Data);
  const claims: Data[] =
      report?.claims || outputs.flatMap((o) => o.claims || []),
    links: Data[] = report?.links || outputs.flatMap((o) => o.reviews || []);
  const lastEvent = run?.events?.at(-1),
    started = run ? Date.parse(run.created_at) : 0,
    elapsed = run
      ? Math.max(
          0,
          Math.round(
            ((running ? now : Date.parse(lastEvent?.at || run.created_at)) -
              started) /
              1000,
          ),
        )
      : 0;
  const progress = report
    ? 4
    : outputs.length >= 4
      ? 3
      : outputs.length
        ? 2
        : evidence.sources
          ? 1
          : 0;
  useEffect(() => {
    if (running && follow.current && conversation.current)
      conversation.current.scrollTop = conversation.current.scrollHeight;
  }, [run?.events?.length]);
  useEffect(() => {
    if (conversation.current && !running) conversation.current.scrollTop = 0;
  }, [run?.id]);
  const selectClaim = (id: string) => {
    setFocused("");
    requestAnimationFrame(() =>
      document
        .getElementById(id)
        ?.scrollIntoView({ behavior: "smooth", block: "nearest" }),
    );
  };
  const messages = Object.entries(stages)
    .filter(([key]) => key.includes(":"))
    .map(([key, value]) => ({ key, ...(value as Data) }) as Data)
    .sort((a, b) => {
      const sequence = (o: Data) =>
        run?.events?.find(
          (e: Data) =>
            e.role === o.role &&
            e.round === o.round &&
            ["agent_completed", "agent_failed"].includes(e.phase),
        )?.seq ?? (o.round === "review" ? 10000 : 0) + roles.indexOf(o.role);
      return sequence(a) - sequence(b);
    });
  const sourceButtons = (ids: string[] = [], quotes: Data[] = []) => (
    <div className="source-chips">
      {[...new Set(ids)].map((id) => (
        <button
          key={id}
          onClick={() => {
            const selected = evidence.sources?.find((s: Data) => s.id === id);
            setSource(
              selected
                ? {
                    ...selected,
                    selectedQuotes: quotes.filter((q) => q.source_id === id),
                  }
                : null,
            );
          }}
        >
          {id} 원문 <ArrowRight size={12} />
        </button>
      ))}
    </div>
  );
  return (
    <div
      className={
        "intelligence-page conversation-page page-enter " +
        (mobileDetail ? "detail-open" : "")
      }
    >
      <header className="room-header">
        <div>
          <span className="eyebrow">VOLGUARD / RESEARCH ROOM</span>
          <h1>네 개의 시선, 하나의 조사</h1>
          <p>
            {run?.stages?.snapshot?.stocks
              ?.map((s: Data) => s.name)
              .join(" · ") || name}{" "}
            · 조사 당시 자료를 기반으로 한 AI 분석
          </p>
        </div>
        <div className="room-actions">
          <button
            className="inspector-toggle"
            onClick={() => setMobileDetail(!mobileDetail)}
          >
            {mobileDetail ? "대화로 돌아가기" : "종합 · 자료 보기"}
          </button>
          <button
            className="primary"
            disabled={busy || !!running || !session.ai?.ready}
            onClick={start}
          >
            {busy || running ? (
              <LoaderCircle size={16} className="spin" />
            ) : (
              <Play size={15} />
            )}{" "}
            {running ? "분석 진행 중" : `${name} 심층 조사`}
          </button>
        </div>
      </header>
      {analysisId && run && run.analysis_id !== analysisId && (
        <div className="inline-warning">
          현재 선택한 시나리오와 아래 저장 보고서의 분석 기준이 다릅니다. 아래
          보고서는 당시 기준으로 보존되며, 심층 조사를 시작하면 현재 분석으로 새
          보고서를 만듭니다.
        </div>
      )}
      <div className="room-meta">
        <span>
          <i className={"status-dot " + (running ? "live" : "")} />
          {run ? labels[run.status] : "시작 대기"}
        </span>
        <span>
          {elapsed}초 · {claims.length}개 주장 · {links.length}개 검토
        </span>
        <select
          aria-label="저장된 조사 선택"
          value={run?.id || ""}
          onChange={(e) => {
            if (e.target.value)
              api(`/investigations/${e.target.value}`)
                .then(setRun)
                .catch((e) => error(e.message));
            else setRun(null);
          }}
        >
          <option value="">새 조사</option>
          {history.map((h) => (
            <option key={h.id} value={h.id}>
              {dateTime(h.created_at)} ·{" "}
              {h.symbols
                .map(
                  (s: string) =>
                    universe.find((u) => u.symbol === s)?.name || s,
                )
                .join(", ")}{" "}
              · {labels[h.status]}
            </option>
          ))}
        </select>
        {run && (
          <a
            href={`/api/investigations/${run.id}/download`}
            aria-label="조사 보고서 내려받기"
          >
            <ArrowDownToLine size={17} />
          </a>
        )}
      </div>
      <div className="progress-track">
        {["자료 확보", "독립 분석", "반론 검토", "교차 검토", "종합 보고"].map(
          (label, i) => (
            <div key={label} className={run && i <= progress ? "done" : ""}>
              <span>
                {i < progress || report ? <Check size={11} /> : i + 1}
              </span>
              {label}
            </div>
          ),
        )}
      </div>
      {streamError && running && (
        <p className="inline-warning">
          진행 연결 복구 중 · 서버의 조사는 계속됩니다.
        </p>
      )}
      <div className="conversation-layout">
        <aside className="participants">
          <button
            className={"all-participants " + (!focused ? "selected" : "")}
            onClick={() => setFocused("")}
          >
            전체 대화 <span>04</span>
          </button>
          {roles.map((role) => {
            const last = run?.events
              ?.filter((e: Data) => e.role === role)
              .at(-1);
            const active =
              !!running &&
              ["agent_started", "tool_started"].includes(last?.phase);
            const output = outputs.filter((o) => o.role === role);
            return (
              <button
                key={role}
                className={
                  "participant " + (focused === role ? "selected" : "")
                }
                onClick={() => setFocused(focused === role ? "" : role)}
              >
                <Avatar role={role} active={active} />
                <div>
                  <strong>{names[role]}</strong>
                  <small>{tools[role]}</small>
                  <span>
                    {active
                      ? last.tool || "분석 중"
                      : output.length
                        ? output.some((o) => o.status !== "completed")
                          ? "일부 결과 제한"
                          : "검토 완료"
                        : "대기 중"}
                  </span>
                </div>
                {active && <LoaderCircle className="spin" size={13} />}
              </button>
            );
          })}
          <p className="room-privacy">
            실제 분석 결과만 대화에 표시합니다.
            <br />새 조사마다 공개 자료와 분석 근거를 OpenAI에 전송합니다. 최대
            {session.ai?.max_calls ?? 11}회 요청(검증 수정 포함).
          </p>
        </aside>
        <section className="chat-column" aria-label="AI 분석 대화">
          <div className="chat-heading">
            <strong>{focused ? names[focused] : "분석가 대화"}</strong>
            <span>독립 분석 → 교차 검토</span>
          </div>
          <div
            className="conversation-scroll"
            ref={conversation}
            onScroll={(e) => {
              const t = e.currentTarget;
              follow.current =
                t.scrollHeight - t.scrollTop - t.clientHeight < 90;
            }}
          >
            {!run && (
              <div className="chat-welcome">
                <div className="portrait-orbit">
                  {roles.map((role) => (
                    <Avatar key={role} role={role} />
                  ))}
                </div>
                <h2>위험의 이유를 함께 살펴보세요.</h2>
                <p>
                  시장 흐름, 기업 공시, 위험 수치.
                  <br />네 분석가가 자료를 확인하고 서로의 주장을 검토합니다.
                </p>
                <span>위의 심층 조사 버튼으로 시작하세요.</span>
              </div>
            )}
            {run && (
              <div className="chat-system">
                {dateTime(run.created_at)} 조사 시작 · 원문 검증을 통과한 결과
              </div>
            )}
            {messages
              .filter((o) => !focused || o.role === focused)
              .map((o) => (
                <article className="chat-message" key={o.key}>
                  <Avatar role={o.role} />
                  <div className="chat-content">
                    <header>
                      <strong>{names[o.role]}</strong>
                      <span>
                        {o.round === "review" ? "교차 검토" : "독립 분석"} ·{" "}
                        {number(o.elapsed_seconds, 1)}초
                      </span>
                    </header>
                    <div className="chat-bubble">
                      {(o.claims || []).map((c: Data) => (
                        <div className="chat-claim" id={c.id} key={c.id}>
                          <span className="claim-tag">{stance[c.stance]}</span>
                          <p>{c.text}</p>
                          {sourceButtons(c.source_ids, c.quotes)}
                        </div>
                      ))}
                      {(o.reviews || []).map((r: Data, i: number) => (
                        <div className="chat-review" key={i}>
                          <button
                            className="reply-reference"
                            onClick={() => selectClaim(r.target_claim_id)}
                          >
                            ↳{" "}
                            {names[
                              claims.find((c) => c.id === r.target_claim_id)
                                ?.role
                            ] || "앞선 분석"}
                            의 주장{" "}
                            <span>
                              {claims.find((c) => c.id === r.target_claim_id)
                                ?.text || r.target_claim_id}
                            </span>
                          </button>
                          <span className={"claim-tag " + r.verdict}>
                            {verdict[r.verdict]}
                          </span>
                          <p>{r.reason}</p>
                          {sourceButtons(r.source_ids, r.quotes)}
                        </div>
                      ))}
                      {(o.responses || []).map((r: Data, i: number) => (
                        <div className="chat-review" key={`response-${i}`}>
                          <span className="claim-tag">
                            반론에 대한{" "}
                            {
                              (
                                {
                                  maintain: "설명",
                                  revise: "주장 수정",
                                  withdraw: "주장 철회",
                                } as Data
                              )[r.action]
                            }
                          </span>
                          <p>{r.reason}</p>
                          {r.action === "revise" && (
                            <p>
                              <b>수정한 주장</b> {r.replacement_text}
                            </p>
                          )}
                          {sourceButtons(r.source_ids, r.quotes)}
                        </div>
                      ))}
                      {o.repair_attempt && (
                        <details className="chat-exclusion">
                          <summary>원문 검증 후 답변 수정 1회</summary>
                          <p>
                            첫 답변의 수치·인용 오류를 분석가에게 돌려줬습니다.
                            아래 최종 답변에도 같은 검증을 적용했습니다.
                          </p>
                          {o.repair_attempt.initial_errors.map(
                            (r: Data, i: number) => (
                              <p key={i}>{r.reason}</p>
                            ),
                          )}
                        </details>
                      )}
                      {!!o.rejected?.length && (
                        <div className="chat-exclusion">
                          원문 대조에서 {o.rejected.length}개 결과 제외{" "}
                          <details>
                            <summary>사유 확인</summary>
                            {o.rejected.map((r: Data, i: number) => (
                              <p key={i}>{r.reason}</p>
                            ))}
                          </details>
                        </div>
                      )}
                      {o.status === "failed" && (
                        <p className="inline-warning">
                          {o.error || "분석 실행 실패"}
                        </p>
                      )}
                      {!o.claims?.length &&
                        !o.reviews?.length &&
                        !o.responses?.length &&
                        !o.rejected?.length && (
                          <p>
                            {o.limitations?.join(" · ") ||
                              "확인된 분석 결과가 없습니다."}
                          </p>
                        )}
                    </div>
                  </div>
                </article>
              ))}
            {running && (
              <div className="chat-system">
                <LoaderCircle className="spin" size={15} />
                {lastEvent?.message || "자료 확인 중"}
              </div>
            )}
            {stages.error && (
              <p className="inline-warning">{stages.error.message}</p>
            )}
          </div>
          <div className="chat-footer">
            <span>
              {running
                ? "실제 실행 이벤트 수신 중"
                : "저장된 분석 · 원문과 함께 확인하세요"}
            </span>
            <button
              onClick={() => {
                follow.current = true;
                conversation.current?.scrollTo({
                  top: conversation.current.scrollHeight,
                  behavior: "smooth",
                });
              }}
            >
              최근 대화 ↓
            </button>
          </div>
        </section>
        <aside className="research-inspector">
          <nav aria-label="조사 상세">
            <button
              className={panel === "summary" ? "selected" : ""}
              onClick={() => setPanel("summary")}
            >
              종합
            </button>
            <button
              className={panel === "data" ? "selected" : ""}
              onClick={() => setPanel("data")}
            >
              자료
            </button>
            <button
              className={panel === "events" ? "selected" : ""}
              onClick={() => setPanel("events")}
            >
              실행 기록
            </button>
          </nav>
          <div className="inspector-scroll">
            {panel === "summary" && (
              <>
                <span className="eyebrow">RESEARCH BRIEF</span>
                <h2>
                  {report?.summary || "조사가 끝나면 핵심 내용을 모아드립니다."}
                </h2>
                {report && (
                  <>
                    {stages.analysis && (
                      <div className="scenario-report-context">
                        <h3>이 보고서의 분석 기준</h3>
                        <p>
                          {stages.analysis.target_date} · 변동폭{" "}
                          {number(stages.analysis.prediction.low, 2)}~
                          {number(stages.analysis.prediction.high, 2)}%
                        </p>
                        <p>
                          방향 확률 미표시: {stages.analysis.direction.reason}
                        </p>
                        <p>
                          선택한 가상 경로:{" "}
                          {
                            (
                              { up: "상승", flat: "횡보", down: "하락" } as Data
                            )[stages.analysis.scenario.direction]
                          }{" "}
                          · 종가 수익률{" "}
                          {number(
                            stages.analysis.scenario.close_return * 100,
                            2,
                          )}
                          %
                        </p>
                        <p>
                          등록 포트폴리오의 가상 손익{" "}
                          {number(stages.analysis.impact.pnl_krw)}원
                        </p>
                        <small>
                          가정에 따른 계산이며 경로 발생 확률이 아닙니다. 분석{" "}
                          {stages.analysis.id.slice(0, 8)}
                        </small>
                      </div>
                    )}
                    <h3>핵심 위험</h3>
                    {report.risk_claim_ids?.length ? (
                      report.risk_claim_ids.map((id: string) => (
                        <button
                          className="brief-claim"
                          key={id}
                          onClick={() => selectClaim(id)}
                        >
                          {claims.find((c) => c.id === id)?.text}
                          <ArrowRight size={15} />
                        </button>
                      ))
                    ) : (
                      <p>
                        확인된 위험 주장 없음 · 위험이 없다는 의미는 아닙니다.
                      </p>
                    )}
                    <h3>반대 근거 · 불확실성</h3>
                    {report.counter_claim_ids?.map((id: string) => (
                      <p key={id}>{claims.find((c) => c.id === id)?.text}</p>
                    ))}
                    {[
                      ...new Set([
                        ...(report.uncertainties || []),
                        ...(evidence.limitations || []),
                      ]),
                    ].map((s: any, i) => (
                      <p key={i}>{s}</p>
                    ))}
                    <h3>다음 확인 사항</h3>
                    {report.next_checks?.map((s: string, i: number) => (
                      <p key={i}>{s}</p>
                    ))}
                    <small>{report.validation_note}</small>
                  </>
                )}
                <div className="inspector-facts">
                  <span>
                    확인된 주장<strong>{claims.length}</strong>
                  </span>
                  <span>
                    교차 검토<strong>{links.length}</strong>
                  </span>
                  <span>
                    원문 근거<strong>{evidence.sources?.length || 0}</strong>
                  </span>
                </div>
              </>
            )}
            {panel === "data" && (
              <>
                {stages.snapshot && (
                  <section className="data-section">
                    <div className="section-title">
                      <h2>예측 · 실제 관측 상태</h2>
                      <span>
                        조사 시점 고정 · {dateTime(stages.snapshot.captured_at)}
                      </span>
                    </div>
                    {stages.snapshot.stocks.map((s: Data) => (
                      <div className="snapshot-stock" key={s.symbol}>
                        <h3>{s.name}</h3>
                        <p>
                          {gaugeLabels[
                            s.gauge?.status as keyof typeof gaugeLabels
                          ] || "상태 확인 필요"}
                        </p>
                        <div className="risk-stats">
                          <div>
                            <span>
                              {s.prediction
                                ? `${s.prediction.method?.toUpperCase()} · ${s.prediction.target_date}`
                                : "현재 발행 예측"}
                            </span>
                            <b>
                              {s.prediction?.status === "ready"
                                ? `${number(s.prediction.low, 2)} ~ ${number(s.prediction.high, 2)}%`
                                : "사용 가능한 예측 없음"}
                            </b>
                          </div>
                          <div>
                            <span>실제 관측 변동폭</span>
                            <b>{number(s.gauge?.range_pct, 2)}%</b>
                          </div>
                        </div>
                      </div>
                    ))}
                    {stages.snapshot.risk && (
                      <div className="snapshot-risk">
                        <h3>포트폴리오 위험</h3>
                        <p>현재 사용자 구성 · 조사 시점 계산 기록</p>
                        <div className="risk-stats">
                          <div>
                            <span>1일 95% VaR</span>
                            <b>
                              {["passed", "cash_only"].includes(
                                stages.snapshot.risk.status,
                              )
                                ? `${number(stages.snapshot.risk.var_krw)}원`
                                : "검증 관문 미통과"}
                            </b>
                          </div>
                          <div>
                            <span>외표본 표본</span>
                            <b>{number(stages.snapshot.risk.n)}일</b>
                          </div>
                        </div>
                        {stages.snapshot.risk.stress?.map(
                          (s: Data, i: number) => (
                            <div className="snapshot-stress" key={i}>
                              <span>{s.scenario}</span>
                              <b>
                                {s.pnl_krw === null
                                  ? "자료 불완전"
                                  : `${number(s.pnl_krw)}원`}
                              </b>
                            </div>
                          ),
                        )}
                      </div>
                    )}
                  </section>
                )}
                {!!evidence.comparisons?.length && (
                  <section className="data-section">
                    <div className="section-title">
                      <h2>시장 · 종목 비교</h2>
                      <span>동일 기준일 확인</span>
                    </div>
                    <div className="table-scroll">
                      <table>
                        <thead>
                          <tr>
                            <th>종목·지수</th>
                            <th>기준일</th>
                            <th>고저 변동폭</th>
                            <th>최근 22일 평균 / 거래량</th>
                          </tr>
                        </thead>
                        <tbody>
                          {evidence.comparisons.map((c: Data, i: number) => (
                            <tr key={i}>
                              <td>
                                {stages.snapshot?.stocks?.find(
                                  (s: Data) => s.symbol === c.symbol,
                                )?.name || c.symbol}
                              </td>
                              <td>{c.as_of}</td>
                              <td>{number(c.range_pct, 2)}%</td>
                              <td>
                                {number(c.mean_range_pct, 2)}% /{" "}
                                {number(c.volume_ratio, 2)}배
                              </td>
                            </tr>
                          ))}
                          {evidence.indices
                            ?.slice(0, 12)
                            .map((c: Data, i: number) => (
                              <tr key={"i" + i}>
                                <td>{c.name}</td>
                                <td>{c.as_of}</td>
                                <td>{number(c.range_pct, 2)}%</td>
                                <td>종목 업종 배속 미확정</td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  </section>
                )}
                {!!evidence.financials?.length && (
                  <section className="data-section">
                    <div className="section-title">
                      <h2>DART 재무정보</h2>
                      <span>원문 금액 · 기간 · 연결/별도 기준 보존</span>
                    </div>
                    <p className="fineprint">
                      큰 숫자는 당기금액이며 손익은 분기값일 수 있습니다.
                      누적금액을 별도로 표시하고, 조·억원 표시는 반올림합니다.
                      정확한 금액은 원문 근거에서 확인하세요.
                    </p>
                    {evidence.financials.map((f: Data) => (
                      <div key={f.id}>
                        <button
                          className="text-action"
                          onClick={() => setSource(f)}
                        >
                          {f.title} · {f.division === "CFS" ? "연결" : "별도"}
                          <ExternalLink size={14} />
                        </button>
                        <div className="financial-grid">
                          {f.facts?.map((r: Data, i: number) => (
                            <div key={i}>
                              <span>{r.account_nm}</span>
                              <strong
                                title={`${r.thstrm_amount} ${r.currency}`}
                              >
                                {financialAmount(r.thstrm_amount, r.currency)}
                              </strong>
                              <small>
                                {r.thstrm_nm} ·{" "}
                                {statementNames[r.sj_div] || r.sj_div}
                              </small>
                              {r.thstrm_add_amount && (
                                <small>
                                  누적{" "}
                                  {financialAmount(
                                    r.thstrm_add_amount,
                                    r.currency,
                                  )}
                                </small>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </section>
                )}
                {!!evidence.sources?.length && (
                  <section className="sources-index">
                    <div className="section-title">
                      <h2>자료 보관함</h2>
                      <span>{evidence.sources.length}개 근거</span>
                    </div>
                    {evidence.sources.map((s: Data) => (
                      <button key={s.id} onClick={() => setSource(s)}>
                        <span>{s.id}</span>
                        <b>{s.title}</b>
                        <small>{dateTime(s.retrieved_at)}</small>
                        <ArrowRight size={14} />
                      </button>
                    ))}
                  </section>
                )}
              </>
            )}
            {panel === "events" && (
              <>
                <h2>실제 실행 기록</h2>
                {run?.events?.map((e: Data) => (
                  <div className="compact-event" key={e.seq}>
                    <time>{dateTime(e.at)}</time>
                    <strong>{names[e.role] || "조사 시스템"}</strong>
                    <p>{e.message}</p>
                    {e.tool && <small>{e.tool}</small>}
                  </div>
                ))}
                {!!legacy.length && (
                  <details>
                    <summary>기존 운영 보고서 {legacy.length}건</summary>
                    {legacy.map((r) => (
                      <p key={r.id}>
                        <a href={`/api/reports/legacy/${r.id}`}>
                          {dateTime(r.created_at)} 보고서 내려받기
                        </a>
                      </p>
                    ))}
                  </details>
                )}
              </>
            )}
          </div>
        </aside>
      </div>
      {source && <Evidence source={source} onClose={() => setSource(null)} />}
    </div>
  );
}
