import json
from datetime import datetime, timezone

import streamlit as st

from .dart_evidence import api_key
from .investigation import Investigations, capture, markdown_report
from .report_ai import configuration


def render(settings, store, members, demo):
    jobs = Investigations(settings.data_dir)
    config = configuration()
    st.caption("위험 상태 → 공시 목록·시장 자료 수집 → 근거·반대 근거·불확실성 구분 → 보고서 보존")
    st.caption("공시 연결: OpenDART 회사별 검색·본문 일부 발췌" if api_key() else "공시 연결: 공개 RSS 목록. DART_API_KEY 설정 시 회사별 검색·본문 발췌 사용")
    if not config["ready"]:
        st.info("근거 수집·정형 보고서는 사용 가능합니다. AI 공급자와 모델을 설정하세요. 로컬 Ollama는 API 키가 필요 없습니다.")
    else:
        st.caption(f"AI 연결: {config['provider']} / {config['model']}. 실제 호출 성공 여부는 보고서별로 표시합니다.")
    worker = store.get("meta", "report_worker", {})
    healthy = False
    if worker.get("heartbeat"):
        healthy = (datetime.now(timezone.utc) - datetime.fromisoformat(worker["heartbeat"])).total_seconds() < 180
    if not healthy:
        st.warning("보고서 수집기 연결을 확인할 수 없습니다. 요청은 저장되며 수집기가 시작되면 처리됩니다.")
    prefs = store.get("meta", "investigation_preferences", {"auto_alerts": True})
    automatic = st.checkbox("80%·100% 알림의 근거 보고서 자동 생성", value=prefs.get("auto_alerts", True))
    if automatic != prefs.get("auto_alerts", True):
        store.put("meta", "investigation_preferences", {"auto_alerts": automatic})
    st.caption("자동 보고서는 근거를 수집하고 정형 보고서를 저장합니다. AI 해석은 아래에서 직접 선택합니다. 공시 본문 미확인 여부를 표시합니다.")
    watched = [w["symbol"] for w in store.watchlist() if w["symbol"] in members]
    symbols = st.multiselect("조사 대상", list(members), default=watched, max_selections=30,
                             format_func=lambda s: f"{s} {members[s]}")
    use_ai = st.checkbox("AI 근거 선별 추가" if config["provider"] == "ollama" else "AI 해석 추가",
                         value=False, disabled=not config["ready"] or demo)
    if use_ai:
        if config["provider"] == "ollama":
            st.caption("이 PC의 Ollama가 검토할 위험 지표·공시 원문을 선별합니다. 자유 서술은 제공하지 않으며 보유 평가금액·현금·인증키는 입력에서 제외합니다.")
        else:
            st.caption("선택 종목의 위험 지표·공시 자료를 설정한 AI로 전송합니다. 보유 평가금액·현금·인증키는 전송하지 않습니다.")
    if st.button("근거 조사 시작", type="primary", disabled=not symbols):
        try:
            job_id = jobs.enqueue(capture(settings, symbols=symbols), use_ai=use_ai)
            st.success(f"조사 요청 저장: {job_id[:12]}")
        except ValueError as exc:
            st.error(str(exc))
    history(jobs)


@st.fragment(run_every="5s")
def history(jobs):
    entries = jobs.recent()
    st.subheader("보고서 이력")
    if not entries:
        st.caption("아직 보고서가 없습니다.")
        return
    st.dataframe(entries, hide_index=True, width="stretch")
    chosen = st.selectbox("보고서 선택", [e["id"] for e in entries],
                          format_func=lambda x: next(f"{e['created_at']} · {e['status']} · {x[:8]}" for e in entries if e["id"] == x))
    report = jobs.read(chosen)
    if report is None:
        st.caption("처리 대기·진행 중이거나 실패한 요청입니다. 실패 요청은 새 조사로 다시 요청할 수 있습니다.")
        return
    st.markdown(markdown_report(report))
    if report["ai"]["status"] == "completed":
        st.caption(report["ai"]["validation"])
    st.download_button("보고서 Markdown 다운로드", markdown_report(report),
                       file_name=f"volguard-risk-{chosen[:12]}.md", mime="text/markdown")
    st.download_button("근거·수치 JSON 다운로드", json.dumps(report, ensure_ascii=False, indent=2),
                       file_name=f"volguard-risk-{chosen[:12]}.json", mime="application/json")
    with st.expander("수집 당시 수치·근거 원문·해시"):
        st.json(report)
