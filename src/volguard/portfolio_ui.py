import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .investigation import Investigations, capture
from .portfolio import calculate, current_result
from .util import digest


def render(settings, store, bars, sessions, members, bars_path):
    st.caption("1일 95% 변동성 조정 역사적 VaR · 롱온리·무차입 · 현재 구성의 가상 과거 검증")
    saved = store.get("meta", "portfolio", {"holdings": {}, "cash": 0.0})
    symbols = sorted({w["symbol"] for w in store.watchlist()} | set(saved["holdings"]))
    if bars.empty or not sessions:
        st.info("일봉 데이터와 거래일 달력이 준비되면 계산할 수 있습니다.")
        return
    with st.form("portfolio"):
        entry = pd.DataFrame([{"코드": s, "종목": members.get(s, s),
                               "평가금액 (원)": float(saved["holdings"].get(s, 0))} for s in symbols],
                             columns=["코드", "종목", "평가금액 (원)"])
        edited = st.data_editor(entry, disabled=["코드", "종목"], hide_index=True, width="stretch",
                                column_config={"평가금액 (원)": st.column_config.NumberColumn(min_value=0., step=100000.)})
        cash = st.number_input("현금 (원)", min_value=0., value=float(saved["cash"]), step=100000.)
        submitted = st.form_submit_button("구성 저장 · 검증 후 계산", type="primary")
    st.caption("관심종목에서 제거해도 저장된 보유는 유지됩니다. 매도·제외한 보유는 평가금액을 0으로 바꾸세요.")
    bars_hash = digest(bars_path.read_bytes())
    if submitted:
        try:
            holdings = {r["코드"]: r["평가금액 (원)"] for r in edited.to_dict("records")}
            if not {s for s, value in holdings.items() if value > 0}.issubset(members):
                raise ValueError("고정 유니버스 밖의 보유는 제외한 뒤 계산하세요")
            with st.spinner("현재 구성의 순차 백테스트와 스트레스 손익을 계산합니다."):
                calculate(store, bars, sessions, holdings, cash, bars_hash)
        except ValueError as exc:
            st.error(str(exc))
    result = current_result(store, bars_hash, sessions)
    if result is None:
        st.info("현재 구성·데이터에 맞는 계산이 필요합니다. 저장 후 계산 버튼을 눌러주세요.")
        return
    if result["status"] in {"passed", "cash_only"}:
        st.metric("1일 95% VaR", f"₩{result['var_krw']:,.0f}")
        st.caption("관문 통과는 손실 상한이나 정확성 보장이 아닙니다. 현재 구성에만 적용됩니다.")
    else:
        st.warning(f"검증 미통과 / 표본 부족 — VaR 수치를 표시하지 않습니다. {result.get('reason', '')}")
    st.caption(f"기준일: {result.get('as_of', '주식 노출 없음')} · 계산: {result['calculated_at']} · 구성 {result['portfolio_hash'][:12]}")
    if result["exposures"]:
        exposure = pd.DataFrame(result["exposures"])
        exposure["종목"] = exposure.symbol.map(members).fillna(exposure.symbol)
        exposure["비중 (%)"] = 100 * exposure.weight
        st.dataframe(exposure[["종목", "value_krw", "비중 (%)"]].rename(columns={"value_krw": "평가금액 (원)"}),
                     hide_index=True, width="stretch")
    st.caption(f"현금 비중: {result['cash_weight']:.1%}")
    st.subheader("스트레스 테스트")
    scenarios = pd.DataFrame([{k: r.get(k) for k in ("scenario", "pnl_krw", "status", "reason")}
                              for r in result["stress"]])
    st.dataframe(scenarios.rename(columns={"scenario": "시나리오", "pnl_krw": "손익 (원)",
                                           "status": "구분", "reason": "제한 사유"}), hide_index=True, width="stretch")
    available = scenarios.dropna(subset=["pnl_krw"])
    if not available.empty:
        fig = go.Figure(go.Bar(x=available.scenario, y=available.pnl_krw, marker_color="#ee7b7b"))
        fig.update_layout(height=280, yaxis_title="가정 손익 (원)", template="plotly_dark")
        st.plotly_chart(fig, width="stretch")
    st.caption("과거 충격의 재현 또는 가정 손익입니다. 발생 확률이나 금리·환율의 인과적 영향 추정이 아닙니다.")
    with st.expander("종목별 스트레스 손익·검정 근거"):
        st.json(result)
    st.download_button("포트폴리오 위험 결과 JSON 다운로드", json.dumps(result, ensure_ascii=False, indent=2),
                       file_name=f"volguard-portfolio-{result['portfolio_hash'][:12]}.json", mime="application/json")
    if st.button("현재 포트폴리오 근거 조사 요청"):
        snapshot = capture(settings, symbols=sorted(result["holdings"]))
        job_id = Investigations(settings.data_dir).enqueue(snapshot)
        st.success(f"조사 요청 저장: {job_id[:12]}. ‘근거 조사 보고서’ 화면에서 진행 상태와 결과를 확인하세요.")
