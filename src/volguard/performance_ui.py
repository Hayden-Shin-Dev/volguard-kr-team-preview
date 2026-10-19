"""Human-readable performance reports, distinct from operational eligibility."""

import pandas as pd
import streamlit as st

NAMES = {"empirical": "최근 20일 경험적 구간", "har": "HAR + Conformal", "ml": "LightGBM + Conformal"}


def render_metrics(report):
    if "eligible_symbols" in report:
        st.caption(
            f"평가 대상 {len(report['eligible_symbols'])}종목 · 학습 표본·HAR 수렴 조건으로 시험 전에 확정. "
            f"{len(report.get('excluded_symbols', []))}종목은 개별 사용 제한이며 전체 결과도 보고서에 보존합니다."
        )
    rows = []
    for method in ("empirical", "har", "ml"):
        values = report.get("metrics", {}).get(method)
        if values:
            rows.append(
                {
                    "모델": NAMES[method],
                    "평가 표본": values["n"],
                    "실제 포함률 (%)": 100 * values["coverage"] if values["coverage"] is not None else None,
                    "평균 구간 폭 (%p)": values["width"],
                    "Interval score ↓": values["score"],
                }
            )
    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "실제 포함률 (%)": st.column_config.NumberColumn(format="%.2f"),
                "평균 구간 폭 (%p)": st.column_config.NumberColumn(format="%.3f"),
                "Interval score ↓": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.caption(
            "포함률 목표 90%, 허용 87–93%. 구간 폭과 이탈을 함께 벌점화한 Interval score는 낮을수록 좋습니다."
        )
    cols = st.columns(3)
    improvement = report.get("improvement")
    upper = report.get("bootstrap_upper")
    cols[0].metric(
        "ML의 HAR 대비 점수 개선", f"{improvement:.2%}" if improvement is not None else "계산 불가"
    )
    cols[1].metric("점수 차이 95% CI 상한", f"{upper:.4f}" if upper is not None else "계산 불가")
    cols[2].metric("세 모델 공통 예측 비율", f"{report.get('availability', 0):.2%}")
    st.caption(
        "개선율은 동일 종목·거래일의 점수를 날짜별 평균한 뒤 비교합니다. CI는 20거래일 블록 2,000회 부트스트랩입니다."
    )
    statistical = report.get("statistical_selection", report.get("selected", "observation_only"))
    st.write(
        f"통계 관문 선택: **{NAMES.get(statistical, '관측 전용')}** · "
        f"HAR {'통과' if report.get('har_pass') else '미통과'} · "
        f"ML {'통과' if report.get('ml_pass') else '미통과'}"
    )
    if report.get("reason"):
        st.warning(report["reason"])


def render_performance(final, development, portfolio):
    if final:
        st.subheader("실제 시장 데이터 예측 성능" if not final.get("demo") else "합성 데이터 예측 성능")
        st.caption(f"잠금 테스트 {final['test_start']} ~ {final['test_end']} · 버전 {final['study_id']}")
        if final.get("deployment_eligible") is False:
            st.caption("이 보고서는 당시 회고 평가 기록입니다. 현재 운영 릴리스 상태는 위의 서비스 상태에서 별도로 표시합니다.")
        render_metrics(final)
        if final.get("by_year"):
            with st.expander("연도별·급변동 구간·종목별 결과"):
                st.dataframe(final["by_year"], hide_index=True)
                st.dataframe(final.get("regimes", []), hide_index=True)
                st.dataframe(final.get("by_symbol", []), hide_index=True)
        if final.get("limitations"):
            with st.expander("이 수치의 해석과 제한", expanded=True):
                for text in final["limitations"]:
                    st.write("• " + text)
    elif not development:
        st.info(
            "실데이터 성능 평가가 아직 완료되지 않았습니다. API 연결 성공이나 코드 테스트 통과는 예측 성능이 아닙니다."
        )
    if development:
        st.subheader("시간순 개발 검증")
        for fold in development.get("folds", []):
            with st.expander(f"{fold['year']}년 외표본 결과", expanded=not bool(final)):
                render_metrics(fold)
                if fold.get("regimes"):
                    st.dataframe(fold["regimes"], hide_index=True)
    if portfolio:
        with st.expander("기존 무조정 역사적 VaR 비교군 — 보존 결과"):
            st.caption("사전에 정한 예시 구성의 회고 검정입니다. 사용자의 실제 보유나 운영 승인이 아닙니다.")
            rows = []
            for item in portfolio["portfolios"]:
                tests = item.get("tests", {})
                rows.append(
                    {
                        "구성": item["name"],
                        "결과": item["status"],
                        "외표본 일수": item["n"],
                        "초과 비율": item.get("exception_rate"),
                        "Kupiec p": tests.get("kupiec_p"),
                        "Christoffersen p": tests.get("christoffersen_p"),
                        "제한 사유": item.get("reason"),
                    }
                )
            st.dataframe(rows, hide_index=True)
            st.caption(
                "각 평가일 이전 500일로 계산합니다. 미통과·불완전 데이터의 VaR 숫자는 제공하지 않습니다."
            )
