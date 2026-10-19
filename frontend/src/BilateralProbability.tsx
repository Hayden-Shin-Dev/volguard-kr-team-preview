import { useState } from 'react';
import { number, type Data } from './api';
const pct=(x:number|null)=>x==null?'—':`${number(x*100,2)}%`;
export function BilateralProbability({data}:{data:Data}){
  const [model,setModel]=useState('initial_refit__sigmoid'),[period,setPeriod]=useState('2024');
  if(!data?.models)return null;
  const selected=data.models[model],r=selected?.periods[period];
  const bands=(r?.bands??[]).filter((b:Data)=>b.low<.6||(b.low===.6&&b.high===.7)||b.low===.7).sort((a:Data,b:Data)=>a.side===b.side?a.low-b.low:a.side==='up'?-1:1);
  return <section className="probability-evidence" aria-label="상승 하락 대칭 확률 검증">
    <span className="eyebrow">상승도 하락도 같은 기준으로</span><h3>예측한 방향과 확률, 실제 결과를 함께 봅니다</h3>
    <p>하락 60%도 상승 60%와 같은 가치의 연구 신호입니다. 평균이 ±5%p 안에 있다는 사실과 충분히 검증됐다는 판정을 구분합니다. 정확히 50%인 예측은 방향 신호에서 제외합니다.</p>
    <label>보존 모델 <select aria-label="양방향 검증 모델" value={model} onChange={e=>setModel(e.target.value)}><option value="initial_refit__sigmoid">초기 모델 · 고정 보정</option><option value="technical_recent__market_sigmoid">최근 연구 모델 · 고정 보정</option></select></label>
    <div className="probability-bins">{['2023','2024','2023-H1','2023-H2','2024-H1','2024-H2'].map(y=><button key={y} aria-pressed={period===y} onClick={()=>setPeriod(y)}>{y.replace('-H1',' 상반기').replace('-H2',' 하반기')}</button>)}</div>
    <p>{number(r?.rows)}건 · {r?.dates}거래일. 이미 사용한 과거 자료의 진단입니다. 이번에 학습·보정·후보 선택을 다시 하지 않았으며 운영 모델도 교체하지 않았습니다.</p>
    <div className="direction-table"><table><thead><tr><th>방향·확률 구간</th><th>예측 평균</th><th>실제 해당 방향 빈도</th><th>실제 빈도 95% 구간</th><th>건수 / 날짜</th><th>전체 중 발생</th><th>연구 해석</th></tr></thead><tbody>{bands.map((b:Data)=><tr key={`${b.side}-${b.low}`}><td>{b.side==='up'?'상승':'하락'} {Math.round(b.low*100)}~{Math.round(b.high*100)}% 미만</td><td>{pct(b.mean_probability)}</td><td>{pct(b.actual_frequency)}</td><td>{b.frequency_ci95?b.frequency_ci95.map(pct).join(' ~ '):'표본 부족'}</td><td>{number(b.n)}건 / {b.dates}일</td><td>{pct(b.coverage)}</td><td>{b.research_label}</td></tr>)}</tbody></table></div>
    <details><summary>여러 시기에도 유지됐나요?</summary><p>아래는 사후 진단이며 운영 승격 기준을 낮추지 않습니다. 반기별 표본 부족도 미확인으로 남깁니다. 서로 다른 날짜도 독립으로 간주하지 않고 20거래일 블록으로 신뢰구간을 계산했습니다.</p><div className="direction-table"><table><thead><tr><th>방향·구간</th><th>두 연도 평균 ±5%p</th><th>네 반기 평균 ±5%p</th><th>네 반기 기존 표본·정합 요건</th></tr></thead><tbody>{selected.stability.filter((b:Data)=>(b.low===.6&&b.high===.7)||b.low===.7).map((b:Data)=><tr key={`${b.side}-${b.low}`}><td>{b.side==='up'?'상승':'하락'} {b.low*100}~{b.high*100}%</td><td>{b.both_year_means_close?'근접':'미충족'}</td><td>{b.all_half_means_close?'근접':'미충족·표본 없음 포함'}</td><td>{b.all_half_supported?'회고 요건 충족':'미확인'}</td></tr>)}</tbody></table></div></details>
    <p>현재 변동폭 구간은 상승·하락에 공통인 예측입니다. 이 표만으로 상승 시·하락 시 변동폭을 별도로 검증했다고 표시하지 않습니다.</p>
  </section>;
}
