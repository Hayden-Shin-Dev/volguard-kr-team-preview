import { useEffect, useRef, useState } from "react";
import { number, type Data } from "./api";
import { scenarioCandle } from "./domain.mjs";

function useDrawProgress(target: number) {
  const [value, setValue] = useState(target);
  const current = useRef(target);
  useEffect(() => {
    if (target === 0 || matchMedia("(prefers-reduced-motion: reduce)").matches) {
      current.current = target; setValue(target); return;
    }
    let frame = 0;
    const start = performance.now(), from = current.current;
    const draw = (now: number) => {
      const t = Math.min(1, (now - start) / 110);
      current.current = from + (target - from) * (1 - (1-t) ** 3);
      setValue(current.current);
      if (t < 1) frame = requestAnimationFrame(draw);
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [target]);
  return value;
}

export function NextDayCanvas({ before, analysis, progress, actual }: {
  before: Data[]; analysis: Data; progress: number; actual?: Data | null;
}) {
  const prediction = analysis.prediction;
  const p = useDrawProgress(Math.min(1, Math.max(0, progress)));
  const zoom = Math.min(1,p/.3), reveal = Math.max(0,(p-.3)/.7);
  const probability = analysis.direction?.probability;
  const past = before.slice(-12), recent = past.slice(-3);
  const anchor = past.at(-1)?.close || analysis.scenario.anchor_price;
  const outer = scenarioCandle(anchor,prediction.high,'up');
  if (!outer || !past.length) return null;
  const extrema = (rows: Data[]) => [Math.min(outer.low,...rows.map(r=>r.low)),Math.max(outer.high,...rows.map(r=>r.high))];
  const [oldLo,oldHi] = extrema(past), [newLo,newHi] = extrema(recent);
  const floor = oldLo+(newLo-oldLo)*zoom, ceil = oldHi+(newHi-oldHi)*zoom;
  const pad = Math.max((ceil-floor)*.13,anchor*.002);
  const y = (v:number) => 445-(v-floor+pad)/(ceil-floor+2*pad)*285;
  const lastX = 380, spacing = 28+76*zoom;
  const maxVolume = Math.max(1,...recent.map(r=>r.volume||0));
  return <div className="candle-stage">
    <div className="candle-stage-intro"><b>최근 캔들에서 다음 거래일로</b><span>실제 일봉 3개 확대 → 오른쪽에 같은 다음 거래일의 두 가정</span></div>
    <div className="candle-stage-scroll">
    <svg viewBox="0 0 1100 590" className="forecast-candle-chart" role="img" aria-label="최근 실제 캔들 확대와 오른쪽 상승·하락 가상 캔들 애니메이션">
      <defs><clipPath id="recent-candle-clip"><rect x="28" y="140" width="410" height="410"/></clipPath></defs>
      <rect x="465" y="12" width="270" height="528" rx="20" fill="#fcf4f4"/>
      <rect x="750" y="12" width="270" height="528" rx="20" fill="#f3f5fe"/>
      {[0,.25,.5,.75,1].map(t=>{const v=floor-pad+t*(ceil-floor+2*pad);return <g key={t}><line x1="28" x2="1020" y1={y(v)} y2={y(v)} stroke="#e7e6ef"/><text x="1030" y={y(v)+4} fill="#938d9f" fontSize="12">{number(v)}</text></g>})}
      <text x="30" y="42" fontSize="16" fill="#51475d">실제 KRX 일봉</text>
      <text x="30" y="68" fontSize="12" fill="#8f869a">최근 3거래일 · 같은 가격 축</text>
      <line x1="28" x2="1020" y1={y(anchor)} y2={y(anchor)} stroke="#a097b1" strokeDasharray="5 5"/>
      <line x1="446" x2="446" y1="140" y2="540" stroke="#c8bfda" strokeDasharray="5 5"/>
      <g clipPath="url(#recent-candle-clip)">
      {past.map((r,i)=>{const x=lastX-(past.length-1-i)*spacing;const color=r.close>=r.open?'#c55454':'#6473de';const bw=14+30*zoom;return <g key={r.time} opacity={i<past.length-3?1-zoom:1}>
        <line x1={x} x2={x} y1={y(r.high)} y2={y(r.low)} stroke={color} strokeWidth="2"/>
        <rect x={x-bw/2} y={y(Math.max(r.open,r.close))} width={bw} height={Math.max(2,Math.abs(y(r.open)-y(r.close)))} fill={color}/>
        <rect x={x-bw/2} y={530-(r.volume||0)/maxVolume*45} width={bw} height={(r.volume||0)/maxVolume*45} fill={color} opacity=".3"/>
        <text x={x} y="550" textAnchor="middle" fontSize="12" fill="#8e849a">{r.time.slice(5)}</text>
      </g>})}
      </g>
      <text x="380" y="470" textAnchor="middle" fontSize="12" fill="#655773">마지막 확정 종가 {number(anchor)}원</text>
      {(['up','down'] as const).map((side,i)=>{
        const x=600+i*285, color=side==='up'?'#bf565d':'#536bc7';
        const val=(v:number)=>anchor+(v-anchor)*reveal;
        const prob=probability==null?null:side==='up'?probability:1-probability;
        return <g key={side}>
          <text x={x-110} y="44" fill={color} fontSize="17" fontWeight="600">{side==='up'?'상승':'하락'}할 경우</text>
          <text x={x+110} y="46" textAnchor="end" fill={color} fontSize="28" fontWeight="600">{prob==null?'—':number(prob*100,1)}%</text>
          <text x={x+110} y="65" textAnchor="end" fill="#93869b" fontSize="10">보합 제외 연구 확률</text>
          {[["하한",prediction.low],["중앙값",prediction.median],["상한",prediction.high]].map(([name,value],j)=><g key={String(name)}><text x={x-88+j*88} y="92" textAnchor="middle" fill="#8d8198" fontSize="11">{name}</text><text x={x-88+j*88} y="118" textAnchor="middle" fill={color} fontSize="21" fontWeight="600">{number(value,2)}%</text></g>)}
          <g opacity={Math.min(1,reveal*4)}>
            {([['하한',prediction.low],['중앙값',prediction.median],['상한',prediction.high]] as const).map(([label,range],j)=>{
              const c=scenarioCandle(anchor,range,side)!;
              const cx=x-88+j*88;
              return <g key={label} aria-label={`${side==='up'?'상승':'하락'} ${label} ${number(range,2)}%, 가상 저가 ${number(c.low)}원, 고가 ${number(c.high)}원`}>
                <rect x={cx-31} y={y(val(c.high))} width="62" height={Math.max(1,y(val(c.low))-y(val(c.high)))} fill={color} opacity=".05" rx="6"/>
                <line x1={cx} x2={cx} y1={y(val(c.high))} y2={y(val(c.low))} stroke={color} strokeWidth="2.5"/>
                <rect x={cx-16} y={y(Math.max(anchor,val(c.close)))} width="32" height={Math.max(2,Math.abs(y(anchor)-y(val(c.close))))} fill={color} fillOpacity={j===1?.8:.45} stroke={color} rx="2"/>
                <text x={cx} y={y(val(c.high))-10} textAnchor="middle" fill={color} fontSize="11" fontWeight="600">{number(c.high)}원</text>
                <text x={cx} y={y(val(c.low))+18} textAnchor="middle" fill={color} fontSize="11" fontWeight="600">{number(c.low)}원</text>
                <text x={cx} y="473" textAnchor="middle" fill={color} fontSize="11">{label} {number(range,2)}%</text>
              </g>;
            })}
            <text x={x} y="496" textAnchor="middle" fill="#8e8398" fontSize="10">위 가격: 가상 고가 / 아래 가격: 가상 저가</text>
          </g>
          <text x={x} y="526" textAnchor="middle" fill={color} fontSize="13">{analysis.target_date}</text>
        </g>;
      })}
      <text x="742" y="571" textAnchor="middle" fontSize="12" fill="#8b7b9c">6개 모두 같은 다음 거래일의 가정입니다. 연속된 6일이 아닙니다.</text>
    </svg>
    </div>
    <section className="scenario-price-panel" aria-label="변동폭별 가상 가격대">
      <div className="scenario-price-heading"><b>이 변동폭을 가격으로 보면</b><span>기준 {number(anchor)}원 · 시가=직전 종가 가정 · 상승·하락 공통</span></div>
      <div className="scenario-price-grid">
        {([['하한',prediction.low],['중앙값',prediction.median],['상한',prediction.high]] as const).map(([label,range])=>{
          const c=scenarioCandle(anchor,range,'up')!;
          return <div className={`scenario-price-item ${label==='중앙값'?'central':''}`} key={label}>
            <span>{label} 변동폭 <b>{number(range,2)}%</b></span>
            <strong>{number(c.low)}<em> ~ </em>{number(c.high)}<small>원</small></strong>
            <div><span>가상 저가 → 가상 고가</span><span>차이 {number(c.high-c.low)}원</span></div>
          </div>;
        })}
      </div>
      <p>변동폭만으로 가격대가 하나로 정해지지는 않습니다. 위 가격은 기준가격을 중심으로 고저 비율을 적용한 예시이며, 가격 목표나 가격의 90% 예측구간이 아닙니다.</p>
    </section>
    <div className="candle-stage-legend"><span>왼쪽부터 하한 · 중앙값 · 상한을 적용한 가상 일봉</span><span>모든 캔들은 같은 가격 축으로 비교합니다.</span></div>
    <p className="lab-assumption">시가=직전 종가, 갭 0을 가정한 그림입니다. 캔들의 몸통·방향·가격 경로는 변동폭 모델의 예측이 아닙니다. 방향별 변동폭 모델이 아니므로 양쪽에 같은 하한·중앙값·상한을 적용합니다.</p>
    <div className="lab-source"><span className="research-badge">연구용 확률</span><p>{analysis.direction?.name || '방향 추론 대기'} · {analysis.anchor_date} 기준{probability==null?` · ${analysis.direction?.reason || '확률 미확보'}`:' · 실제 적중률 운영 검증 전'}</p></div>
    {actual && <p className="lab-assumption">실제 관측 {actual.time}: 고가 {number(actual.high)}원 / 저가 {number(actual.low)}원</p>}
  </div>;
}
