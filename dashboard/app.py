"""GapBridge 대시보드 — 외국인 할인 프로모션 배치 의사결정 도구

화면: ① 진단 지도 → ② 동 상세 → ③ 상권 지도 → ④ 예산 시뮬레이터
실행: streamlit run app.py
데이터: data/ (prepare_data.py로 생성)

[TODO-A] = 관광 접근성(팀원 A) 결과 (존 카드·지도·공고·영업 우선 리스트 연결 완료)
[TODO-B] = 후보 선정 트랙(팀원 B) 결과 (할인 존 / QR 확충 존, 연결 완료)
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from scipy.spatial import ConvexHull

st.set_page_config(page_title='GapBridge 대시보드', layout='wide')

# 실행 위치와 상관없이 app.py 옆의 data/ 폴더를 읽음 (Streamlit Cloud는 저장소 맨 위에서 실행)
DATA = Path(__file__).resolve().parent / 'data'

# 배경지도: Esri 회색 지도 (API 키 불필요)
# 한국은 13레벨까지만 있어서, 더 확대하면 13레벨 그림을 늘려서 보여줌 (빈칸 방지)
ESRI_GRAY = 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}'
ESRI_ATTR = 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ'

def base_map(location, zoom):
    m = folium.Map(location=location, zoom_start=zoom, tiles=None, max_zoom=18)
    folium.TileLayer(ESRI_GRAY, attr=ESRI_ATTR, name='회색 지도', max_native_zoom=13, max_zoom=18).add_to(m)
    return m

COLOR = {'전환형': '#c0392b', '객단가형': '#e67e22', '대상 아님': '#8e959c', '수요 부족': '#f2f3f4'}
# 상권 존 트랙 (팀원 B 후보 선정): 지도 색·배지
TRACK = {'할인 존': {'color': '#2e86ab', 'icon': '💳', 'dash': None},
         'QR 확충 존': {'color': '#e67e22', 'icon': '🔧', 'dash': '6 4'}}

def how(title, items):
    """기술 설명은 본문에 늘어놓지 않고, 접힌 칸 안에 불렛으로"""
    with st.expander(title):
        st.markdown('\n'.join(f'- {i}' for i in items))

def legend(items):
    """지도 범례: 색 칩 + 짧은 이름"""
    chip = lambda c, dash: (f"<span style='display:inline-block;width:12px;height:12px;background:{c}55;"
                            f"border:2px {'dashed' if dash else 'solid'} {c};margin-right:4px;vertical-align:-1px'></span>")
    st.markdown(' &nbsp;&nbsp; '.join(f"{chip(c, dash)}{t}" for c, dash, t in items), unsafe_allow_html=True)

OPS = {'상권 집중형': '할인 존을 지정해 상권 위주로 할인', '혼합형': '할인 존 + 동 전역 가맹점 할인',
       '분산형': '동 전역 가맹점 위주로 할인', '할인 존 없음': '동 전역 가맹점 할인', '존 없음': '동 전역 가맹점 할인'}

def qr_needed(n, q, n_min=11, ratio=0.5):
    """할인 존 기준(QR 11곳 이상 또는 확인 비율 50% 이상)을 채우려면 QR을 몇 곳 더 설치해야 하나"""
    return int(max(0, min(n_min - q, np.ceil(ratio * n - q))))


# ---------------- 데이터 ----------------
@st.cache_data
def load(data_version):
    # data_version: data 폴더 파일들의 (이름, 수정 시각, 크기). 파일이 바뀌면 캐시를 새로 읽음
    # 주의: 이름이 '_'로 시작하는 인자는 st.cache_data가 캐시 키에서 빼므로 '_version'으로 쓰면 안 됨
    r = lambda f: pd.read_csv(DATA / f, encoding='utf-8-sig')
    geo = json.load(open(DATA / 'jeju_dong.geojson', encoding='utf-8'))
    return geo, r('dong.csv'), r('zones.csv'), r('zone_budget.csv'), r('stores.csv'), r('sim_input.csv')

if not (DATA / 'stores.csv').exists():
    st.error('data/stores.csv가 없습니다. 가게 단위 목록은 공개 저장소에 포함하지 않았습니다. '
             'README의 "데이터" 항목을 참고해 7_zone/zone_detect.py → dashboard/prepare_data.py 순서로 생성하세요.')
    st.stop()

data_version = tuple(sorted((f.name, f.stat().st_mtime, f.stat().st_size) for f in DATA.iterdir()))
geo, dong, zones, zbud, stores, sim = load(data_version)
target = dong[dong['구분'] == '할인 대상'].sort_values('순위')
dong_info = dong.set_index('행정동명')

def won(x):
    return '-' if pd.isna(x) else (f'{x/1e8:.2f}억' if x >= 1e8 else f'{x/1e4:,.0f}만')

@st.cache_data
def zone_hulls():
    out = {}
    for zid, g in stores.dropna(subset=['존ID']).groupby('존ID'):
        pts = np.unique(g[['위도', '경도']].values, axis=0)
        if len(pts) >= 3:
            try:
                out[zid] = pts[ConvexHull(pts).vertices].tolist()
            except Exception:
                pass
    return out

hulls = zone_hulls()

def promo_card(zid, zr, r, d, months=2):
    """선택한 존의 프로모션을 공고문처럼 보여주는 카드"""
    minpay = f"{d['최소결제금액']/1e4:,.0f}만 원" if pd.notna(d['최소결제금액']) else '없음'
    minpay_sub = '이상 결제 시 할인' if pd.notna(d['최소결제금액']) else '모든 결제 할인'
    n_case = r['존예산'] / d['1인당한도'] if d['1인당한도'] else 0
    road = zr['대표도로'].split()[-1]
    spots = ' · '.join(s.rsplit(' (', 1)[0] for s in str(zr['주변관광지TOP3']).split(' / '))
    big = lambda v, t: (f"<div style='flex:1;text-align:center'><div style='font-size:26px;font-weight:700;color:#c0392b'>{v}</div>"
                        f"<div style='font-size:12px;color:#666'>{t}</div></div>")
    html = f"""
<div style='border:1.5px solid #c0392b;border-radius:10px;padding:16px 20px;margin:14px 0 18px;background:#fffaf9;color:#1f2328'>
  <div style='font-size:12px;color:#c0392b;font-weight:700'>프로모션 공고 (안)</div>
  <div style='font-size:17px;font-weight:700;margin:2px 0 10px'>{zid} · {road} 상권 외국인 간편결제 할인</div>
  <div style='display:flex;gap:6px;margin-bottom:10px'>
    {big(f"{d['할인율']:.0%}", '할인율')}{big(minpay, f'최소 결제금액 · {minpay_sub}')}{big(f"{d['1인당한도']:,.0f}원", '1회 최대 할인')}{big(won(r['존예산']), '예산')}
  </div>
  <div style='font-size:13px;line-height:1.75'>
    📍 <b>어디서</b> 이 상권의 QR 가맹점 {int(r['QR확인'])}곳 · 근처 관광지 {spots}<br>
    📅 <b>언제</b> {months}개월<br>
    👥 <b>누가</b> 해외 간편결제(알리페이 등) 이용 외국인 · 타깃 {d['타깃 국적 (비중)']}<br>
    💰 <b>예산</b> 할인액만큼 가맹점에 보전 · 소진되면 종료
  </div>
  <div style='border-top:1px dashed #e0b4ad;margin-top:10px;padding-top:8px;font-size:13px;color:#444'>
    최대 할인 기준 약 <b>{n_case:,.0f}건</b>을 지원할 수 있는 규모예요.<br>
    예상 추가 소비 <b>{won(r['존예산']*1.10)}</b> (낮음 {won(r['존예산']*0.44)} ~ 높음 {won(r['존예산']*2.22)}, 시뮬레이션)
  </div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)

def expand_card(zid, zr, r, sel):
    """QR 확충 존: 존 단위 공고 대신 '동 전역 할인에 포함' 안내 + QR 확충 목표"""
    n_t, q_t = int(zr['할인대상_가게수']), int(zr['할인대상_QR확인'])
    html = f"""
<div style='border:1.5px dashed #e67e22;border-radius:10px;padding:16px 20px;margin:14px 0 18px;background:#fffaf3;color:#1f2328'>
  <div style='font-size:12px;color:#e67e22;font-weight:700'>🔧 QR 확충 존 · 존 단위 프로모션 공고 없음</div>
  <div style='font-size:17px;font-weight:700;margin:2px 0 8px'>{zid} 가게는 {sel} 동 전역 할인에 포함돼요</div>
  <div style='font-size:13px;line-height:1.75'>
    · QR 가맹점이 {q_t}곳 (할인 대상 동 가게 {n_t}곳 중 {q_t / n_t:.0%})이라 상권 단위 할인을 따로 걸기엔 쓸 수 있는 가게가 적어요.<br>
    · 이 존 몫 예산 <b>{won(r['원래존예산'])}</b>은 {sel}의 존 밖 가맹점(동 전역 할인) 예산에 합쳤어요.<br>
    · QR <b>{qr_needed(n_t, q_t)}곳</b>을 더 설치하면 할인 존으로 바꿀 수 있어요. 아래 영업 우선 리스트부터 찾아가요.
  </div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)

# 선택된 동 (화면 간 공유)
# 지도 클릭으로 바뀐 동은 위젯(selectbox)이 만들어지기 전에 반영해야 함 (만든 뒤에는 값을 바꿀 수 없음)
if '_clicked_dong' in st.session_state:
    st.session_state['dong'] = st.session_state.pop('_clicked_dong')
if 'dong' not in st.session_state:
    st.session_state['dong'] = target['행정동명'].iloc[0]

st.title('GapBridge: 외국인 할인 프로모션 배치 대시보드')
st.caption('다음 회차 외국인 할인 프로모션을 기획·집행하는 담당자용 · 어디에 → 누구에게 → 어느 상권에서 → 얼마나')

with st.sidebar:
    st.header('동 선택')
    names = target['행정동명'].tolist()
    st.selectbox('할인 대상 18개 동', names, key='dong',
                 format_func=lambda n: f"{n} ({dong_info.loc[n, '등급']}등급)")
    st.caption('진단 지도에서 동을 눌러도 선택돼요.')

tab1, tab2, tab3, tab4 = st.tabs(['① 진단 지도', '② 동 상세', '③ 상권 지도', '④ 예산 시뮬레이터'])

# ================= ① 진단 지도 =================
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('할인 대상 동', '18개', '43개 동 중')
    c2.metric('전환형 / 객단가형', '13 / 5')
    c3.metric('대상 동의 외국인 방문 비중', '41%')
    c4.metric('대상 동의 음식·소매 소비 비중', '15%')

    def style(feat):
        n = feat['properties']['행정동명']; r = dong_info.loc[n]
        key = r['원인'] if r['구분'] == '할인 대상' else r['구분']
        return {'fillColor': COLOR[key], 'color': '#555', 'weight': 0.6, 'fillOpacity': 0.75}

    for f in geo['features']:
        n = f['properties']['행정동명']; r = dong_info.loc[n]
        f['properties']['상태'] = (f"할인 대상 · {r['원인']} · {r['등급']}등급" if r['구분'] == '할인 대상'
                                 else r['구분'])
    m = base_map([33.38, 126.55], 10)
    folium.GeoJson(geo, style_function=style,
                   highlight_function=lambda f: {'weight': 2.5, 'color': '#000'},
                   tooltip=folium.GeoJsonTooltip(['행정동명', '상태'], aliases=['동', '진단'])).add_to(m)
    left, right = st.columns([3, 1])
    with left:
        out = st_folium(m, height=560, use_container_width=True, key='map1',
                        returned_objects=['last_active_drawing'])
    with right:
        st.markdown('**범례**')
        for k, v in [('전환형 (정률 할인)', COLOR['전환형']), ('객단가형 (최소결제 조건)', COLOR['객단가형']),
                     ('대상 아님', COLOR['대상 아님']), ('수요 부족', COLOR['수요 부족'])]:
            st.markdown(f"<span style='display:inline-block;width:14px;height:14px;background:{v};"
                        f"border:1px solid #888;margin-right:6px'></span>{k}", unsafe_allow_html=True)
        st.markdown('---')
        st.markdown('**진단 규칙**')
        st.markdown('1. 수요 필터: 외국인 방문 또는 결제가 하위 25% 초과\n'
                    '2. 소비갭: 1회 방문당 소비가 중앙값(89,806원) 미만\n'
                    '3. 원인 분해: 전환(결제 횟수) vs 객단가(결제 금액)')
    # 지도 클릭 → 다음 실행 때 selectbox 값으로 반영
    # (지도는 마지막 클릭을 계속 돌려주므로, 새로 누른 경우에만 처리)
    clicked = (out or {}).get('last_active_drawing')
    if clicked:
        n = clicked['properties']['행정동명']
        if n != st.session_state.get('_last_click'):
            st.session_state['_last_click'] = n
            if n in names and n != st.session_state['dong']:
                st.session_state['_clicked_dong'] = n
                st.rerun()

# ================= ② 동 상세 =================
sel = st.session_state['dong']
d = dong_info.loc[sel]
with tab2:
    WHY = {'전환형': '외국인이 방문은 하지만 결제 자체를 적게 해요. 문턱 낮은 정률 할인으로 결제 횟수를 늘려요.', '객단가형': '결제는 일어나지만 한 번에 쓰는 금액이 작아요. 최소 결제금액 조건을 걸어 한 번에 더 쓰게 해요.'}
    st.subheader(f"{sel} · {d['등급']}등급 · {d['원인']}")
    c1, c2, c3 = st.columns(3)
    c1.metric('우선순위', f"{int(d['순위'])}위 / 18")
    c2.metric('동 예산 (3억·2개월 기준)', won(d['최종배분예산']))
    c3.metric('운영 방식', d['운영방식'].split(' (')[0])

    a, b = st.columns(2)
    with a:
        st.markdown('#### 왜·어떻게')
        st.markdown(f"- **소비갭 원인:** {d['원인']} — {WHY[d['원인']]}")
        op = d['운영방식'].split(' (')[0]
        st.markdown(f"- **운영 방식:** {op} — {OPS.get(op, '')}")
        how('운영 방식은 어떻게 정하나', [
            f"이 동의 QR 가게 중 할인 존 안 비중: **{d['존안QR비중']:.0%}** (할인 존 {int(d['할인존수'])}개)",
            '50% 이상 상권 집중형 · 30~50% 혼합형 · 30% 미만 분산형',
            'QR 확충 존은 동 전역 할인에 포함되므로 세지 않음'])
    with b:
        st.markdown('#### 누구에게')
        st.markdown(f"- **타깃 국적:** {d['타깃 국적 (비중)']}")
        st.markdown(f"- **결제 근거:** {d['결제 근거']}")
    st.caption('구체적인 할인 조건은 ③ 상권 지도의 프로모션 공고에 있어요.')

    st.markdown('#### 18개 동 한눈에 보기')
    show = target[['행정동명', '등급', '원인', '최종배분예산', '운영방식']].rename(columns={'최종배분예산': '동 예산'}).copy()
    show['동 예산'] = show['동 예산'].map(won)
    st.dataframe(show, hide_index=True, width='stretch')

# ================= ③ 상권 지도 =================
with tab3:
    zb = zbud[zbud['행정동명'] == sel]
    # 할인 존 먼저(예산 큰 순), 그다음 QR 확충 존(관광 접근성 높은 순)
    zin = (zb[zb['구분'] == '상권 존'].merge(zones[['존ID', '관광접근성_77개존순위']], on='존ID')
             .assign(_k=lambda x: x['트랙'] != '할인 존')
             .sort_values(['_k', '존예산', '관광접근성_77개존순위'], ascending=[True, False, True]).drop(columns='_k'))
    zout = zb[zb['구분'] == '존 밖 가맹점']
    n_promo = int((zin['트랙'] == '할인 존').sum())
    moved = zout['존밖합산예산'].sum()
    st.subheader(f'{sel}의 상권 존')
    c1, c2, c3 = st.columns(3)
    c1.metric('상권 존', f'{len(zin)}개', f'할인 존 {n_promo} · QR 확충 존 {len(zin) - n_promo}', delta_color='off')
    c2.metric('할인 존 예산', won(zin['존예산'].sum()))
    c3.metric('존 밖 가맹점 예산 (동 전역 할인)', won(zout['존예산'].sum()),
              f'QR 확충 존 몫 {won(moved)} 포함' if moved > 0 else None, delta_color='off')

    ds = stores[stores['행정동명'] == sel]
    left, right = st.columns([3, 2])

    # 오른쪽: 존 선택 (지도보다 먼저 정해야 지도에서 강조 가능)
    with right:
        if len(zin) == 0:
            st.info('이 동은 상권 존이 없어요. 동 전역 가맹점 할인으로 운영해요.')
            zsel = None
        else:
            track = zin.set_index('존ID')['트랙']
            zsel = st.selectbox('존 선택', zin['존ID'].tolist(), key=f'zsel_{sel}',
                                format_func=lambda z: f"{z} · {track[z]}")
            r = zin[zin['존ID'] == zsel].iloc[0]
            zr = zones[zones['존ID'] == zsel].iloc[0]
            badge = TRACK[zr['트랙']]
            st.markdown(f"<span style='background:{badge['color']};color:#fff;border-radius:4px;padding:2px 8px;"
                        f"font-size:13px;font-weight:700'>{badge['icon']} {zr['트랙']}</span> "
                        f"**{zsel}** · {zr['대표도로']} · 반경 {int(zr['반경90_m'])}m", unsafe_allow_html=True)
            a, b, c = st.columns(3)
            a.metric('가게 수', f"{int(r['존내_동가게수'])}곳")
            b.metric('QR 확인', f"{r['QR확인비율']:.0%}")
            c.metric('관광 접근성', f"{int(zr['관광접근성_77개존순위'])}위", f"77개 존 중 · {zr['접근성점수']:.0f}점", delta_color='off')
            st.markdown(f"- **주요 업종:** {zr['주요업종']}")
            if zr['동수'] > 1:
                st.markdown(f"- **여러 동에 걸친 상권:** {zr['동구성']} (이 중 {sel} 부분만 표시)")
            # [TODO-A] 관광 접근성 (팀원 A): 주변 관광지 인기도(VISIT JEJU 조회수) × 거리 가중치
            st.markdown(f"- **주변 관광지:** {zr['주변관광지TOP3']}  \n"
                        f"  <span style='color:#888;font-size:13px'>500m 안 {int(zr['관광지수_500m'])}곳 · 1km 안 {int(zr['관광지수_1km'])}곳</span>",
                        unsafe_allow_html=True)
            # [TODO-B] 후보 선정 (팀원 B): 할인 대상 동 가게 기준 QR 11곳 이상 또는 확인 비율 50% 이상 → 할인 존
            n_t, q_t = int(zr['할인대상_가게수']), int(zr['할인대상_QR확인'])
            scope = f" ({zr['할인대상_행정동']} 합계)" if n_t != int(r['존내_동가게수']) else ''
            if zr['트랙'] == '할인 존':
                st.markdown(f"- **추천 지원:** 💳 **바로 할인** · QR 가맹점 {q_t}곳{scope}에서 운영  \n"
                            f"  <span style='color:#888;font-size:13px'>QR 미확인 {n_t - q_t}곳은 확충 대상</span>",
                            unsafe_allow_html=True)
            else:
                need = qr_needed(n_t, q_t)
                st.markdown(f"- **추천 지원:** 🔧 **QR 설치 영업 먼저** · {need}곳 더 설치하면 할인 존 전환  \n"
                            f"  <span style='color:#888;font-size:13px'>지금 QR 가맹점 {q_t}곳({q_t / n_t:.0%}) · 동 전역 할인에는 포함</span>",
                            unsafe_allow_html=True)
            if pd.notna(zr['대표후보_검토상태']):
                st.markdown(f"- **대표 상권 검토:** {zr['대표후보_검토상태']}")
            how('지표 설명', [
                '**관광 접근성**: 주변 관광지 인기도(VISIT JEJU 조회수) × 거리 가중치, 1위 존 = 100점',
                '**할인 존 기준**: 할인 대상 동 가게 중 QR 11곳 이상 또는 확인 비율 50% 이상',
                '**QR 확인**: 제로페이 가맹점과 매칭된 가게 (미확인 = 미가맹 확정 아님)'])

    # 왼쪽: 지도 (선택한 존은 진하게 강조하고 가운데로, 나머지 존은 흐리게)
    with left:
        if zsel:
            zc = zones[zones['존ID'] == zsel].iloc[0]
            m3 = base_map([zc['중심위도'], zc['중심경도']], 15)
        else:
            m3 = base_map([ds['위도'].mean(), ds['경도'].mean()], 14)
        gj = {'type': 'FeatureCollection', 'features': [f for f in geo['features'] if f['properties']['행정동명'] == sel]}
        folium.GeoJson(gj, style_function=lambda f: {'fill': False, 'color': '#333', 'weight': 2, 'dashArray': '5'}).add_to(m3)
        for _, p in ds[ds['QR상태'] != 'QR 있음'].iterrows():
            folium.CircleMarker([p['위도'], p['경도']], radius=1.5, color='#aaa', fill=True,
                                tooltip=f"{p['상호명']} · {p['세부업종']} (QR 미확인)").add_to(m3)
        for _, r in zin.sort_values('존ID', key=lambda x: x == zsel).iterrows():   # 선택한 존을 맨 위에 그림
            zr = zones[zones['존ID'] == r['존ID']].iloc[0]
            on = r['존ID'] == zsel
            tk = TRACK[zr['트랙']]
            budget = (f"존 예산 {won(r['존예산'])}" if zr['트랙'] == '할인 존'
                      else f"동 전역 할인에 포함 (예산 {won(r['원래존예산'])} → 존 밖)")
            tip = (f"<b>{tk['icon']} {r['존ID']} · {zr['트랙']}</b><br>{zr['대표도로']}<br>{sel} 가게 {int(r['존내_동가게수'])}곳 · "
                   f"QR {r['QR확인비율']:.0%}<br>관광 접근성 {zr['접근성점수']:.0f}점<br>{budget}")
            if r['존ID'] in hulls:
                folium.Polygon(hulls[r['존ID']], color='#111' if on else tk['color'], dash_array=tk['dash'],
                               weight=4 if on else 1.8, fill=True, fill_color=tk['color'],
                               fill_opacity=0.45 if on else 0.18, tooltip=tip).add_to(m3)
            folium.Marker([zr['중심위도'], zr['중심경도']], icon=folium.DivIcon(
                icon_size=(60, 18), icon_anchor=(30, 9),
                html=(f"<div style='font:{'bold 13px' if on else '11px'} sans-serif;color:{'#000' if on else '#555'};"
                      f"text-align:center;text-shadow:0 0 3px #fff,0 0 3px #fff'>{tk['icon']}{r['존ID'].split('-')[-1]}</div>"))).add_to(m3)
        st_folium(m3, height=520, use_container_width=True, key=f'map3_{sel}_{zsel}', returned_objects=[])
        legend([('#2e86ab', False, '💳 할인 존'), ('#e67e22', True, '🔧 QR 확충 존'),
                ('#111111', False, '선택한 존'), ('#aaaaaa', False, 'QR 미확인 가게')])

    if zsel:
        zr_sel, r_sel = zones[zones['존ID'] == zsel].iloc[0], zin[zin['존ID'] == zsel].iloc[0]
        if zr_sel['트랙'] == '할인 존':
            promo_card(zsel, zr_sel, r_sel, d)
        else:
            expand_card(zsel, zr_sel, r_sel, sel)
        st.markdown(f'### {zsel}의 QR 미확인 가게')
        zs = ds[ds['존ID'] == zsel].copy()
        miss = zs[zs['QR상태'] != 'QR 있음'].copy()
        q1, q2, q3 = st.tabs(['업종별로 보기', 'QR 설치 시뮬레이션', '영업 우선 리스트'])

        with q1:
            cnt = (zs.assign(QR=np.where(zs['QR상태'] == 'QR 있음', 'QR 있음', 'QR 미확인'))
                     .groupby(['세부업종', 'QR']).size().unstack(fill_value=0)
                     .reindex(columns=['QR 미확인', 'QR 있음'], fill_value=0))
            cnt = cnt.sort_values('QR 미확인', ascending=False).head(10)
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown('**업종별 가게 수 (QR 미확인이 많은 순, 상위 10개)**')
                st.bar_chart(cnt, horizontal=True, color=['#c0392b', '#2e86ab'], height=320)
            with c2:
                pick = st.multiselect('업종 고르기', sorted(miss['세부업종'].unique()),
                                      placeholder='전체 업종', key=f'pick_{zsel}')
                view = miss if not pick else miss[miss['세부업종'].isin(pick)]
                st.metric('QR 미확인 가게', f'{len(view)}곳', f"존 전체 {len(zs)}곳 중", delta_color='off')
            lst = view[['상호명', '세부업종', '도로명주소']]
            st.dataframe(lst, hide_index=True, width='stretch', height=240)
            st.download_button('목록 내려받기 (CSV)', lst.to_csv(index=False).encode('utf-8-sig'),
                               file_name=f'{zsel}_QR미확인가게.csv', mime='text/csv')

        with q2:
            # 동 흡수 한도(cap)를 동 QR 가게 수로 나눠 "QR 가게 1곳당 쓸 수 있는 할인 예산"으로 보고, 설치 수만큼 더함
            per_hi = d['cap'] / d['동QR가게']          # 한도 도달 가정 (2개월)
            per_lo = d['cap_보수'] / d['동QR가게']      # 현재 소비 가정 (2개월)
            n_now, n_all = int((zs['QR상태'] == 'QR 있음').sum()), len(zs)
            n_add = st.slider('미확인 가게 중 QR을 설치할 가게 수', 0, len(miss), min(10, len(miss)), key=f'qradd_{zsel}')
            k1, k2, k3 = st.columns(3)
            k1.metric('QR 가맹 비율', f'{(n_now + n_add) / n_all:.0%}', f'{n_add / n_all:+.0%}p (현재 {n_now / n_all:.0%})')
            k2.metric('QR 가맹점 수', f'{n_now + n_add}곳', f'+{n_add}곳')
            k3.metric('추가로 쓸 수 있는 할인 예산 (2개월)', won(n_add * per_hi),
                      f'현재 소비 기준 {won(n_add * per_lo)}', delta_color='off')
            how('계산 가정', [
                f"QR 가맹점 1곳이 2개월간 쓸 수 있는 할인 예산 = {sel} 평균",
                f"한도 도달 {won(per_hi)} · 현재 소비 {won(per_lo)}",
                '새로 설치한 가게도 기존 가맹점만큼 이용된다고 가정'])
            if zr_sel['트랙'] == 'QR 확충 존':
                need = qr_needed(int(zr_sel['할인대상_가게수']), int(zr_sel['할인대상_QR확인']))
                if n_add >= need:
                    st.success(f'QR {n_add}곳을 설치하면 할인 존으로 바꿀 수 있어요 (필요 {need}곳).')
                else:
                    st.info(f'할인 존까지 QR {need - n_add}곳 더 필요해요 (필요 {need}곳).')

        with q3:
            # 우선순위: ① 동의 QR 설치 우선 업종(TOP3) ② 가장 가까운 관광지까지 거리 ③ 상권 중심에서 가까운 순
            top3 = [t.rsplit(' ', 1)[0] for t in str(d['QR 설치 우선 업종 TOP3']).split(' · ')]
            lat0 = np.deg2rad(33.4)
            miss['거리(m)'] = np.hypot((miss['경도'] - zc['중심경도']) * 111320 * np.cos(lat0),
                                     (miss['위도'] - zc['중심위도']) * 110574).round(0).astype(int)
            miss['우선 업종'] = miss['세부업종'].isin(top3)
            rank = (miss.sort_values(['우선 업종', '관광지거리_m', '거리(m)'], ascending=[False, True, True])
                        .head(20).reset_index(drop=True))
            rank.insert(0, '순위', range(1, len(rank) + 1))
            rank['가까운 관광지'] = rank['가까운관광지'] + ' (' + rank['관광지거리_m'].astype(str) + 'm)'
            rank['이유'] = (np.where(rank['우선 업종'], '동 QR 설치 우선 업종 · ', '')
                          + '상권 중심에서 ' + rank['거리(m)'].astype(str) + 'm')
            st.markdown('**먼저 찾아갈 가게 TOP 20**')
            st.dataframe(rank[['순위', '상호명', '세부업종', '가까운 관광지', '도로명주소', '이유']],
                         hide_index=True, width='stretch', height=320)
            how('정렬 기준', [
                f"① {sel}의 QR 설치 우선 업종 ({', '.join(top3)})",
                '② 가까운 관광지 순 (관광지 표준데이터, 의료·축제 제외, 직선거리)',
                '③ 상권 중심에서 가까운 순',
                '관광객 동선에 가까운 가게부터 설치해야 할인이 빨리 쓰여요'])

# ================= ④ 예산 시뮬레이터 =================
def allocate(weight, cap, total, max_share=0.20):
    """3-2 배분 알고리즘 (gapbridge_3_1_3_2.py와 동일): 가중 비례배분 + cap 초과분 재배분"""
    ub = np.minimum(cap, max_share * total)
    alloc = np.zeros(len(weight)); remain = float(total); active = ub > 0
    for _ in range(100):
        if remain < 1 or not active.any():
            break
        share = np.where(active, weight, 0) / np.where(active, weight, 0).sum()
        alloc = np.minimum(alloc + remain * share, ub)
        remain = total - alloc.sum()
        active = alloc < ub - 1e-6
    return alloc

with tab4:
    st.subheader('예산 시뮬레이터')
    st.caption('조건을 바꾸면 18개 동 배분과 집행액이 다시 계산돼요. 기본값은 보고서 기준(3억·2개월·20/15/10%)이에요.')
    s1, s2, s3 = st.columns(3)
    total = s1.slider('총예산 (억 원)', 1.0, 6.0, 3.0, 0.5) * 1e8
    months = s1.slider('시행 기간 (개월)', 1, 6, 2)
    ra = s2.slider('A등급 할인율 (%)', 5, 30, 20) / 100
    rb = s2.slider('B등급 할인율 (%)', 5, 30, 15) / 100
    rc = s2.slider('C등급 할인율 (%)', 5, 30, 10) / 100
    mode = s3.radio('집행 가정', ['한도 도달 (5만 원 결제)', '현재 소비 수준'])
    # [TODO-C] 시행 계절: 그 계절 실제 방문 ÷ 연평균 방문(4-2 민감도와 같은 배율)을 결제가능방문에 곱함
    SEASONS = {'연평균': None, '여름 (6~7월)': '여름배율', '겨울 (12~1월)': '겨울배율'}
    has_season = all(c in sim.columns for c in ['여름배율', '겨울배율'])
    season = s3.radio('시행 계절', list(SEASONS), horizontal=True, disabled=not has_season,
                      help='여름 = 2026년 6~7월, 겨울 = 2025년 12월~2026년 1월 실제 방문 기준 (4-2 민감도와 같음)')
    if not has_season:
        s3.caption('계절 배율 데이터가 아직 없어 연평균만 볼 수 있어요.')
    elif SEASONS[season]:
        s3.caption('그 계절 방문에 맞춰 배분을 다시 계산해요.')

    t = sim.copy()
    if SEASONS[season]:
        t['결제가능방문'] = t['결제가능방문'] * t[SEASONS[season]]
    t['할인율'] = t['등급'].map({'A': ra, 'B': rb, 'C': rc})
    t['한도'] = t['할인율'] * 50_000
    t['cap'] = t['결제가능방문'] * t['QR가맹률'] * t['한도'] * months / 12
    t['cap_보수'] = t['결제가능방문'] * t['QR가맹률'] * np.minimum(t['할인율'] * t['방문당소비'], t['한도']) * months / 12
    t['우리 배분'] = allocate((t['한도'] * t['타깃도달방문']).values, t['cap'].values, total)
    t['균등 배분'] = total / len(t)
    capx = t['cap'] if mode.startswith('한도') else t['cap_보수']
    t['우리 집행'] = np.minimum(t['우리 배분'], capx)
    t['균등 집행'] = np.minimum(t['균등 배분'], capx)

    ours, unif = t['우리 집행'].sum(), t['균등 집행'].sum()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric('집행 가능액 (우리 배분)', won(ours), f'균등 대비 {ours/unif:.2f}배')
    k2.metric('집행 가능액 (균등 배분)', won(unif))
    k3.metric('미집행액 (우리 배분)', won(total - ours))
    k4.metric('예상 추가 소비 (우리 배분)', won(ours * 1.10), f'낮음 {won(ours * 0.44)} ~ 높음 {won(ours * 2.22)}', delta_color='off')

    chart = t.set_index('행정동명')[['우리 집행', '균등 집행']] / 1e4
    st.markdown('**동별 집행 가능액 (만 원)**')
    st.bar_chart(chart, stack=False, height=320)
    cols = ['행정동명', '등급', '할인율'] + (['계절 배율'] if SEASONS[season] else []) + ['우리 배분', '우리 집행', '균등 집행']
    view = t.assign(**{'계절 배율': t[SEASONS[season]] if SEASONS[season] else 1.0})[cols].copy()
    view['할인율'] = view['할인율'].map('{:.0%}'.format)
    if SEASONS[season]:
        view['계절 배율'] = view['계절 배율'].map('×{:.2f}'.format)
    for c in ['우리 배분', '우리 집행', '균등 집행']:
        view[c] = view[c].map(won)
    st.dataframe(view, hide_index=True, width='stretch')
    how('계산 가정', [
        '배분: 3-2 알고리즘 (동별 흡수 한도 안에서 가중 비례배분, 동당 최대 20%)',
        '추가 소비: 문헌 효과값 할인 1원당 0.44 / 1.10 / 2.22원 (한국관광공사 2026), 기본 1.10원 · 실측 아님',
        '계절 선택 시 그 계절 방문으로 배분을 다시 계산 → 연평균 배분을 고정한 4-2 민감도와 우리 배분 집행액이 다를 수 있음 (균등 배분은 같음)'])
