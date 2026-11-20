"""상권 존 지도: 인터랙티브(HTML) + 보고서용 정적 그림(PNG)"""
import pandas as pd, numpy as np, folium
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.spatial import ConvexHull

st = pd.read_csv('out/store_zone.csv', encoding='utf-8-sig')
z = pd.read_csv('out/zones.csv', encoding='utf-8-sig')

def color(q):
    return '#c0392b' if q < 0.33 else ('#e67e22' if q < 0.40 else '#2e86ab')

def hull(pts):
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return None
    try:
        return pts[ConvexHull(pts).vertices]
    except Exception:
        return None

# ---------- 인터랙티브 ----------
# 배경지도: API 키가 필요 없고 파일로 열어도 막히지 않는 Esri 타일 사용
m = folium.Map(location=[33.38, 126.55], zoom_start=10, tiles=None)
folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
                 attr='Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ', name='회색 지도',
                 max_native_zoom=13, max_zoom=19).add_to(m)   # 한국은 13레벨까지만 있어서 확대 시 13레벨 타일을 늘려 보여줌
folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                 attr='Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics', name='위성 지도 (골목 확인용)',
                 max_zoom=19).add_to(m)
for zid, g in st.dropna(subset=['존ID']).groupby('존ID'):
    r = z[z['존ID'] == zid].iloc[0]
    tip = (f"<b>{zid}</b> {'(할인 대상 동 포함)' if r['할인대상포함'] else ''}<br>가게 {r['가게수']}곳 · QR 확인 {r['QR확인비율']:.0%}"
           f"<br>동 구성: {r['동구성']}<br>{r['대표도로']}<br>{r['주요업종']}")
    h = hull(g[['위도', '경도']].values)
    if h is not None:
        folium.Polygon(h.tolist(), color=color(r['QR확인비율']), weight=2, fill=True, fill_opacity=0.25,
                       tooltip=tip, dash_array=None if r['할인대상포함'] else '6').add_to(m)
    folium.CircleMarker([r['중심위도'], r['중심경도']], radius=3, color='#333', fill=True, tooltip=tip).add_to(m)
fg = folium.FeatureGroup(name='존 안의 QR 미확인 가게', show=False)
for _, p in st[(st['QR상태'] != 'QR 있음') & st['존ID'].notna()].iterrows():
    folium.CircleMarker([p['위도'], p['경도']], radius=1.5, color='#666', fill=True,
                        tooltip=f"{p['상호명']} · {p['세부업종']}").add_to(fg)
fg.add_to(m)
folium.LayerControl().add_to(m)
m.get_root().html.add_child(folium.Element(
    '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:#fff;padding:10px 12px;'
    'border:1px solid #ccc;font:13px sans-serif"><b>상권 존</b> (반경 60m 안 가게 10곳 이상)<br>'
    '<span style="color:#c0392b">■</span> QR 확인 33% 미만<br><span style="color:#e67e22">■</span> 33~40%<br>'
    '<span style="color:#2e86ab">■</span> 40% 이상<br>점선 = 할인 대상 동이 없는 존</div>'))
m.save('out/zone_map.html')

# ---------- 정적 그림 ----------
fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(fp)
plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
plt.rcParams['axes.unicode_minus'] = False

def panel(ax, dongs, title):
    g = st[st['행정동명'].isin(dongs)]
    ax.scatter(g['경도'], g['위도'], s=3, c='#cccccc', lw=0, label='존 밖 가게')
    for zid, c in g.dropna(subset=['존ID']).groupby('존ID'):
        r = z[z['존ID'] == zid].iloc[0]
        col = color(r['QR확인비율'])
        ax.scatter(c['경도'], c['위도'], s=5, c=col, lw=0)
        h = hull(c[['경도', '위도']].values)
        if h is not None:
            ax.fill(h[:, 0], h[:, 1], color=col, alpha=0.18, lw=1.2, ec=col)
        if r['가게수'] >= 120 or (r['가게수'] >= 40 and len(dongs) == 1):
            ax.annotate(f"{zid} ({r['가게수']})", (r['중심경도'], r['중심위도']),
                        fontsize=8, ha='center', va='center', weight='bold')
    ax.set_title(title, fontsize=11, loc='left')
    ax.set_aspect(1 / np.cos(np.deg2rad(33.4)))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color('#ddd')

fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
panel(axes[0], ['일도1동', '이도1동', '삼도2동', '건입동'], '제주시 원도심: 동문시장 상권 (일도1동·이도1동 등)')
panel(axes[1], ['중앙동', '정방동', '천지동'], '서귀포 원도심: 올레시장 상권 (중앙동·정방동·천지동)')
panel(axes[2], ['한림읍'], '읍면 사례: 한림읍')
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color='#c0392b', alpha=.5, label='QR 확인 33% 미만'),
                    Patch(color='#e67e22', alpha=.5, label='33~40%'),
                    Patch(color='#2e86ab', alpha=.5, label='40% 이상'),
                    Patch(color='#cccccc', label='존 밖 가게')],
           loc='lower center', ncol=4, frameon=False, fontsize=10)
fig.suptitle('상권 존 탐지 결과 (제주 전체 기준 DBSCAN: 반경 60m 안 관광 업종 가게 10곳 이상, 존 반경 400m 이하)',
             fontsize=13, x=0.01, ha='left')
plt.tight_layout(rect=(0, 0.06, 1, 0.95))
plt.savefig('out/zone_map_예시.png', dpi=150)
print('saved')
