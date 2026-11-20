"""상권 존 탐지 (제주 전체 기준)
① 가게 정리: 제주 전체 관광 업종 가게, 중복 제거
② 존 탐지: DBSCAN (반경 EPS m 안에 가게 MIN_SAMPLES곳 이상 = 상권 핵심)
   - 상권은 행정 경계를 따르지 않으므로 동 경계와 상관없이 제주 전체에서 탐지
   - 존 반경(90분위)이 R_MAX를 넘으면 반경을 줄여 다시 쪼갬 (최대 3단계)
③ 존 × 동 연결: 한 존이 여러 동에 걸치면 동별 부분을 따로 기록 → 동 단위 조건(할인 방식·예산)을 그 부분에 적용

사용: python3 zone_detect.py [EPS=60] [MIN_SAMPLES=10] [R_MAX=400] [OUT=out]
"""
import sys, os, json
import numpy as np, pandas as pd
from sklearn.cluster import DBSCAN

EPS = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0        # 이웃 반경(m)
MIN_SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 10    # 상권 핵심 판정 최소 가게 수
R_MAX = float(sys.argv[3]) if len(sys.argv) > 3 else 400.0     # 존 반경 상한(m): 걸어서 5분 안팎
OUT = sys.argv[4] if len(sys.argv) > 4 else 'out'
MIN_ZONE = 15                                                   # 존 최소 가게 수

d = pd.read_csv('QR_가게단위_매칭.csv', encoding='utf-8-sig')
TARGET = pd.read_csv('../3-1_3-2/3-1_3-2_최종표.csv',
                     encoding='utf-8-sig')['행정동명'].tolist()

# ---------- ① 가게 정리 ----------
s = d.copy()
log = {'원자료(제주 전체)': len(s)}
# 같은 상호+주소 2~3건 → 같은 점포의 중복 등록으로 보고 1건만 남김 (QR 있음 우선)
# 4건 이상은 야시장 판매대처럼 같은 이름의 서로 다른 점포라 유지
s['_g'] = s.groupby(['상호명', '도로명주소'])['상가업소번호'].transform('size')
s['_qr'] = (s['QR상태'] == 'QR 있음').astype(int)
s = s.sort_values(['_qr', '관광업종'], ascending=False)
dup = s.duplicated(['상호명', '도로명주소'], keep='first') & s['_g'].between(2, 3)
log['중복 제거'] = int(dup.sum())
s = s[~dup]
log['비관광 업종 제외'] = int((~s['관광업종']).sum())
s = s[s['관광업종']].copy()
log['분석 대상 가게(제주 전체)'] = len(s)
s['QR'] = (s['QR상태'] == 'QR 있음').astype(int)
s['할인대상동'] = s['행정동명'].isin(TARGET)

# ---------- ② 존 탐지 ----------
lat0 = np.deg2rad(33.4)                       # 위경도 → 미터 근사 (제주 위도 기준)
s['x'] = s['경도'] * 111320 * np.cos(lat0)
s['y'] = s['위도'] * 110574

def r90(c):
    return np.hypot(c['x'] - c['x'].mean(), c['y'] - c['y'].mean()).quantile(0.9)

def detect(g, eps, depth=0):
    if len(g) < MIN_SAMPLES:
        return []
    lab = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit_predict(g[['x', 'y']].values)
    out = []
    for l in set(lab) - {-1}:
        c = g[lab == l]
        if len(c) < MIN_ZONE:
            continue
        if r90(c) <= R_MAX:
            out.append(c.index)
        elif depth < 3:
            out += detect(c, eps * 0.7, depth + 1)   # 너무 넓으면 반경을 줄여 다시 쪼갬
    return out

parts = detect(s, EPS)
# 존 이름: 가게가 가장 많은 동 + 그 동 안에서 가게 수 순번
meta = []
for idx in parts:
    g = s.loc[idx]
    meta.append((idx, g['행정동명'].value_counts().index[0], len(g)))
meta.sort(key=lambda t: (t[1], -t[2]))
s['존ID'] = None
seq = {}
for idx, main, n in meta:
    seq[main] = seq.get(main, 0) + 1
    s.loc[idx, '존ID'] = f'{main}-{seq[main]:02d}'

# ---------- 존 요약 ----------
z = s[s['존ID'].notna()].copy()
cx = z.groupby('존ID')[['x', 'y']].transform('mean')
z['_r'] = np.hypot(z['x'] - cx['x'], z['y'] - cx['y'])

def top3(x):
    return ' · '.join(f'{k} {v}' for k, v in x.value_counts().head(3).items())

zones = z.groupby('존ID').agg(
    가게수=('상가업소번호', 'size'), QR확인=('QR', 'sum'),
    대상동가게수=('할인대상동', 'sum'),
    동구성=('행정동명', lambda x: ' / '.join(f'{k} {v}' for k, v in x.value_counts().items())),
    동수=('행정동명', 'nunique'),
    주요업종=('세부업종', top3),
    대표도로=('도로명주소', lambda x: x.str.split().str[2:-1].str.join(' ').mode().iat[0]),
    중심위도=('위도', 'mean'), 중심경도=('경도', 'mean'),
    반경90_m=('_r', lambda r: round(r.quantile(0.9))),
).reset_index()
zones['QR미확인'] = zones['가게수'] - zones['QR확인']
zones['QR확인비율'] = (zones['QR확인'] / zones['가게수']).round(3)
zones['할인대상포함'] = zones['대상동가게수'] > 0
zones = zones.sort_values(['할인대상포함', '가게수'], ascending=[False, False])

# 존 × 동 연결표 (할인 대상 동 부분만): 동 단위 조건·예산을 붙이는 단위
zd = z[z['할인대상동']].groupby(['존ID', '행정동명']).agg(
    가게수=('상가업소번호', 'size'), QR확인=('QR', 'sum')).reset_index()
zd['QR미확인'] = zd['가게수'] - zd['QR확인']
zd['QR확인비율'] = (zd['QR확인'] / zd['가게수']).round(3)
zd = zd.merge(zones[['존ID', '가게수', '동구성']].rename(columns={'가게수': '존전체가게수'}), on='존ID')
zd['존내비중'] = (zd['가게수'] / zd['존전체가게수']).round(3)
zd = zd.sort_values(['행정동명', '가게수'], ascending=[True, False])

os.makedirs(OUT, exist_ok=True)
s[['상가업소번호', '행정동명', '할인대상동', '상호명', '도로명주소', '대분류', '세부업종',
   '위도', '경도', 'QR상태', '존ID']].to_csv(f'{OUT}/store_zone.csv', index=False, encoding='utf-8-sig')
zones.to_csv(f'{OUT}/zones.csv', index=False, encoding='utf-8-sig')
zd.to_csv(f'{OUT}/zone_dong.csv', index=False, encoding='utf-8-sig')

t = s[s['할인대상동']]
rel = zones[zones['할인대상포함']]
log.update({
    '설정': dict(eps_m=EPS, min_samples=MIN_SAMPLES, r_max_m=R_MAX, min_zone=MIN_ZONE),
    '존 수(제주 전체)': int(len(zones)),
    '할인 대상 동이 포함된 존': int(len(rel)),
    '그중 여러 동에 걸친 존': int((rel['동수'] >= 2).sum()),
    '존이 있는 할인 대상 동 수': int(zd['행정동명'].nunique()),
    '할인 대상 동 가게 중 존 소속 비율': round(float(t['존ID'].notna().mean()), 3),
})
json.dump(log, open(f'{OUT}/log.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(log, ensure_ascii=False))
