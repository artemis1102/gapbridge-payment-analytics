"""대시보드용 데이터 만들기 (분석 결과 → dash/data/)
분석 결과가 바뀌면 이 스크립트만 다시 돌리면 된다.
입력 경로는 아래 SRC에서 바꾼다.
"""
import os
import numpy as np, pandas as pd

SRC = {
    'final': '../3-1_3-2/3-1_3-2_최종표.csv',         # 3-1·3-2 등급·예산
    'place': '../1-5_reach/1-5_프로모션배치표.csv',    # 1-5 타깃 국적·QR 우선 업종
    'zones': '../7_zone/out/zones.csv',                              # 상권 존
    'zone_budget': '../7_zone/out/zone_budget.csv',
    'dong_check': '../7_zone/out/zone_budget_동별점검.csv',
    'stores': '../7_zone/out/store_zone.csv',
    'access': '../7_zone/out/zone_access.csv',            # 관광 접근성 (팀원 A, 175개 존)
    'candidate': '../7_zone/out/77개존_후보비교표.csv',    # 후보 선정 트랙 (팀원 B, 77개 존)
    'spots': '../7_zone/관광지_행정동매핑.csv',            # 관광지 좌표 (드라이브, 원천이라 커밋 안 함)
}
OUT = 'data'
os.makedirs(OUT, exist_ok=True)
rd = lambda k: pd.read_csv(SRC[k], encoding='utf-8-sig')

# ---------- 동 (43개) ----------
DEMAND_FAIL = ['추자면', '삼양동', '봉개동', '효돈동', '영천동', '서홍동']   # 1-1 수요 필터 탈락
st = rd('stores')
dong = pd.DataFrame({'행정동명': sorted(st['행정동명'].unique())})
f = rd('final'); p = rd('place'); c = rd('dong_check')
dong['구분'] = np.where(dong['행정동명'].isin(f['행정동명']), '할인 대상',
                np.where(dong['행정동명'].isin(DEMAND_FAIL), '수요 부족', '대상 아님'))
dong = dong.merge(f[['행정동명', '등급', '순위', '원인', '할인방식', '할인율', '1인당한도', '최소결제금액',
                     '방문', '도달비중', '최종배분예산', 'cap', 'cap_보수']], on='행정동명', how='left')
dong = dong.merge(p[['행정동명', '타깃 국적 (비중)', '결제 근거', 'QR 설치 우선 업종 TOP3',
                     'QR 없는 관광 가게', 'QR 없음 비율(%)']], on='행정동명', how='left')
dong = dong.merge(c[['행정동명', '존수', '존예산합', '존밖예산', '동QR가게', '존안QR비중']], on='행정동명', how='left')
dong = dong.rename(columns={'존안QR비중': '존안QR비중_전체존'})   # 참고용: 확충 존까지 포함한 원래 값
# 운영 방식: 동 QR 가게 중 '할인 존' 안에 있는 비중으로 구분
#   QR 확충 존은 존 단위 프로모션을 하지 않고 동 전역 할인에 포함되므로 제외 (TODO-B (가) 결정과 맞춤)
#   50% 이상 = 상권 집중형 / 30~50% = 혼합형(존 + 동 전역) / 30% 미만 = 분산형 / 할인 존 없음
_zb = rd('zone_budget')
_promo_ids = rd('candidate').loc[lambda x: x['운영방향'] == '할인 우선', '존ID']
_promo = _zb[(_zb['구분'] == '상권 존') & _zb['존ID'].isin(_promo_ids)].groupby('행정동명')
dong['할인존수'] = dong['행정동명'].map(_promo.size()).fillna(0).astype(int)
dong['존안QR비중'] = (dong['행정동명'].map(_promo['QR확인'].sum()).fillna(0) / dong['동QR가게']).round(4)
dong['운영방식'] = np.select(
    [dong['구분'] != '할인 대상', dong['존수'].fillna(0) == 0, dong['할인존수'] == 0,
     dong['존안QR비중'] >= 0.5, dong['존안QR비중'] >= 0.3],
    [None, '존 없음 (동 전역 할인)', '할인 존 없음 (동 전역 할인 + QR 확충)', '상권 집중형', '혼합형 (상권 존 + 동 전역)'],
    default='분산형 (동 전역 할인)')
dong.to_csv(f'{OUT}/dong.csv', index=False, encoding='utf-8-sig')

# ---------- 상권 존 ----------
z = rd('zones')
z = z[z['할인대상포함']]
# [TODO-A] 관광 접근성: 점수·순위(175개 존 기준)·77개 존 내 순위·주변 관광지 TOP3
a = rd('access').rename(columns={'접근성 점수': '접근성점수', '접근성 순위': '접근성순위_175', '주변 관광지 TOP3': '주변관광지TOP3'})
cand = rd('candidate')
z = z.merge(a[['존ID', '접근성점수', '접근성순위_175', '관광지수_500m', '관광지수_1km', '주변관광지TOP3']], on='존ID', how='left')
# [TODO-B] 후보 선정 트랙: QR 11곳 이상 또는 확인 비율 50% 이상 → 할인 존, 둘 다 미달 → QR 확충 존
cand['트랙'] = cand['운영방향'].map({'할인 우선': '할인 존', '확충 우선': 'QR 확충 존'})
z = z.merge(cand[['존ID', '관광접근성_77개존순위', '트랙', 'QR공동기준_유형', '할인대상_행정동', '할인대상_가게수', '할인대상_QR확인',
                  '관광QR_예비선정근거', '대표후보_검토상태']], on='존ID', how='left')
assert z['트랙'].notna().all() and z['접근성점수'].notna().all(), '접근성·트랙이 없는 존이 있음'
z.to_csv(f'{OUT}/zones.csv', index=False, encoding='utf-8-sig')

# 존 예산: QR 확충 존은 프로모션 공고 없이 동 전역 할인에 포함 → 그 존 예산을 같은 동의 '존 밖 가맹점' 예산에 합침
zb = rd('zone_budget').merge(z[['존ID', '트랙']], on='존ID', how='left')
move = (zb['구분'] == '상권 존') & (zb['트랙'] == 'QR 확충 존')
zb['원래존예산'] = zb['존예산']
zb['존밖합산예산'] = np.where(move, zb['존예산'], 0.0)
zb.loc[move, '존예산'] = 0.0
add = zb[move].groupby('행정동명')['존밖합산예산'].sum()
out_row = zb['구분'] == '존 밖 가맹점'
assert set(add.index) <= set(zb.loc[out_row, '행정동명']), '존 밖 행이 없는 동이 있음'
zb.loc[out_row, '존예산'] += zb.loc[out_row, '행정동명'].map(add).fillna(0)
zb.loc[out_row, '존밖합산예산'] = zb.loc[out_row, '행정동명'].map(add).fillna(0)
before, after = rd('zone_budget').groupby('행정동명')['존예산'].sum(), zb.groupby('행정동명')['존예산'].sum()
assert np.allclose(before, after.reindex(before.index)), '동 예산 합계가 바뀜'
zb.to_csv(f'{OUT}/zone_budget.csv', index=False, encoding='utf-8-sig')

# ---------- 가게 (할인 대상 동, 존 안팎 모두) ----------
s = st[st['할인대상동']][['존ID', '행정동명', '상호명', '도로명주소', '세부업종', '위도', '경도', 'QR상태']].copy()
# [TODO-A] 가게마다 가장 가까운 관광지와 거리(m): 영업 우선 리스트 기준
#   관광지 표준데이터 중 상시 방문지만 사용 (의료·축제 제외, 행정동 매칭 이상치 제외)
from scipy.spatial import cKDTree
sp = rd('spots')
sp = sp[~sp['관광지 분야'].isin(['의료', '축제', '축제/행사']) & (sp['매칭방식'] != '이상치')].reset_index(drop=True)
xy = lambda d: np.c_[d['경도'] * 111320 * np.cos(np.deg2rad(33.4)), d['위도'] * 110574]   # 제주 위도 기준 평면(m)
dist, idx = cKDTree(xy(sp)).query(xy(s))
s['가까운관광지'] = sp.loc[idx, '관광지명'].values
s['관광지거리_m'] = dist.round(0).astype(int)
s.to_csv(f'{OUT}/stores.csv', index=False, encoding='utf-8-sig')

# ---------- 시뮬레이터 입력 ----------
sim = f[['행정동명', '등급', '원인', '방문', '결제가능방문', '타깃도달방문', 'QR가맹률', '소비갭']].copy()
sim['방문당소비'] = (89806 / np.exp(sim['소비갭'])).round(0)    # 소비갭 = ln(중앙값 89,806 / 방문당소비)
sim = sim.drop(columns='소비갭')

# [TODO-C] 시행 계절 배율 (4-2 민감도와 같은 방식): 그 계절 2개월 실제 방문 ÷ 연평균 2개월 방문
#   마스터_B_동월.csv는 원천 데이터라 저장소에 올리지 않음 → 드라이브에서 받아 아래 경로에 두고 실행
#   파일이 없으면 기존 data/sim_input.csv의 배율을 그대로 유지
MONTHLY = '../4-2_sensitivity/마스터_B_동월.csv'
SEASON = {'여름배율': [202606, 202607], '겨울배율': [202512, 202601]}
if os.path.exists(MONTHLY):
    B = pd.read_csv(MONTHLY, encoding='utf-8-sig')
    year = B[B['기준연월'].between(202508, 202607)].groupby('행정동명')['방문'].sum()
    for col, months in SEASON.items():
        s = B[B['기준연월'].isin(months)].groupby('행정동명')['방문'].sum()
        sim[col] = sim['행정동명'].map(s / (year * 2 / 12)).round(4)
elif os.path.exists(f'{OUT}/sim_input.csv'):
    old = pd.read_csv(f'{OUT}/sim_input.csv', encoding='utf-8-sig')
    keep = [c for c in SEASON if c in old.columns]
    if keep:
        sim = sim.merge(old[['행정동명'] + keep], on='행정동명', how='left')
    else:
        print(f'[알림] {MONTHLY} 없음 → 시뮬레이터 계절 선택은 연평균만 가능')
sim.to_csv(f'{OUT}/sim_input.csv', index=False, encoding='utf-8-sig')

print({k: len(pd.read_csv(f'{OUT}/{k}.csv', encoding='utf-8-sig'))
       for k in ['dong', 'zones', 'zone_budget', 'stores', 'sim_input']})
