"""존 예산 배분: 5장 동별 배분예산 → 존 단위로 내림

존 예산 = 동 배분예산 × (존 안의 그 동 QR 가게 수 ÷ 그 동 전체 QR 가게 수)
존 밖 예산 = 동 배분예산 − 그 동 존 예산 합   (존에 속하지 않은 가맹점 몫. 할인 범위는 동 내 가맹점 전체이므로 유지)

- 할인은 QR 가맹점에서만 쓰이므로 면적이 아니라 QR 가게 수로 나눈다.
- 여러 동에 걸친 존은 할인 대상 동 부분만 그 동의 예산·할인 조건을 받는다.
  (할인 대상이 아닌 동의 가게는 예산 없음)
- 존이 없는 동(예래동)은 예산 전액이 존 밖(동 전역 가맹점) 몫이 된다.

사용: python3 zone_budget.py [OUT=out]   (zone_detect.py 실행 후)
"""
import sys
import pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else 'out'

dong = pd.read_csv('../3-1_3-2/3-1_3-2_최종표.csv', encoding='utf-8-sig')
dong = dong[['행정동명', '등급', '원인', '할인방식', '할인율', '1인당한도', '최소결제금액', '최종배분예산']]
zd = pd.read_csv(f'{OUT}/zone_dong.csv', encoding='utf-8-sig')
st = pd.read_csv(f'{OUT}/store_zone.csv', encoding='utf-8-sig')

# 동 전체 QR 가게 중 존 안에 있는 비중 (참고용)
t = st[st['할인대상동']]
dq = t.groupby('행정동명').agg(동QR가게=('QR상태', lambda x: (x == 'QR 있음').sum()))
dq['존안QR가게'] = t[t['존ID'].notna()].groupby('행정동명')['QR상태'].apply(lambda x: (x == 'QR 있음').sum())
dq = dq.fillna(0).astype(int)
dq['존안QR비중'] = (dq['존안QR가게'] / dq['동QR가게']).round(3)

# 가중치: 존 안의 그 동 QR 가게 수 ÷ 그 동 전체 QR 가게 수
zd = zd.merge(dq['동QR가게'], left_on='행정동명', right_index=True)
zd['동내비중'] = zd['QR확인'] / zd['동QR가게']

zb = zd.merge(dong, on='행정동명', how='left')
zb['존예산'] = (zb['최종배분예산'] * zb['동내비중']).round(0)
zb = zb[['존ID', '행정동명', '등급', '할인율', '1인당한도', '할인방식', '최소결제금액',
         '가게수', 'QR확인', 'QR미확인', 'QR확인비율', '존내비중', '동구성',
         '동내비중', '최종배분예산', '존예산']]
zb = zb.rename(columns={'가게수': '존내_동가게수', '존내비중': '존에서_이동비중',
                        '동내비중': '동예산중_존몫', '최종배분예산': '동배분예산'})
zb['동예산중_존몫'] = zb['동예산중_존몫'].round(3)
# 존 밖 몫 (존이 없는 동은 전부 존 밖)
rest = dong.merge(zb.groupby('행정동명')['존예산'].sum().rename('존합'), on='행정동명', how='left').fillna({'존합': 0})
rest = rest.merge(dq, on='행정동명', how='left')
rest = pd.DataFrame({'존ID': rest['행정동명'] + '-존밖', '행정동명': rest['행정동명'], '등급': rest['등급'],
                     '할인율': rest['할인율'], '1인당한도': rest['1인당한도'], '할인방식': rest['할인방식'],
                     '최소결제금액': rest['최소결제금액'], 'QR확인': rest['동QR가게'] - rest['존안QR가게'],
                     '동내비중': 1 - rest['존안QR비중'], '최종배분예산': rest['최종배분예산'],
                     '존예산': (rest['최종배분예산'] - rest['존합']).round(0)})
zb = pd.concat([zb, rest.rename(columns={'동내비중': '동예산중_존몫', '최종배분예산': '동배분예산'})], ignore_index=True)
zb['동예산중_존몫'] = zb['동예산중_존몫'].round(3)
zb['구분'] = zb['존ID'].str.endswith('-존밖').map({True: '존 밖 가맹점', False: '상권 존'})
zb = zb.sort_values(['등급', '행정동명', '구분', '존예산'], ascending=[True, True, False, False])
zb.to_csv(f'{OUT}/zone_budget.csv', index=False, encoding='utf-8-sig')

# 동별 점검표
chk = dong[['행정동명', '등급', '최종배분예산']].merge(
    zb[zb['구분'] == '상권 존'].groupby('행정동명').agg(존수=('존ID', 'size'), 존예산합=('존예산', 'sum')).reset_index(),
    on='행정동명', how='left').merge(dq.reset_index(), on='행정동명', how='left')
chk['존수'] = chk['존수'].fillna(0).astype(int)
chk['존예산합'] = chk['존예산합'].fillna(0)
chk['존밖예산'] = chk['최종배분예산'] - chk['존예산합']
chk['비고'] = chk['존수'].map(lambda n: '존 없음 → 전액 존 밖(동 전역)' if n == 0 else '')
chk.to_csv(f'{OUT}/zone_budget_동별점검.csv', index=False, encoding='utf-8-sig')

# 검증: 동마다 (존 예산 + 존 밖 예산) = 동 배분예산
diff = (zb.groupby('행정동명')['존예산'].sum() - dong.set_index('행정동명')['최종배분예산']).abs().max()
assert diff < 2, f'동 예산 합 불일치 {diff}'
z_in = zb.loc[zb['구분'] == '상권 존', '존예산'].sum()
print(f"총 {zb['존예산'].sum()/1e8:.3f}억 = 상권 존 {z_in/1e8:.3f}억 ({z_in/zb['존예산'].sum():.0%}) + 존 밖 가맹점 {(zb['존예산'].sum()-z_in)/1e8:.3f}억")
