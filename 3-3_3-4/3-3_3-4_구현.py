import argparse
import csv
import math
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOTAL_BUDGET = 300_000_000
PROMOTION_MONTHS = 2
DATA_MONTHS = 12
EFFECTS = {'낮음': Decimal('0.44'), '중간': Decimal('1.10'), '높음': Decimal('2.22')}
CASES = {'할인한도 도달': 'cap', '현재 소비 수준': 'cap_보수'}
DETAIL_COLUMNS = [
    '행정동명', '등급', '집행가정', '배정예산_원', '적용상한_원',
    '집행가능액_원', '미집행액_원', '집행률_%',
    *[f'추가소비_{label}({value:.2f}배)_원' for label, value in EFFECTS.items()]]
SUMMARY_COLUMNS = ['집행가정', '시행개월', '배정예산_원', '집행가능액_원',
                   '미집행액_원', '집행률_%', *DETAIL_COLUMNS[8:]]
COMPARISON_COLUMNS = [
    '집행가정', '배분 방식', '집행가능액_원', '미집행액_원', '집행률',
    '추가소비_낮음_원', '추가소비_중간_원', '추가소비_높음_원', '총예산_원', '시행개월']
GINI_COLUMNS = ['집행가정', '효과시나리오', '평가동수', '시행전_지니계수',
                '우리배분 후_지니계수', '균등배분 후_지니계수']
DIFFERENCE_COLUMNS = ['행정동', '낮음차이_원', '중간차이_원', '높음차이_원']


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def won(value):
    """금액은 최종 출력할 때만 정수 원 단위로 반올림한다."""
    return int(value.quantize(Decimal('1'), rounding=ROUND_HALF_UP))


def percent(spent, budget):
    return float((spent / budget * 100).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)) if budget else 0.0


def load_inputs(allocation_path, master_path):
    _, allocation = read_csv(allocation_path)
    _, master = read_csv(master_path)
    required = ['행정동명', '등급', '순위', '최종배분예산', 'cap', 'cap_보수']
    if len(allocation) != 18 or len({r['행정동명'] for r in allocation}) != 18:
        raise ValueError('최종 배분표에는 중복 없이 18개 동이 있어야 합니다.')
    for row in allocation:
        if any(not row.get(c) for c in required):
            raise ValueError(f'최종 배분표의 필수 값 누락: {row}')
        if row['등급'] not in ['A', 'B', 'C']:
            raise ValueError('등급은 A/B/C여야 합니다.')
        for key in ['최종배분예산', 'cap', 'cap_보수']:
            if not Decimal(row[key]).is_finite() or Decimal(row[key]) < 0:
                raise ValueError(f'{key}는 0 이상 유한한 금액이어야 합니다.')
        if Decimal(row['cap_보수']) > Decimal(row['cap']):
            raise ValueError('현재 소비 기준 상한이 한도 도달 상한보다 큽니다.')
    allocation.sort(key=lambda r: (r['등급'], Decimal(r['순위'])))
    if abs(sum(Decimal(r['최종배분예산']) for r in allocation) - TOTAL_BUDGET) > Decimal('0.01'):
        raise ValueError('이 코드는 총예산 3억 원의 배분표를 재현합니다.')
    if len(master) != 43 or len({r['행정동명'] for r in master}) != 43:
        raise ValueError('마스터에는 중복 없이 43개 동이 있어야 합니다.')
    baseline = {r['행정동명']: float(Decimal(r['금액_음식소매']) * PROMOTION_MONTHS / DATA_MONTHS)
                for r in master}
    if not all(math.isfinite(v) and v >= 0 for v in baseline.values()):
        raise ValueError('마스터 소비액은 0 이상 유한한 값이어야 합니다.')
    if not {r['행정동명'] for r in allocation}.issubset(baseline):
        raise ValueError('할인 대상 동이 마스터에 누락되어 있습니다.')
    return allocation, baseline


def simulate(allocation, uniform=False):
    """3-3 또는 3-4 균등 배분의 동별 36행과 요약 2행을 계산한다.

    상한은 3-2 최종표의 cap / cap_보수를 그대로 사용한다.
    cap = 결제가능방문 × QR가맹률 × 1인당한도 × (2/12)
    1인당한도 = 할인율 × 50,000
    cap_보수 = 결제가능방문 × QR가맹률 × 1인당할인_보수 × (2/12)
    1인당할인_보수 = min(할인율 × 방문당소비_음식소매, 1인당한도)
    cap를 반올림된 마스터 열에서 다시 계산하면 기존 원자료와 차이가 생길 수
    있으므로, 확정 배분표의 소수 정밀도 값을 사용한다.
    """
    if uniform:
        # 정수 원으로 3억 원을 나눔: 앞의 12개 동 +1원, 동별 차이 최대 1원.
        base, remainder = divmod(TOTAL_BUDGET, len(allocation))
        budgets = {r['행정동명']: Decimal(base + int(i < remainder)) for i, r in enumerate(allocation)}
    else:
        budgets = {r['행정동명']: Decimal(r['최종배분예산']) for r in allocation}

    detail, summary = [], []
    for case, cap_col in CASES.items():
        allocated_total = spent_total = Decimal(0)
        for r in allocation:
            budget = budgets[r['행정동명']]
            cap = Decimal(r[cap_col])
            spent = min(budget, cap)
            unspent = budget - spent  # 남는 예산을 다른 동으로 재배분하지 않음.
            allocated_total += budget
            spent_total += spent
            row = {'행정동명': r['행정동명'], '등급': r['등급'], '집행가정': case,
                   '배정예산_원': won(budget), '적용상한_원': won(cap),
                   '집행가능액_원': won(spent), '미집행액_원': won(unspent),
                   '집행률_%': percent(spent, budget)}
            for label, effect in EFFECTS.items():
                row[f'추가소비_{label}({effect:.2f}배)_원'] = won(spent * effect)
            detail.append(row)

        # 동별 반올림 결과를 더하지 않고, 반올림 전 총액에서 요약을 계산한다.
        row = {'집행가정': case, '시행개월': PROMOTION_MONTHS,
               '배정예산_원': won(allocated_total), '집행가능액_원': won(spent_total),
               '미집행액_원': won(allocated_total - spent_total),
               '집행률_%': percent(spent_total, allocated_total)}
        for label, effect in EFFECTS.items():
            row[f'추가소비_{label}({effect:.2f}배)_원'] = won(spent_total * effect)
        summary.append(row)
    for row in summary:
        parts = [r for r in detail if r['집행가정'] == row['집행가정']]
        assert len(parts) == 18
        # 우리 배분은 원자료의 소수 배정액을 동별로 반올림하므로 합계에 몇 원 차이 가능.
        rounded_budget_sum = sum(r['배정예산_원'] for r in parts)
        assert abs(rounded_budget_sum - TOTAL_BUDGET) <= (0 if uniform else 18)
        for col in SUMMARY_COLUMNS:
            if col.endswith('_원'):
                assert abs(sum(r[col] for r in parts) - row[col]) <= 18
    return detail, summary


def make_comparison(own_summary, equal_summary):
    """두 요약표를 합쳐 예산·추가 소비 비교표 4행을 만든다."""
    rows = []
    for case in CASES:
        for method, source in [('우리 배분', own_summary), ('균등 배분', equal_summary)]:
            s = next(r for r in source if r['집행가정'] == case)
            row = {'집행가정': '한도 도달' if case == '할인한도 도달' else '현재 소비',
                   '배분 방식': method, '집행가능액_원': s['집행가능액_원'],
                   '미집행액_원': s['미집행액_원'], '집행률': s['집행률_%']}
            for label, effect in EFFECTS.items():
                row[f'추가소비_{label}_원'] = s[f'추가소비_{label}({effect:.2f}배)_원']
            row.update({'총예산_원': s['배정예산_원'], '시행개월': PROMOTION_MONTHS})
            rows.append(row)
    return rows


def gini(values):
    """정렬된 소비액 x_i: G = 2Σ(i×x_i)/(n×Σx_i) − (n+1)/n."""
    values = sorted(values)
    n = len(values)
    total = sum(values)
    if not total:
        return 0.0
    value = 2*sum(i*x for i, x in enumerate(values, 1))/(n*total) - (n+1)/n
    independent = sum(abs(x-y) for x in values for y in values)/(2*n*total)
    assert abs(value - independent) < 1e-12
    return value


def make_gini(own_detail, equal_detail, baseline, names):
    """18개 대상 동 또는 전체 43개 동의 시행 전후 지니계수 6행.

    시뮬레이션 CSV에 기록된 원 단위 추가 소비를 기존 2개월 환산 소비에 더한다.
    43개 동 비교에서는 비대상 25개 동에 추가 소비 0을 적용한다.
    """
    before = gini([baseline[name] for name in names])
    lookups = [{(r['집행가정'], r['행정동명']): r for r in detail}
               for detail in [own_detail, equal_detail]]
    rows = []
    for case in CASES:
        for label, effect in EFFECTS.items():
            col = f'추가소비_{label}({effect:.2f}배)_원'
            after = [gini([baseline[name] + lookup.get((case, name), {}).get(col, 0)
                           for name in names]) for lookup in lookups]
            rows.append({'집행가정': case, '효과시나리오': label, '평가동수': len(names),
                         '시행전_지니계수': f'{before:.8f}',
                         '우리배분 후_지니계수': f'{after[0]:.8f}',
                         '균등배분 후_지니계수': f'{after[1]:.8f}'})
    return rows


def make_difference(own_detail, equal_detail):
    """현재 소비 가정: 동별 원 단위 추가 소비 차이 = 우리 − 균등. 합계 포함."""
    own = [r for r in own_detail if r['집행가정'] == '현재 소비 수준']
    equal = {r['행정동명']: r for r in equal_detail if r['집행가정'] == '현재 소비 수준'}
    rows = []
    for r in own:
        row = {'행정동': r['행정동명']}
        for label, effect in EFFECTS.items():
            col = f'추가소비_{label}({effect:.2f}배)_원'
            row[f'{label}차이_원'] = r[col] - equal[r['행정동명']][col]
        rows.append(row)
    rows.append({'행정동': '합계', **{f'{label}차이_원': sum(r[f'{label}차이_원'] for r in rows)
                                    for label in EFFECTS}})
    return rows


def write_csv(path, columns, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def verify_csv(generated_path, reference_path):
    """컬럼·행 순서·문자열·수치 비교. 지니계수만 1e-8 허용, 금액은 정확히 일치."""
    columns, actual = read_csv(generated_path)
    expected_columns, expected = read_csv(reference_path)
    if columns != expected_columns or len(actual) != len(expected):
        raise ValueError(f'검증 실패(구조): {reference_path.name}')
    for i, (a, b) in enumerate(zip(actual, expected), 2):
        for column in columns:
            if a[column] == b[column]:
                continue
            try:
                difference = abs(Decimal(a[column]) - Decimal(b[column]))
            except InvalidOperation:
                raise ValueError(f'검증 실패: {reference_path.name} {i}행 {column}')
            tolerance = Decimal('0.00000001') if '지니계수' in column else Decimal(0)
            if difference > tolerance:
                raise ValueError(f'검증 실패: {reference_path.name} {i}행 {column}: {a[column]} != {b[column]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allocation', type=Path, default=ROOT/'work/final.csv', help='3-1·3-2 최종 배분표')
    parser.add_argument('--master', type=Path, default=ROOT/'work/master.csv', help='마스터_A_동.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/재현결과')
    parser.add_argument('--verify-dir', type=Path, help='현재 8개 CSV와 비교할 때 지정하는 폴더')
    args = parser.parse_args()
    if args.output_dir.resolve() == (ROOT/'outputs').resolve():
        parser.error('기존 결과를 보호하기 위해 outputs 바로 아래에는 저장하지 않습니다.')
    if args.verify_dir and args.output_dir.resolve() == args.verify_dir.resolve():
        parser.error('출력 폴더와 검증 대상 폴더는 달라야 합니다.')
    allocation, baseline = load_inputs(args.allocation, args.master)
    own_detail, own_summary = simulate(allocation)
    equal_detail, equal_summary = simulate(allocation, uniform=True)
    target_names = [r['행정동명'] for r in allocation]
    outputs = [
        ('3-3_시나리오요약.csv', SUMMARY_COLUMNS, own_summary),
        ('3-3_동별시뮬레이션.csv', DETAIL_COLUMNS, own_detail),
        ('3-4_균일배분_시나리오요약.csv', SUMMARY_COLUMNS, equal_summary),
        ('3-4_균일배분_동별시뮬레이션.csv', DETAIL_COLUMNS, equal_detail),
        ('3-4_지니비교_43개동.csv', GINI_COLUMNS, make_gini(own_detail, equal_detail, baseline, list(baseline))),
        ('3-4_지니비교_18개동.csv', GINI_COLUMNS, make_gini(own_detail, equal_detail, baseline, target_names)),
        ('3-4_배분비교표.csv', COMPARISON_COLUMNS, make_comparison(own_summary, equal_summary)),
        ('3-4_동별_예상추가소비차이_현재소비.csv', DIFFERENCE_COLUMNS, make_difference(own_detail, equal_detail))]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, columns, rows in outputs:
        path = args.output_dir / filename
        write_csv(path, columns, rows)
        saved_columns, saved = read_csv(path)
        assert saved_columns == columns and len(saved) == len(rows)
        assert all(all(r[c] != '' for c in columns) for r in saved)
        if args.verify_dir:
            verify_csv(path, args.verify_dir/filename)
        print(f'{filename}: {len(rows)}행' + (' / 기존 파일과 일치' if args.verify_dir else ''))
    print(f'저장 위치: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
