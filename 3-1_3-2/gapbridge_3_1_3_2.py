"""
제주페이 갭브릿지 | 3-1 우선순위 등급화 + 3-2 예산 배분
입력 : 1-5_전달_3-1_3-2용.csv (필수), 마스터_A_동.csv (객단가형 최소결제금액 산정용 객단가만 사용)
범위 : 흡수 가능 예산 = [cap_보수(현재 방문당 소비 반영), cap(1인당 한도 소진 가정 상한)]
한계 : cap의 시행기간 조정은 12개월 cap × (시행개월/12) 단순 비례로, 월별 방문 계절성을 반영하지 않는다.
원칙 : 1~5단계 결과(18개 동, 전환형/객단가형, 도달비중, QR가맹률)는 그대로 사용하고 재계산하지 않는다.
"""
import itertools
import numpy as np
import pandas as pd

# =====================================================================
# [A] 정책 파라미터 (외생) — 회의에서 확정. 아래 값은 모두 '예시'다.
# =====================================================================
TOTAL_BUDGET = 300_000_000            # 총예산(원) ← 사용자 입력. 예시 3억
BUDGET_SCENARIOS = [200_000_000, 300_000_000, 400_000_000, 500_000_000]   # 가상 시나리오
PROMOTION_MONTHS = 2                  # 프로모션 시행기간(개월) ← 정책 파라미터
DATA_MONTHS = 12                      # 방문 데이터 기준기간(최근 12개월) — 데이터 정의, 변경 금지
# --- 할인 구조 (할인율·1인당 한도를 하나의 식으로 묶음) ---------------------
#   1인당 할인액   = min(결제액 × 할인율_g, 1인당 한도_g)
#   1인당 한도_g   = 할인율_g × LIMIT_REACH_SPEND   (한도에 도달하는 결제액은 모든 등급 동일)
#   → 할인율은 '기울기', 한도는 '끝점'. 둘이 같은 파라미터로 묶여 중복·모순이 없다.
DISCOUNT_MODE = "graded"              # 'graded'(추천: 등급별 할인율) | 'uniform'(전 동 동일 할인율)
DISCOUNT_RATE = {"A": 0.20, "B": 0.15, "C": 0.10}   # graded 모드 등급별 할인율 ← 예시
UNIFORM_RATE = 0.20                   # uniform 모드 할인율 ← 예시
LIMIT_REACH_SPEND = 50_000            # 한도 도달 결제액(원) ← 예시. A 20%×5만원 = 1인당 한도 1만원
GRADE_MULT = {"A": 1.5, "B": 1.0, "C": 0.7}         # uniform 모드에서만 쓰는 등급별 배분 가중 ← 예시
MAX_SHARE = 0.20                      # 한 동이 총예산에서 가져갈 수 있는 최대 비중 ← 예시
MIN_PAY_UPLIFT = 0.10                 # 객단가형 최소결제금액 = 현재 객단가 × (1+α) ← 예시 α
MIN_PAY_ROUND = 5_000                 # 최소결제금액 올림 단위(원)

# 가중치 시나리오 (w_gap, w_visit, w_reach), 합=1
WEIGHT_SCENARIOS = {
    "S1_기본안(추천)":      (0.40, 0.30, 0.30),
    "S2_소비갭중시안":      (0.60, 0.20, 0.20),
    "S3_도달·집행효율중시안": (0.30, 0.30, 0.40),
}
BASE_SCENARIO = "S1_기본안(추천)"

# =====================================================================
# [B] 방법론 설정 (분석자 설계)
# =====================================================================
NORMALIZATION = "percentile"   # 'percentile'(추천) | 'minmax' | 'zscore'
LOG_VISIT = True               # 방문 규모 log 변환 (minmax/zscore에서만 의미 있음)
GRADE_METHOD = "30-40-30"      # '30-40-30'(추천) | 'jenks'

PATH_MAIN = "/mnt/user-data/uploads/1-5_전달_3-1_3-2용.csv"
PATH_MASTER_A = "마스터_A_동.csv"


# ---------------------------------------------------------------------
def load(path_main=PATH_MAIN, path_a=PATH_MASTER_A):
    d = pd.read_csv(path_main, encoding="utf-8-sig")
    t = d[d["구분"] == "할인 배치 대상"].copy().reset_index(drop=True)
    t = t.rename(columns={"소비갭(로그)": "소비갭", "도달비중(%)": "도달비중",
                          "QR가맹률_음식소매": "QR가맹률", "결제가능_방문(추정)": "결제가능방문"})
    try:
        a = pd.read_csv(path_a, encoding="utf-8-sig")[["행정동명", "객단가_음식소매"]]
        t = t.merge(a, on="행정동명", how="left")
    except FileNotFoundError:
        t["객단가_음식소매"] = np.nan
    return t


def validate_input(t):
    assert len(t) == 18, f"대상 동 {len(t)}개"
    assert t["원인"].value_counts().to_dict() == {"전환형": 13, "객단가형": 5}
    key = ["방문", "소비갭", "도달비중", "QR가맹률", "결제가능방문"]
    assert t[key].isna().sum().sum() == 0, "결측 존재"
    # 결제가능방문 = 방문 × 결제가능_비중 검증 (비중이 소수 1자리 반올림이라 ±0.05%p 오차 허용)
    diff = (t["방문"] * t["결제가능_비중(%)"] / 100 - t["결제가능방문"]).abs()
    assert (diff <= t["방문"] * 0.0005 + 1).all()
    # 소비갭 = ln(37개 동 중앙값 89,806 / 방문당소비)
    assert np.allclose(np.log(89806 / t["방문당소비_음식소매"]), t["소비갭"], atol=2e-3)


# ---------------------------------------------------------------------
# 3-1 우선순위
# ---------------------------------------------------------------------
def normalize(s, method):
    if method == "percentile":          # 순위 백분위 0~100 (동점은 평균순위)
        return (s.rank(method="average") - 1) / (len(s) - 1) * 100
    if method == "minmax":
        return (s - s.min()) / (s.max() - s.min()) * 100
    if method == "zscore":
        return (s - s.mean()) / s.std(ddof=0)
    raise ValueError(method)


def score(t, weights, method=NORMALIZATION, log_visit=LOG_VISIT):
    v = np.log(t["방문"]) if (log_visit and method != "percentile") else t["방문"]
    out = pd.DataFrame({
        "점수_소비갭": normalize(t["소비갭"], method),      # 갭 클수록 ↑
        "점수_방문": normalize(v, method),                 # 방문 많을수록 ↑
        "점수_도달": normalize(t["도달비중"], method),       # 도달 높을수록 ↑
    })
    w1, w2, w3 = weights
    assert abs(w1 + w2 + w3 - 1) < 1e-9
    out["종합점수"] = w1 * out["점수_소비갭"] + w2 * out["점수_방문"] + w3 * out["점수_도달"]
    out["순위"] = out["종합점수"].rank(ascending=False, method="min").astype(int)
    return out


def jenks3(x):
    """18개 표본이므로 3구간 natural break를 전수탐색(GVF 최대)."""
    xs = np.sort(np.asarray(x))[::-1]
    n, best = len(xs), (None, np.inf)
    for i, j in itertools.combinations(range(1, n), 2):
        sse = sum(((g - g.mean()) ** 2).sum() for g in (xs[:i], xs[i:j], xs[j:]))
        if sse < best[1]:
            best = ((i, j), sse)
    return best[0]                       # A 개수 = i, B 개수 = j-i


def grade(rank_series, comp, method=GRADE_METHOD):
    n = len(rank_series)
    if method == "30-40-30":
        nA = int(round(n * 0.3)); nC = nA          # 18 → 5 / 8 / 5
        nB = n - nA - nC
    elif method == "jenks":
        i, j = jenks3(comp); nA, nB = i, j - i
    order = rank_series.rank(method="first")
    return np.where(order <= nA, "A", np.where(order <= nA + nB, "B", "C"))


# ---------------------------------------------------------------------
# 3-2 예산
# ---------------------------------------------------------------------
def min_payment(t):
    m = np.ceil(t["객단가_음식소매"] * (1 + MIN_PAY_UPLIFT) / MIN_PAY_ROUND) * MIN_PAY_ROUND
    return np.where(t["원인"] == "객단가형", m, np.nan)


def allocate(weight, cap, total, max_share=MAX_SHARE):
    """우선순위 가중 비례배분 + cap 초과분 회수 후 반복 재배분(water-filling)."""
    ub = np.minimum(cap, max_share * total)            # 동별 상한 = min(cap, 집중도 상한)
    alloc = np.zeros(len(weight)); remain = float(total)
    active = ub > 0
    for _ in range(100):
        if remain < 1 or not active.any():
            break
        share = np.where(active, weight, 0) / np.where(active, weight, 0).sum()
        alloc = alloc + remain * share
        over = alloc > ub
        alloc = np.minimum(alloc, ub)                  # 초과분 제거
        remain = total - alloc.sum()                   # 잔여예산
        active = alloc < ub - 1e-6                     # 여유 있는 동만 다음 라운드
        if not over.any():
            break
    return alloc, max(remain, 0.0)


def run(total_budget=TOTAL_BUDGET, verbose=True, mode=None):
    mode = mode or DISCOUNT_MODE
    assert 0 < PROMOTION_MONTHS <= DATA_MONTHS, "시행기간은 1~12개월"
    t = load(); validate_input(t)

    # ---------- 3-1 ----------
    sc = score(t, WEIGHT_SCENARIOS[BASE_SCENARIO])
    t = pd.concat([t, sc], axis=1)
    t["등급"] = grade(t["순위"], t["종합점수"])

    # ---------- 3-2 ----------
    t["할인방식"] = np.where(t["원인"] == "전환형", "문턱 낮은 정률 할인", "최소결제금액 조건부 할인")
    if mode == "graded":
        t["할인율"] = t["등급"].map(DISCOUNT_RATE)
        grade_w = 1.0                                   # 등급 차이는 할인율(→한도)에 이미 반영
    elif mode == "uniform":
        t["할인율"] = UNIFORM_RATE
        grade_w = t["등급"].map(GRADE_MULT)              # 할인율이 같으므로 배분가중으로만 차등
    else:
        raise ValueError(mode)
    t["1인당한도"] = t["할인율"] * LIMIT_REACH_SPEND      # 1인당 최대 할인액
    t["최소결제금액"] = min_payment(t)
    assert (t["최소결제금액"].fillna(0) < LIMIT_REACH_SPEND).all(), "최소결제금액이 한도 도달 결제액 이상"
    t["타깃도달방문"] = t["방문"] * t["도달비중"] / 100
    # cap = 결제가능방문 × QR가맹률 × 1인당한도 × (시행개월/12)  ※ 계절성 미반영 단순 비례
    t["cap_12개월"] = t["결제가능방문"] * t["QR가맹률"] * t["1인당한도"]
    t["cap"] = t["cap_12개월"] * (PROMOTION_MONTHS / DATA_MONTHS)
    # [보수 시나리오] 실제 방문당 소비 수준을 반영한 1인당 할인액 → cap_보수 (배분 로직에는 쓰지 않음, 참고 지표)
    t["1인당할인_보수"] = np.minimum(t["할인율"] * t["방문당소비_음식소매"], t["1인당한도"])
    t["cap_보수"] = (t["결제가능방문"] * t["QR가맹률"] * t["1인당할인_보수"]
                    * (PROMOTION_MONTHS / DATA_MONTHS))
    # 배분가중 = (등급가중) × 1인당한도 × 타깃도달방문  → graded면 타깃 1명당 지원액이 할인율 비례
    t["배분가중"] = grade_w * t["1인당한도"] * t["타깃도달방문"]
    t["최종배분예산"], left = allocate(t["배분가중"].values, t["cap"].values, total_budget)
    t["cap소진율"] = t["최종배분예산"] / t["cap"]

    # ---------- 검증 ----------
    assert (t["최종배분예산"] <= t["cap"] + 1).all()
    assert t["최종배분예산"].sum() <= total_budget + 1
    assert (t["최종배분예산"] <= MAX_SHARE * total_budget + 1).all()
    assert t["원인"].value_counts().to_dict() == {"전환형": 13, "객단가형": 5}
    if verbose:
        print(f"총예산 {total_budget:,.0f} | 배분 {t['최종배분예산'].sum():,.0f} | 미배분 {left:,.0f} | cap합({PROMOTION_MONTHS}개월) {t['cap'].sum():,.0f}")
    return t.sort_values("순위").reset_index(drop=True), left


def sensitivity(step=0.1, wmin=0.1):
    """명시 시나리오 + 가중치 격자(각 w ≥ 0.1, 0.1 간격, 36개 조합) 민감도."""
    t = load()
    grid = [(a, b, round(1 - a - b, 2)) for a in np.arange(wmin, 1, step) for b in np.arange(wmin, 1, step)
            if 1 - a - b >= wmin - 1e-9]
    ranks, grades = {}, {}
    for w in grid:
        s = score(t, w); ranks[w] = s["순위"]; grades[w] = grade(s["순위"], s["종합점수"])
    R = pd.DataFrame(ranks); G = pd.DataFrame(grades)
    res = pd.DataFrame({"행정동명": t["행정동명"],
                        "최고순위": R.min(axis=1), "최저순위": R.max(axis=1), "순위폭": R.max(axis=1) - R.min(axis=1),
                        "A비율": (G == "A").mean(axis=1), "B비율": (G == "B").mean(axis=1), "C비율": (G == "C").mean(axis=1)})
    return res, len(grid)


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    res, _ = run()
    cols = ["행정동명", "등급", "순위", "원인", "할인방식", "할인율", "1인당한도", "최소결제금액", "소비갭", "방문", "도달비중",
            "점수_소비갭", "점수_방문", "점수_도달", "종합점수", "결제가능_비중(%)", "결제가능방문", "타깃도달방문",
            "QR가맹률", "cap_12개월", "cap", "1인당할인_보수", "cap_보수", "배분가중", "최종배분예산", "cap소진율"]
    res[cols].round(4).to_csv("3-1_3-2_최종표.csv", index=False, encoding="utf-8-sig")
    print(res[cols].round(1).to_string())

    # 가중치 시나리오별 순위·등급
    t = load(); sc = {}
    for k, w in WEIGHT_SCENARIOS.items():
        s_ = score(t, w); sc[k + "_순위"] = s_["순위"].values; sc[k + "_등급"] = grade(s_["순위"], s_["종합점수"])
    sc = pd.DataFrame(sc, index=t["행정동명"])
    sens, n = sensitivity()
    sc.join(sens.set_index("행정동명")).sort_values(list(sc)[0]).to_csv("민감도_시나리오.csv", encoding="utf-8-sig")

    # 총예산 가상 시나리오 (현재 파라미터 기준)
    alloc = {f"배분_{B/1e8:.0f}억(가상)": run(B, verbose=True)[0].set_index("행정동명")["최종배분예산"]
             for B in BUDGET_SCENARIOS}
    pd.DataFrame(alloc).loc[res["행정동명"]].round(0).to_csv("예산시나리오.csv", encoding="utf-8-sig")

    # 할인율 방안 비교: ① uniform(전 동 20%) vs ② graded(A20/B15/C10)
    rows = []
    for mode in ["uniform", "graded"]:
        for B in BUDGET_SCENARIOS:
            r, left = run(B, verbose=False, mode=mode)
            g = r.groupby("등급")
            per = g["최종배분예산"].sum() / g["타깃도달방문"].sum()          # 타깃 1명당 배분액
            rows.append({"방안": mode, "총예산(가상)": B, "cap합": r["cap"].sum(),
                         "배분": r["최종배분예산"].sum(), "미배분": left,
                         "cap도달동수": int((r["cap소진율"] > 0.999).sum()),
                         **{f"{k}_예산": g["최종배분예산"].sum()[k] for k in "ABC"},
                         **{f"{k}_타깃1인당": per[k] for k in "ABC"},
                         "C/A_1인당비율": per["C"] / per["A"]})
    pd.DataFrame(rows).round(3).to_csv("할인방안비교.csv", index=False, encoding="utf-8-sig")
