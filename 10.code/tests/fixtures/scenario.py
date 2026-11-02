"""PRD 3.1 재구매 시나리오의 LLM 응답 스크립트."""

DOCUMENT = """마케팅팀 요구사항 정의서
첫 구매 고객 중 재구매 가능성이 낮은 고객에게 쿠폰을 보내고 싶다.
첫 주문 금액과 결제 할부 정보를 활용하면 좋겠다.
쿠폰 발송은 첫 구매 직후에 한다."""

ITEMS = [
    {"id": "R1", "text": "첫 구매 고객 대상", "kind": "population"},
    {"id": "R2", "text": "재구매 여부 예측", "kind": "target"},
    {"id": "R3", "text": "첫 주문 금액 활용", "kind": "feature"},
    {"id": "R4", "text": "결제 할부 정보 활용", "kind": "feature"},
    {"id": "R5", "text": "취소 주문 제외", "kind": "filter"},
    {"id": "R6", "text": "쿠폰 발송", "kind": "constraint"},
]

SPEC_ROUND1 = {
    "business_goal": "재구매 가능성이 낮은 첫 구매 고객에게 쿠폰을 보내 재구매를 늘린다",
    "prediction_target": "첫 구매 고객",
    "analysis_unit": "고객",
    "prediction_time": "첫 구매 직후",
    "target_definition": None,
    "success_metric": "상위 20% 중 미재구매자 비율",
    "requirement_items": ITEMS[:4] + ITEMS[5:],
    "open_questions": [
        {
            "id": "Q1",
            "question": "재구매 판단 기간은 첫 구매 후 며칠로 할까요?",
            "target_field": "target_definition",
            "options": ["30일", "60일", "90일"],
            "default_option": "90일",
        },
        {
            "id": "Q2",
            "question": "취소된 주문도 첫 구매로 볼까요?",
            "target_field": "prediction_target",
            "options": ["포함", "제외"],
            "default_option": "제외",
        },
    ],
    "rationale": "재구매 판단 기간과 취소 주문 처리가 없어 질문으로 남김",
}

SPEC_ROUND2 = {
    **SPEC_ROUND1,
    "prediction_target": "취소되지 않은 첫 구매 고객",
    "target_definition": "첫 구매 후 90일 내 두 번째 주문",
    "requirement_items": ITEMS,
    "open_questions": [],
    "rationale": "사용자 답변으로 타깃 기간(90일)과 취소 제외를 반영",
}

MAPPING = {
    "analysis_key": {"table": "customers", "column": "customer_unique_id"},
    "reference_time": {"table": "orders", "column": "order_purchase_timestamp"},
    "items": [
        {
            "requirement_id": "R1",
            "role": "key",
            "columns": [{"table": "customers", "column": "customer_unique_id"}],
            "evidence": ["customers.customer_unique_id"],
        },
        {
            "requirement_id": "R2",
            "role": "target",
            "columns": [{"table": "orders", "column": "order_purchase_timestamp"}],
            "evidence": ["orders.order_purchase_timestamp"],
        },
        {
            "requirement_id": "R3",
            "role": "feature",
            "columns": [
                {"table": "order_payments", "column": "payment_value"},
                {"table": "orders", "column": "fake_amount"},
            ],
            "evidence": ["order_payments.payment_value"],
        },
        {
            "requirement_id": "R4",
            "role": "feature",
            "columns": [{"table": "order_payments", "column": "payment_installments"}],
            "evidence": ["order_payments.payment_installments"],
        },
        {
            "requirement_id": "R5",
            "role": "filter",
            "columns": [{"table": "orders", "column": "order_status"}],
            "evidence": ["orders.order_status"],
        },
    ],
    "join_paths": [
        {
            "left": {"table": "orders", "column": "customer_id"},
            "right": {"table": "customers", "column": "customer_id"},
        },
        {
            "left": {"table": "order_payments", "column": "order_id"},
            "right": {"table": "orders", "column": "order_id"},
        },
    ],
    "rationale": "customer_id는 주문 수와 고유값이 같아 customer_unique_id를 고객 키로 사용",
}

_PLAN = {
    "analysis_unit": "고객",
    "entity_key": "customers.customer_unique_id",
    "reference_time": "orders.order_purchase_timestamp",
    "cohort": "first_event",
    "population_filters": ["orders.order_status <> 'canceled'"],
    "target": {
        "kind": "repeat_event",
        "window_days": 90,
        "description": "첫 구매 후 90일 내 두 번째 주문",
    },
}
_FEATURES = [
    {
        "name": "first_amount",
        "source_column": "order_payments.payment_value",
        "agg": "sum",
        "requirement_ids": ["R3"],
    },
    {
        "name": "installments",
        "source_column": "order_payments.payment_installments",
        "agg": "max",
        "requirement_ids": ["R4"],
    },
]

# v1: 리뷰 점수(배송 후 작성) 누수
DESIGN_V1 = {
    **_PLAN,
    "features": [
        *_FEATURES,
        {"name": "review_score", "source_column": "order_reviews.review_score", "agg": "avg"},
    ],
    "rationale": "리뷰 점수가 재구매와 관련이 클 것으로 보고 포함",
}

# v2: 피드백 반영 (누수 피처 제거). 관측 기간 절단·분할은 SQL 빌더가 처리한다.
DESIGN_V2 = {**_PLAN, "features": _FEATURES, "rationale": "검증 피드백에 따라 리뷰 점수를 제거"}
