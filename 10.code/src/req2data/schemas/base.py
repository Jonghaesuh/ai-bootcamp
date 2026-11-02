from pydantic import BaseModel, ConfigDict


class LLMOutput(BaseModel):
    """LLM이 채우는 스키마. 모르는 키를 거부해야 모델이 임의 구조로 감싼 출력({"data_mapping": {...}})이
    빈 기본값으로 통과하지 않고 검증 실패 → 재요청 경로를 탄다. JSON Schema에도 additionalProperties: false가 들어간다."""

    model_config = ConfigDict(extra="forbid")
