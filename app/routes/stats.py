from fastapi import APIRouter

from app.metrics import metrics

router = APIRouter(tags=["operations"])


@router.get("/stats")
def stats() -> dict:
    """Return process-local latency, cache, and embedding measurements."""
    return metrics.snapshot()
