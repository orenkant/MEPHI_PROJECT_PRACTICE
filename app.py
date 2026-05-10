import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predict import load_artifact, predict_one, predict_batch


# артефакт модели в памяти процесса
ARTIFACT = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.environ.get("MODEL_PATH", "model.pkl")
    t0 = time.time()
    art = load_artifact(model_path)
    ARTIFACT["data"] = art
    ARTIFACT["loaded_in_s"] = round(time.time() - t0, 2)
    print(f"Модель загружена: {model_path} за {ARTIFACT['loaded_in_s']} с")
    print(f"ROC-AUC test: {art.get('roc_auc_test', 'unknown')}")
    yield
    ARTIFACT.clear()


app = FastAPI(
    title="СберАвтоподписка — предсказание целевого действия",
    description="Возвращает вероятность того, что визит закончится целевым действием",
    version="1.0.0",
    lifespan=lifespan,
)


# схема одного визита, все поля опциональны
class VisitInput(BaseModel):
    utm_source: Optional[str] = None
    utm_medium: Optional[str] = None
    utm_campaign: Optional[str] = None
    utm_adcontent: Optional[str] = None
    device_category: Optional[str] = Field(None, description="mobile / desktop / tablet")
    device_os: Optional[str] = None
    device_brand: Optional[str] = None
    device_browser: Optional[str] = None
    geo_country: Optional[str] = None
    geo_city: Optional[str] = None
    visit_hour: Optional[int] = Field(None, ge=0, le=23)
    visit_dayofweek: Optional[int] = Field(None, ge=0, le=6, description="0=Пн, 6=Вс")
    visit_month: Optional[int] = Field(None, ge=1, le=12)
    visit_number: Optional[int] = Field(None, ge=1)
    screen_w: Optional[int] = Field(None, ge=0, le=10000)
    screen_h: Optional[int] = Field(None, ge=0, le=10000)
    client_id: Optional[str] = Field(None, description="ID клиента для подтягивания истории")


class PredictResponse(BaseModel):
    probability: float = Field(..., description="вероятность целевого действия [0..1]")
    label: int = Field(..., description="бинарный прогноз 0/1")
    latency_ms: float


class BatchRequest(BaseModel):
    visits: List[VisitInput]
    threshold: float = Field(0.5, ge=0, le=1)


class BatchResponse(BaseModel):
    n: int
    predictions: List[PredictResponse]
    total_latency_ms: float


@app.get("/health")
def health():
    art = ARTIFACT.get("data")
    if art is None:
        raise HTTPException(503, "model not loaded")
    return {
        "status": "ok",
        "model_loaded_in_s": ARTIFACT.get("loaded_in_s"),
        "roc_auc_val": art.get("roc_auc_val"),
        "roc_auc_test": art.get("roc_auc_test"),
        "n_features": len(art.get("feature_cols", [])),
        "n_clients_in_history": len(art.get("client_history_map", {})),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(visit: VisitInput, threshold: float = 0.5):
    art = ARTIFACT.get("data")
    if art is None:
        raise HTTPException(503, "model not loaded")
    payload = visit.model_dump(exclude_none=True)
    res = predict_one(payload, art, threshold=threshold)
    return PredictResponse(**res)


@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch_endpoint(req: BatchRequest):
    art = ARTIFACT.get("data")
    if art is None:
        raise HTTPException(503, "model not loaded")
    if not req.visits:
        raise HTTPException(400, "empty visits list")

    payloads = [v.model_dump(exclude_none=True) for v in req.visits]
    probas, labels, dt_ms = predict_batch(payloads, art, threshold=req.threshold)

    per_item = dt_ms / max(len(payloads), 1)
    items = [
        PredictResponse(probability=float(p), label=int(l), latency_ms=round(per_item, 3))
        for p, l in zip(probas, labels)
    ]
    return BatchResponse(n=len(items), predictions=items, total_latency_ms=round(dt_ms, 3))


# точка входа: запуск uvicorn с обработкой занятых портов
if __name__ == "__main__":
    import argparse
    import socket
    import sys
    import uvicorn

    parser = argparse.ArgumentParser(description="REST API для предсказания целевого действия")
    parser.add_argument("--host", default=os.environ.get("APP_HOST", "127.0.0.1"),
                        help="интерфейс для прослушивания (env APP_HOST)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("APP_PORT", "8000")),
                        help="порт (env APP_PORT)")
    parser.add_argument("--public", action="store_true",
                        help="слушать на 0.0.0.0")
    args = parser.parse_args()

    host = "0.0.0.0" if args.public else args.host

    # проверка доступности порта до запуска uvicorn
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, args.port))
    except OSError as e:
        print(f"Не удалось забиндить {host}:{args.port} ({e}).")
        print("Подсказка: укажите другой порт через --port 8765 или используйте --public.")
        sys.exit(2)
    finally:
        probe.close()

    print(f"Listening on http://{host}:{args.port}  (Swagger UI: /docs)", flush=True)

    uvicorn.run("app:app", host=host, port=args.port, reload=False)
