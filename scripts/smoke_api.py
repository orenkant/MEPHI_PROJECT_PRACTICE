# самопроверка app.py через сетевой запуск uvicorn
import os
import socket
import subprocess
import sys
import time

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app.py")


# свободный порт на localhost
def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ожидание готовности /health
def wait_ready(base_url, timeout_s=30):
    t0 = time.time()
    last_err = None
    while time.time() - t0 < timeout_s:
        try:
            r = httpx.get(f"{base_url}/health", timeout=2.0, trust_env=False)
            if r.status_code == 200:
                return time.time() - t0
        except Exception as e:
            last_err = e
        time.sleep(0.3)
    raise RuntimeError(f"сервер не поднялся за {timeout_s} с: {last_err}")


def main():
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    print(f"Smoke-test API через сетевой запуск: {base_url}")

    env = os.environ.copy()
    env["MODEL_PATH"] = os.path.join(ROOT, "model.pkl")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    proc = subprocess.Popen(
        [sys.executable, APP, "--port", str(port)],
        cwd=ROOT,
        env=env,
    )

    try:
        ready_in = wait_ready(base_url)
        print(f"  сервер готов за {ready_in:.1f} с")

        with httpx.Client(base_url=base_url, timeout=10.0, trust_env=False) as client:
            t0 = time.time()
            r = client.get("/health")
            dt = round((time.time() - t0) * 1000, 1)
            print(f"  GET /health             -> {r.status_code}  {dt} мс")
            assert r.status_code == 200, r.text
            body = r.json()
            print(f"    roc_auc_test = {body.get('roc_auc_test')}, n_features = {body.get('n_features')}")

            payload = {"utm_medium": "cpc", "geo_city": "Moscow", "visit_hour": 14}
            t0 = time.time()
            r = client.post("/predict", json=payload)
            dt = round((time.time() - t0) * 1000, 1)
            print(f"  POST /predict           -> {r.status_code}  {dt} мс  {r.json()}")
            assert r.status_code == 200, r.text

            batch = {"visits": [payload, {"utm_medium": "organic", "geo_city": "Saint Petersburg"}], "threshold": 0.5}
            t0 = time.time()
            r = client.post("/predict_batch", json=batch)
            dt = round((time.time() - t0) * 1000, 1)
            print(f"  POST /predict_batch n=2 -> {r.status_code}  {dt} мс")
            assert r.status_code == 200, r.text

        print("OK: API исправен.")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    main()
