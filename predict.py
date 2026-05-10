import argparse
import json
import math
import sys
import time
import numpy as np

try:
    import joblib
    import pandas as pd
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Установите зависимости: pip install -r requirements.txt")
    sys.exit(1)


# проверка на отсутствующее значение
def _is_missing(val):
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    return False


# дефолты для полей визита
DEFAULTS = {
    "utm_source": "unknown",
    "utm_medium": "unknown",
    "utm_campaign": "unknown",
    "utm_adcontent": "unknown",
    "device_category": "unknown",
    "device_os": "unknown",
    "device_brand": "unknown",
    "device_browser": "unknown",
    "geo_country": "Russia",
    "geo_city": "unknown",
    "visit_hour": 14,
    "visit_dayofweek": 2,
    "visit_month": 9,
    "visit_number": 1,
    "screen_w": 1280,
    "screen_h": 720,
    "client_id": None,
}


def load_artifact(path="model.pkl"):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        print(f"Файл {path} не найден. Сначала обучите модель: запустите notebook.ipynb.")
        sys.exit(1)


# feature engineering для одного визита
def build_feature_row(payload, art):
    rare_cities = set(art.get("rare_cities", []))
    rare_maps = art.get("rare_maps", {})
    cross_rare_maps = art.get("cross_rare_maps", {})
    client_history_map = art.get("client_history_map", {})
    gm_train = float(art.get("gm_train", 0.04))
    smooth_client = int(art.get("smooth_client", 10))

    row = dict(DEFAULTS)
    for key, val in payload.items():
        if not _is_missing(val) and key in row:
            row[key] = val

    # схлопывание редких городов
    if row["geo_city"] in rare_cities:
        row["geo_city"] = "other"

    # схлопывание редких UTM-значений
    for col, rare_list in rare_maps.items():
        if col in row and row[col] in set(rare_list):
            row[col] = "other"

    # обрезка visit_number
    row["visit_number_capped"] = min(int(row["visit_number"]), 50)

    # обрезка размеров экрана
    row["screen_w"] = max(0, min(int(row["screen_w"]), 4000))
    row["screen_h"] = max(0, min(int(row["screen_h"]), 4000))
    row["screen_area"] = row["screen_w"] * row["screen_h"]

    # циклические признаки часа и дня недели
    h = int(row["visit_hour"])
    dow = int(row["visit_dayofweek"])
    row["hour_sin"] = np.sin(2 * np.pi * h / 24)
    row["hour_cos"] = np.cos(2 * np.pi * h / 24)
    row["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    row["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    row["is_weekend"] = int(dow >= 5)

    # доменные флаги канала
    um = str(row["utm_medium"]).lower()
    row["is_organic"] = int(row["utm_medium"] in ("organic", "referral", "(none)"))
    row["is_paid"] = 1 - row["is_organic"]
    row["is_social"] = int("social" in um or "smm" in um)

    # склейки категорий
    row["utm_src_x_device"] = f'{row["utm_source"]}|{row["device_category"]}'
    row["utm_camp_x_city"] = f'{row["utm_campaign"]}|{row["geo_city"]}'
    row["utm_src_x_medium"] = f'{row["utm_source"]}|{row["utm_medium"]}'
    for col, rare_list in cross_rare_maps.items():
        if row.get(col) in set(rare_list):
            row[col] = "other"

    # история клиента из словаря по client_id
    cid = row.get("client_id")
    if cid is not None and cid in client_history_map:
        hist = client_history_map[cid]
        prev_sessions = int(hist["sessions"])
        prev_conv = int(hist["conversions"])
    else:
        prev_sessions = 0
        prev_conv = 0

    row["client_prev_sessions"] = prev_sessions
    row["client_prev_conv"] = prev_conv
    row["client_prev_cr"] = (prev_conv + smooth_client * gm_train) / (prev_sessions + smooth_client)

    return row


def prepare_X(payloads, art):
    base_feature_cols = art.get("base_feature_cols", art["feature_cols"])
    cat_features = art["cat_features"]
    te_cols = art.get("te_cols", [])
    te_maps = art.get("te_maps", {})
    feature_cols = art["feature_cols"]
    enc = art["encoder"]

    rows = [build_feature_row(p, art) for p in payloads]
    X = pd.DataFrame(rows)[base_feature_cols].copy()

    # сырые категории сохраняем до кодирования
    raw_cat = {c: X[c].astype(str).values for c in te_cols}

    # ordinal-кодирование категорий
    X[cat_features] = enc.transform(X[cat_features].astype(str))

    # подстановка target-encoded значений из словарей
    for col in te_cols:
        te_map, gm = te_maps[col]
        X[f"te_{col}"] = [float(te_map.get(v, gm)) for v in raw_cat[col]]

    return X[feature_cols]


def predict_one(payload, art, threshold=0.5):
    model = art["model"]
    X = prepare_X([payload], art)

    t0 = time.time()
    proba = float(model.predict_proba(X)[0, 1])
    dt_ms = (time.time() - t0) * 1000

    return {
        "probability": proba,
        "label": int(proba >= threshold),
        "latency_ms": round(dt_ms, 3),
    }


def predict_batch(payloads, art, threshold=0.5):
    model = art["model"]
    X = prepare_X(payloads, art)

    t0 = time.time()
    probas = model.predict_proba(X)[:, 1]
    dt_ms = (time.time() - t0) * 1000

    labels = (probas >= threshold).astype(int)
    return probas, labels, dt_ms


def cli_single(args, art):
    payload = {k: v for k, v in vars(args).items() if v is not None and k in DEFAULTS}
    res = predict_one(payload, art, threshold=args.threshold)

    if args.format == "json":
        print(json.dumps(res, ensure_ascii=False))
        return

    print()
    print(f"  Вероятность конверсии : {res['probability']:.4f} ({res['probability']*100:.2f}%)")
    print(f"  Бинарный прогноз ({args.threshold}): {res['label']}")
    print(f"  {'Целевое действие ожидается' if res['label'] else 'Целевое действие не ожидается'}")
    print(f"  Время инференса       : {res['latency_ms']:.2f} мс")
    print()


# чтение входа из JSON-файла или stdin
def _load_json_input(path):
    if path == "-":
        raw = sys.stdin.read()
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    data = json.loads(raw)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("JSON должен быть объектом визита или массивом таких объектов")


def cli_json(args, art):
    payloads = _load_json_input(args.json)

    if not payloads:
        print("Пустой список визитов в JSON", file=sys.stderr)
        sys.exit(2)

    probas, labels, dt_ms = predict_batch(payloads, art, threshold=args.threshold)
    per_item = dt_ms / len(payloads)

    results = [
        {
            "probability": round(float(p), 6),
            "label": int(l),
            "latency_ms": round(per_item, 3),
        }
        for p, l in zip(probas, labels)
    ]

    if args.format == "json":
        if len(results) == 1:
            print(json.dumps(results[0], ensure_ascii=False))
        else:
            print(json.dumps(results, ensure_ascii=False))
        return

    print(f"Обработано визитов: {len(results)}")
    print(f"Время: {dt_ms:.2f} мс ({per_item:.3f} мс/визит)")
    print()
    for i, r in enumerate(results, 1):
        verdict = "ожидается" if r["label"] else "не ожидается"
        print(f"  [{i}] p={r['probability']:.4f}  label={r['label']}  целевое действие {verdict}")


def cli_batch(args, art):
    df_in = pd.read_csv(args.input)
    print(f"Загружено строк из {args.input}: {len(df_in):,}")

    payloads = df_in.to_dict("records")

    probas, labels, dt_ms = predict_batch(payloads, art, threshold=args.threshold)
    print(f"Время инференса: {dt_ms:.1f} мс на {len(payloads)} строк ({dt_ms/len(payloads):.3f} мс/строка)")

    out = df_in.copy()
    out["probability"] = probas
    out["label"] = labels

    out_path = args.output or args.input.rsplit(".", 1)[0] + "_pred.csv"
    out.to_csv(out_path, index=False)
    print(f"Результат сохранён: {out_path}")
    print(f"Доля прогнозов 1: {labels.mean():.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Предсказание целевого действия пользователя на sberauto.com",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model", default="model.pkl", help="путь к файлу модели")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="порог отсечения для бинарного прогноза")
    parser.add_argument("--format", default="text", choices=["text", "json"],
                        help="формат вывода: text или json")

    # batch-режим CSV
    parser.add_argument("--input", default=None, help="CSV-файл с входными данными для batch-режима")
    parser.add_argument("--output", default=None,
                        help="CSV-файл для записи результата (по умолчанию <input>_pred.csv)")

    # JSON-вход: путь к файлу или '-' для stdin
    parser.add_argument("--json", default=None,
                        help="JSON-файл с визитом(ами); '-' для чтения из stdin")

    # utm
    parser.add_argument("--utm_source", default=None)
    parser.add_argument("--utm_medium", default=None)
    parser.add_argument("--utm_campaign", default=None)
    parser.add_argument("--utm_adcontent", default=None)

    # устройство
    parser.add_argument("--device_category", default=None,
                        help="mobile/desktop/tablet или иное")
    parser.add_argument("--device_os", default=None)
    parser.add_argument("--device_brand", default=None)
    parser.add_argument("--device_browser", default=None)

    # гео
    parser.add_argument("--geo_country", default=None)
    parser.add_argument("--geo_city", default=None)

    # время
    parser.add_argument("--visit_hour", type=int, default=None)
    parser.add_argument("--visit_dayofweek", type=int, default=None, help="0=Пн, 6=Вс")
    parser.add_argument("--visit_month", type=int, default=None)
    parser.add_argument("--visit_number", type=int, default=None,
                        help="порядковый номер визита клиента")

    # экран
    parser.add_argument("--screen_w", type=int, default=None, help="ширина экрана, px")
    parser.add_argument("--screen_h", type=int, default=None, help="высота экрана, px")

    # клиент
    parser.add_argument("--client_id", default=None,
                        help="client_id для подтягивания истории")

    args = parser.parse_args()
    art = load_artifact(args.model)

    # выбор режима: CSV-batch > JSON > одиночный CLI
    if args.input:
        cli_batch(args, art)
    elif args.json:
        cli_json(args, art)
    else:
        cli_single(args, art)


if __name__ == "__main__":
    main()
