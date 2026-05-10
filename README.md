# СберАвтоподписка — анализ сайта и предсказание целевого действия

Учебный проект (МИФИ, магистратура). По данным визита на сайте sberauto.com строится бинарный классификатор: совершит ли пользователь целевое действие (оставить заявку, заказать звонок, открыть диалог и т.д.).

Метрика: ROC-AUC. Ориентир задания: ≥ 0.65.

## Результаты

| Модель | ROC-AUC val | ROC-AUC test |
|---|---|---|
| Dummy (most_frequent) | 0.5000 | 0.5000 |
| LogisticRegression + OHE | 0.6436 | 0.6374 |
| **LightGBM** (выигравший конфиг `strong_reg`) | **0.6799** | **0.6554** |
| Ориентир задания | — | 0.6500 |

Запас над ориентиром на тесте: +0.0054.

## Структура проекта

```
PROJECT_PRACTICE_DEV/
├── DATA/                          # исходные данные (не коммитятся)
│   ├── ga_sessions.pkl/csv        # сессии GA
│   ├── ga_hits.pkl/csv            # хиты GA
│   ├── Описание данных.pdf
│   └── Принципы вариации для выполнения проекта. ROC-AUC.pdf
├── Задача/
│   ├── бриф Учебная задача «Анализ сайта».docx
│   └── Задача.txt
├── notebook.ipynb                 # EDA, ML, интерпретация, выводы
├── predict.py                     # .py-файл с моделью: CLI / JSON / CSV-batch
├── app.py                         # FastAPI: GET /health, POST /predict, POST /predict_batch
├── examples/
│   ├── visit_example.json         # одиночный визит для --json
│   ├── visits_example.json        # массив визитов
│   └── visits_example.csv         # CSV для --input
├── scripts/
│   ├── smoke_predict.py           # самопроверка predict.py
│   └── smoke_api.py               # самопроверка app.py через сетевой запуск
├── model.pkl                      # обученный артефакт
├── img_*.png                      # графики из ноутбука
├── requirements.txt
└── README.md
```

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Требования: Python 3.10+, ~16 ГБ RAM для переобучения (загрузка `ga_hits.pkl`), ~2 ГБ RAM для инференса.

## Воспроизведение результата

Файлы `DATA/ga_sessions.pkl` и `DATA/ga_hits.pkl` кладутся в папку `DATA/`. Запуск ноутбука:

```bash
python -m nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb --ExecutePreprocessor.timeout=3600
```

После выполнения создаются `model.pkl` и `img_*.png`. Время полного прогона: 7–11 минут.

## Использование модели

### CLI: одиночный визит

```bash
python predict.py --utm_medium cpc --device_category mobile --geo_city Moscow --visit_hour 14
```

С историей клиента:

```bash
python predict.py --utm_medium cpc --geo_city Moscow --visit_hour 14 --client_id 1000775308.1627486259
```

JSON-вывод:

```bash
python predict.py --utm_medium cpc --geo_city Moscow --visit_hour 14 --format json
```

Все параметры:

```bash
python predict.py --help
```

### JSON-вход

```bash
python predict.py --json examples/visit_example.json --format json
python predict.py --json examples/visits_example.json --format json
```

JSON через stdin:

```powershell
Get-Content examples/visit_example.json | python predict.py --json - --format json
```

```bash
cat examples/visit_example.json | python predict.py --json - --format json
```

### CSV-batch

```bash
python predict.py --input examples/visits_example.csv --output visits_pred.csv
```

На выходе — исходный CSV + колонки `probability` и `label`.

### Самопроверка

```bash
python scripts/smoke_predict.py
```

Запускает `predict.py` во всех режимах (CLI, JSON, JSON через stdin, CSV-batch) и проверяет вывод.

### REST API (FastAPI)

```bash
python app.py                  # 127.0.0.1:8000
python app.py --port 8765
python app.py --public         # 0.0.0.0
```

Swagger UI: http://127.0.0.1:8000/docs.

#### `GET /health`

```json
{
  "status": "ok",
  "model_loaded_in_s": 6.43,
  "roc_auc_val": 0.6799,
  "roc_auc_test": 0.6554,
  "n_features": 34,
  "n_clients_in_history": 1391719
}
```

#### `POST /predict`

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"utm_medium":"cpc","device_category":"mobile","geo_city":"Moscow","visit_hour":14}'
```

Ответ:

```json
{"probability": 0.4030, "label": 0, "latency_ms": 3.816}
```

#### `POST /predict_batch`

```json
{
  "visits": [
    {"utm_medium": "cpc", "geo_city": "Moscow", "visit_hour": 14},
    {"utm_medium": "organic", "geo_city": "Saint Petersburg", "visit_hour": 3}
  ],
  "threshold": 0.5
}
```

#### Самопроверка API

```bash
python scripts/smoke_api.py
```

Запускает `app.py` в отдельном процессе на свободном порту, ждёт готовности через `/health` и делает реальные HTTP-вызовы к `/health`, `/predict`, `/predict_batch`.

## Описание входных полей

| Поле | Тип | Диапазон | По умолчанию |
|---|---|---|---|
| `utm_source` | str | — | `unknown` |
| `utm_medium` | str | — | `unknown` |
| `utm_campaign` | str | — | `unknown` |
| `utm_adcontent` | str | — | `unknown` |
| `device_category` | str | — | `unknown` |
| `device_os` | str | — | `unknown` |
| `device_brand` | str | — | `unknown` |
| `device_browser` | str | — | `unknown` |
| `geo_country` | str | — | `Russia` |
| `geo_city` | str | — | `unknown` |
| `visit_hour` | int | 0–23 | 14 |
| `visit_dayofweek` | int | 0–6 | 2 |
| `visit_month` | int | 1–12 | 9 |
| `visit_number` | int | ≥ 1 | 1 |
| `screen_w` | int | 0–10000 | 1280 |
| `screen_h` | int | 0–10000 | 720 |
| `client_id` | str | — | `null` |

При наличии `client_id` в `client_history_map` подтягиваются накопленные счётчики (`client_prev_sessions`, `client_prev_conv`, `client_prev_cr`).

## Что внутри модели

### Целевая переменная

Бинарный таргет: была ли в сессии хотя бы одна целевая активность. Целевые `event_action` отбираются по регулярному выражению:

```
claim | callback | call_number | open_dialog | submit_success | start_chat
```

Найдено 14 целевых значений `event_action`. CR по датасету: 4.19 %, дисбаланс 1:23.

### Признаки (34 шт.)

Сырые категории (заполнение пропусков `unknown`, схлопывание редких в `other`):

- UTM: `utm_source`, `utm_medium`, `utm_campaign`, `utm_adcontent`
- Device: `device_category`, `device_os`, `device_brand`, `device_browser`
- Geo: `geo_country`, `geo_city`

Доменные флаги канала:

- `is_organic` = `utm_medium ∈ {'organic', 'referral', '(none)'}`
- `is_paid` = `1 - is_organic`
- `is_social` = в `utm_medium` есть `'social'` или `'smm'`

Циклические признаки времени (sin/cos на 24 ч и 7 дней):

- `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_weekend`, `visit_month`

Взаимодействия категорий:

- `utm_src_x_device` = `utm_source × device_category`
- `utm_camp_x_city` = `utm_campaign × geo_city`
- `utm_src_x_medium` = `utm_source × utm_medium`

Параметры экрана (с обрезкой выбросов до 4000 px):

- `screen_w`, `screen_h`, `screen_area`

Глубина визита:

- `visit_number_capped` (обрезан сверху до 50)

Out-of-fold target encoding (5-фолдов, smooth=100):

- `te_utm_campaign`, `te_utm_source`, `te_utm_adcontent`, `te_geo_city`, `te_utm_medium`

История клиента по `client_id`:

- `client_prev_sessions`, `client_prev_conv`, `client_prev_cr` (smooth=10)

### Модели

| Модель | Назначение | Кодирование категорий |
|---|---|---|
| `DummyClassifier(strategy='most_frequent')` | Sanity-check | — |
| `LogisticRegression(class_weight='balanced')` | Линейный бейзлайн | OneHotEncoder с `min_frequency=0.001` |
| `LightGBMClassifier` | Основная модель | OrdinalEncoder + `categorical_feature` (native) |

### Подбор гиперпараметров LightGBM

Мини-сетка из 4 конфигов с разной регуляризацией. Лучший выбирается по val-AUC:

```python
candidates = [
    dict(learning_rate=0.03, num_leaves=63, min_child_samples=100, reg_alpha=0.05, reg_lambda=0.5),
    dict(learning_rate=0.03, num_leaves=31, min_child_samples=200, reg_alpha=0.1,  reg_lambda=2.0),
    dict(learning_rate=0.02, num_leaves=31, min_child_samples=300, reg_alpha=0.1,  reg_lambda=2.0),
    dict(learning_rate=0.05, num_leaves=31, min_child_samples=500, reg_alpha=0.5,  reg_lambda=5.0),
]
# общие: n_estimators=3000, colsample_bytree=0.8, subsample=0.8,
#        scale_pos_weight ≈ 20.9, early_stopping(300) по val-AUC
```

Результаты последнего прогона:

| Конфиг | val_auc | best_iter |
|---|---|---|
| `base` | 0.6770 | 30 |
| `shallow` | 0.6773 | 14 |
| `slow_lr` | 0.6785 | 14 |
| **`strong_reg`** | **0.6799** | 24 |

Финальная модель — `strong_reg`.

### Методология

- Временной сплит 70 / 10 / 20 train / val / test (по `visit_dt`).
- Early stopping LightGBM по validation. Тест замеряется один раз.
- OOF target encoding (KFold, `random_state=42`).
- `gm_train` считается только по train.
- Кумулятивные счётчики клиента смотрят только в прошлое (`cumcount`, `cumsum - self`).

## Интерпретация

- Feature Importance (gain) — `img_feature_importance.png`. Топ-10: `te_utm_campaign`, `utm_campaign`, `client_prev_cr`, `geo_city`, `utm_src_x_medium`, `utm_adcontent`, `utm_src_x_device`, `utm_camp_x_city`, `client_prev_conv`, `visit_month`.
- SHAP summary plot — `img_shap_summary.png`.
- SHAP violin plot — `img_shap_violin.png`.

## Производительность

### Обучение

- Полный прогон ноутбука: 7–11 минут на CPU.
- LightGBM (4 конфига): ~107 с.
- LogReg: ~112 с.
- SHAP на подвыборке 5000: ~1–2 минуты.

### Инференс

| Сценарий | Время |
|---|---|
| Загрузка `model.pkl` | ~6.5 с |
| `/predict` через сеть | ~3 мс |
| `/predict` внутри Python | ~0.9 мс |
| `/predict_batch` 10 000 визитов | ~5.8 мс (≈0.0006 мс/визит) |
| CLI `--input` 4 строки | ~2 мс |

Время ответа на визит — миллисекунды, при пороге задания 3 секунды.

### Память

- Старт Python + импорты: ~150 МБ.
- После `joblib.load('model.pkl')`: ~620 МБ (основное — `client_history_map` на 1.39 М клиентов).
