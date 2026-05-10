# самопроверка predict.py во всех режимах ввода
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICT = os.path.join(ROOT, "predict.py")
EXAMPLES = os.path.join(ROOT, "examples")


def run(args, stdin_data=None):
    # принудительно UTF-8 для stdout дочернего процесса (на Windows локаль может быть CP1251)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    cmd = [sys.executable, PREDICT] + args
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        input=stdin_data,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
        env=env,
    )
    dt = round((time.time() - t0) * 1000, 1)
    return proc.returncode, proc.stdout, proc.stderr, dt


def main():
    print("Smoke-test predict.py: проверка скрипта без сети")
    cold_times = []
    inference_latencies = []

    # одиночный визит через CLI-аргументы, текстовый вывод
    print()
    print("[1] CLI-аргументы, текстовый вывод")
    rc, out, err, dt = run([
        "--utm_medium", "cpc",
        "--device_category", "mobile",
        "--geo_city", "Moscow",
        "--visit_hour", "14",
    ])
    print(f"    rc={rc}, time={dt} мс")
    assert rc == 0, f"ненулевой код возврата:\nSTDOUT:\n{out}\nSTDERR:\n{err}"
    assert "Вероятность конверсии" in out, f"нет ожидаемого текста:\n{out}"
    assert "Бинарный прогноз" in out, f"нет бинарного прогноза:\n{out}"
    cold_times.append(dt)
    # парсинг latency_ms из строки "Время инференса: X мс"
    for line in out.splitlines():
        if "Время инференса" in line:
            try:
                inference_latencies.append(float(line.split(":")[1].strip().split()[0]))
            except (ValueError, IndexError):
                pass
    print("    OK: текстовый вывод содержит вероятность и бинарный прогноз")

    # одиночный визит через CLI-аргументы, JSON-вывод
    print()
    print("[2] CLI-аргументы, JSON-вывод")
    rc, out, err, dt = run([
        "--utm_medium", "cpc",
        "--device_category", "mobile",
        "--geo_city", "Moscow",
        "--visit_hour", "14",
        "--format", "json",
    ])
    print(f"    rc={rc}, time={dt} мс")
    assert rc == 0, f"ненулевой код возврата:\n{err}"
    parsed = json.loads(out.strip())
    assert "probability" in parsed and "label" in parsed, f"неверный JSON: {parsed}"
    assert parsed["label"] in (0, 1), f"label не 0/1: {parsed}"
    assert 0.0 <= parsed["probability"] <= 1.0, f"вероятность вне [0,1]: {parsed}"
    cold_times.append(dt)
    inference_latencies.append(float(parsed["latency_ms"]))
    print(f"    OK: probability={parsed['probability']:.4f}, label={parsed['label']}, "
          f"latency_ms={parsed['latency_ms']}")

    # JSON-файл с одиночным визитом
    print()
    print("[3] JSON-файл, одиночный визит")
    rc, out, err, dt = run([
        "--json", os.path.join(EXAMPLES, "visit_example.json"),
        "--format", "json",
    ])
    print(f"    rc={rc}, time={dt} мс")
    assert rc == 0, f"ненулевой код возврата:\n{err}"
    parsed = json.loads(out.strip())
    assert isinstance(parsed, dict), f"для одиночного визита ждём объект, получили {type(parsed)}"
    assert parsed["label"] in (0, 1)
    cold_times.append(dt)
    inference_latencies.append(float(parsed["latency_ms"]))
    print(f"    OK: probability={parsed['probability']:.4f}, label={parsed['label']}, "
          f"latency_ms={parsed['latency_ms']}")

    # JSON-файл с массивом визитов
    print()
    print("[4] JSON-файл, массив визитов")
    rc, out, err, dt = run([
        "--json", os.path.join(EXAMPLES, "visits_example.json"),
        "--format", "json",
    ])
    print(f"    rc={rc}, time={dt} мс")
    assert rc == 0, f"ненулевой код возврата:\n{err}"
    parsed = json.loads(out.strip())
    assert isinstance(parsed, list), f"для массива визитов ждём массив, получили {type(parsed)}"
    assert len(parsed) >= 2
    for r in parsed:
        assert r["label"] in (0, 1)
        assert 0.0 <= r["probability"] <= 1.0
    cold_times.append(dt)
    inference_latencies.extend(float(r["latency_ms"]) for r in parsed)
    print(f"    OK: получено {len(parsed)} прогнозов, latency_ms на визит ~ {parsed[0]['latency_ms']}")

    # JSON через stdin
    print()
    print("[5] JSON через stdin")
    visit = {"utm_medium": "organic", "geo_city": "Saint Petersburg", "visit_hour": 3}
    rc, out, err, dt = run(["--json", "-", "--format", "json"], stdin_data=json.dumps(visit))
    print(f"    rc={rc}, time={dt} мс")
    assert rc == 0, f"ненулевой код возврата:\n{err}"
    parsed = json.loads(out.strip())
    assert parsed["label"] in (0, 1)
    cold_times.append(dt)
    inference_latencies.append(float(parsed["latency_ms"]))
    print(f"    OK: probability={parsed['probability']:.4f}, label={parsed['label']}, "
          f"latency_ms={parsed['latency_ms']}")

    # CSV-batch
    print()
    print("[6] CSV-batch")
    with tempfile.TemporaryDirectory() as td:
        out_csv = os.path.join(td, "out.csv")
        rc, out, err, dt = run([
            "--input", os.path.join(EXAMPLES, "visits_example.csv"),
            "--output", out_csv,
        ])
        print(f"    rc={rc}, time={dt} мс")
        assert rc == 0, f"ненулевой код возврата:\n{err}"
        assert os.path.isfile(out_csv), "выходной CSV не создан"
        with open(out_csv, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) >= 2
        assert "probability" in lines[0] and "label" in lines[0]
        cold_times.append(dt)
        # парсинг latency на строку из "Время инференса: X мс на N строк (Y мс/строка)"
        for line in out.splitlines():
            if "мс/строка" in line:
                try:
                    per = float(line.split("(")[1].split(" мс")[0])
                    inference_latencies.append(per)
                except (ValueError, IndexError):
                    pass
        print(f"    OK: записано строк (с шапкой): {len(lines)}")

    # сводка по времени
    print()
    print("Сводка по времени:")
    print(f"  Cold start процесса (запуск Python + загрузка model.pkl):")
    print(f"     min={min(cold_times):.0f} мс, max={max(cold_times):.0f} мс, "
          f"avg={sum(cold_times)/len(cold_times):.0f} мс")
    print(f"  Время ответа модели (latency_ms):")
    print(f"     min={min(inference_latencies):.3f} мс, max={max(inference_latencies):.3f} мс, "
          f"avg={sum(inference_latencies)/len(inference_latencies):.3f} мс")
    print(f"  Порог брифа: 3000 мс на визит. Запас: "
          f"x{3000 / max(inference_latencies):.0f} от худшего инференса.")
    print()
    print("OK: все проверки пройдены.")


if __name__ == "__main__":
    main()
