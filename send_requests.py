import requests
import random

URL = "http://localhost:8000/predict"

random.seed(42)

for i in range(1, 401):  # 400 запросов
    payload = {
        "user_id": i,  # даст и A, и B
        "Pclass": random.choice([1, 2, 3]),
        "Sex": random.choice(["male", "female"]),
        "Age": random.randint(1, 80),
        "Fare": round(random.uniform(5, 150), 2),
        # ground truth для метрик (пусть будет условный, но стабильный)
        "Survived": 1 if (i % 5 == 0) else 0,
    }
    r = requests.post(URL, json=payload, timeout=10)
    if r.status_code != 200:
        print("ERR", i, r.status_code, r.text)

print("done")
