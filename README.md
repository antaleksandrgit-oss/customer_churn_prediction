# Customer Churn Prediction

[![API tests](https://github.com/antaleksandrgit-oss/customer_churn_prediction/actions/workflows/tests.yml/badge.svg)](https://github.com/antaleksandrgit-oss/customer_churn_prediction/actions/workflows/tests.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antaleksandrgit-oss/customer_churn_prediction/blob/main/customer_churn_prediction.ipynb)
[![FastAPI](https://img.shields.io/badge/FastAPI-Live-009688?logo=fastapi&logoColor=white)](https://customer-churn-api-7gq1.onrender.com/docs)

Проект бинарной классификации для прогнозирования ухода банковских клиентов.

Модель оценивает вероятность ухода и помогает сформировать список клиентов для программы удержания. Бизнес-ограничение проекта - возможность связаться примерно с 25% клиентской базы.

## Данные

Использован датасет [Bank Customer Churn](https://www.openml.org/search?id=46911&type=data) из OpenML:

- 10 000 клиентов
- 10 признаков
- положительный класс - уход клиента
- доля ушедших клиентов - 20.4%
- числовые и категориальные признаки

Данные разделены на train и test со стратификацией. Тестовая выборка использовалась один раз для финальной оценки.

## Модели

Были сравнены:

- DummyClassifier
- Logistic Regression
- Logistic Regression с class weights
- Random Forest
- Histogram Gradient Boosting

Модели оценивались с помощью стратифицированной 5-кратной кросс-валидации. Основная метрика выбора - PR-AUC.

Для итоговой модели выбран Gradient Boosting. Порог классификации подобран по out-of-fold прогнозам с учётом лимита на контакт примерно с 25% клиентов.

## Результаты

Результаты на отложенной тестовой выборке:

| Метрика | Dummy | Gradient Boosting |
|---|---:|---:|
| Accuracy | 0.796 | 0.828 |
| Precision | 0.000 | 0.562 |
| Recall | 0.000 | 0.710 |
| F1 | 0.000 | 0.628 |
| ROC-AUC | 0.500 | 0.877 |
| PR-AUC | 0.204 | 0.730 |
| Brier score | 0.162 | 0.098 |
| Contact rate | 0.0% | 25.7% |

Матрица ошибок итоговой модели:

```text
TN = 1368    FP = 225
FN = 118     TP = 289
```

При том же лимите контактов Gradient Boosting обнаружил на 230 уходящих клиентов больше, чем Logistic Regression.

## Интерпретация

Наиболее важные признаки по permutation importance:

- возраст
- количество продуктов
- активность клиента
- баланс
- страна

Самая высокая наблюдаемая доля ухода была у клиентов 50-59 лет, клиентов с 3-4 продуктами и неактивных клиентов.

Эти результаты показывают ассоциации, но не доказывают причинное влияние признаков на уход.

## API

Интерактивная документация:

https://customer-churn-api-7gq1.onrender.com/docs

Проверка состояния:

https://customer-churn-api-7gq1.onrender.com/health

Пример запроса:

```bash
curl -X POST \
  "https://customer-churn-api-7gq1.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "credit_score": 620,
    "country": "Germany",
    "gender": "Female",
    "age": 50,
    "tenure": 4,
    "balance": 120000,
    "products_number": 3,
    "credit_card": 1,
    "active_member": 0,
    "estimated_salary": 90000
  }'
```

Пример ответа:

```json
{
  "churn_probability": 0.965,
  "threshold": 0.2523,
  "predicted_churn": true,
  "recommended_action": "contact"
}
```

## Локальный запуск

Установить зависимости:

```bash
pip install -r requirements.txt
```

Запустить API:

```bash
uvicorn app:app --reload
```

Открыть документацию:

```text
http://127.0.0.1:8000/docs
```

## Тесты

Запуск тестов:

```bash
python -m pytest -v
```

GitHub Actions автоматически запускает тесты после каждого push и pull request.

Тесты проверяют:

- загрузку модели
- health endpoint
- корректный прогноз
- структуру ответа
- валидацию страны
- запрет отрицательного баланса
- обязательные поля

## Структура проекта

```text
.
├── .github/workflows/tests.yml
├── .python-version
├── app.py
├── customer_churn_model.joblib
├── customer_churn_prediction.ipynb
├── requirements.txt
├── test_app.py
└── README.md
```

## Ограничения

- Проект использует учебный датасет и не предназначен для принятия реальных банковских решений.
- Порог зависит от принятого ограничения на размер программы удержания.
- В продакшене необходимо отслеживать изменение распределения данных и качества модели.
- Выявленные зависимости не следует интерпретировать как причинно-следственные.

## Ссылки
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DuvHpYcBfeQQhwRpdqdGXSMaTAIs5n6G?usp=sharing)
[![Kaggle Profile](https://img.shields.io/badge/Kaggle-Profile-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/potsml)
[![FastAPI](https://img.shields.io/badge/FastAPI-Live-009688?logo=fastapi&logoColor=white)](https://customer-churn-api-7gq1.onrender.com/docs)
[![API tests](https://github.com/antaleksandrgit-oss/customer_churn_prediction/actions/workflows/tests.yml/badge.svg)](https://github.com/antaleksandrgit-oss/customer_churn_prediction/actions/workflows/tests.yml)

## Контакты
- Telegram: @antaleksandr
- Email: AntAleksandrGit@gmail.com
