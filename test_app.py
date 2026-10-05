from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


VALID_CUSTOMER = {
    "credit_score": 620,
    "country": "Germany",
    "gender": "Female",
    "age": 50,
    "tenure": 4,
    "balance": 120000.0,
    "products_number": 3,
    "credit_card": 1,
    "active_member": 0,
    "estimated_salary": 90000.0,
}


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["feature_count"] == 10


def test_predict_valid_customer():
    response = client.post(
        "/predict",
        json=VALID_CUSTOMER,
    )

    assert response.status_code == 200

    body = response.json()

    assert 0 <= body["churn_probability"] <= 1
    assert 0 <= body["threshold"] <= 1
    assert isinstance(body["predicted_churn"], bool)

    if body["predicted_churn"]:
        assert body["recommended_action"] == "contact"
    else:
        assert body["recommended_action"] == "no_contact"


def test_invalid_country():
    invalid_customer = VALID_CUSTOMER.copy()
    invalid_customer["country"] = "Italy"

    response = client.post(
        "/predict",
        json=invalid_customer,
    )

    assert response.status_code == 422


def test_negative_balance():
    invalid_customer = VALID_CUSTOMER.copy()
    invalid_customer["balance"] = -1000

    response = client.post(
        "/predict",
        json=invalid_customer,
    )

    assert response.status_code == 422


def test_missing_required_field():
    invalid_customer = VALID_CUSTOMER.copy()
    invalid_customer.pop("age")

    response = client.post(
        "/predict",
        json=invalid_customer,
    )

    assert response.status_code == 422
