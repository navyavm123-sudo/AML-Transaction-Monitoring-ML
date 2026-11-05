"""Unit and integration tests for FastAPI monitoring service."""

import pytest
from fastapi.testclient import TestClient
from src.api import app


@pytest.fixture
def client():
    """Create test client with lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Test that the root endpoint returns 200 OK and expected metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert (
        "FastAPI lets other applications use your trained model" in data["description"]
    )


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_name" in data


def test_predict_single_endpoint(client):
    """Test single transaction prediction."""
    payload = {
        "Sender_account": "ACC-9901",
        "Receiver_account": "ACC-9902",
        "Amount": 15000.0,
        "Payment_currency": "US Dollar",
        "Received_currency": "Euro",
        "Sender_bank_location": "USA",
        "Receiver_bank_location": "Switzerland",
        "Payment_type": "Cross-border",
        "Date": "2023-10-15",
        "Time": "14:20:00",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "is_suspicious" in data
    assert "risk_score" in data
    assert "risk_level" in data
    assert data["transaction_id"] == "ACC-9901"


def test_predict_batch_endpoint(client):
    """Test batch transaction prediction."""
    payload = {
        "transactions": [
            {"Amount": 200.0, "Payment_type": "Credit card"},
            {"Amount": 250000.0, "Payment_type": "Cross-border"},
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["results"]) == 2


def test_model_reload_endpoint(client):
    """Test hot reload of model artifacts."""
    response = client.post("/model/reload")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


def test_explain_endpoint_when_model_not_loaded(client):
    """Test /explain endpoint returns 503 Service Unavailable when in mock mode."""
    from src.api import model_artifacts

    original_is_loaded = model_artifacts["is_loaded"]
    original_model = model_artifacts["model"]
    original_prep = model_artifacts["preprocessor"]

    try:
        model_artifacts["is_loaded"] = False
        model_artifacts["model"] = None

        payload = {
            "Amount": 45000.0,
            "Payment_type": "Cross-border",
            "Sender_bank_location": "USA",
            "Receiver_bank_location": "Switzerland",
        }
        response = client.post("/explain", json=payload)
        assert response.status_code == 503
        data = response.json()
        assert "Machine learning model is not loaded" in data["detail"]
    finally:
        model_artifacts["is_loaded"] = original_is_loaded
        model_artifacts["model"] = original_model
        model_artifacts["preprocessor"] = original_prep


def test_explain_endpoint_with_active_model(client):
    """Test /explain endpoint returns dual-granularity SHAP explanation when model is active."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from src.api import model_artifacts, explainability_service

    # Create synthetic active model and background
    np.random.seed(42)
    X_syn = np.random.randn(30, 4)
    y_syn = (X_syn[:, 0] > 0.1).astype(int)
    feat_names = [
        "num__Amount",
        "cat__Payment_type_Cross-border",
        "cat__Payment_type_ACH",
        "cat__Sender_bank_location_Switzerland",
    ]
    rf = RandomForestClassifier(n_estimators=5, random_state=42)
    rf.fit(X_syn, y_syn)

    original_is_loaded = model_artifacts["is_loaded"]
    original_model = model_artifacts["model"]
    original_prep = model_artifacts["preprocessor"]
    original_selector = model_artifacts.get("selector")
    original_bg = explainability_service.background_data

    class MockPreprocessor:
        def transform(self, df):
            # Transform dataframe into 4 numeric feature values for the mock model
            arr = np.zeros((len(df), 4), dtype=float)
            for i, (_, row) in enumerate(df.iterrows()):
                arr[i, 0] = float(row.get("Amount", 0.0)) / 1000.0
                arr[i, 1] = 1.0 if row.get("Payment_type") == "Cross-border" else 0.0
                arr[i, 2] = 1.0 if row.get("Payment_type") == "ACH" else 0.0
                arr[i, 3] = (
                    1.0 if row.get("Sender_bank_location") == "Switzerland" else 0.0
                )
            return arr

        def get_feature_names_out(self):
            return np.array(feat_names)

    mock_prep = MockPreprocessor()

    try:
        model_artifacts["is_loaded"] = True
        model_artifacts["model"] = rf
        model_artifacts["model_name"] = "RandomForestClassifier"
        model_artifacts["preprocessor"] = mock_prep

        explainability_service.set_model_artifacts(
            model=rf,
            preprocessor=mock_prep,
            feature_names=feat_names,
            background_data=X_syn,
        )

        payload = {
            "Sender_account": "ACC-9901",
            "Amount": 75000.0,
            "Payment_type": "Cross-border",
            "Sender_bank_location": "Switzerland",
        }

        # 1. Default SHAP explanation
        response = client.post("/explain?method=shap&top_k=5", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["method"] == "shap"
        assert data["model_name"] == "RandomForestClassifier"
        assert data["prediction_class"] == "Suspicious"
        assert "raw_contributions" in data
        assert "domain_contributions" in data
        assert "disclaimer" in data
        assert len(data["domain_contributions"]) > 0

        # 2. On-demand LIME explanation
        response_lime = client.post("/explain?method=lime&top_k=3", json=payload)
        assert response_lime.status_code == 200
        data_lime = response_lime.json()
        assert data_lime["method"] == "lime"
        assert len(data_lime["domain_contributions"]) > 0

        # 3. Invalid method returns 400 Bad Request
        bad_response = client.post("/explain?method=invalid_algo", json=payload)
        assert bad_response.status_code == 400
        assert (
            "Supported methods are 'shap' and 'lime'" in bad_response.json()["detail"]
        )

    finally:
        model_artifacts["is_loaded"] = original_is_loaded
        model_artifacts["model"] = original_model
        model_artifacts["preprocessor"] = original_prep
        model_artifacts["selector"] = original_selector
        if original_model is not None:
            explainability_service.set_model_artifacts(
                model=original_model,
                preprocessor=original_prep,
                selector=original_selector,
                feature_names=model_artifacts.get("feature_names"),
                background_data=original_bg,
            )


def test_xgboost_no_smote_model_active(client):
    """Verify that the selected XGBoost No SMOTE model is loaded and operational."""
    from src.api import model_artifacts

    assert model_artifacts["is_loaded"] is True
    assert "XGB" in model_artifacts["model_name"]

    health = client.get("/health").json()
    assert health["is_model_loaded"] is True
    assert "XGB" in health["model_name"]


def test_xgboost_explain_real_transaction(client):
    """Verify that /explain produces live SHAP explanations on the XGBoost model."""
    payload = {
        "Sender_account": "ACC-891024",
        "Receiver_account": "ACC-381940",
        "Amount": 12500.0,
        "Payment_currency": "US Dollar",
        "Received_currency": "Euro",
        "Sender_bank_location": "USA",
        "Receiver_bank_location": "Switzerland",
        "Payment_type": "Cross-border",
        "Date": "2023-10-15",
        "Time": "23:45:10",
    }
    response = client.post("/explain?method=shap&top_k=5", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "shap"
    assert "XGB" in data["model_name"]
    assert len(data["domain_contributions"]) > 0
    assert len(data["raw_contributions"]) > 0
    assert "disclaimer" in data


def test_model_info_and_cors_preflight(client):
    """Verify /model/info metadata and CORS preflight headers for frontend clients."""
    # 1. /model/info endpoint
    info_resp = client.get("/model/info")
    assert info_resp.status_code == 200
    info_data = info_resp.json()
    assert info_data["status"] == "healthy"
    assert info_data["is_model_loaded"] is True
    assert "threshold" in info_data
    assert "feature_count" in info_data

    # 2. CORS preflight for frontend origins (Streamlit, React, etc.)
    for origin in ["http://localhost:8501", "http://localhost:3000"]:
        preflight = client.options(
            "/predict",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers.get("access-control-allow-origin") == origin
