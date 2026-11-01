"""Unit and integration tests for Explainable AI (XAI) Stage 1 modules."""

import pytest
from src.explainability.config import (
    AUDIT_DISCLAIMER,
    DEFAULT_CONFIG,
    ExplainabilityConfig,
)
from src.explainability.schemas import (
    DirectionalImpact,
    DomainContribution,
    ExplanationMethod,
    GlobalExplanation,
    GlobalFeatureImportance,
    LocalExplanation,
    RawContribution,
)
from src.explainability.base import (
    BaseExplainer,
    ContaminatedBackgroundError,
    ExplainabilityError,
    IncompatibleFeatureSchemaError,
    UnsupportedModelExplainerError,
)
from src.explainability.aggregator import FeatureAggregator
from src.explainability.shap_adapter import ShapAdapter
from src.explainability.lime_adapter import LimeAdapter
from src.explainability.background import BackgroundSelector
from src.explainability.explainer_factory import ExplainerFactory
from src.explainability.service import ExplainabilityService
import numpy as np
import scipy.sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

# ==============================================================================
# 1. CONFIGURATION TESTS
# ==============================================================================


def test_explainability_config_defaults():
    """Verify default configuration parameters and immutability."""
    config = ExplainabilityConfig()
    assert config.default_method == "shap"
    assert config.default_background_size == 200
    assert config.default_top_k == 10
    assert config.target_class_name == "Suspicious"
    assert config.baseline_class_name == "Legitimate"
    assert "Amount" in config.known_domain_features
    assert "Payment_type" in config.known_domain_features

    # Verify immutability (frozen dataclass)
    with pytest.raises(Exception):
        config.default_top_k = 25  # type: ignore


# ==============================================================================
# 2. SCHEMA AND DTO TESTS
# ==============================================================================


def test_raw_contribution_schema():
    """Verify RawContribution validation and fields."""
    contrib = RawContribution(
        feature_name="cat__Payment_type_Cross-border",
        feature_value=1.0,
        attribution=0.345,
        absolute_attribution=0.345,
        direction=DirectionalImpact.SUSPICIOUS_PUSHER,
        rank=1,
    )
    assert contrib.feature_name == "cat__Payment_type_Cross-border"
    assert contrib.attribution == 0.345
    assert contrib.direction == DirectionalImpact.SUSPICIOUS_PUSHER


def test_domain_contribution_schema():
    """Verify DomainContribution validation, subcomponents, and serialization."""
    sub1 = RawContribution(
        feature_name="cat__Payment_type_Cross-border",
        feature_value=1.0,
        attribution=0.30,
        absolute_attribution=0.30,
        direction=DirectionalImpact.SUSPICIOUS_PUSHER,
    )
    sub2 = RawContribution(
        feature_name="cat__Payment_type_ACH",
        feature_value=0.0,
        attribution=-0.05,
        absolute_attribution=0.05,
        direction=DirectionalImpact.LEGITIMACY_ANCHOR,
    )
    domain_c = DomainContribution(
        domain_feature="Payment_type",
        domain_value="Cross-border",
        net_attribution=0.25,
        absolute_attribution=0.35,
        direction=DirectionalImpact.SUSPICIOUS_PUSHER,
        sub_components=[sub1, sub2],
        rank=1,
    )
    assert domain_c.domain_feature == "Payment_type"
    assert domain_c.domain_value == "Cross-border"
    assert domain_c.net_attribution == 0.25
    assert domain_c.absolute_attribution == 0.35
    assert len(domain_c.sub_components) == 2


def test_local_explanation_schema():
    """Verify complete LocalExplanation model instantiation."""
    explanation = LocalExplanation(
        transaction_id="TXN-10029",
        method=ExplanationMethod.SHAP,
        model_name="LGBMClassifier",
        predicted_risk_score=0.88,
        base_value=0.015,
        raw_contributions=[],
        domain_contributions=[],
        latency_ms=12.4,
        metadata={"tree_limit": 100},
    )
    assert explanation.prediction_class == "Suspicious"
    assert explanation.disclaimer == AUDIT_DISCLAIMER
    assert explanation.predicted_risk_score == 0.88
    assert explanation.latency_ms == 12.4


def test_global_explanation_schema():
    """Verify GlobalExplanation instantiation."""
    global_exp = GlobalExplanation(
        method=ExplanationMethod.SHAP,
        model_name="RandomForestClassifier",
        sample_size=500,
        raw_feature_importances=[
            GlobalFeatureImportance(
                feature_name="num__Amount", mean_absolute_value=0.22, rank=1
            )
        ],
        domain_feature_importances=[
            GlobalFeatureImportance(
                feature_name="Amount", mean_absolute_value=0.22, rank=1
            )
        ],
    )
    assert global_exp.sample_size == 500
    assert len(global_exp.domain_feature_importances) == 1


# ==============================================================================
# 3. BASE CLASS & EXCEPTION TESTS
# ==============================================================================


def test_unsupported_model_explainer_error():
    """Verify custom exception formatted message."""
    err = UnsupportedModelExplainerError(
        model="DummyKerasModel",
        explainer_type="TreeExplainer",
        reason="Model is not a decision tree ensemble.",
    )
    assert "TreeExplainer" in str(err)
    assert "DummyKerasModel" in str(err)
    assert isinstance(err, ExplainabilityError)


def test_base_explainer_cannot_be_instantiated():
    """Verify that BaseExplainer enforces abstract methods."""

    class IncompleteExplainer(BaseExplainer):
        pass

    with pytest.raises(TypeError):
        IncompleteExplainer()  # type: ignore


# ==============================================================================
# 4. FEATURE AGGREGATOR TESTS
# ==============================================================================


@pytest.fixture
def aggregator():
    return FeatureAggregator()


def test_resolve_domain_name_patterns(aggregator):
    """Test resolution of various sklearn ColumnTransformer feature name formats."""
    # scikit-learn OneHotEncoder format
    assert (
        aggregator.resolve_domain_name("cat__Payment_type_Cross-border")
        == "Payment_type"
    )
    assert aggregator.resolve_domain_name("cat__Payment_type_ACH") == "Payment_type"
    assert (
        aggregator.resolve_domain_name("cat__Sender_bank_location_Switzerland")
        == "Sender_bank_location"
    )
    assert (
        aggregator.resolve_domain_name("cat__Receiver_bank_location_Cayman_Islands")
        == "Receiver_bank_location"
    )
    assert aggregator.resolve_domain_name("cat__Time_14:20:00") == "Time"
    assert aggregator.resolve_domain_name("cat__Date_2023-10-15") == "Date"

    # Numeric scalers
    assert aggregator.resolve_domain_name("num__Amount") == "Amount"
    assert aggregator.resolve_domain_name("num__Sender_account") == "Sender_account"
    assert aggregator.resolve_domain_name("num__Receiver_account") == "Receiver_account"

    # Direct match or fallback
    assert aggregator.resolve_domain_name("Amount") == "Amount"
    assert (
        aggregator.resolve_domain_name("unknown_custom_metric")
        == "unknown_custom_metric"
    )


def test_aggregator_mathematical_conservation(aggregator):
    """Verify mathematical conservation of net and absolute attributions across aggregation."""
    raw_list = [
        RawContribution(
            feature_name="cat__Payment_type_Cross-border",
            feature_value=1.0,
            attribution=0.40,
            absolute_attribution=0.40,
            direction=DirectionalImpact.SUSPICIOUS_PUSHER,
        ),
        RawContribution(
            feature_name="cat__Payment_type_ACH",
            feature_value=0.0,
            attribution=-0.10,
            absolute_attribution=0.10,
            direction=DirectionalImpact.LEGITIMACY_ANCHOR,
        ),
        RawContribution(
            feature_name="num__Amount",
            feature_value=1.85,
            attribution=0.25,
            absolute_attribution=0.25,
            direction=DirectionalImpact.SUSPICIOUS_PUSHER,
        ),
        RawContribution(
            feature_name="cat__Sender_bank_location_Switzerland",
            feature_value=1.0,
            attribution=0.15,
            absolute_attribution=0.15,
            direction=DirectionalImpact.SUSPICIOUS_PUSHER,
        ),
    ]

    raw_record = {
        "Payment_type": "Cross-border",
        "Amount": 45000.00,
        "Sender_bank_location": "Switzerland",
    }

    domain_results = aggregator.aggregate_contributions(raw_list, raw_record=raw_record)

    # 1. Total net sum must match exactly
    raw_net_sum = sum(r.attribution for r in raw_list)
    domain_net_sum = sum(d.net_attribution for d in domain_results)
    assert pytest.approx(domain_net_sum, abs=1e-5) == raw_net_sum

    # 2. Total absolute sum must match exactly
    raw_abs_sum = sum(r.absolute_attribution for r in raw_list)
    domain_abs_sum = sum(d.absolute_attribution for d in domain_results)
    assert pytest.approx(domain_abs_sum, abs=1e-5) == raw_abs_sum

    # 3. Check Payment_type aggregated results
    payment_contrib = next(
        d for d in domain_results if d.domain_feature == "Payment_type"
    )
    assert pytest.approx(payment_contrib.net_attribution, abs=1e-5) == 0.30
    assert pytest.approx(payment_contrib.absolute_attribution, abs=1e-5) == 0.50
    assert payment_contrib.domain_value == "Cross-border"
    assert payment_contrib.direction == DirectionalImpact.SUSPICIOUS_PUSHER
    assert len(payment_contrib.sub_components) == 2

    # 4. Check Amount aggregated results
    amount_contrib = next(d for d in domain_results if d.domain_feature == "Amount")
    assert pytest.approx(amount_contrib.net_attribution, abs=1e-5) == 0.25
    assert amount_contrib.domain_value == 45000.00

    # 5. Check ranking order (Payment_type with abs=0.50 should rank higher than Amount with abs=0.25)
    assert domain_results[0].domain_feature == "Payment_type"
    assert domain_results[0].rank == 1


def test_aggregator_top_k_truncation(aggregator):
    """Verify that top_k correctly limits returned domain contributions."""
    raw_list = [
        RawContribution(
            feature_name="num__Amount",
            attribution=0.5,
            absolute_attribution=0.5,
            direction=DirectionalImpact.SUSPICIOUS_PUSHER,
        ),
        RawContribution(
            feature_name="cat__Payment_type_Cash",
            attribution=-0.3,
            absolute_attribution=0.3,
            direction=DirectionalImpact.LEGITIMACY_ANCHOR,
        ),
        RawContribution(
            feature_name="cat__Sender_bank_location_USA",
            attribution=0.1,
            absolute_attribution=0.1,
            direction=DirectionalImpact.SUSPICIOUS_PUSHER,
        ),
    ]

    domain_results = aggregator.aggregate_contributions(raw_list, top_k=2)
    assert len(domain_results) == 2
    assert domain_results[0].domain_feature == "Amount"
    assert domain_results[1].domain_feature == "Payment_type"


# ==============================================================================
# 5. SHAP ADAPTER TESTS (STAGE 2)
# ==============================================================================


@pytest.fixture
def synthetic_tabular_data():
    """Generate small synthetic transaction feature matrix and labels."""
    np.random.seed(42)
    X = np.random.randn(60, 4)
    # Ensure some correlation: column 0 (Amount) positive implies higher risk
    y = ((X[:, 0] + X[:, 1] * 0.5) > 0.2).astype(int)
    feature_names = [
        "num__Amount",
        "cat__Payment_type_Cross-border",
        "cat__Payment_type_ACH",
        "cat__Sender_bank_location_Switzerland",
    ]
    return X, y, feature_names


def test_shap_tree_adapter_random_forest(synthetic_tabular_data):
    """Verify SHAP TreeExplainer dispatch and output contract with RandomForest."""
    X, y, feature_names = synthetic_tabular_data
    rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    rf.fit(X, y)

    adapter = ShapAdapter(model=rf, feature_names=feature_names)
    assert adapter.explainer_type == "TreeExplainer"
    assert adapter.get_feature_names() == feature_names

    raw_rec = {
        "Payment_type": "Cross-border",
        "Amount": 25000.0,
        "Sender_bank_location": "Switzerland",
    }
    explanation = adapter.explain_instance(X[0], raw_record=raw_rec, top_k=5)

    assert isinstance(explanation, LocalExplanation)
    assert explanation.method == ExplanationMethod.SHAP
    assert explanation.model_name == "RandomForestClassifier"
    assert explanation.prediction_class == "Suspicious"
    assert 0.0 <= explanation.predicted_risk_score <= 1.0
    assert explanation.latency_ms is not None and explanation.latency_ms > 0
    assert len(explanation.raw_contributions) == 4
    assert len(explanation.domain_contributions) > 0

    # Verify domain aggregation conservation
    raw_net = sum(c.attribution for c in explanation.raw_contributions)
    dom_net = sum(d.net_attribution for d in explanation.domain_contributions)
    assert pytest.approx(dom_net, abs=1e-4) == raw_net


def test_shap_tree_adapter_lightgbm(synthetic_tabular_data):
    """Verify SHAP TreeExplainer with LGBMClassifier and global summary."""
    X, y, feature_names = synthetic_tabular_data
    lgb = LGBMClassifier(n_estimators=10, max_depth=3, random_state=42, verbose=-1)
    lgb.fit(X, y)

    adapter = ShapAdapter(model=lgb, feature_names=feature_names)
    assert adapter.explainer_type == "TreeExplainer"

    local_exp = adapter.explain_instance(X[0])
    assert local_exp.method == ExplanationMethod.SHAP
    assert local_exp.prediction_class == "Suspicious"

    global_exp = adapter.explain_global(X, top_k=3)
    assert isinstance(global_exp, GlobalExplanation)
    assert global_exp.sample_size == len(X)
    assert len(global_exp.raw_feature_importances) == 3
    assert len(global_exp.domain_feature_importances) > 0


def test_shap_linear_adapter_logistic_regression(synthetic_tabular_data):
    """Verify SHAP LinearExplainer dispatch with LogisticRegression."""
    X, y, feature_names = synthetic_tabular_data
    lr = LogisticRegression(random_state=42)
    lr.fit(X, y)

    # LinearExplainer requires background data
    adapter = ShapAdapter(model=lr, background_data=X[:20], feature_names=feature_names)
    assert adapter.explainer_type == "LinearExplainer"

    explanation = adapter.explain_instance(X[0])
    assert explanation.method == ExplanationMethod.SHAP
    assert explanation.model_name == "LogisticRegression"
    assert explanation.prediction_class == "Suspicious"


def test_shap_linear_adapter_missing_background_raises_error(synthetic_tabular_data):
    """Verify that initializing LinearExplainer without background data raises ExplainabilityError."""
    X, y, feature_names = synthetic_tabular_data
    lr = LogisticRegression(random_state=42)
    lr.fit(X, y)

    with pytest.raises(ExplainabilityError) as exc_info:
        ShapAdapter(model=lr, background_data=None, feature_names=feature_names)
    assert "Background data is strictly required" in str(exc_info.value)


def test_shap_unsupported_model_raises_error():
    """Verify unsupported model raises UnsupportedModelExplainerError."""

    class UnsupportedCustomModel:
        pass

    with pytest.raises(UnsupportedModelExplainerError) as exc_info:
        ShapAdapter(model=UnsupportedCustomModel(), background_data=None)
    assert "No compatible SHAP explainer available" in str(exc_info.value)


# ==============================================================================
# 6. LIME ADAPTER TESTS (STAGE 2)
# ==============================================================================


def test_lime_adapter_random_forest(synthetic_tabular_data):
    """Verify LIME tabular explainer initialization and local explanation."""
    X, y, feature_names = synthetic_tabular_data
    rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    rf.fit(X, y)

    adapter = LimeAdapter(model=rf, training_data=X, feature_names=feature_names)
    assert adapter.get_feature_names() == feature_names

    raw_rec = {
        "Payment_type": "Cross-border",
        "Amount": 18000.0,
        "Sender_bank_location": "Switzerland",
    }
    explanation = adapter.explain_instance(
        X[0], raw_record=raw_rec, top_k=3, num_samples=200
    )

    assert isinstance(explanation, LocalExplanation)
    assert explanation.method == ExplanationMethod.LIME
    assert explanation.prediction_class == "Suspicious"
    assert 0.0 <= explanation.predicted_risk_score <= 1.0
    assert len(explanation.raw_contributions) == 3
    assert len(explanation.domain_contributions) > 0
    assert explanation.metadata["num_samples"] == 200


def test_lime_adapter_sparse_matrix_support(synthetic_tabular_data):
    """Verify LIME adapter transparently handles models fitted on sparse matrices."""
    X, y, feature_names = synthetic_tabular_data
    X_sparse = scipy.sparse.csr_matrix(X)
    rf = RandomForestClassifier(n_estimators=5, random_state=42)
    rf.fit(X_sparse, y)

    adapter = LimeAdapter(model=rf, training_data=X_sparse, feature_names=feature_names)
    explanation = adapter.explain_instance(X_sparse[0], top_k=2, num_samples=150)

    assert explanation.method == ExplanationMethod.LIME
    assert len(explanation.raw_contributions) == 2


def test_lime_adapter_stability_evaluation(synthetic_tabular_data):
    """Verify LIME stability evaluation produces valid Jaccard index between 0.0 and 1.0."""
    X, y, feature_names = synthetic_tabular_data
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    adapter = LimeAdapter(model=rf, training_data=X, feature_names=feature_names)
    stability = adapter.evaluate_stability(X[0], num_runs=3, top_k=2, num_samples=150)

    assert isinstance(stability, float)
    assert 0.0 <= stability <= 1.0


# ==============================================================================
# 7. BACKGROUND SELECTOR TESTS (STAGE 3)
# ==============================================================================


def test_background_selector_sampling(synthetic_tabular_data):
    """Verify background selector strictly extracts legitimate (class 0) transactions."""
    X, y, _ = synthetic_tabular_data
    selector = BackgroundSelector()

    sampled_bg = selector.sample_legitimate_background(data=X, labels=y, sample_size=15)
    assert sampled_bg.shape[0] == 15
    assert sampled_bg.shape[1] == X.shape[1]
    assert selector.get_cached_background() is not None


def test_background_selector_contaminated_detection(synthetic_tabular_data):
    """Verify background selector raises ContaminatedBackgroundError if no legitimate records exist."""
    X, _, _ = synthetic_tabular_data
    # Synthetic / resampled-only batch with all positive labels
    all_positive_y = np.ones(len(X), dtype=int)
    selector = BackgroundSelector()

    with pytest.raises(ContaminatedBackgroundError) as exc_info:
        selector.sample_legitimate_background(
            data=X, labels=all_positive_y, sample_size=10
        )
    assert "contains no legitimate (class 0) transactions" in str(exc_info.value)


# ==============================================================================
# 8. EXPLAINER FACTORY TESTS (STAGE 3)
# ==============================================================================


def test_explainer_factory_dispatch(synthetic_tabular_data):
    """Verify factory properly creates SHAP and LIME adapters based on requested method."""
    X, y, feature_names = synthetic_tabular_data
    rf = RandomForestClassifier(n_estimators=5, random_state=42)
    rf.fit(X, y)

    # 1. SHAP dispatch
    shap_exp = ExplainerFactory.create_explainer(
        method="shap",
        model=rf,
        background_data=X,
        feature_names=feature_names,
    )
    assert isinstance(shap_exp, ShapAdapter)

    # 2. LIME dispatch
    lime_exp = ExplainerFactory.create_explainer(
        method="lime",
        model=rf,
        background_data=X,
        feature_names=feature_names,
    )
    assert isinstance(lime_exp, LimeAdapter)

    # 3. Invalid method raises error
    with pytest.raises(UnsupportedModelExplainerError):
        ExplainerFactory.create_explainer(
            method="unknown_algo",
            model=rf,
            background_data=X,
        )


# ==============================================================================
# 9. EXPLAINABILITY SERVICE TESTS (STAGE 3)
# ==============================================================================


def test_explainability_service_end_to_end(synthetic_tabular_data):
    """Verify ExplainabilityService end-to-end local and global explanations."""
    X, y, feature_names = synthetic_tabular_data
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    service = ExplainabilityService(
        model=rf,
        feature_names=feature_names,
        background_data=X,
    )

    raw_rec = {
        "Payment_type": "Cross-border",
        "Amount": 35000.0,
        "Sender_bank_location": "Switzerland",
    }

    # 1. Default SHAP local explanation
    shap_explanation = service.explain_transaction(
        X[0], raw_record=raw_rec, method="shap"
    )
    assert shap_explanation.method == ExplanationMethod.SHAP
    assert shap_explanation.model_name == "RandomForestClassifier"
    assert len(shap_explanation.domain_contributions) > 0
    assert shap_explanation.prediction_class == "Suspicious"

    # 2. On-demand LIME local explanation
    lime_explanation = service.explain_transaction(
        X[0], raw_record=raw_rec, method="lime", num_samples=150
    )
    assert lime_explanation.method == ExplanationMethod.LIME
    assert lime_explanation.model_name == "RandomForestClassifier"
    assert len(lime_explanation.domain_contributions) > 0

    # 3. On-demand stability evaluation
    stability = service.evaluate_stability(X[0], num_runs=2, top_k=2, num_samples=100)
    assert 0.0 <= stability <= 1.0

    # 4. Global explanation
    global_exp = service.get_global_explanation(dataset=X[:20], method="shap", top_k=3)
    assert global_exp.method == ExplanationMethod.SHAP
    assert len(global_exp.raw_feature_importances) == 3


def test_explainability_service_hot_reload(synthetic_tabular_data):
    """Verify service dynamically updates when model artifacts are reloaded."""
    X, y, feature_names = synthetic_tabular_data
    rf = RandomForestClassifier(n_estimators=5, random_state=42)
    rf.fit(X, y)
    lr = LogisticRegression(random_state=42)
    lr.fit(X, y)

    service = ExplainabilityService(
        model=rf, feature_names=feature_names, background_data=X
    )
    assert service.model == rf

    # Hot reload to LogisticRegression
    service.set_model_artifacts(
        model=lr, feature_names=feature_names, background_data=X
    )
    assert service.model == lr

    exp = service.explain_transaction(X[0], method="shap")
    assert exp.model_name == "LogisticRegression"


def test_shap_tree_adapter_xgboost(synthetic_tabular_data):
    """Verify ShapAdapter correctly identifies and runs on XGBClassifier."""
    from xgboost import XGBClassifier

    X, y, feature_names = synthetic_tabular_data
    xgb = XGBClassifier(n_estimators=10, max_depth=3, random_state=42, eval_metric="aucpr")
    xgb.fit(X, y)

    adapter = ShapAdapter(model=xgb, feature_names=feature_names)
    assert adapter.explainer_type == "TreeExplainer"

    local_exp = adapter.explain_instance(X[0], top_k=5)
    assert local_exp.method == ExplanationMethod.SHAP
    assert local_exp.model_name == "XGBClassifier"
    assert len(local_exp.raw_contributions) == min(5, len(feature_names))
    assert len(local_exp.domain_contributions) > 0
    assert 0.0 <= local_exp.predicted_risk_score <= 1.0

