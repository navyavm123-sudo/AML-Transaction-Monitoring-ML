"""Explainability Service: Unified facade coordinating preprocessing, background data, and explainers."""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import scipy.sparse

from src.explainability.aggregator import FeatureAggregator
from src.explainability.background import BackgroundSelector
from src.explainability.base import (
    BaseExplainer,
    ExplainabilityError,
    IncompatibleFeatureSchemaError,
)
from src.explainability.config import DEFAULT_CONFIG, ExplainabilityConfig
from src.explainability.explainer_factory import ExplainerFactory
from src.explainability.schemas import (
    ExplanationMethod,
    GlobalExplanation,
    LocalExplanation,
)

logger = logging.getLogger("aml_explainability.service")


class ExplainabilityService:
    """Unified service facade for real-time and audit XAI explanations in AML Transaction Monitoring.

    Responsibilities:
    - Encapsulates model artifacts (model, preprocessor, selector, feature names).
    - Manages uncontaminated empirical reference background data.
    - Caches configured SHAP and LIME adapters for low-latency scoring.
    - Seamlessly accepts either raw transaction dictionaries or transformed model matrices.
    - Delivers dual-granularity outputs with mathematical conservation.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        preprocessor: Optional[Any] = None,
        selector: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        background_data: Optional[Union[np.ndarray, scipy.sparse.spmatrix]] = None,
        config: Optional[ExplainabilityConfig] = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.aggregator = FeatureAggregator(self.config.known_domain_features)
        self.background_selector = BackgroundSelector(self.config)

        self.model: Optional[Any] = None
        self.preprocessor: Optional[Any] = None
        self.selector: Optional[Any] = None
        self.feature_names: List[str] = feature_names or []
        self.background_data: Optional[Union[np.ndarray, scipy.sparse.spmatrix]] = (
            background_data
        )

        # Explainer cache: {'shap': ShapAdapter, 'lime': LimeAdapter}
        self._explainers: Dict[str, BaseExplainer] = {}

        if model is not None:
            self.set_model_artifacts(
                model=model,
                preprocessor=preprocessor,
                selector=selector,
                feature_names=feature_names,
                background_data=background_data,
            )

    def set_model_artifacts(
        self,
        model: Any,
        preprocessor: Optional[Any] = None,
        selector: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        background_data: Optional[Union[np.ndarray, scipy.sparse.spmatrix]] = None,
    ) -> None:
        """Update model artifacts and re-initialize explainer cache."""
        self.model = model
        self.preprocessor = preprocessor
        self.selector = selector
        self._explainers.clear()

        # Update background if provided, or retain existing if dimensions match
        if background_data is not None:
            self.background_data = background_data
        elif self.background_data is not None and self.feature_names:
            if (
                hasattr(self.background_data, "shape")
                and self.background_data.shape[1] != len(self.feature_names)
            ):
                self.background_data = None

        # Derive feature names if not provided
        if feature_names:
            self.feature_names = list(feature_names)
        else:
            self.feature_names = self._infer_feature_names()

        logger.info(
            f"ExplainabilityService configured for model '{type(model).__name__}' "
            f"with {len(self.feature_names)} features."
        )

    def _infer_feature_names(self) -> List[str]:
        """Infer feature names from fitted preprocessor or selector where available."""
        names: List[str] = []
        if self.preprocessor is not None and hasattr(
            self.preprocessor, "get_feature_names_out"
        ):
            try:
                prep_names = self.preprocessor.get_feature_names_out()
                if self.selector is not None and hasattr(self.selector, "get_support"):
                    support = self.selector.get_support()
                    if len(support) == len(prep_names):
                        return list(prep_names[support])
                return list(prep_names)
            except Exception as e:
                logger.debug(f"Could not infer feature names from preprocessor: {e}")

        if self.background_data is not None and hasattr(self.background_data, "shape"):
            n_cols = self.background_data.shape[1]
            return [f"f_{i}" for i in range(n_cols)]

        return names

    def get_background_data(self) -> Union[np.ndarray, scipy.sparse.spmatrix]:
        """Retrieve or lazily sample uncontaminated background reference data."""
        if self.background_data is None:
            self.background_data = (
                self.background_selector.sample_legitimate_background()
            )
        return self.background_data

    def get_explainer(self, method: str = "shap") -> BaseExplainer:
        """Get or lazily instantiate an explainer adapter for the specified method."""
        if self.model is None:
            raise ExplainabilityError("No ML model loaded in ExplainabilityService.")

        method_key = method.lower().strip()
        if method_key not in self._explainers:
            bg_data = self.get_background_data()
            explainer = ExplainerFactory.create_explainer(
                method=method_key,
                model=self.model,
                background_data=bg_data,
                feature_names=self.feature_names,
                config=self.config,
                aggregator=self.aggregator,
            )
            self._explainers[method_key] = explainer

        return self._explainers[method_key]

    def transform_raw_record(
        self, raw_record: Dict[str, Any]
    ) -> Union[np.ndarray, scipy.sparse.spmatrix]:
        """Apply feature engineering, preprocessor and selector pipelines to a raw transaction dictionary."""
        from src.features.pipeline import engineer_features

        # Sanitize raw record: prevent target leakage
        clean_dict = {
            k: v
            for k, v in raw_record.items()
            if k != "Is_laundering"
        }
        if "Laundering_type" not in clean_dict or clean_dict["Laundering_type"] is None:
            clean_dict["Laundering_type"] = "Unknown"

        df = engineer_features(clean_dict)

        if self.preprocessor is not None:
            try:
                X_proc = self.preprocessor.transform(df)
                if self.selector is not None:
                    try:
                        X_proc = self.selector.transform(X_proc)
                    except Exception as sel_err:
                        logger.debug(f"Selector transform skipped: {sel_err}")
                return X_proc
            except Exception as e:
                raise IncompatibleFeatureSchemaError(
                    f"Failed transforming raw transaction through preprocessor pipeline: {e}"
                ) from e

        # If no preprocessor is fitted, align with registered feature_names if present
        if self.feature_names:
            matching_cols = [col for col in self.feature_names if col in df.columns]
            if len(matching_cols) == len(self.feature_names):
                return df[self.feature_names].to_numpy()

        # Fallback to dense numeric values
        return df.to_numpy()

    def explain_transaction(
        self,
        transaction: Union[
            Dict[str, Any], np.ndarray, scipy.sparse.spmatrix, pd.DataFrame, pd.Series
        ],
        raw_record: Optional[Dict[str, Any]] = None,
        method: str = "shap",
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> LocalExplanation:
        """Generate dual-granularity local explanation for an incoming transaction.

        Args:
            transaction: Raw transaction dictionary or preprocessed feature vector/matrix.
            raw_record: Optional unencoded transaction attributes for domain value binding.
            method: Explanation algorithm ('shap' by default, 'lime' on demand).
            top_k: Maximum number of top features to return in raw and domain views.
            **kwargs: Extra parameters passed to the explainer (e.g. num_samples for LIME).

        Returns:
            LocalExplanation containing both raw model attributions and aggregated domain features.
        """
        top_k = top_k or self.config.default_top_k
        explainer = self.get_explainer(method=method)

        # 1. Handle raw dictionary input
        if isinstance(transaction, dict):
            raw_dict = transaction
            x_input = self.transform_raw_record(raw_dict)
            return explainer.explain_instance(
                instance=x_input,
                raw_record=raw_dict,
                top_k=top_k,
                **kwargs,
            )

        # 2. Handle transformed feature vector / matrix
        return explainer.explain_instance(
            instance=transaction,
            raw_record=raw_record,
            top_k=top_k,
            **kwargs,
        )

    def get_global_explanation(
        self,
        dataset: Optional[
            Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame]
        ] = None,
        method: str = "shap",
        sample_size: int = 100,
        top_k: Optional[int] = None,
    ) -> GlobalExplanation:
        """Generate global feature-importance explanation across a representative dataset sample."""
        top_k = top_k or 20
        explainer = self.get_explainer(method=method)

        if dataset is None:
            dataset = self.get_background_data()

        if hasattr(dataset, "shape") and dataset.shape[0] > sample_size:
            dataset = dataset[:sample_size]

        return explainer.explain_global(dataset=dataset, top_k=top_k)

    def evaluate_stability(
        self,
        instance: Union[np.ndarray, scipy.sparse.spmatrix],
        num_runs: int = 3,
        top_k: int = 5,
        num_samples: int = 500,
    ) -> float:
        """Evaluate LIME perturbation stability on demand."""
        lime_explainer = self.get_explainer("lime")
        if hasattr(lime_explainer, "evaluate_stability"):
            inst_1d = (
                instance.toarray().ravel()
                if scipy.sparse.issparse(instance)
                else np.asarray(instance).ravel()
            )
            return lime_explainer.evaluate_stability(
                instance=inst_1d,
                num_runs=num_runs,
                top_k=top_k,
                num_samples=num_samples,
            )
        return 1.0
