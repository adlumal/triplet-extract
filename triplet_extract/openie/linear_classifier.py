"""
Linear Classifier for Clause Splitting

Loads the pre-trained Stanford Linear Classifier weights and performs
3-class classification for clause splitting decisions.

Classifier labels:
- NOT_A_CLAUSE: Edge should not split into a clause
- CLAUSE_SPLIT: Edge should split, creating independent clause
- CLAUSE_INTERM: Edge should split, but dependent on parent context
"""

import logging
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


class LinearClassifier:
    """
    Linear classifier using pre-trained Stanford OpenIE weights.

    This classifier was trained on the LSOIE dataset and uses 14,834 features
    to predict clause split decisions.
    """

    def __init__(self, model_path: str | None = None):
        """
        Initialize the linear classifier.

        Args:
            model_path: Path to extracted_model.pkl
                       If None, uses package's built-in model
        """
        if model_path is None:
            # Use package-relative path
            model_path = (
                Path(__file__).parent.parent / "models" / "clause_model" / "extracted_model.pkl"
            )

        self.model_path = Path(model_path)
        self.weights: np.ndarray | None = None
        self.features: list[str] | None = None
        self.labels: list[str] | None = None
        self.feature_to_index: dict[str, int] | None = None

        self._load_model()

    def _load_model(self):
        """Load the pre-trained model from pickle file."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Please run scripts/extract_clause_model.py first."
            )

        logging.info(f"Loading linear classifier from {self.model_path}")

        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)

        # Extract components
        self.weights = np.array(model_data["weights"])  # Shape: (14834, 3)
        self.features = model_data["features"]  # List of 14834 feature names
        self.labels = model_data["labels"]  # ['NOT_A_CLAUSE', 'CLAUSE_SPLIT', 'CLAUSE_INTERM']

        # Create feature name -> index mapping for fast lookup
        self.feature_to_index = {name: idx for idx, name in enumerate(self.features)}

        logging.info("Loaded classifier:")
        logging.info(f"  Features: {len(self.features)}")
        logging.info(f"  Labels: {self.labels}")
        logging.info(f"  Weight matrix shape: {self.weights.shape}")

    def featurize_to_vector(self, feature_counter: Counter) -> np.ndarray:
        """
        Convert a feature counter to a dense feature vector.

        Args:
            feature_counter: Counter mapping feature names to counts/values

        Returns:
            Dense numpy array of shape (n_features,) with feature values
        """
        # Initialize zero vector
        feature_vector = np.zeros(len(self.features), dtype=np.float32)

        # Fill in feature values
        for feature_name, value in feature_counter.items():
            if feature_name in self.feature_to_index:
                idx = self.feature_to_index[feature_name]
                feature_vector[idx] = float(value)

        return feature_vector

    def predict(self, feature_counter: Counter) -> str:
        """
        Predict the label for given features.

        Args:
            feature_counter: Counter of feature name -> value

        Returns:
            Predicted label string (NOT_A_CLAUSE, CLAUSE_SPLIT, or CLAUSE_INTERM)
        """
        scores = self.predict_scores(feature_counter)
        best_label_idx = np.argmax(scores)
        return self.labels[best_label_idx]

    def predict_scores(self, feature_counter: Counter) -> np.ndarray:
        """
        Compute raw scores for each label.

        Args:
            feature_counter: Counter of feature name -> value

        Returns:
            Array of shape (3,) with scores for each label
        """
        # Convert features to vector
        feature_vector = self.featurize_to_vector(feature_counter)

        # Compute dot product: scores = weights.T @ features
        # weights shape: (14834, 3)
        # feature_vector shape: (14834,)
        # result shape: (3,)
        scores = feature_vector @ self.weights

        return scores

    def predict_probabilities(self, feature_counter: Counter) -> dict[str, float]:
        """
        Predict label probabilities using softmax.

        Args:
            feature_counter: Counter of feature name -> value

        Returns:
            Dictionary mapping label names to probabilities
        """
        scores = self.predict_scores(feature_counter)

        # Apply softmax
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        probabilities = exp_scores / np.sum(exp_scores)

        return {label: float(prob) for label, prob in zip(self.labels, probabilities, strict=True)}

    def predict_with_confidence(self, feature_counter: Counter) -> tuple[str, float]:
        """
        Predict label with confidence score.

        Args:
            feature_counter: Counter of feature name -> value

        Returns:
            Tuple of (predicted_label, confidence) where confidence is the softmax probability
        """
        probabilities = self.predict_probabilities(feature_counter)
        predicted_label = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted_label]

        return predicted_label, confidence
