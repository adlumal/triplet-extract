"""
Affinity Models for Natural Logic Entailment

These models contain prepositional phrase attachment probabilities used for
determining whether a PP should be deleted during natural logic inference.

Format: TSV files with counts (probability = count / total_count)
- pp.tab.gz: verb, prep, count, total_count
- subj_pp.tab.gz: verb, subj, prep, count, total_count
- obj.tab.gz: verb, count, total_count
- subj_obj_pp.tab.gz: verb, subj, obj, prep, count, total_count
- subj_pp_obj.tab.gz: verb, subj, prep, obj, count, total_count
- subj_pp_pp.tab.gz: verb, subj, prep1, prep2, count, total_count
"""

import gzip
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AffinityModels:
    """
    Loads and provides access to PP attachment affinity models.

    These models are used by the ForwardEntailer to determine deletion
    probabilities for prepositional phrases during natural logic inference.
    """

    def __init__(self, models_dir: str = None):
        """
        Load affinity models from directory.

        Args:
            models_dir: Directory containing .tab.gz affinity model files
                       If None, uses package's built-in models
        """
        if models_dir is None:
            # Use package-relative path
            models_dir = Path(__file__).parent.parent / "models" / "affinities"
        self.models_dir = Path(models_dir)

        # Dictionaries for different affinity types
        # Keys are tuples of (verb, prep, ...), values are probabilities
        self.pp_affinity: dict[tuple[str, str], float] = {}
        self.subj_pp_affinity: dict[tuple[str, str, str], float] = {}
        self.obj_affinity: dict[str, float] = {}
        self.subj_obj_pp_affinity: dict[tuple[str, str, str, str], float] = {}
        self.subj_pp_obj_affinity: dict[tuple[str, str, str, str], float] = {}
        self.subj_pp_pp_affinity: dict[tuple[str, str, str, str], float] = {}

        # Load all models
        self._load_models()

    def _load_model(self, filename: str, key_fields: int) -> dict:
        """
        Load a single affinity model from .tab.gz file.

        Args:
            filename: Name of .tab.gz file in models_dir
            key_fields: Number of fields in key tuple

        Returns:
            Dictionary mapping key tuple to probability
        """
        filepath = self.models_dir / filename
        affinities = {}

        try:
            with gzip.open(filepath, "rt") as f:
                for line in f:
                    fields = line.strip().split("\t")

                    if len(fields) < key_fields + 1:
                        continue  # Skip malformed lines

                    # Extract key fields
                    key = tuple(fields[:key_fields])

                    # The probability is in fields[key_fields]
                    # (following Stanford's Java code: Double.parseDouble(fields[2]) for pp.tab.gz)
                    try:
                        prob = float(fields[key_fields])
                    except (ValueError, IndexError):
                        # If it's an integer count, we need to divide by total
                        # Try interpreting as count / total_count
                        try:
                            count = int(fields[key_fields])
                            total = int(fields[key_fields + 1])
                            prob = count / total if total > 0 else 0.0
                        except Exception:
                            continue  # Skip this line

                    # Store affinity
                    affinities[key] = prob

        except FileNotFoundError:
            logger.warning("Affinity model not found: %s", filepath)
        except Exception as e:
            logger.error("Failed to load affinity model %s: %s", filename, e)

        return affinities

    def _load_models(self):
        """Load all affinity models."""
        logger.debug("Loading affinity models")

        # pp.tab.gz: (verb, prep) -> probability
        self.pp_affinity = self._load_model("pp.tab.gz", 2)
        logger.debug("Loaded %d PP affinities", len(self.pp_affinity))

        # subj_pp.tab.gz: (verb, subj, prep) -> probability
        self.subj_pp_affinity = self._load_model("subj_pp.tab.gz", 3)
        logger.debug("Loaded %d Subj-PP affinities", len(self.subj_pp_affinity))

        # obj.tab.gz: (verb,) -> probability
        obj_dict = self._load_model("obj.tab.gz", 1)
        self.obj_affinity = {k[0]: v for k, v in obj_dict.items()}
        logger.debug("Loaded %d Obj affinities", len(self.obj_affinity))

        # subj_obj_pp.tab.gz: (verb, subj, obj, prep) -> probability
        self.subj_obj_pp_affinity = self._load_model("subj_obj_pp.tab.gz", 4)
        logger.debug("Loaded %d Subj-Obj-PP affinities", len(self.subj_obj_pp_affinity))

        # subj_pp_obj.tab.gz: (verb, subj, prep, obj) -> probability
        self.subj_pp_obj_affinity = self._load_model("subj_pp_obj.tab.gz", 4)
        logger.debug("Loaded %d Subj-PP-Obj affinities", len(self.subj_pp_obj_affinity))

        # subj_pp_pp.tab.gz: (verb, subj, prep1, prep2) -> probability
        self.subj_pp_pp_affinity = self._load_model("subj_pp_pp.tab.gz", 4)
        logger.debug("Loaded %d Subj-PP-PP affinities", len(self.subj_pp_pp_affinity))

        logger.info("Affinity models loaded")

    def get_pp_affinity(self, verb: str, prep: str) -> float | None:
        """Get PP attachment probability for (verb, prep)."""
        return self.pp_affinity.get((verb, prep))

    def get_subj_pp_affinity(self, verb: str, subj: str, prep: str) -> float | None:
        """Get PP attachment probability for (verb, subj, prep)."""
        return self.subj_pp_affinity.get((verb, subj, prep))

    def get_obj_affinity(self, verb: str) -> float | None:
        """Get object deletion probability for verb."""
        return self.obj_affinity.get(verb)

    def get_subj_obj_pp_affinity(self, verb: str, subj: str, obj: str, prep: str) -> float | None:
        """Get PP attachment probability for (verb, subj, obj, prep)."""
        return self.subj_obj_pp_affinity.get((verb, subj, obj, prep))

    def get_subj_pp_obj_affinity(self, verb: str, subj: str, prep: str, obj: str) -> float | None:
        """Get PP attachment probability for (verb, subj, prep, obj)."""
        return self.subj_pp_obj_affinity.get((verb, subj, prep, obj))

    def get_subj_pp_pp_affinity(self, verb: str, subj: str, prep1: str, prep2: str) -> float | None:
        """Get PP attachment probability for (verb, subj, prep1, prep2)."""
        return self.subj_pp_pp_affinity.get((verb, subj, prep1, prep2))
