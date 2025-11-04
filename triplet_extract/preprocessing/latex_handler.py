"""LaTeX preprocessing for scientific text.

Handles LaTeX mathematical notation by replacing it with placeholders
that are treated as atomic nouns during extraction, then restoring the
original notation in the final output.
"""

import re


class LatexPreprocessor:
    """Preprocess and postprocess LaTeX notation in text."""

    def __init__(self):
        """Initialize the LaTeX preprocessor."""
        # Patterns for detecting LaTeX (ordered by precedence)
        self.patterns = [
            # Display math: $$...$$
            (r"\$\$(.*?)\$\$", "display"),
            # Inline math: $...$
            (r"\$(.*?)\$", "inline"),
            # LaTeX commands: \command{...}
            (r"\\[a-zA-Z]+\{[^}]*\}", "command"),
            # Just the command (no braces)
            (r"\\[a-zA-Z]+", "command_simple"),
        ]

    def preprocess(self, text: str) -> tuple[str, dict[str, str]]:
        """
        Replace LaTeX notation with placeholders.

        Args:
            text: Input text potentially containing LaTeX

        Returns:
            Tuple of (processed_text, latex_map)
            - processed_text: Text with placeholders
            - latex_map: Dict mapping placeholders to original LaTeX
        """
        latex_map = {}
        processed = text
        counter = 1

        # Process each pattern in order
        for pattern, _latex_type in self.patterns:
            matches = list(re.finditer(pattern, processed))

            # Process matches in reverse order to preserve indices
            for match in reversed(matches):
                original = match.group(0)
                placeholder = f"LATEX_MATH_{counter}"

                # Store mapping
                latex_map[placeholder] = original

                # Replace in text
                start, end = match.span()
                processed = processed[:start] + placeholder + processed[end:]

                counter += 1

        return processed, latex_map

    def postprocess_triplet(
        self, subject: str, relation: str, obj: str, latex_map: dict[str, str]
    ) -> tuple[str, str, str]:
        """
        Restore LaTeX notation in a single triplet.

        Args:
            subject: Subject string (may contain placeholders)
            relation: Relation string (may contain placeholders)
            obj: Object string (may contain placeholders)
            latex_map: Mapping of placeholders to original LaTeX

        Returns:
            Tuple of (subject, relation, object) with LaTeX restored
        """
        # Replace placeholders in each component
        for placeholder, latex in latex_map.items():
            subject = subject.replace(placeholder, latex)
            relation = relation.replace(placeholder, latex)
            obj = obj.replace(placeholder, latex)

        return subject, relation, obj

    def postprocess_triplets(self, triplets: list, latex_map: dict[str, str]) -> list:
        """
        Restore LaTeX notation in a list of triplets.

        Args:
            triplets: List of triplet objects with subject/relation/object
            latex_map: Mapping of placeholders to original LaTeX

        Returns:
            List of triplets with LaTeX restored
        """
        if not latex_map:
            return triplets  # No LaTeX to restore

        restored = []
        for triplet in triplets:
            # Restore in each component
            subject = triplet.subject
            relation = triplet.relation
            obj = triplet.object

            for placeholder, latex in latex_map.items():
                subject = subject.replace(placeholder, latex)
                relation = relation.replace(placeholder, latex)
                obj = obj.replace(placeholder, latex)

            # Create new triplet with restored LaTeX
            # (Assuming triplet is a dataclass or similar)
            restored_triplet = type(triplet)(
                subject=subject,
                relation=relation,
                object=obj,
                confidence=triplet.confidence if hasattr(triplet, "confidence") else 1.0,
                # Copy other attributes if they exist
                **{
                    k: getattr(triplet, k)
                    for k in dir(triplet)
                    if not k.startswith("_")
                    and k not in ["subject", "relation", "object", "confidence"]
                    and not callable(getattr(triplet, k))
                },
            )

            restored.append(restored_triplet)

        return restored


def test_latex_preprocessor():
    """Test the LaTeX preprocessor."""
    preprocessor = LatexPreprocessor()

    # Test cases
    test_texts = [
        "The value is $x + y$.",
        "Equation $$E = mc^2$$ is famous.",
        "We use \\ensuremath{\\alpha} here.",
        "Complex: $a$ and $$b^2$$ with \\beta.",
    ]

    for text in test_texts:
        print(f"\nOriginal: {text}")
        processed, latex_map = preprocessor.preprocess(text)
        print(f"Processed: {processed}")
        print(f"Mapping: {latex_map}")

        # Test restoration
        for placeholder, latex in latex_map.items():
            assert placeholder in processed
            assert latex not in processed  # Original LaTeX should be gone


if __name__ == "__main__":
    test_latex_preprocessor()
