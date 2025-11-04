"""
Quantity normalizer for post-processing triplet text.

This module provides functions to normalize quantity expressions in extracted
triplet text, ensuring consistent formatting for percentages and scientific units.
"""

import re

# Percent and per-mille signs (no space between number and symbol)
PERCENT_PATTERN = re.compile(r"(\d+(?:[.,]\d+)?)\s*([%‰])")

# Common scientific/metric units (normalized with single space)
UNIT_NAMES = {
    # Mass
    "mg",
    "g",
    "kg",
    "µg",
    "ug",
    "ng",
    # Volume
    "ml",
    "mL",
    "l",
    "L",
    "µl",
    "uL",
    # Length
    "mm",
    "cm",
    "m",
    "km",
    # Frequency
    "hz",
    "Hz",
    "kHz",
    "MHz",
    "GHz",
    # Power/Energy
    "w",
    "W",
    "kw",
    "kW",
    "v",
    "V",
    "mv",
    "mV",
    "j",
    "J",
    "kj",
    "kJ",
    # Chemistry
    "mol",
    "mmol",
    "nM",
    "uM",
    "µM",
    "mM",
    # Pressure
    "mmHg",
    "Pa",
    "kPa",
    "bar",
}

UNIT_PATTERN = re.compile(r"(\d+(?:[.,]\d+)?)\s*([A-Za-zµμ][A-Za-zµμ/]*)")


def normalize_percent(text: str) -> str:
    """
    Remove space before percent signs.

    Examples:
        >>> normalize_percent("95 % of people")
        '95% of people'
        >>> normalize_percent("2.5 %")
        '2.5%'
        >>> normalize_percent("10.25 ‰")
        '10.25‰'

    Args:
        text: Input text with potential spaced percentages

    Returns:
        Text with percentages normalized (no space)
    """
    return PERCENT_PATTERN.sub(r"\1\2", text)


def normalize_units(text: str) -> str:
    """
    Add proper spacing for scientific units.

    Only normalizes known scientific/metric units from UNIT_NAMES.
    Leaves other numeric patterns (like "2024 study") unchanged.

    Examples:
        >>> normalize_units("10mg dose")
        '10 mg dose'
        >>> normalize_units("2.5 mL solution")
        '2.5 mL solution'
        >>> normalize_units("2024 study")
        '2024 study'

    Args:
        text: Input text with potential unit expressions

    Returns:
        Text with units normalized (single space)
    """

    def repl(m):
        num, unit = m.groups()
        # Check if it's a known unit (case-insensitive)
        if unit in UNIT_NAMES or unit.lower() in {u.lower() for u in UNIT_NAMES}:
            return f"{num} {unit}"
        return m.group(0)  # Leave unknown patterns unchanged

    return UNIT_PATTERN.sub(repl, text)


def normalize_quantities(text: str) -> str:
    """
    Normalize all quantity expressions in text.

    Applies percent and unit normalization to ensure consistent formatting
    of numeric expressions in extracted triplets.

    Examples:
        >>> normalize_quantities("95 % of people used 10mg")
        '95% of people used 10 mg'

    Args:
        text: Input text

    Returns:
        Text with normalized quantity expressions
    """
    text = normalize_percent(text)
    text = normalize_units(text)
    return text
