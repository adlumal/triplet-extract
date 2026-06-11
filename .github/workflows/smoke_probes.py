"""Behavioral smoke probes run against the BUILT WHEEL in a clean venv.

Shared by the release workflow (and mirroring release.sh step 5): a wheel
must demonstrate, before publication, the version it claims and one probe
per recent soundness class — negation polarity, asserter chains, quote
attribution. Run from a directory that does NOT contain the source tree,
so the installed package is what gets imported.
"""
import sys

import triplet_extract

expected = sys.argv[1]
assert triplet_extract.__version__ == expected, (
    f"wheel reports {triplet_extract.__version__}, expected {expected}")

from triplet_extract import OpenIEExtractor  # noqa: E402

ex = OpenIEExtractor()


def texts(s):
    return [f"{t.subject}|{t.relation}|{t.object}" for t in ex.extract_triplet_objects(s)]


# negation polarity: no rendering may drop a parsed negation
out = texts("Tom did not say that Sarah burned the pasta.")
assert not any(t.startswith("Tom|did|") for t in out), f"polarity drop: {out}"

# asserter chains: nested reported content carries its chain
trips = ex.extract_triplet_objects("Tom said Sarah claimed the chef burned the pasta.")
inner = [t for t in trips if t.subject == "the chef"]
assert inner and inner[0].asserter_chain == ["Tom", "Sarah"], "chains missing"

# quote attribution: quoted content must not be author-direct
trips = ex.extract_triplet_objects('"Sarah ate the cake," said Tom.')
quoted = [t for t in trips if t.subject == "Sarah" and "ate" in t.relation]
assert quoted and quoted[0].asserter_chain == ["Tom"], "quote attribution missing"

print("smoke probes passed:", triplet_extract.__version__)
