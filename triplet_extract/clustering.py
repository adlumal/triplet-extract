"""
Source-identity clustering for asserter and subject mentions.

Surface-string identity fragments a single source across its mentions:
"My friend Sarah", "my friend", "Sarah", and (once resolved) "she" are
one person, but compared as raw strings they are four. For downstream
consumers that track per-source reliability, that fragmentation is a bug.

This assigns a cluster id to mention strings using the lexicon-free,
high-precision STRUCTURAL sieves of Stanford CoreNLP's deterministic
coreference system (Lee et al., 2011), ported from Rules.java:

- Exact string match (entityExactStringMatch, Rules.java:331): identical
  normalized strings, including the possessive "'s" variant.
- Relaxed exact match (entityRelaxedExactStringMatch, Rules.java:359):
  proper-noun mentions equal after dropping the phrase following the head
  ("Mr. Bickford" ~ "Mr. Bickford, an 18-year mediation veteran").
- Head match with a compatible-modifier guard (entityHaveIncompatibleModifier,
  Rules.java): same head noun and no conflicting proper-noun modifier, so
  "my friend" merges with "My friend Sarah" but "my friend Sarah" does NOT
  merge with "my friend Mary".
- Shared proper-noun name: two mentions naming the same person ("Sarah"
  inside "My friend Sarah", and a later "Sarah") merge.

No lexicons. Operates on the emitted mention strings (re-parsed), which
sidesteps the reparsed-fragment token-identity problem and clusters
exactly the strings that appear in the output.

Metadata only: rendered triplet strings are never changed.
"""


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _mention_keys(parsed):
    """
    Derive matching keys for a parsed mention string:
    - full: normalized whole string
    - head: root token lemma (the noun the mention is about)
    - names: set of proper-noun token texts (lowercased)
    - common_head: True if the head is a common noun (eligible for the
      head-match sieve; proper-noun heads must match by name)
    """
    tokens = [t for t in parsed if not t.is_punct]
    full = _normalize(" ".join(t.text for t in tokens))
    names = {t.text.lower() for t in parsed if t.pos_ == "PROPN"}
    root = next((t for t in parsed if t.head.i == t.i), None)
    head = root.lemma_.lower() if root is not None else None
    common_head = root is not None and root.pos_ == "NOUN"
    return {"full": full, "head": head, "names": names, "common_head": common_head}


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _should_merge(a, b) -> bool:
    # Exact / possessive-variant match
    if a["full"] == b["full"]:
        return True
    if a["full"] == b["full"] + " 's" or b["full"] == a["full"] + " 's":
        return True

    shared_names = a["names"] & b["names"]
    conflicting_names = bool(a["names"] and b["names"] and not shared_names)

    # Shared proper-noun name (and no competing distinct names)
    if shared_names and not conflicting_names:
        return True

    # Head match with compatible-modifier guard: same common-noun head and
    # no conflicting proper-noun modifier
    if (
        a["head"]
        and a["head"] == b["head"]
        and (a["common_head"] or b["common_head"])
        and not conflicting_names
    ):
        return True

    return False


def cluster_mentions(strings, nlp) -> dict[str, int]:
    """
    Cluster mention strings by source identity.

    Args:
        strings: iterable of mention strings (subjects, asserter names)
        nlp: spaCy Language used to re-parse the mentions

    Returns:
        dict mapping each input string to a stable cluster id (small ints,
        assigned in first-appearance order).
    """
    unique = list(dict.fromkeys(s for s in strings if s and s.strip()))
    if not unique:
        return {}

    parsed = list(nlp.pipe(unique))
    keys = [_mention_keys(p) for p in parsed]

    uf = _UnionFind(len(unique))
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            if _should_merge(keys[i], keys[j]):
                uf.union(i, j)

    # Assign cluster ids in first-appearance order for stability
    root_to_id: dict[int, int] = {}
    result: dict[str, int] = {}
    for idx, s in enumerate(unique):
        root = uf.find(idx)
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id)
        result[s] = root_to_id[root]
    return result
