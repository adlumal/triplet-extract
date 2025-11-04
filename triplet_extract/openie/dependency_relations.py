"""
Dependency Relation to Natural Logic Relation Mappings

Maps dependency arc types to natural logic relations for insertion and deletion.

Ports the insertArcToNaturalLogicRelation dictionary from:
NaturalLogicRelation.java (lines 201-533)
"""

from .polarity import NaturalLogicRelation

# Dependency relation to natural logic relation for INSERTION
# 326 mappings from Stanford CoreNLP
DEPENDENCY_INSERTION_RELATIONS: dict[str, NaturalLogicRelation] = {
    "acomp": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "advcl": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "acl": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "acl:relcl": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "advmod": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "agent": NaturalLogicRelation.INDEPENDENCE,
    "amod": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "appos": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "aux": NaturalLogicRelation.INDEPENDENCE,
    "aux:pass": NaturalLogicRelation.INDEPENDENCE,
    "auxpass": NaturalLogicRelation.INDEPENDENCE,
    "comp": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "ccomp": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "cc": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "compound": NaturalLogicRelation.INDEPENDENCE,
    "flat": NaturalLogicRelation.INDEPENDENCE,
    "mwe": NaturalLogicRelation.INDEPENDENCE,
    "conj:and/or": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "conj:and": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "conj:both": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "conj:but": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "conj:nor": NaturalLogicRelation.FORWARD_ENTAILMENT,
    "conj:or": NaturalLogicRelation.FORWARD_ENTAILMENT,
    "conj:plus": NaturalLogicRelation.FORWARD_ENTAILMENT,
    "conj": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "conj_x": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "cop": NaturalLogicRelation.INDEPENDENCE,
    "csubj": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "csubj:pass": NaturalLogicRelation.INDEPENDENCE,
    "csubjpass": NaturalLogicRelation.INDEPENDENCE,
    "dep": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "det": NaturalLogicRelation.FORWARD_ENTAILMENT,
    "discourse": NaturalLogicRelation.EQUIVALENT,
    "obj": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "dobj": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "expl": NaturalLogicRelation.EQUIVALENT,
    "goeswith": NaturalLogicRelation.EQUIVALENT,
    "infmod": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "iobj": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "mark": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "neg": NaturalLogicRelation.NEGATION,
    "nn": NaturalLogicRelation.INDEPENDENCE,
    "npadvmod": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nsubj": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nsubj:pass": NaturalLogicRelation.INDEPENDENCE,
    "nsubjpass": NaturalLogicRelation.INDEPENDENCE,
    "number": NaturalLogicRelation.INDEPENDENCE,
    "nummod": NaturalLogicRelation.INDEPENDENCE,
    "op": NaturalLogicRelation.INDEPENDENCE,
    "parataxis": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "partmod": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "pcomp": NaturalLogicRelation.INDEPENDENCE,
    "pobj": NaturalLogicRelation.INDEPENDENCE,
    "possessive": NaturalLogicRelation.INDEPENDENCE,
    "poss": NaturalLogicRelation.FORWARD_ENTAILMENT,
    "nmod:poss": NaturalLogicRelation.FORWARD_ENTAILMENT,
    "preconj": NaturalLogicRelation.INDEPENDENCE,
    "predet": NaturalLogicRelation.INDEPENDENCE,
    "case": NaturalLogicRelation.INDEPENDENCE,
    "prep": NaturalLogicRelation.INDEPENDENCE,
    # nmod relations (180+ variations)
    "nmod:aboard": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:about": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:above": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:according_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:across_from": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:across": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:after": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:against": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:ahead_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:along": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:alongside_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:alongside": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:along_with": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:amid": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:among": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:anti": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:apart_from": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:around": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:as_for": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:as_from": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:aside_from": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:as_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:as_per": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:as": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:as_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:at": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:away_from": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:based_on": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:because_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:before": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:behind": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:below": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:beneath": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:beside": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:besides": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:between": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:beyond": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:but": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:by_means_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:by": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:depending_on": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:dep": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:despite": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:down": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:due_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:during": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:en": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:except_for": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:excepting": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:except": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:excluding": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:exclusive_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:followed_by": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:following": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:for": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:from": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:if": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:in_accordance_with": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:in_addition_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:in_case_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:including": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:in_front_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:in_lieu_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:in_place_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:in": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:inside_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:inside": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:in_spite_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:instead_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:into": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:irrespective_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:like": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:minus": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:near": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:near_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:next_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:off_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:off": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:on_account_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:on_behalf_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:on": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:on_top_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:onto": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:opposite": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:out_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:out": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:outside_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:outside": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:over": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:owing_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:past": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:per": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:plus": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:preliminary_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:preparatory_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:previous_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:prior_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:pursuant_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:regarding": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:regardless_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:round": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:save": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:since": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:subsequent_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:such_as": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:thanks_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:than": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:throughout": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:through": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:together_with": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:toward": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:towards": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:underneath": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:under": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:unlike": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:until": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:upon": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:up": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:versus": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:via": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:vs.": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:whether": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:within": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:without": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:with_regard_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:with_respect_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "nmod:with": NaturalLogicRelation.REVERSE_ENTAILMENT,
    # obl relations (same as nmod - Universal Dependencies v2)
    "obl:aboard": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:about": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:above": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:according_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:across_from": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:across": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:after": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:against": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:ahead_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:along": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:alongside_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:alongside": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:along_with": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:amid": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:among": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:anti": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:apart_from": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:around": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:as_for": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:as_from": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:aside_from": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:as_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:as_per": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:as": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:as_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:at": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:away_from": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:based_on": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:because_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:before": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:behind": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:below": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:beneath": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:beside": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:besides": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:between": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:beyond": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:but": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:by_means_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:by": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:depending_on": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:dep": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:despite": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:down": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:due_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:during": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:en": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:except_for": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:excepting": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:except": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:excluding": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:exclusive_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:followed_by": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:following": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:for": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:from": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:if": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:in_accordance_with": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:in_addition_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:in_case_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:including": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:in_front_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:in_lieu_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:in_place_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:in": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:inside_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:inside": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:in_spite_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:instead_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:into": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:irrespective_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:like": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:minus": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:near": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:near_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:next_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:off_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:off": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:on_account_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:on_behalf_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:on": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:on_top_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:onto": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:opposite": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:out_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:out": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:outside_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:outside": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:over": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:owing_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:past": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:per": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:plus": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:preliminary_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:preparatory_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:previous_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:prior_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:pursuant_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:regarding": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:regardless_of": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:round": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:save": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:since": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:subsequent_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:such_as": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:thanks_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:than": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:throughout": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:through": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:together_with": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:toward": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:towards": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:underneath": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:under": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:unlike": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:until": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:upon": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:up": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:versus": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:via": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:vs.": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:whether": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:within": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:without": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:with_regard_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:with_respect_to": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "obl:with": NaturalLogicRelation.REVERSE_ENTAILMENT,
    # Other relations
    "prt": NaturalLogicRelation.INDEPENDENCE,
    "punct": NaturalLogicRelation.EQUIVALENT,
    "purpcl": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "quantmod": NaturalLogicRelation.FORWARD_ENTAILMENT,
    "ref": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "rcmod": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "root": NaturalLogicRelation.INDEPENDENCE,
    "tmod": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "vmod": NaturalLogicRelation.REVERSE_ENTAILMENT,
    "xcomp": NaturalLogicRelation.REVERSE_ENTAILMENT,
}


def for_dependency_insertion(
    dependency_label: str, is_subject: bool = True, dependent_word: str | None = None
) -> NaturalLogicRelation:
    """
    Get the natural logic relation for inserting a dependency arc.

    Ports NaturalLogicRelation.forDependencyInsertion()

    Args:
        dependency_label: Dependency relation (e.g., "nsubj", "dobj", "nmod:in")
        is_subject: Whether this is on the subject side (for conj:or/nor)
        dependent_word: The dependent word (for special cases like "neither")

    Returns:
        Natural logic relation for insertion
    """
    label_lower = dependency_label.lower()

    # Special case: "or" in object position behaves as "and"
    if not is_subject:
        if label_lower in ("conj:or", "conj:nor"):
            return for_dependency_insertion("conj:and", False)
        if label_lower == "cc:preconj":
            if dependent_word and dependent_word.lower() == "neither":
                return NaturalLogicRelation.INDEPENDENCE
            else:
                return NaturalLogicRelation.REVERSE_ENTAILMENT

    # Direct lookup
    if label_lower in DEPENDENCY_INSERTION_RELATIONS:
        return DEPENDENCY_INSERTION_RELATIONS[label_lower]

    # Fallback patterns
    if label_lower.startswith("nmod:"):
        return NaturalLogicRelation.REVERSE_ENTAILMENT
    elif label_lower.startswith("obl:"):
        return NaturalLogicRelation.REVERSE_ENTAILMENT
    elif label_lower.startswith("conj"):
        return NaturalLogicRelation.REVERSE_ENTAILMENT
    elif label_lower.startswith("advcl"):
        return NaturalLogicRelation.REVERSE_ENTAILMENT
    else:
        return NaturalLogicRelation.INDEPENDENCE


def insertion_to_deletion(insertion_rel: NaturalLogicRelation) -> NaturalLogicRelation:
    """
    Convert an insertion relation to a deletion relation.

    Deletion is the inverse of insertion:
    - FORWARD_ENTAILMENT <-> REVERSE_ENTAILMENT
    - ALTERNATION <-> COVER
    - Others remain the same

    Ports NaturalLogicRelation.insertionToDeletion()

    Args:
        insertion_rel: Natural logic relation for insertion

    Returns:
        Natural logic relation for deletion
    """
    if insertion_rel == NaturalLogicRelation.EQUIVALENT:
        return NaturalLogicRelation.EQUIVALENT
    elif insertion_rel == NaturalLogicRelation.FORWARD_ENTAILMENT:
        return NaturalLogicRelation.REVERSE_ENTAILMENT
    elif insertion_rel == NaturalLogicRelation.REVERSE_ENTAILMENT:
        return NaturalLogicRelation.FORWARD_ENTAILMENT
    elif insertion_rel == NaturalLogicRelation.NEGATION:
        return NaturalLogicRelation.NEGATION
    elif insertion_rel == NaturalLogicRelation.ALTERNATION:
        return NaturalLogicRelation.COVER
    elif insertion_rel == NaturalLogicRelation.COVER:
        return NaturalLogicRelation.ALTERNATION
    elif insertion_rel == NaturalLogicRelation.INDEPENDENCE:
        return NaturalLogicRelation.INDEPENDENCE
    else:
        raise ValueError(f"Unknown natural logic relation: {insertion_rel}")


def for_dependency_deletion(
    dependency_label: str, is_subject: bool = True, dependent_word: str | None = None
) -> NaturalLogicRelation:
    """
    Get the natural logic relation for deleting a dependency arc.

    Deletion relation = inverse of insertion relation.

    Ports NaturalLogicRelation.forDependencyDeletion()

    Args:
        dependency_label: Dependency relation (e.g., "nsubj", "dobj", "nmod:in")
        is_subject: Whether this is on the subject side (for conj:or/nor)
        dependent_word: The dependent word (for special cases)

    Returns:
        Natural logic relation for deletion
    """
    insertion_rel = for_dependency_insertion(dependency_label, is_subject, dependent_word)
    return insertion_to_deletion(insertion_rel)
