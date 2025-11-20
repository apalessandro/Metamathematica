"""
Propositional logic system with formula representation, enumeration, and visualization.

Main features:
- Formula AST: Var, Const, Not, And, Or, Implies with evaluation and variable extraction
- Formula enumeration: Generate formulas up to a given syntactic depth
- Semantic entailment: Check if axioms semantically entail a formula
- Logic graphs: Build directed graphs showing inference relationships
  * Syntactic derivability (build_syntactic_graph): Proof-based, forward-chaining with inference rules
  * Semantic entailment (build_semantic_graph): Truth-based, exhaustive formula enumeration
- Inference rules: modus ponens, modus tollens, disjunctive syllogism,
  hypothetical syllogism, conjunction elimination/introduction
- Interactive visualization: Export graphs to HTML with pyvis or display with matplotlib
- Boolean function utilities: Enumerate truth tables, convert to DNF, create truth table DataFrames

Key distinction:
- Syntactic ⊆ Semantic: Everything provable is semantically entailed, but not vice versa
- Syntactic: Shows derivation paths, focused (20-100 formulas typical)
- Semantic: Complete for propositional logic, comprehensive (1000s of formulas)
"""

from __future__ import annotations
from dataclasses import dataclass
from itertools import product, islice
from typing import Iterable, List, Sequence, Tuple, Dict, Set, Any
import pandas as pd
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import os


# ---------------------------------------------------------------------------
# Formula AST for propositional logic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Formula:
    def evaluate(self, valuation: Dict[str, int]) -> int:  # returns 0/1
        raise NotImplementedError

    def variables(self) -> Set[str]:
        raise NotImplementedError


@dataclass(frozen=True)
class Var(Formula):
    name: str

    def evaluate(self, valuation: Dict[str, int]) -> int:
        return int(valuation[self.name])

    def variables(self) -> Set[str]:
        return {self.name}

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Const(Formula):
    value: int  # 0 or 1

    def evaluate(self, valuation: Dict[str, int]) -> int:
        return self.value

    def variables(self) -> Set[str]:
        return set()

    def __str__(self) -> str:
        return "⊤" if self.value == 1 else "⊥"


@dataclass(frozen=True)
class Not(Formula):
    inner: Formula

    def evaluate(self, valuation: Dict[str, int]) -> int:
        return 1 - self.inner.evaluate(valuation)

    def variables(self) -> Set[str]:
        return self.inner.variables()

    def __str__(self) -> str:
        return f"¬({self.inner})"


@dataclass(frozen=True)
class And(Formula):
    left: Formula
    right: Formula

    def evaluate(self, valuation: Dict[str, int]) -> int:
        return self.left.evaluate(valuation) & self.right.evaluate(valuation)

    def variables(self) -> Set[str]:
        return self.left.variables() | self.right.variables()

    def __str__(self) -> str:
        return f"({self.left} ∧ {self.right})"


@dataclass(frozen=True)
class Or(Formula):
    left: Formula
    right: Formula

    def evaluate(self, valuation: Dict[str, int]) -> int:
        return self.left.evaluate(valuation) | self.right.evaluate(valuation)

    def variables(self) -> Set[str]:
        return self.left.variables() | self.right.variables()

    def __str__(self) -> str:
        return f"({self.left} ∨ {self.right})"


@dataclass(frozen=True)
class Implies(Formula):
    antecedent: Formula
    consequent: Formula

    def evaluate(self, valuation: Dict[str, int]) -> int:
        a = self.antecedent.evaluate(valuation)
        b = self.consequent.evaluate(valuation)
        # a -> b is equivalent to ¬a ∨ b
        return (1 - a) | b

    def variables(self) -> Set[str]:
        return self.antecedent.variables() | self.consequent.variables()

    def __str__(self) -> str:
        return f"({self.antecedent} → {self.consequent})"


def all_valuations(var_names: Sequence[str]) -> List[Dict[str, int]]:
    """Generate all valuations (dict variable -> 0/1) for given variables."""
    return [dict(zip(var_names, vals)) for vals in all_inputs(len(var_names))]


def formula_truth_vector(formula: Formula, var_names: Sequence[str]) -> Tuple[int, ...]:
    """Return truth vector of a formula over ordered valuations of var_names."""
    return tuple(formula.evaluate(v) for v in all_valuations(var_names))


def enumerate_formulas(var_names: Sequence[str], max_depth: int) -> List[Formula]:
    """Enumerate formulas up to given syntactic depth.

    Depth counts nesting of operators. Variables and constants have depth 0.
    This grows quickly; keep max_depth small (e.g., 2 or 3).
    """
    # depth -> set[Formula]
    levels: List[Set[Formula]] = []
    base: Set[Formula] = set([Var(name) for name in var_names])
    levels.append(base)

    def combine(a: Formula, b: Formula) -> List[Formula]:
        return [And(a, b), Or(a, b), Implies(a, b)]

    for d in range(1, max_depth + 1):
        new_set: Set[Formula] = set()
        # Unary NOT applied to any formula of depth < d
        for prev in levels[d - 1]:
            new_set.add(Not(prev))
        # Binary operations combining formulas whose max subdepth < d
        # We allow reuse of earlier depths for combinatorial richness.
        all_previous = set.union(*levels[:d])
        for a in all_previous:
            for b in all_previous:
                for f in combine(a, b):
                    new_set.add(f)
        levels.append(new_set)

    # Aggregate unique by string representation to filter syntactic duplicates
    unique: Dict[str, Formula] = {}
    for s in levels:
        for f in s:
            unique[str(f)] = f
    return list(unique.values())


def satisfying_valuations(
    axioms: Sequence[Formula], var_names: Sequence[str]
) -> List[Dict[str, int]]:
    """Return valuations that make all axioms true."""
    vals = all_valuations(var_names)
    res = [v for v in vals if all(a.evaluate(v) == 1 for a in axioms)]
    return res


def entails(
    axioms: Sequence[Formula], formula: Formula, var_names: Sequence[str]
) -> bool:
    """Semantic entailment: axioms |= formula.

    If no valuation satisfies axioms (inconsistent), every formula is entailed.
    """
    sat = satisfying_valuations(axioms, var_names)
    if not sat:
        return True  # explosion from inconsistency
    return all(formula.evaluate(v) == 1 for v in sat)


def _sample_formulas(
    formulas: List[Formula],
    axiom_set: Set[str],
    var_names: Sequence[str],
    axioms: Sequence[Formula],
    max_nodes: int,
) -> List[Formula]:
    """Intelligently sample formulas to keep the most interesting ones.

    Priority order:
    1. Axioms (always included)
    2. Entailed formulas (consequences of axioms)
    3. Tautologies
    4. Formulas with shorter string representation (simpler)
    5. Random sample of the rest
    """
    # Categorize formulas
    axiom_formulas = []
    entailed_formulas = []
    tautology_formulas = []
    other_formulas = []

    for f in formulas:
        f_str = str(f)
        tv = formula_truth_vector(f, var_names)
        is_taut = all(bit == 1 for bit in tv)
        is_ent = entails(axioms, f, var_names)

        if f_str in axiom_set:
            axiom_formulas.append(f)
        elif is_ent:
            entailed_formulas.append(f)
        elif is_taut:
            tautology_formulas.append(f)
        else:
            other_formulas.append(f)

    # Sort each category by simplicity (shorter string = simpler)
    entailed_formulas.sort(key=lambda f: len(str(f)))
    tautology_formulas.sort(key=lambda f: len(str(f)))
    other_formulas.sort(key=lambda f: len(str(f)))

    # Build result prioritizing categories
    result = []
    result.extend(axiom_formulas)  # Always include all axioms

    remaining = max_nodes - len(result)
    if remaining <= 0:
        return result

    # Take entailed formulas (most important)
    take_entailed = min(remaining, len(entailed_formulas))
    result.extend(entailed_formulas[:take_entailed])
    remaining -= take_entailed

    if remaining <= 0:
        return result

    # Take some tautologies
    take_taut = min(remaining // 3, len(tautology_formulas))
    result.extend(tautology_formulas[:take_taut])
    remaining -= take_taut

    if remaining <= 0:
        return result

    # Reserve space for false statements (at least 20% of remaining or 5, whichever is larger)
    min_false = max(5, remaining // 5)
    take_false = min(min_false, len(other_formulas), remaining)

    # Fill remaining with simplest other formulas (false statements)
    result.extend(other_formulas[:take_false])

    print(
        f"📊 Sampled {len(result)} formulas from {len(formulas)}: "
        f"{len(axiom_formulas)} axioms, {take_entailed} entailed, "
        f"{take_taut} tautologies, {take_false} false statements"
    )

    return result


def build_semantic_graph(
    var_names: Sequence[str],
    axioms: Sequence[Formula],
    max_depth: int = 2,
    max_nodes: int | None = None,
    rules: Set[str] | None = None,
) -> Any:
    """Build a directed graph using SEMANTIC ENTAILMENT.

    This approach generates all possible formulas up to a depth, then checks which ones
    are semantically entailed (true in all models where axioms are true).

    Semantic entailment: A formula is entailed if it's true in every truth assignment
    that satisfies all axioms. This is complete for propositional logic but generates
    many more formulas than syntactic derivation.

    Nodes: all enumerated formulas (or sampled subset if max_nodes is set).
    Node attributes:
        truth_vector: tuple of 0/1 over all valuations
        tautology: bool (true on all valuations)
        entailed: bool (semantic consequence of axioms)

    Edges:
        - From each axiom node to entailed formulas (that are not axioms themselves).
        - Inference edges: modus ponens, modus tollens, disjunctive syllogism,
          hypothetical syllogism, conjunction elimination, disjunction introduction.

    Parameters
    ----------
    max_depth:
        Maximum syntactic depth for formula enumeration.
    max_nodes:
        If set, intelligently sample the graph to keep only the most interesting nodes.
        Prioritizes: axioms, entailed formulas, tautologies, simpler formulas.
    rules:
        Set of rule names to apply for edges. If None, applies all available rules.
        Available rules: 'MP', 'MT', 'DS', 'HS', '∧E', '∧I', '∨I', 'DNE',
        'CP', 'MI', 'RMI', 'DeM', 'RDeM', 'LEM', 'ID'
    """
    # Default: use all rules if none specified
    if rules is None:
        rules = {
            "MP",
            "MT",
            "DS",
            "HS",
            "∧E",
            "∧I",
            "∨I",
            "DNE",
            "CP",
            "MI",
            "RMI",
            "DeM",
            "RDeM",
            "LEM",
            "ID",
        }

    formulas = enumerate_formulas(var_names, max_depth)
    axiom_set = {str(a) for a in axioms}

    # If max_nodes is set and formula count exceeds it, do smart sampling
    if max_nodes and len(formulas) > max_nodes:
        formulas = _sample_formulas(formulas, axiom_set, var_names, axioms, max_nodes)

    g = nx.DiGraph()

    for f in formulas:
        tv = formula_truth_vector(f, var_names)
        taut = all(bit == 1 for bit in tv)
        ent = entails(axioms, f, var_names)
        g.add_node(
            str(f),
            truth_vector=tv,
            tautology=taut,
            entailed=ent,
            is_axiom=str(f) in axiom_set,
        )

    # Add inference edges
    for f in formulas:
        # Modus Ponens: (A → B), A ⊢ B
        if "MP" in rules and isinstance(f, Implies):
            impl_str = str(f)
            ant_str = str(f.antecedent)
            cons_str = str(f.consequent)
            if (
                impl_str in g
                and ant_str in g
                and cons_str in g
                and g.nodes[impl_str]["entailed"]
                and g.nodes[ant_str]["entailed"]
                and not g.nodes[cons_str]["is_axiom"]  # Don't point to axioms
            ):
                g.add_edge(ant_str, cons_str, reason="MP", rule="modus_ponens")

        # Modus Tollens: (A → B), ¬B ⊢ ¬A
        if "MT" in rules and isinstance(f, Implies):
            impl_str = str(f)
            not_b_str = str(Not(f.consequent))
            not_a_str = str(Not(f.antecedent))
            if (
                impl_str in g
                and not_b_str in g
                and not_a_str in g
                and g.nodes[impl_str]["entailed"]
                and g.nodes[not_b_str]["entailed"]
                and not g.nodes[not_a_str]["is_axiom"]  # Don't point to axioms
            ):
                g.add_edge(not_b_str, not_a_str, reason="MT", rule="modus_tollens")

        # Disjunctive Syllogism: (A ∨ B), ¬A ⊢ B
        if "DS" in rules and isinstance(f, Or):
            or_str = str(f)
            not_a_str = str(Not(f.left))
            b_str = str(f.right)
            if (
                or_str in g
                and not_a_str in g
                and b_str in g
                and g.nodes[or_str]["entailed"]
                and g.nodes[not_a_str]["entailed"]
                and not g.nodes[b_str]["is_axiom"]  # Don't point to axioms
            ):
                g.add_edge(not_a_str, b_str, reason="DS", rule="disjunctive_syllogism")

            # Symmetric: (A ∨ B), ¬B ⊢ A
            not_b_str = str(Not(f.right))
            a_str = str(f.left)
            if (
                or_str in g
                and not_b_str in g
                and a_str in g
                and g.nodes[or_str]["entailed"]
                and g.nodes[not_b_str]["entailed"]
                and not g.nodes[a_str]["is_axiom"]  # Don't point to axioms
            ):
                g.add_edge(not_b_str, a_str, reason="DS", rule="disjunctive_syllogism")

        # Hypothetical Syllogism: (A → B), (B → C) ⊢ (A → C)
        if "HS" in rules:
            for f2 in formulas:
                if isinstance(f, Implies) and isinstance(f2, Implies):
                    if str(f.consequent) == str(f2.antecedent):
                        impl1_str = str(f)
                        impl2_str = str(f2)
                        concl_str = str(Implies(f.antecedent, f2.consequent))
                        if (
                            impl1_str in g
                            and impl2_str in g
                            and concl_str in g
                            and g.nodes[impl1_str]["entailed"]
                            and g.nodes[impl2_str]["entailed"]
                            and not g.nodes[concl_str][
                                "is_axiom"
                            ]  # Don't point to axioms
                        ):
                            g.add_edge(
                                impl1_str,
                                concl_str,
                                reason="HS",
                                rule="hypothetical_syllogism",
                            )

        # Conjunction Elimination: (A ∧ B) ⊢ A, (A ∧ B) ⊢ B
        if "∧E" in rules and isinstance(f, And):
            and_str = str(f)
            left_str = str(f.left)
            right_str = str(f.right)
            if (
                and_str in g
                and left_str in g
                and g.nodes[and_str]["entailed"]
                and not g.nodes[left_str]["is_axiom"]
            ):
                g.add_edge(
                    and_str, left_str, reason="∧E-L", rule="conjunction_elim_left"
                )
            if (
                and_str in g
                and right_str in g
                and g.nodes[and_str]["entailed"]
                and not g.nodes[right_str]["is_axiom"]
            ):
                g.add_edge(
                    and_str, right_str, reason="∧E-R", rule="conjunction_elim_right"
                )

    # Add additional inference rules for semantic graph
    for f in formulas:
        for f2 in formulas:
            # Conjunction Introduction: A, B ⊢ (A ∧ B)
            if "∧I" in rules:
                and_formula = And(f, f2)
                and_str = str(and_formula)
                f_str = str(f)
                f2_str = str(f2)
                if (
                    and_str in g
                    and f_str in g
                    and f2_str in g
                    and g.nodes[f_str]["entailed"]
                    and g.nodes[f2_str]["entailed"]
                    and g.nodes[and_str]["entailed"]
                    and not g.nodes[and_str]["is_axiom"]
                ):
                    g.add_edge(f_str, and_str, reason="∧I", rule="conjunction_intro")
                    if f_str != f2_str:
                        g.add_edge(
                            f2_str, and_str, reason="∧I", rule="conjunction_intro"
                        )

            # Disjunction Introduction: A ⊢ (A ∨ B)
            if "∨I" in rules:
                or_formula = Or(f, f2)
                or_str = str(or_formula)
                f_str = str(f)
                if (
                    or_str in g
                    and f_str in g
                    and g.nodes[f_str]["entailed"]
                    and g.nodes[or_str]["entailed"]
                    and not g.nodes[or_str]["is_axiom"]
                ):
                    g.add_edge(f_str, or_str, reason="∨I", rule="disjunction_intro")

        # Double Negation Elimination: ¬¬A ⊢ A
        if "DNE" in rules and isinstance(f, Not) and isinstance(f.inner, Not):
            dne_str = str(f)
            inner_str = str(f.inner.inner)
            if (
                dne_str in g
                and inner_str in g
                and g.nodes[dne_str]["entailed"]
                and not g.nodes[inner_str]["is_axiom"]
            ):
                g.add_edge(
                    dne_str, inner_str, reason="DNE", rule="double_negation_elim"
                )

        # Contraposition: (A → B) ⊢ (¬B → ¬A)
        if "CP" in rules and isinstance(f, Implies):
            contra = Implies(Not(f.consequent), Not(f.antecedent))
            impl_str = str(f)
            contra_str = str(contra)
            if (
                impl_str in g
                and contra_str in g
                and g.nodes[impl_str]["entailed"]
                and not g.nodes[contra_str]["is_axiom"]
            ):
                g.add_edge(impl_str, contra_str, reason="CP", rule="contraposition")

        # Material Implication: (A → B) ⊢ (¬A ∨ B)
        if "MI" in rules and isinstance(f, Implies):
            mat_impl = Or(Not(f.antecedent), f.consequent)
            impl_str = str(f)
            mat_str = str(mat_impl)
            if (
                impl_str in g
                and mat_str in g
                and g.nodes[impl_str]["entailed"]
                and not g.nodes[mat_str]["is_axiom"]
            ):
                g.add_edge(impl_str, mat_str, reason="MI", rule="material_implication")

        # Reverse Material Implication: (¬A ∨ B) ⊢ (A → B)
        if "RMI" in rules and isinstance(f, Or) and isinstance(f.left, Not):
            rev_mat = Implies(f.left.inner, f.right)
            or_str = str(f)
            rev_str = str(rev_mat)
            if (
                or_str in g
                and rev_str in g
                and g.nodes[or_str]["entailed"]
                and not g.nodes[rev_str]["is_axiom"]
            ):
                g.add_edge(or_str, rev_str, reason="RMI", rule="reverse_material_impl")

        # De Morgan's Laws: ¬(A ∧ B) ⊢ (¬A ∨ ¬B)
        if "DeM" in rules and isinstance(f, Not) and isinstance(f.inner, And):
            dem_result = Or(Not(f.inner.left), Not(f.inner.right))
            not_and_str = str(f)
            dem_str = str(dem_result)
            if (
                not_and_str in g
                and dem_str in g
                and g.nodes[not_and_str]["entailed"]
                and not g.nodes[dem_str]["is_axiom"]
            ):
                g.add_edge(not_and_str, dem_str, reason="DeM", rule="de_morgan_and")

        # De Morgan's Laws: ¬(A ∨ B) ⊢ (¬A ∧ ¬B)
        if "DeM" in rules and isinstance(f, Not) and isinstance(f.inner, Or):
            dem_result = And(Not(f.inner.left), Not(f.inner.right))
            not_or_str = str(f)
            dem_str = str(dem_result)
            if (
                not_or_str in g
                and dem_str in g
                and g.nodes[not_or_str]["entailed"]
                and not g.nodes[dem_str]["is_axiom"]
            ):
                g.add_edge(not_or_str, dem_str, reason="DeM", rule="de_morgan_or")

        # Reverse De Morgan: (¬A ∨ ¬B) ⊢ ¬(A ∧ B)
        if (
            "RDeM" in rules
            and isinstance(f, Or)
            and isinstance(f.left, Not)
            and isinstance(f.right, Not)
        ):
            rdem_result = Not(And(f.left.inner, f.right.inner))
            or_not_str = str(f)
            rdem_str = str(rdem_result)
            if (
                or_not_str in g
                and rdem_str in g
                and g.nodes[or_not_str]["entailed"]
                and not g.nodes[rdem_str]["is_axiom"]
            ):
                g.add_edge(
                    or_not_str, rdem_str, reason="RDeM", rule="reverse_de_morgan_and"
                )

        # Reverse De Morgan: (¬A ∧ ¬B) ⊢ ¬(A ∨ B)
        if (
            "RDeM" in rules
            and isinstance(f, And)
            and isinstance(f.left, Not)
            and isinstance(f.right, Not)
        ):
            rdem_result = Not(Or(f.left.inner, f.right.inner))
            and_not_str = str(f)
            rdem_str = str(rdem_result)
            if (
                and_not_str in g
                and rdem_str in g
                and g.nodes[and_not_str]["entailed"]
                and not g.nodes[rdem_str]["is_axiom"]
            ):
                g.add_edge(
                    and_not_str, rdem_str, reason="RDeM", rule="reverse_de_morgan_or"
                )

    # Add tautologies if LEM or ID rules are enabled
    if "LEM" in rules or "ID" in rules:
        for var_name in var_names:
            v = Var(var_name)
            # Law of Excluded Middle: ⊢ (A ∨ ¬A)
            if "LEM" in rules:
                lem = Or(v, Not(v))
                lem_str = str(lem)
                if lem_str in g and g.nodes[lem_str]["tautology"]:
                    # LEM is a tautology, no premises needed
                    pass

            # Law of Identity: ⊢ (A → A)
            if "ID" in rules:
                id_formula = Implies(v, v)
                id_str = str(id_formula)
                if id_str in g and g.nodes[id_str]["tautology"]:
                    # ID is a tautology, no premises needed
                    pass

    return g


def visualize_logic_graph(
    g: Any,
    layout: str = "spring",
    figsize: Tuple[int, int] = (10, 8),
    show_truth: bool = False,
) -> None:
    """Visualize logic graph using matplotlib."""

    if layout == "spring":
        pos = nx.spring_layout(g, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(g)
    else:
        pos = nx.circular_layout(g)

    plt.figure(figsize=figsize)
    node_colors = []
    labels: Dict[str, str] = {}
    for node, data in g.nodes(data=True):
        if data.get("is_axiom"):
            node_colors.append("orange")
        elif data.get("tautology"):
            node_colors.append("lightgreen")
        elif data.get("entailed"):
            node_colors.append("skyblue")
        else:
            node_colors.append("lightgray")
        if show_truth:
            labels[node] = f"{node}\n{''.join(map(str, data.get('truth_vector', [])))}"
        else:
            labels[node] = node

    nx.draw_networkx_nodes(g, pos, node_color=node_colors, edgecolors="black")
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=8)
    edge_colors = [
        "red" if d.get("reason") == "MP" else "black" for _, _, d in g.edges(data=True)
    ]
    nx.draw_networkx_edges(g, pos, edge_color=edge_colors, arrows=True, arrowsize=10)
    mp_edges = sum(1 for _, _, d in g.edges(data=True) if d.get("reason") == "MP")
    plt.title(f"Logic Graph (MP edges: {mp_edges})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def get_available_rules() -> Dict[str, str]:
    """Return a dictionary of available inference rules and their descriptions.

    Returns
    -------
    dict
        Mapping from rule abbreviation to full description.
    """
    return {
        "MP": "Modus Ponens: (A → B), A ⊢ B",
        "MT": "Modus Tollens: (A → B), ¬B ⊢ ¬A",
        "DS": "Disjunctive Syllogism: (A ∨ B), ¬A ⊢ B",
        "HS": "Hypothetical Syllogism: (A → B), (B → C) ⊢ (A → C)",
        "∧E": "Conjunction Elimination: (A ∧ B) ⊢ A, (A ∧ B) ⊢ B",
        "∧I": "Conjunction Introduction: A, B ⊢ (A ∧ B)",
        "∨I": "Disjunction Introduction: A ⊢ (A ∨ B)",
        "DNE": "Double Negation Elimination: ¬¬A ⊢ A",
        "CP": "Contraposition: (A → B) ⊢ (¬B → ¬A)",
        "MI": "Material Implication: (A → B) ⊢ (¬A ∨ B)",
        "RMI": "Reverse Material Implication: (¬A ∨ B) ⊢ (A → B)",
        "DeM": "De Morgan's Laws: ¬(A ∧ B) ⊢ (¬A ∨ ¬B), ¬(A ∨ B) ⊢ (¬A ∧ ¬B)",
        "RDeM": "Reverse De Morgan's Laws: (¬A ∨ ¬B) ⊢ ¬(A ∧ B), (¬A ∧ ¬B) ⊢ ¬(A ∨ B)",
        "LEM": "Law of Excluded Middle: ⊢ (A ∨ ¬A)",
        "ID": "Law of Identity: ⊢ (A → A)",
    }


def get_rule_presets() -> Dict[str, Set[str]]:
    """Return predefined sets of inference rules for common use cases.

    Returns
    -------
    dict
        Mapping from preset name to set of rule abbreviations.
    """
    return {
        "minimal": {"MP", "∧E", "∧I"},
        "classical": {"MP", "MT", "DS", "HS", "∧E", "∧I"},
        "extended": {"MP", "MT", "DS", "HS", "∧E", "∧I", "∨I", "DNE", "CP"},
        "complete": {
            "MP",
            "MT",
            "DS",
            "HS",
            "∧E",
            "∧I",
            "∨I",
            "DNE",
            "CP",
            "MI",
            "RMI",
            "DeM",
            "RDeM",
        },
        "all": {
            "MP",
            "MT",
            "DS",
            "HS",
            "∧E",
            "∧I",
            "∨I",
            "DNE",
            "CP",
            "MI",
            "RMI",
            "DeM",
            "RDeM",
            "LEM",
            "ID",
        },
    }


def build_syntactic_graph(
    var_names: Sequence[str],
    axioms: Sequence[Formula],
    max_iterations: int = 3,
    rules: Set[str] | None = None,
) -> Any:
    """Build a directed graph using SYNTACTIC DERIVABILITY (forward-chaining).

    Start with axioms and iteratively apply inference rules to generate
    new statements that can be proven step-by-step.

    Syntactic derivability: A formula is derivable if it can be proven by applying
    a finite sequence of inference rules. This is focused and efficient but may not
    derive all semantically entailed formulas (incomplete for propositional logic).

    Parameters
    ----------
    var_names:
        List of variable names used in formulas.
    axioms:
        Starting formulas (axioms).
    max_iterations:
        Maximum number of iterations for rule application.
        Each iteration applies all possible rules to current knowledge base.
    rules:
        Set of rule names to apply. If None, applies all available rules.
        Available rules: 'MP', 'MT', 'DS', 'HS', '∧E', '∧I', '∨I', 'DNE',
        'CP', 'MI', 'RMI', 'DeM', 'RDeM', 'LEM', 'ID'

    Returns
    -------
    NetworkX DiGraph with nodes representing formulas and edges showing
    inference steps from premises to conclusions.
    """
    # Default: use all rules if none specified
    if rules is None:
        rules = {
            "MP",
            "MT",
            "DS",
            "HS",
            "∧E",
            "∧I",
            "∨I",
            "DNE",
            "CP",
            "MI",
            "RMI",
            "DeM",
            "RDeM",
            "LEM",
            "ID",
        }
    g = nx.DiGraph()

    # Initialize with axioms
    knowledge_base: Set[Formula] = set(axioms)

    # Add axiom nodes
    for axiom in axioms:
        tv = formula_truth_vector(axiom, var_names)
        g.add_node(
            str(axiom),
            truth_vector=tv,
            tautology=all(bit == 1 for bit in tv),
            entailed=True,
            is_axiom=True,
            generation=0,
        )

    # Iteratively apply rules
    for iteration in range(max_iterations):
        iteration_new: Set[Formula] = set()

        # Apply inference rules to formulas in knowledge base
        for f in knowledge_base:
            # Modus Ponens: If we have (A → B) and A, derive B
            if "MP" in rules and isinstance(f, Implies):
                if f.antecedent in knowledge_base:
                    conclusion = f.consequent
                    if conclusion not in knowledge_base:
                        iteration_new.add(conclusion)
                        # Add to graph
                        tv = formula_truth_vector(conclusion, var_names)
                        g.add_node(
                            str(conclusion),
                            truth_vector=tv,
                            tautology=all(bit == 1 for bit in tv),
                            entailed=True,
                            is_axiom=False,
                            generation=iteration + 1,
                        )
                        # Add edge showing the inference
                        g.add_edge(
                            str(f.antecedent),
                            str(conclusion),
                            reason="MP",
                            rule="modus_ponens",
                            via=str(f),
                        )

            # Conjunction Elimination: If we have (A ∧ B), derive A and B
            if "∧E" in rules and isinstance(f, And):
                for part, label in [(f.left, "∧E-L"), (f.right, "∧E-R")]:
                    if part not in knowledge_base:
                        iteration_new.add(part)
                        tv = formula_truth_vector(part, var_names)
                        g.add_node(
                            str(part),
                            truth_vector=tv,
                            tautology=all(bit == 1 for bit in tv),
                            entailed=True,
                            is_axiom=False,
                            generation=iteration + 1,
                        )
                        g.add_edge(
                            str(f),
                            str(part),
                            reason=label,
                            rule=f"conjunction_elim_{'left' if label == '∧E-L' else 'right'}",
                        )

        # Two-formula rules (need to check pairs)
        kb_list = list(knowledge_base)
        for i, f1 in enumerate(kb_list):
            for f2 in kb_list[i:]:
                # Modus Tollens: (A → B) and ¬B, derive ¬A
                if "MT" in rules and isinstance(f1, Implies) and isinstance(f2, Not):
                    if f2.inner == f1.consequent:
                        conclusion = Not(f1.antecedent)
                        if conclusion not in knowledge_base:
                            iteration_new.add(conclusion)
                            tv = formula_truth_vector(conclusion, var_names)
                            g.add_node(
                                str(conclusion),
                                truth_vector=tv,
                                tautology=all(bit == 1 for bit in tv),
                                entailed=True,
                                is_axiom=False,
                                generation=iteration + 1,
                            )
                            g.add_edge(
                                str(f2),
                                str(conclusion),
                                reason="MT",
                                rule="modus_tollens",
                                via=str(f1),
                            )

                # Symmetric case for MT
                if "MT" in rules and isinstance(f2, Implies) and isinstance(f1, Not):
                    if f1.inner == f2.consequent:
                        conclusion = Not(f2.antecedent)
                        if conclusion not in knowledge_base:
                            iteration_new.add(conclusion)
                            tv = formula_truth_vector(conclusion, var_names)
                            g.add_node(
                                str(conclusion),
                                truth_vector=tv,
                                tautology=all(bit == 1 for bit in tv),
                                entailed=True,
                                is_axiom=False,
                                generation=iteration + 1,
                            )
                            g.add_edge(
                                str(f1),
                                str(conclusion),
                                reason="MT",
                                rule="modus_tollens",
                                via=str(f2),
                            )

                # Disjunctive Syllogism: (A ∨ B) and ¬A, derive B
                if "DS" in rules and isinstance(f1, Or) and isinstance(f2, Not):
                    if f2.inner == f1.left:
                        conclusion = f1.right
                        if conclusion not in knowledge_base:
                            iteration_new.add(conclusion)
                            tv = formula_truth_vector(conclusion, var_names)
                            g.add_node(
                                str(conclusion),
                                truth_vector=tv,
                                tautology=all(bit == 1 for bit in tv),
                                entailed=True,
                                is_axiom=False,
                                generation=iteration + 1,
                            )
                            g.add_edge(
                                str(f2),
                                str(conclusion),
                                reason="DS",
                                rule="disjunctive_syllogism",
                                via=str(f1),
                            )
                    elif f2.inner == f1.right:
                        conclusion = f1.left
                        if conclusion not in knowledge_base:
                            iteration_new.add(conclusion)
                            tv = formula_truth_vector(conclusion, var_names)
                            g.add_node(
                                str(conclusion),
                                truth_vector=tv,
                                tautology=all(bit == 1 for bit in tv),
                                entailed=True,
                                is_axiom=False,
                                generation=iteration + 1,
                            )
                            g.add_edge(
                                str(f2),
                                str(conclusion),
                                reason="DS",
                                rule="disjunctive_syllogism",
                                via=str(f1),
                            )

                # Symmetric cases for DS
                if "DS" in rules and isinstance(f2, Or) and isinstance(f1, Not):
                    if f1.inner == f2.left:
                        conclusion = f2.right
                        if conclusion not in knowledge_base:
                            iteration_new.add(conclusion)
                            tv = formula_truth_vector(conclusion, var_names)
                            g.add_node(
                                str(conclusion),
                                truth_vector=tv,
                                tautology=all(bit == 1 for bit in tv),
                                entailed=True,
                                is_axiom=False,
                                generation=iteration + 1,
                            )
                            g.add_edge(
                                str(f1),
                                str(conclusion),
                                reason="DS",
                                rule="disjunctive_syllogism",
                                via=str(f2),
                            )
                    elif f1.inner == f2.right:
                        conclusion = f2.left
                        if conclusion not in knowledge_base:
                            iteration_new.add(conclusion)
                            tv = formula_truth_vector(conclusion, var_names)
                            g.add_node(
                                str(conclusion),
                                truth_vector=tv,
                                tautology=all(bit == 1 for bit in tv),
                                entailed=True,
                                is_axiom=False,
                                generation=iteration + 1,
                            )
                            g.add_edge(
                                str(f1),
                                str(conclusion),
                                reason="DS",
                                rule="disjunctive_syllogism",
                                via=str(f2),
                            )

                # Hypothetical Syllogism: (A → B) and (B → C), derive (A → C)
                if (
                    "HS" in rules
                    and isinstance(f1, Implies)
                    and isinstance(f2, Implies)
                ):
                    if f1.consequent == f2.antecedent:
                        conclusion = Implies(f1.antecedent, f2.consequent)
                        if conclusion not in knowledge_base:
                            iteration_new.add(conclusion)
                            tv = formula_truth_vector(conclusion, var_names)
                            g.add_node(
                                str(conclusion),
                                truth_vector=tv,
                                tautology=all(bit == 1 for bit in tv),
                                entailed=True,
                                is_axiom=False,
                                generation=iteration + 1,
                            )
                            g.add_edge(
                                str(f1),
                                str(conclusion),
                                reason="HS",
                                rule="hypothetical_syllogism",
                                via=str(f2),
                            )

                # Conjunction Introduction: A and B, derive (A ∧ B)
                # (Only add if both are simple enough to avoid explosion)
                if (
                    "∧I" in rules
                    and not isinstance(f1, And)
                    and not isinstance(f2, And)
                ):
                    conclusion = And(f1, f2)
                    if conclusion not in knowledge_base and len(str(conclusion)) < 50:
                        iteration_new.add(conclusion)
                        tv = formula_truth_vector(conclusion, var_names)
                        g.add_node(
                            str(conclusion),
                            truth_vector=tv,
                            tautology=all(bit == 1 for bit in tv),
                            entailed=True,
                            is_axiom=False,
                            generation=iteration + 1,
                        )
                        # Add edges from both premises
                        g.add_edge(
                            str(f1),
                            str(conclusion),
                            reason="∧I",
                            rule="conjunction_intro",
                        )
                        if str(f1) != str(f2):  # Avoid self-loops
                            g.add_edge(
                                str(f2),
                                str(conclusion),
                                reason="∧I",
                                rule="conjunction_intro",
                            )

        # Single-formula transformations (apply to all formulas in knowledge base)
        for f in knowledge_base:
            # Disjunction Introduction: A ⊢ (A ∨ B) for any B in knowledge_base
            # Only add simple cases to avoid explosion
            if "∨I" in rules:
                for other in list(knowledge_base)[
                    :5
                ]:  # Limit to avoid exponential growth
                    if str(f) != str(other) and len(str(Or(f, other))) < 50:
                        conclusion = Or(f, other)
                        if conclusion not in knowledge_base:
                            iteration_new.add(conclusion)
                            tv = formula_truth_vector(conclusion, var_names)
                            g.add_node(
                                str(conclusion),
                                truth_vector=tv,
                                tautology=all(bit == 1 for bit in tv),
                                entailed=True,
                                is_axiom=False,
                                generation=iteration + 1,
                            )
                            g.add_edge(
                                str(f),
                                str(conclusion),
                                reason="∨I",
                                rule="disjunction_intro",
                            )

            # Double Negation Elimination: ¬¬A ⊢ A
            if "DNE" in rules and isinstance(f, Not) and isinstance(f.inner, Not):
                conclusion = f.inner.inner
                if conclusion not in knowledge_base:
                    iteration_new.add(conclusion)
                    tv = formula_truth_vector(conclusion, var_names)
                    g.add_node(
                        str(conclusion),
                        truth_vector=tv,
                        tautology=all(bit == 1 for bit in tv),
                        entailed=True,
                        is_axiom=False,
                        generation=iteration + 1,
                    )
                    g.add_edge(
                        str(f),
                        str(conclusion),
                        reason="DNE",
                        rule="double_negation_elim",
                    )

            # Contraposition: (A → B) ⊢ (¬B → ¬A)
            if "CP" in rules and isinstance(f, Implies):
                conclusion = Implies(Not(f.consequent), Not(f.antecedent))
                if conclusion not in knowledge_base:
                    iteration_new.add(conclusion)
                    tv = formula_truth_vector(conclusion, var_names)
                    g.add_node(
                        str(conclusion),
                        truth_vector=tv,
                        tautology=all(bit == 1 for bit in tv),
                        entailed=True,
                        is_axiom=False,
                        generation=iteration + 1,
                    )
                    g.add_edge(
                        str(f),
                        str(conclusion),
                        reason="CP",
                        rule="contraposition",
                    )

            # Material Implication: (A → B) ⊢ (¬A ∨ B)
            if "MI" in rules and isinstance(f, Implies):
                conclusion = Or(Not(f.antecedent), f.consequent)
                if conclusion not in knowledge_base and len(str(conclusion)) < 50:
                    iteration_new.add(conclusion)
                    tv = formula_truth_vector(conclusion, var_names)
                    g.add_node(
                        str(conclusion),
                        truth_vector=tv,
                        tautology=all(bit == 1 for bit in tv),
                        entailed=True,
                        is_axiom=False,
                        generation=iteration + 1,
                    )
                    g.add_edge(
                        str(f),
                        str(conclusion),
                        reason="MI",
                        rule="material_implication",
                    )

            # Reverse Material Implication: (¬A ∨ B) ⊢ (A → B)
            if "RMI" in rules and isinstance(f, Or) and isinstance(f.left, Not):
                conclusion = Implies(f.left.inner, f.right)
                if conclusion not in knowledge_base:
                    iteration_new.add(conclusion)
                    tv = formula_truth_vector(conclusion, var_names)
                    g.add_node(
                        str(conclusion),
                        truth_vector=tv,
                        tautology=all(bit == 1 for bit in tv),
                        entailed=True,
                        is_axiom=False,
                        generation=iteration + 1,
                    )
                    g.add_edge(
                        str(f),
                        str(conclusion),
                        reason="RMI",
                        rule="reverse_material_impl",
                    )

            # De Morgan's Laws: ¬(A ∧ B) ⊢ (¬A ∨ ¬B)
            if "DeM" in rules and isinstance(f, Not) and isinstance(f.inner, And):
                conclusion = Or(Not(f.inner.left), Not(f.inner.right))
                if conclusion not in knowledge_base and len(str(conclusion)) < 50:
                    iteration_new.add(conclusion)
                    tv = formula_truth_vector(conclusion, var_names)
                    g.add_node(
                        str(conclusion),
                        truth_vector=tv,
                        tautology=all(bit == 1 for bit in tv),
                        entailed=True,
                        is_axiom=False,
                        generation=iteration + 1,
                    )
                    g.add_edge(
                        str(f),
                        str(conclusion),
                        reason="DeM",
                        rule="de_morgan_and",
                    )

            # De Morgan's Laws: ¬(A ∨ B) ⊢ (¬A ∧ ¬B)
            if "DeM" in rules and isinstance(f, Not) and isinstance(f.inner, Or):
                conclusion = And(Not(f.inner.left), Not(f.inner.right))
                if conclusion not in knowledge_base and len(str(conclusion)) < 50:
                    iteration_new.add(conclusion)
                    tv = formula_truth_vector(conclusion, var_names)
                    g.add_node(
                        str(conclusion),
                        truth_vector=tv,
                        tautology=all(bit == 1 for bit in tv),
                        entailed=True,
                        is_axiom=False,
                        generation=iteration + 1,
                    )
                    g.add_edge(
                        str(f),
                        str(conclusion),
                        reason="DeM",
                        rule="de_morgan_or",
                    )

            # Reverse De Morgan: (¬A ∨ ¬B) ⊢ ¬(A ∧ B)
            if "RDeM" in rules and (
                isinstance(f, Or)
                and isinstance(f.left, Not)
                and isinstance(f.right, Not)
            ):
                conclusion = Not(And(f.left.inner, f.right.inner))
                if conclusion not in knowledge_base and len(str(conclusion)) < 50:
                    iteration_new.add(conclusion)
                    tv = formula_truth_vector(conclusion, var_names)
                    g.add_node(
                        str(conclusion),
                        truth_vector=tv,
                        tautology=all(bit == 1 for bit in tv),
                        entailed=True,
                        is_axiom=False,
                        generation=iteration + 1,
                    )
                    g.add_edge(
                        str(f),
                        str(conclusion),
                        reason="RDeM",
                        rule="reverse_de_morgan_and",
                    )

            # Reverse De Morgan: (¬A ∧ ¬B) ⊢ ¬(A ∨ B)
            if "RDeM" in rules and (
                isinstance(f, And)
                and isinstance(f.left, Not)
                and isinstance(f.right, Not)
            ):
                conclusion = Not(Or(f.left.inner, f.right.inner))
                if conclusion not in knowledge_base and len(str(conclusion)) < 50:
                    iteration_new.add(conclusion)
                    tv = formula_truth_vector(conclusion, var_names)
                    g.add_node(
                        str(conclusion),
                        truth_vector=tv,
                        tautology=all(bit == 1 for bit in tv),
                        entailed=True,
                        is_axiom=False,
                        generation=iteration + 1,
                    )
                    g.add_edge(
                        str(f),
                        str(conclusion),
                        reason="RDeM",
                        rule="reverse_de_morgan_or",
                    )

        # Generate basic tautologies for each variable (only in first iteration)
        if iteration == 0:
            for var_name in var_names:
                v = Var(var_name)
                # Law of Excluded Middle: A ∨ ¬A
                if "LEM" in rules:
                    tautology_lem = Or(v, Not(v))
                    if tautology_lem not in knowledge_base:
                        iteration_new.add(tautology_lem)
                        tv = formula_truth_vector(tautology_lem, var_names)
                        g.add_node(
                            str(tautology_lem),
                            truth_vector=tv,
                            tautology=True,
                            entailed=True,
                            is_axiom=False,
                            generation=iteration + 1,
                        )

                # Law of Identity: A → A
                if "ID" in rules:
                    tautology_id = Implies(v, v)
                    if tautology_id not in knowledge_base:
                        iteration_new.add(tautology_id)
                        tv = formula_truth_vector(tautology_id, var_names)
                        g.add_node(
                            str(tautology_id),
                            truth_vector=tv,
                            tautology=True,
                            entailed=True,
                            is_axiom=False,
                            generation=iteration + 1,
                        )

        if not iteration_new:
            break

        knowledge_base.update(iteration_new)
    return g


def export_to_html(
    g: Any,
    output_file: str = "logic_graph.html",
    height: str = "800px",
    width: str = "100%",
    notebook: bool = False,
) -> None:
    """Export logic graph to interactive HTML using pyvis.

    Parameters
    ----------
    g:
        NetworkX DiGraph from build_logic_graph.
    output_file:
        Path to save HTML file.
    height:
        Height of visualization area.
    width:
        Width of visualization area.
    notebook:
        If True, configure for Jupyter notebook display.
    """

    net = Network(height=height, width=width, directed=True, notebook=notebook)

    # Color scheme
    color_map = {
        "axiom": "#FF8C00",  # orange for axiom formulas
        "tautology": "#90EE90",  # lightgreen for tautologies
        "entailed": "#87CEEB",  # skyblue for entailed formulas
        "other": "#FF0000",  # red for statements not entailed (false under axioms)
    }

    # Add nodes with styling and tooltips
    for node, data in g.nodes(data=True):
        # Determine color
        if data.get("is_axiom"):
            color = color_map["axiom"]
            category = "Axiom"
        elif data.get("tautology"):
            color = color_map["tautology"]
            category = "Tautology"
        elif data.get("entailed"):
            color = color_map["entailed"]
            category = "Entailed"
        else:
            color = color_map["other"]
            category = "Not entailed"

        # Build tooltip
        tv = data.get("truth_vector", ())
        truth_str = "".join(map(str, tv))
        tooltip = f"{node}\n"
        tooltip += f"Category: {category}\n"
        tooltip += f"Truth vector: {truth_str}\n"
        tooltip += f"Tautology: {data.get('tautology', False)}\n"
        tooltip += f"Entailed: {data.get('entailed', False)}"

        net.add_node(
            node,
            label=node,
            color=color,
            title=tooltip,
            size=20,
            font={"size": 12},
        )

    # Add edges with labels and colors
    edge_color_map = {
        "MP": "#E91E63",  # hot pink - modus ponens
        "MT": "#9C27B0",  # deep purple - modus tollens
        "DS": "#FF9800",  # amber - disjunctive syllogism
        "HS": "#34495E",  # dark gray-blue - hypothetical syllogism
        "∧E-L": "#3F51B5",  # indigo - conjunction elim left
        "∧E-R": "#795548",  # brown - conjunction elim right
        "∧I": "#9E9E9E",  # grey - conjunction intro
        "∨I": "#757575",  # dark grey - disjunction intro
        "DNE": "#673AB7",  # deep purple - double negation elimination
        "CP": "#FF5722",  # deep orange - contraposition
        "MI": "#FFC107",  # amber - material implication
        "RMI": "#FFEB3B",  # yellow - reverse material implication
        "DeM": "#8BC34A",  # light green - de morgan's laws
        "RDeM": "#CDDC39",  # lime - reverse de morgan's laws
    }

    for source, target, data in g.edges(data=True):
        reason = data.get("reason", "entailed")
        rule = data.get("rule", reason)
        color = edge_color_map.get(reason, "#808080")

        # Build edge tooltip
        edge_title = f"{reason}"
        if rule and rule != reason:
            edge_title += f"\nRule: {rule}"

        net.add_edge(
            source,
            target,
            color=color,
            title=edge_title,
            arrows="to",
            label=reason,
        )

    # Configure physics for interactive layout
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 150,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 0.1
        },
        "stabilization": {
          "enabled": true,
          "iterations": 200
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      },
      "nodes": {
        "font": {
          "multi": "html"
        }
      }
    }
    """)

    # Create graphs directory if it doesn't exist

    graphs_dir = "graphs"
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)

    # Save to graphs folder if output_file doesn't already specify a path
    if os.path.dirname(output_file) == "":
        output_file = os.path.join(graphs_dir, output_file)

    net.save_graph(output_file)
    print(f"Interactive graph saved to {output_file}")


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------


def all_inputs(n: int) -> List[Tuple[int, ...]]:
    """
    Return a list of all 2^n input combinations of n boolean variables.

    Each combination is a tuple of 0 or 1, ordered lexicographically.
    Example for n = 2:
        [(0, 0), (0, 1), (1, 0), (1, 1)]
    """
    return list(product([0, 1], repeat=n))


def all_boolean_functions(n: int) -> Iterable[Tuple[int, ...]]:
    """
    Generate all boolean functions of n variables.

    Each function is represented as a tuple of outputs (0 or 1) ordered
    according to the input combinations from all_inputs(n).

    There are 2^(2^n) such functions, so be careful with large n.
    """
    inputs = all_inputs(n)
    num_rows = len(inputs)

    # Each function is a tuple of length num_rows with entries in {0,1}
    for outputs in product([0, 1], repeat=num_rows):
        yield outputs


# ---------------------------------------------------------------------------
# Expression generation (DNF)
# ---------------------------------------------------------------------------


def function_to_dnf(
    outputs: Sequence[int],
    var_names: Sequence[str],
    use_unicode_not: bool = True,
    use_unicode_and_or: bool = True,
) -> str:
    """
    Given outputs for all input combinations (ordered like all_inputs(len(var_names))),
    return a string with the DNF expression.

    Parameters
    ----------
    outputs:
        Sequence of 0 or 1, length must match len(all_inputs(len(var_names))).
    var_names:
        Sequence of variable names, for example ["x", "y", "z"].
    use_unicode_not:
        If True, use "¬" for negation, otherwise use "~".
    use_unicode_and_or:
        If True, use "∧" and "∨". If False, use "and" and "or".

    Returns
    -------
    str
        A DNF expression such as:
            (x ∧ ¬y) ∨ (¬x ∧ y)
        or
            (x and ~y) or (~x and y)
    """
    n = len(var_names)
    inputs = all_inputs(n)

    if len(outputs) != len(inputs):
        raise ValueError(
            "Length of outputs does not match number of input combinations"
        )

    not_symbol = "¬" if use_unicode_not else "~"
    if use_unicode_and_or:
        and_symbol = " ∧ "
        or_symbol = " ∨ "
    else:
        and_symbol = " and "
        or_symbol = " or "

    minterms = []

    for inp, out in zip(inputs, outputs):
        if out == 1:
            literals = []
            for var, val in zip(var_names, inp):
                if val == 1:
                    literals.append(var)
                else:
                    literals.append(f"{not_symbol}{var}")
            minterms.append("(" + and_symbol.join(literals) + ")")

    if not minterms:
        return "FALSE"

    return or_symbol.join(minterms)


# ---------------------------------------------------------------------------
# Matrix representation for many functions
# ---------------------------------------------------------------------------


def boolean_function_matrix(
    n: int,
    max_functions: int | None = None,
) -> np.ndarray:
    """
    Create a matrix where each row is one boolean function,
    each column is an input combination, and cell values are 0 or 1.

    Parameters
    ----------
    n:
        Number of variables.
    max_functions:
        If not None, limit to the first max_functions functions
        to avoid gigantic matrices.

    Returns
    -------
    numpy.ndarray
        Shape (num_functions, 2^n).
    """

    inputs = all_inputs(n)
    num_rows = len(inputs)

    if max_functions is None:
        funcs = list(all_boolean_functions(n))
    else:
        funcs = list(islice(all_boolean_functions(n), max_functions))

    matrix = np.array(funcs, dtype=int)
    if matrix.shape[1] != num_rows:
        raise RuntimeError("Internal error: unexpected matrix shape")

    return matrix


def plot_boolean_function_matrix(
    n: int,
    max_functions: int | None = None,
    title: str | None = None,
) -> None:
    """
    Plot a matrix of boolean functions using matplotlib.

    Each row is a boolean function, each column an input combination.

    Parameters
    ----------
    n:
        Number of variables.
    max_functions:
        If not None, limit to first max_functions functions for plotting.
    title:
        Optional custom plot title.
    """

    matrix = boolean_function_matrix(n, max_functions=max_functions)

    plt.imshow(matrix, aspect="auto", interpolation="nearest")
    plt.xlabel("Input combination index")
    plt.ylabel("Function index")

    if title is None:
        if max_functions is None:
            title = f"All boolean functions of {n} variables"
        else:
            title = f"First {matrix.shape[0]} boolean functions of {n} variables"
    plt.title(title)

    plt.colorbar(label="Output")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Truth table representation for a single function
# ---------------------------------------------------------------------------


def make_truth_table(
    var_names: Sequence[str],
    outputs: Sequence[int],
) -> pd.DataFrame:
    """
    Build a pandas DataFrame representing the truth table of a boolean function.

    Parameters
    ----------
    var_names:
        Names of the variables, for example ["x", "y"].
    outputs:
        Sequence of 0 or 1, one per input combination in all_inputs(len(var_names)).

    Returns
    -------
    pandas.DataFrame
        Columns: var_names plus "f" (the function value).
    """
    n = len(var_names)
    inputs = all_inputs(n)

    if len(outputs) != len(inputs):
        raise ValueError(
            "Length of outputs does not match number of input combinations"
        )

    data = {name: [inp[i] for inp in inputs] for i, name in enumerate(var_names)}
    data["f"] = list(outputs)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Optional demo in script mode
# ---------------------------------------------------------------------------


def _demo() -> None:
    """
    Simple demo when running this file directly.
    Does not run in notebook unless called explicitly.
    """
    print("Demo for n = 2\n")
    n = 2
    var_names = ["x", "y"]

    # Get all functions for n = 2 (there are 16)
    funcs = list(all_boolean_functions(n))

    # Show first few functions and their DNF
    for idx, outputs in enumerate(funcs[:4]):
        print(f"Function {idx}: outputs = {outputs}")
        expr = function_to_dnf(outputs, var_names)
        print(f"  DNF: {expr}")
        print()

    # Show a truth table for function 5, as an example
    print("Truth table for function 5:")
    outputs_5 = funcs[5]
    df = make_truth_table(var_names, outputs_5)
    print(df)

    # Optionally, plot a matrix of the first 8 functions
    try:
        plot_boolean_function_matrix(n, max_functions=8)
    except Exception as e:
        print("Plotting failed (probably missing matplotlib or numpy):", e)

    # Demonstrate logic graph with simple axioms if networkx available
    try:
        ax1 = Var("x")
        ax2 = Implies(Var("x"), Var("y"))
        axioms = [ax1, ax2]
        lg = build_syntactic_graph(var_names, axioms, max_iterations=2)
        print("Built logic graph with", len(lg.nodes), "nodes")
        visualize_logic_graph(lg, layout="spring")
    except ImportError as e:
        print("Logic graph demo skipped (missing dependency):", e)
    except Exception as e:
        print("Logic graph demo failed:", e)


if __name__ == "__main__":
    _demo()
