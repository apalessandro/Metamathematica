"""
Utility functions to enumerate and visualize boolean functions (logical statements)
over n boolean variables.

Main features:
- Generate all input combinations (truth table rows)
- Enumerate all boolean functions as output vectors
- Convert output vectors to DNF expressions
- Build a matrix of many functions for visualization
- Build a pandas DataFrame truth table for a single function
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
    take_taut = min(remaining // 2, len(tautology_formulas))
    result.extend(tautology_formulas[:take_taut])
    remaining -= take_taut

    if remaining <= 0:
        return result

    # Fill remaining with simplest other formulas
    result.extend(other_formulas[:remaining])

    print(
        f"📊 Sampled {len(result)} formulas from {len(formulas)}: "
        f"{len(axiom_formulas)} axioms, {take_entailed} entailed, "
        f"{take_taut} tautologies, {len(result) - len(axiom_formulas) - take_entailed - take_taut} others"
    )

    return result


def build_logic_graph(
    var_names: Sequence[str],
    axioms: Sequence[Formula],
    max_depth: int = 2,
    include_all: bool = True,
    add_inference_edges: bool = True,
    max_nodes: int | None = None,
) -> Any:
    """Build a directed graph of formulas.

    Nodes: all enumerated formulas (or only entailed ones if include_all=False).
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
    max_nodes:
        If set, intelligently sample the graph to keep only the most interesting nodes.
        Prioritizes: axioms, direct consequences, nodes with many connections, tautologies.
    """
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
        if include_all or ent:
            g.add_node(
                str(f),
                truth_vector=tv,
                tautology=taut,
                entailed=ent,
                is_axiom=str(f) in axiom_set,
            )
    # Edges from each axiom to entailed formulas (excluding axioms themselves)
    for node, data in list(g.nodes(data=True)):
        if data.get("entailed") and not data.get("is_axiom"):
            # Add edge from each axiom to this entailed formula
            for axiom in axioms:
                axiom_str = str(axiom)
                if axiom_str in g:
                    g.add_edge(axiom_str, node, reason="entailed")

    if add_inference_edges:
        # Helper to ensure node exists in graph
        def ensure_node(formula: Formula) -> str:
            f_str = str(formula)
            if f_str not in g:
                f_tv = formula_truth_vector(formula, var_names)
                g.add_node(
                    f_str,
                    truth_vector=f_tv,
                    tautology=all(b == 1 for b in f_tv),
                    entailed=entails(axioms, formula, var_names),
                    is_axiom=f_str in axiom_set,
                )
            return f_str

        for f in formulas:
            # Modus Ponens: (A → B), A ⊢ B
            if isinstance(f, Implies):
                impl_str = str(f)
                ant_str = str(f.antecedent)
                cons_str = ensure_node(f.consequent)
                if (
                    impl_str in g
                    and ant_str in g
                    and g.nodes[impl_str]["entailed"]
                    and g.nodes[ant_str]["entailed"]
                ):
                    g.add_edge(ant_str, cons_str, reason="MP", rule="modus_ponens")

            # Modus Tollens: (A → B), ¬B ⊢ ¬A
            if isinstance(f, Implies):
                impl_str = str(f)
                not_b = Not(f.consequent)
                not_a = Not(f.antecedent)
                not_b_str = str(not_b)
                not_a_str = ensure_node(not_a)
                if (
                    impl_str in g
                    and not_b_str in g
                    and g.nodes[impl_str]["entailed"]
                    and g.nodes[not_b_str]["entailed"]
                ):
                    g.add_edge(not_b_str, not_a_str, reason="MT", rule="modus_tollens")

            # Disjunctive Syllogism: (A ∨ B), ¬A ⊢ B
            if isinstance(f, Or):
                or_str = str(f)
                not_a = Not(f.left)
                not_a_str = str(not_a)
                b_str = ensure_node(f.right)
                if (
                    or_str in g
                    and not_a_str in g
                    and g.nodes[or_str]["entailed"]
                    and g.nodes[not_a_str]["entailed"]
                ):
                    g.add_edge(
                        not_a_str, b_str, reason="DS", rule="disjunctive_syllogism"
                    )

                # Symmetric: (A ∨ B), ¬B ⊢ A
                not_b = Not(f.right)
                not_b_str = str(not_b)
                a_str = ensure_node(f.left)
                if (
                    or_str in g
                    and not_b_str in g
                    and g.nodes[or_str]["entailed"]
                    and g.nodes[not_b_str]["entailed"]
                ):
                    g.add_edge(
                        not_b_str, a_str, reason="DS", rule="disjunctive_syllogism"
                    )

            # Hypothetical Syllogism: (A → B), (B → C) ⊢ (A → C)
            if isinstance(f, Implies) and isinstance(f.consequent, Implies):
                # Pattern: A → (B → C) represents curried form; skip for now
                pass
            # Check if we have two separate implications
            for f2 in formulas:
                if isinstance(f, Implies) and isinstance(f2, Implies):
                    if str(f.consequent) == str(f2.antecedent):
                        # f: A → B, f2: B → C
                        impl1_str = str(f)
                        impl2_str = str(f2)
                        conclusion = Implies(f.antecedent, f2.consequent)
                        concl_str = ensure_node(conclusion)
                        if (
                            impl1_str in g
                            and impl2_str in g
                            and g.nodes[impl1_str]["entailed"]
                            and g.nodes[impl2_str]["entailed"]
                        ):
                            # Add edge from both premises (we'll pick one arbitrarily)
                            g.add_edge(
                                impl1_str,
                                concl_str,
                                reason="HS",
                                rule="hypothetical_syllogism",
                            )

            # Conjunction Elimination: (A ∧ B) ⊢ A, (A ∧ B) ⊢ B
            if isinstance(f, And):
                and_str = str(f)
                left_str = ensure_node(f.left)
                right_str = ensure_node(f.right)
                if and_str in g and g.nodes[and_str]["entailed"]:
                    g.add_edge(
                        and_str, left_str, reason="∧E-L", rule="conjunction_elim_left"
                    )
                    g.add_edge(
                        and_str, right_str, reason="∧E-R", rule="conjunction_elim_right"
                    )

            # Disjunction Introduction: A ⊢ (A ∨ B) for any B
            # This can create many edges; we'll limit to formulas already in graph
            if not isinstance(f, Or):
                f_str = str(f)
                if f_str in g and g.nodes[f_str]["entailed"]:
                    for candidate in formulas:
                        if isinstance(candidate, Or):
                            if str(candidate.left) == f_str:
                                or_str = ensure_node(candidate)
                                g.add_edge(
                                    f_str,
                                    or_str,
                                    reason="∨I-L",
                                    rule="disjunction_intro_left",
                                )
                            elif str(candidate.right) == f_str:
                                or_str = ensure_node(candidate)
                                g.add_edge(
                                    f_str,
                                    or_str,
                                    reason="∨I-R",
                                    rule="disjunction_intro_right",
                                )
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
        "tautology": "#90EE90",  # lightgreen
        "entailed": "#87CEEB",  # skyblue
        "other": "#D3D3D3",  # lightgray
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
        "entailed": "#000000",  # black
        "MP": "#FF0000",  # red - modus ponens
        "MT": "#FF1493",  # deep pink - modus tollens
        "DS": "#8B008B",  # dark magenta - disjunctive syllogism
        "HS": "#4B0082",  # indigo - hypothetical syllogism
        "∧E-L": "#006400",  # dark green - conjunction elim left
        "∧E-R": "#228B22",  # forest green - conjunction elim right
        "∨I-L": "#0000CD",  # medium blue - disjunction intro left
        "∨I-R": "#4169E1",  # royal blue - disjunction intro right
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
    import os
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
        lg = build_logic_graph(var_names, axioms, max_depth=2)
        print("Built logic graph with", len(lg.nodes), "nodes")
        visualize_logic_graph(lg, layout="spring")
    except ImportError as e:
        print("Logic graph demo skipped (missing dependency):", e)
    except Exception as e:
        print("Logic graph demo failed:", e)


if __name__ == "__main__":
    _demo()
