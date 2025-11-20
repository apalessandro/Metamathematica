# Metamathematica: Boolean Logic Visualization

A Python toolkit for visualizing propositional logic as interactive graphs. Build graphs of logical formulas where nodes are statements and edges represent inference rules (modus ponens, modus tollens, etc.). Explore how theorems derive from axioms through **syntactic derivability** (proof-based) and **semantic entailment** (truth-based).

## Key Concepts

### Syntactic Derivability vs Semantic Entailment

This library provides two fundamentally different approaches to logic:

#### Syntactic Derivability (Proof-Based)**

- Uses `build_syntactic_graph()`
- Starts with axioms and applies **inference rules** step-by-step
- Generates only formulas that can be **proven** through a finite chain of rules
- Focused, efficient, and shows the **derivation path**
- May not derive all true consequences (incomplete for propositional logic)
- Example: From `p` and `p → q`, derive `q` via Modus Ponens

#### Semantic Entailment (Truth-Based)

- Uses `build_semantic_graph()`
- Generates all possible formulas, checks which are **true in all models** where axioms hold
- Complete for propositional logic - finds **all** logical consequences
- More comprehensive but generates many more formulas
- Example: `(p ∧ q) → r` is semantically entailed from `{p, p → q, q → r}` even though it's not directly derivable

**Relationship**: `Syntactic ⊆ Semantic`

- Everything syntactically derivable IS semantically entailed
- NOT everything semantically entailed can be syntactically derived with a finite set of inference rules

## Features

- **Formula AST**: Represent propositional formulas with `Var`, `Const`, `Not`, `And`, `Or`, `Implies`
- **Syntactic Derivability**: Start from axioms and apply inference rules to derive new statements (proof-based)
- **Semantic Entailment**: Generate all formulas and check which are true under axioms (truth-based)
- **Dual Approaches**:
  - `build_syntactic_graph()` - Forward-chaining with inference rules (focused, shows derivations)
  - `build_semantic_graph()` - Exhaustive formula enumeration (complete, explores entire space)
- **Semantic Entailment Checking**: Verify which formulas follow from axioms via truth tables
- **Inference Rules**:
  - Modus Ponens (MP): `(A → B), A ⊢ B`
  - Modus Tollens (MT): `(A → B), ¬B ⊢ ¬A`
  - Disjunctive Syllogism (DS): `(A ∨ B), ¬A ⊢ B`
  - Hypothetical Syllogism (HS): `(A → B), (B → C) ⊢ (A → C)`
  - Conjunction Elimination (∧E): `(A ∧ B) ⊢ A, B`
  - Conjunction Introduction (∧I): `A, B ⊢ (A ∧ B)`
- **Interactive Visualization**: Export to HTML with pyvis for draggable, zoomable graphs
- **Static Visualization**: Matplotlib-based plotting for notebooks and scripts

## Installation

```bash
pip install -r requirements.txt
```

Requirements:

- `networkx` - graph data structures
- `matplotlib` - static plotting
- `pyvis` - interactive HTML visualization
- `pandas` - truth tables
- `numpy` - matrix operations

## Quick Start

### Syntactic Derivability (Proof-Based Approach)

```python
from calculus_ratiocinator import (
    Var, Implies, build_syntactic_graph, export_to_html
)

# Define axioms: p, (p → q)
p = Var("p")
q = Var("q")
axioms = [p, Implies(p, q)]

# Build graph using syntactic derivability (proof-based)
# Starts from axioms and applies inference rules to derive new statements
g = build_syntactic_graph(
    var_names=["p", "q"],
    axioms=axioms,
    max_iterations=3  # Apply rules for 3 iterations
)

# Export to interactive HTML
export_to_html(g, "syntactic_proof.html")
# Result: Shows derivation of q via Modus Ponens with clear proof path
```

### Semantic Entailment (Truth-Based Approach)

```python
from calculus_ratiocinator import (
    Var, Implies, build_semantic_graph, export_to_html
)

# Generate all formulas up to a depth, check which are semantically entailed
g = build_semantic_graph(
    var_names=["p", "q"],
    axioms=axioms,
    max_depth=2,
    max_nodes=300  # Optional: intelligent sampling to limit graph size
)

export_to_html(g, "semantic_entailment.html")
# Result: Shows ALL formulas true under axioms, including tautologies
```

Open `syntactic_proof.html` or `semantic_entailment.html` in your browser to:

- **Drag** nodes to rearrange
- **Hover** over nodes for truth vectors and metadata
- **Hover** over edges to see inference rule names
- **Zoom** with mouse wheel
- **Pan** by dragging background

### Color Coding

- **Orange**: Axiom formulas (starting points)
- **Light green**: Tautologies (always true)
- **Sky blue**: Entailed formulas (derivable/true under axioms)
- **Red**: Not entailed (false under axioms) - only in semantic approach

### Edge Colors

- **Hot Pink**: Modus Ponens (MP)
- **Deep Purple**: Modus Tollens (MT)
- **Amber**: Disjunctive Syllogism (DS)
- **Dark Gray-Blue**: Hypothetical Syllogism (HS)
- **Indigo**: Conjunction Elimination Left (∧E-L)
- **Brown**: Conjunction Elimination Right (∧E-R)
- **Gray**: Conjunction Introduction (∧I)

### Main Functions

#### `build_syntactic_graph(var_names, axioms, max_iterations)`

Build a directed graph using **syntactic derivability** (proof-based, forward-chaining).

Start with axioms and iteratively apply inference rules to generate new statements that can be proven step-by-step. Only generates formulas derivable through the implemented inference rules.

**Parameters:**

- `var_names`: List of variable names
- `axioms`: List of Formula objects representing axioms
- `max_iterations`: Maximum iterations for rule application (default: 3)

**Returns:** NetworkX DiGraph with node attributes:

- `truth_vector`: Tuple of 0/1 over all valuations
- `tautology`: Boolean
- `entailed`: Boolean (always True in syntactic approach)
- `is_axiom`: Boolean
- `generation`: Integer (0 for axioms, 1+ for derived)

And edges showing inference steps with `reason` (MP, MT, DS, HS, ∧E-L, ∧E-R, ∧I) and optional `via` (supporting formula).

#### `build_semantic_graph(var_names, axioms, max_depth, max_nodes)`

Build a directed graph using **semantic entailment** (truth-based).

Generate all formulas up to a depth, then check which are semantically entailed (true in all models where axioms hold). This is complete for propositional logic but generates many more formulas than syntactic derivability.

**Parameters:**

- `var_names`: List of variable names
- `axioms`: List of Formula objects representing axioms
- `max_depth`: Formula enumeration depth (default: 2)
- `max_nodes`: Optional limit on graph size with intelligent sampling (prioritizes axioms → entailed → tautologies → false statements)

**Returns:** NetworkX DiGraph with node attributes:

- `truth_vector`: Tuple of 0/1 over all valuations
- `tautology`: Boolean
- `entailed`: Boolean (semantic consequence of axioms)
- `is_axiom`: Boolean

And edges with inference rule labels (MP, MT, DS, HS, ∧E-L, ∧E-R).

#### `export_to_html(g, output_file, height, width, notebook)`

Export graph to interactive HTML.

**Parameters:**

- `g`: NetworkX DiGraph from `build_logic_graph`
- `output_file`: Path to save HTML (default: "logic_graph.html")
- `height`: Visualization height (default: "800px")
- `width`: Visualization width (default: "100%")
- `notebook`: Configure for Jupyter (default: False)

#### `visualize_logic_graph(g, layout, figsize, show_truth)`

Static matplotlib visualization.

**Parameters:**

- `g`: NetworkX DiGraph
- `layout`: "spring", "kamada_kawai", or "circular" (default: "spring")
- `figsize`: Tuple (width, height) in inches (default: (10, 8))
- `show_truth`: Show truth vectors on nodes (default: False)

### Truth Table Functions

```python
from calculus_ratiocinator import make_truth_table

df = make_truth_table(["x", "y"], outputs=(0, 1, 1, 1))
# Returns pandas DataFrame with columns x, y, f
```

### Formula Enumeration

```python
from calculus_ratiocinator import enumerate_formulas

# Generate all formulas up to depth (used by backward-chaining)
formulas = enumerate_formulas(["x", "y"], max_depth=2)
# Returns list of Formula objects (keep max_depth ≤ 3 to avoid explosion)
```

## License

MIT License - See repository for details.

<img width="842" height="772" alt="image" src="https://github.com/user-attachments/assets/0e09a7d2-8bff-4da0-bfaa-6db3ccc1d255" />


## References

- Propositional logic: <https://en.wikipedia.org/wiki/Propositional_calculus>
- Natural deduction: <https://en.wikipedia.org/wiki/Natural_deduction>
- NetworkX: <https://networkx.org/>
- Pyvis: <https://pyvis.readthedocs.io/>
