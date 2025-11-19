# Metamathematica: Boolean Logic Visualization

A Python toolkit for visualizing propositional logic as interactive graphs. Build graphs of logical formulas where nodes are statements and edges represent inference rules (modus ponens, modus tollens, etc.). Explore how theorems derive from axioms through forward-chaining inference and semantic entailment.

## Features

- **Formula AST**: Represent propositional formulas with `Var`, `Const`, `Not`, `And`, `Or`, `Implies`
- **Forward-Chaining**: Start from axioms and apply inference rules to derive new statements (default)
- **Backward-Chaining**: Generate all formulas up to a depth and find relationships to axioms (alternative)
- **Semantic Entailment**: Check which formulas follow from axioms via truth tables
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

### Basic Usage (Forward-Chaining)

```python
from calculus_ratiocinator import (
    Var, Implies, build_logic_graph, export_to_html
)

# Define axioms: p, (p → q)
p = Var("p")
q = Var("q")
axioms = [p, Implies(p, q)]

# Build graph using forward-chaining (default)
# Starts from axioms and applies inference rules to derive new statements
g = build_logic_graph(
    var_names=["p", "q"],
    axioms=axioms,
    max_iterations=3  # Apply rules for 3 iterations
)

# Export to interactive HTML
export_to_html(g, "my_logic_graph.html")
```

### Alternative: Backward-Chaining

```python
from calculus_ratiocinator import (
    Var, Implies, build_backward_logic_graph, export_to_html
)

# Generate all formulas up to a depth, then find relationships
g = build_backward_logic_graph(
    var_names=["p", "q"],
    axioms=axioms,
    max_depth=2,
    max_nodes=300  # Optional: intelligent sampling to limit graph size
)

export_to_html(g, "backward_graph.html")
```

Open `my_logic_graph.html` in your browser to:

- **Drag** nodes to rearrange
- **Hover** over nodes for truth vectors and metadata
- **Hover** over edges to see inference rule names
- **Zoom** with mouse wheel
- **Pan** by dragging background

### Color Coding

- **Orange**: Axiom formulas (starting points)
- **Light green**: Tautologies (always true)
- **Sky blue**: Entailed formulas (derivable from axioms)
- **Red**: Not entailed (false under axioms) - only in backward-chaining

### Edge Colors

- **Hot Pink**: Modus Ponens (MP)
- **Deep Purple**: Modus Tollens (MT)
- **Amber**: Disjunctive Syllogism (DS)
- **Dark Gray-Blue**: Hypothetical Syllogism (HS)
- **Indigo**: Conjunction Elimination Left (∧E-L)
- **Brown**: Conjunction Elimination Right (∧E-R)
- **Gray**: Conjunction Introduction (∧I)

## Demo Script

Run the comprehensive demo to see all inference rules in action:

```bash
python logic_graph_demo.py
```

This generates 7 HTML files demonstrating:

1. Modus ponens
2. Modus tollens
3. Disjunctive syllogism
4. Hypothetical syllogism
5. Conjunction elimination
6. Complex multi-step proofs
7. Complete formula space exploration

## Examples

### Modus Ponens (Forward-Chaining)

```python
from calculus_ratiocinator import Var, Implies, build_logic_graph, export_to_html

x = Var("x")
y = Var("y")
axioms = [x, Implies(x, y)]

g = build_logic_graph(["x", "y"], axioms, max_iterations=2)
export_to_html(g, "modus_ponens.html")
# Result: y is derived via MP, graph shows clear derivation path
```

### Modus Tollens

```python
from calculus_ratiocinator import Var, Not, Implies, build_logic_graph, export_to_html

p = Var("p")
q = Var("q")
axioms = [Implies(p, q), Not(q)]

g = build_logic_graph(["p", "q"], axioms, max_iterations=2)
export_to_html(g, "modus_tollens.html")
# Result: ¬p is derived via MT edge from ¬q
```

### Chain of Implications

```python
from calculus_ratiocinator import Var, Implies, build_logic_graph, export_to_html

a = Var("a")
b = Var("b")
c = Var("c")
axioms = [a, Implies(a, b), Implies(b, c)]

g = build_logic_graph(["a", "b", "c"], axioms, max_iterations=3)
export_to_html(g, "chain.html")
# Result: b, c, and (a → c) all derived with generation tracking
```

### Exploring All Formulas (Backward-Chaining)

```python
from calculus_ratiocinator import build_backward_logic_graph, export_to_html

# Generate all formulas and identify tautologies
g = build_backward_logic_graph(["x", "y"], axioms=[], max_depth=2, max_nodes=500)
export_to_html(g, "all_formulas.html")
# Explore tautologies (green nodes) in the formula space
```

## API Reference

### Formula Classes

- `Var(name: str)` - Variable
- `Const(value: int)` - Constant (0 or 1)
- `Not(inner: Formula)` - Negation
- `And(left: Formula, right: Formula)` - Conjunction
- `Or(left: Formula, right: Formula)` - Disjunction
- `Implies(antecedent: Formula, consequent: Formula)` - Implication

### Main Functions

#### `build_logic_graph(var_names, axioms, max_iterations)`

Build a directed graph by forward-chaining from axioms (default approach).

Start with axioms and iteratively apply inference rules to generate new entailed statements.

**Parameters:**

- `var_names`: List of variable names
- `axioms`: List of Formula objects representing axioms
- `max_iterations`: Maximum iterations for rule application (default: 3)

**Returns:** NetworkX DiGraph with node attributes:

- `truth_vector`: Tuple of 0/1 over all valuations
- `tautology`: Boolean
- `entailed`: Boolean (always True in forward-chaining)
- `is_axiom`: Boolean
- `generation`: Integer (0 for axioms, 1+ for derived)

And edges showing inference steps with `reason` (MP, MT, DS, HS, ∧E-L, ∧E-R, ∧I) and optional `via` (supporting formula).

#### `build_backward_logic_graph(var_names, axioms, max_depth, max_nodes)`

Build a directed graph using backward-chaining (alternative approach).

Generate all formulas up to a depth, then find relationships to axioms.

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

## Advanced Usage

### Custom Axiom Systems

```python
# Law of Excluded Middle: p ∨ ¬p
from calculus_ratiocinator import Var, Or, Not, build_logic_graph

p = Var("p")
axioms = [Or(p, Not(p))]
g = build_logic_graph(["p"], axioms, max_iterations=2)
export_to_html(g, "excluded_middle.html")
```

### Comparing Forward and Backward Chaining

```python
from calculus_ratiocinator import (
    Var, Implies, build_logic_graph, build_backward_logic_graph, export_to_html
)

axioms = [Var("p"), Implies(Var("p"), Var("q"))]

# Forward: focused, shows derivation steps
g_forward = build_logic_graph(["p", "q"], axioms, max_iterations=3)
print(f"Forward: {g_forward.number_of_nodes()} nodes")

# Backward: comprehensive, explores formula space
g_backward = build_backward_logic_graph(["p", "q"], axioms, max_depth=2, max_nodes=200)
print(f"Backward: {g_backward.number_of_nodes()} nodes")

export_to_html(g_forward, "forward.html")
export_to_html(g_backward, "backward.html")
```

### Using Smart Sampling (Backward-Chaining)

```python
from calculus_ratiocinator import Var, build_backward_logic_graph

# Limit graph size with intelligent sampling
# Prioritizes: axioms → entailed → tautologies → false statements
g = build_backward_logic_graph(
    ["x", "y"],
    axioms=[Var("x")],
    max_depth=3,
    max_nodes=200  # Keep only 200 most important formulas
)
export_to_html(g, "sampled_graph.html")
```

### Working with Three Variables

```python
# Forward-chaining scales better with more variables
from calculus_ratiocinator import Var, Implies, build_logic_graph

axioms = [Var("a"), Implies(Var("a"), Var("b")), Implies(Var("b"), Var("c"))]
g = build_logic_graph(["a", "b", "c"], axioms, max_iterations=3)
export_to_html(g, "three_vars.html")

# Backward-chaining: keep depth low for 3+ variables
g = build_backward_logic_graph(["a", "b", "c"], axioms, max_depth=1, max_nodes=100)
export_to_html(g, "three_vars_backward.html")
```

## Performance Notes

**Forward-chaining** (default):

- Grows based on what can be derived from axioms
- Efficient and focused
- Scales well with more variables
- Best for exploring derivations from specific axioms

**Backward-chaining** (alternative):

- Formula enumeration is **exponential in both depth and variable count**:

| Variables | Depth | Approx. Formulas |
|-----------|-------|------------------|
| 2         | 1     | ~20              |
| 2         | 2     | ~200             |
| 2         | 3     | ~10,000          |
| 3         | 1     | ~30              |
| 3         | 2     | ~1,000           |

**Recommendations:**

- **Forward-chaining**: Use `max_iterations=2-5` for most cases
- **Backward-chaining**: Use `max_depth=2` for 2-3 variables
- **Backward-chaining with sampling**: Use `max_depth=3-4` with `max_nodes=200-300` for focused exploration
- **3+ variables**: Prefer forward-chaining or use `max_depth=1` with backward-chaining

## Troubleshooting

### Missing Dependencies

```bash
pip install networkx matplotlib pyvis pandas numpy
```

### Graph Too Large

- **Forward-chaining**: Reduce `max_iterations` or add axioms that are more constrained
- **Backward-chaining**: Reduce `max_depth` or set `max_nodes` to limit graph size with intelligent sampling

### HTML File Won't Open

Some browsers block local file access. Try:

- Firefox (usually works)
- `python -m http.server` and browse to <http://localhost:8000>

## License

MIT License - See repository for details.

## References

- Propositional logic: <https://en.wikipedia.org/wiki/Propositional_calculus>
- Natural deduction: <https://en.wikipedia.org/wiki/Natural_deduction>
- NetworkX: <https://networkx.org/>
- Pyvis: <https://pyvis.readthedocs.io/>
