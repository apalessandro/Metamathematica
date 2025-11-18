# Metamathematica: Boolean Logic Visualization

A Python toolkit for visualizing propositional logic as interactive graphs. Build graphs of logical formulas where nodes are statements and edges represent inference rules (modus ponens, modus tollens, etc.). Explore how theorems derive from axioms through semantic entailment and proof rules.

## Features

- **Formula AST**: Represent propositional formulas with `Var`, `Const`, `Not`, `And`, `Or`, `Implies`
- **Formula Enumeration**: Generate all formulas up to a given syntactic depth
- **Semantic Entailment**: Check which formulas follow from axioms via truth tables
- **Inference Rules**:
  - Modus Ponens (MP): `(A → B), A ⊢ B`
  - Modus Tollens (MT): `(A → B), ¬B ⊢ ¬A`
  - Disjunctive Syllogism (DS): `(A ∨ B), ¬A ⊢ B`
  - Hypothetical Syllogism (HS): `(A → B), (B → C) ⊢ (A → C)`
  - Conjunction Elimination (∧E): `(A ∧ B) ⊢ A, B`
  - Disjunction Introduction (∨I): `A ⊢ (A ∨ B)`
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

### Basic Usage

```python
from boolean_functions import (
    Var, Implies, build_logic_graph, export_to_html
)

# Define axioms: p, (p → q)
p = Var("p")
q = Var("q")
axioms = [p, Implies(p, q)]

# Build graph
g = build_logic_graph(
    var_names=["p", "q"],
    axioms=axioms,
    max_depth=2,
    include_all=True
)

# Export to interactive HTML
export_to_html(g, "my_logic_graph.html")
```

Open `my_logic_graph.html` in your browser to:

- **Drag** nodes to rearrange
- **Hover** over nodes for truth vectors and metadata
- **Hover** over edges to see inference rule names
- **Zoom** with mouse wheel
- **Pan** by dragging background

### Color Coding

- **Gold star** (⭐): AXIOMS meta-node
- **Orange**: Axiom formulas
- **Light green**: Tautologies (always true)
- **Sky blue**: Entailed formulas (derivable from axioms)
- **Light gray**: Not entailed

### Edge Colors

- **Black**: Entailment from axioms
- **Red**: Modus Ponens (MP)
- **Deep Pink**: Modus Tollens (MT)
- **Dark Magenta**: Disjunctive Syllogism (DS)
- **Indigo**: Hypothetical Syllogism (HS)
- **Dark/Forest Green**: Conjunction Elimination (∧E)
- **Blue shades**: Disjunction Introduction (∨I)

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

### Modus Ponens

```python
from boolean_functions import Var, Implies, build_logic_graph, export_to_html

x = Var("x")
y = Var("y")
axioms = [x, Implies(x, y)]

g = build_logic_graph(["x", "y"], axioms, max_depth=2)
export_to_html(g, "modus_ponens.html")
# Result: y is entailed via MP edge from x
```

### Modus Tollens

```python
from boolean_functions import Var, Not, Implies, build_logic_graph, export_to_html

p = Var("p")
q = Var("q")
axioms = [Implies(p, q), Not(q)]

g = build_logic_graph(["p", "q"], axioms, max_depth=2)
export_to_html(g, "modus_tollens.html")
# Result: ¬p is entailed via MT edge from ¬q
```

### Chain of Implications

```python
from boolean_functions import Var, Implies, build_logic_graph, export_to_html

a = Var("a")
b = Var("b")
c = Var("c")
axioms = [a, Implies(a, b), Implies(b, c)]

g = build_logic_graph(["a", "b", "c"], axioms, max_depth=2)
export_to_html(g, "chain.html")
# Result: b, c, and (a → c) all entailed
```

### Exploring All Formulas

```python
from boolean_functions import build_logic_graph, export_to_html

# No axioms - show the complete space
g = build_logic_graph(["x", "y"], axioms=[], max_depth=2, include_all=True)
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

#### `enumerate_formulas(var_names, max_depth)`

Generate all unique formulas up to syntactic depth.

**Parameters:**

- `var_names`: List of variable names
- `max_depth`: Maximum nesting depth (keep ≤ 3 to avoid explosion)

**Returns:** List of Formula objects

#### `build_logic_graph(var_names, axioms, max_depth, include_all, add_inference_edges)`

Build a directed graph of logical formulas.

**Parameters:**

- `var_names`: List of variable names
- `axioms`: List of Formula objects representing axioms
- `max_depth`: Formula enumeration depth (default: 2)
- `include_all`: Include non-entailed formulas (default: True)
- `add_inference_edges`: Add inference rule edges (default: True)

**Returns:** NetworkX DiGraph with node attributes:

- `truth_vector`: Tuple of 0/1 over all valuations
- `tautology`: Boolean
- `entailed`: Boolean (semantic consequence of axioms)
- `is_axiom`: Boolean

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
from boolean_functions import make_truth_table

df = make_truth_table(["x", "y"], outputs=(0, 1, 1, 1))
# Returns pandas DataFrame with columns x, y, f
```

## Advanced Usage

### Custom Axiom Systems

```python
# Law of Excluded Middle: p ∨ ¬p
from boolean_functions import Var, Or, Not

p = Var("p")
axioms = [Or(p, Not(p))]
g = build_logic_graph(["p"], axioms, max_depth=2)
export_to_html(g, "excluded_middle.html")
```

### Filtering to Entailed Only

```python
# Show only provable formulas
g = build_logic_graph(
    ["x", "y"],
    axioms=[Var("x")],
    max_depth=2,
    include_all=False  # Filter out non-entailed
)
export_to_html(g, "entailed_only.html")
```

### Disabling Inference Edges

```python
# Show only semantic entailment, no proof rules
g = build_logic_graph(
    ["x", "y"],
    axioms=[Var("x")],
    max_depth=2,
    add_inference_edges=False
)
export_to_html(g, "semantic_only.html")
```

### Working with Three Variables

```python
# Warning: grows exponentially!
g = build_logic_graph(
    ["a", "b", "c"],
    axioms=[Var("a")],
    max_depth=1  # Keep depth low for 3+ variables
)
export_to_html(g, "three_vars.html")
```

## Performance Notes

Formula enumeration is **exponential in both depth and variable count**:

| Variables | Depth | Approx. Formulas |
|-----------|-------|------------------|
| 2         | 1     | ~20              |
| 2         | 2     | ~200             |
| 2         | 3     | ~10,000          |
| 3         | 1     | ~30              |
| 3         | 2     | ~1,000           |

**Recommendations:**

- Use `max_depth=2` for 2-3 variables
- Use `max_depth=1` for 4+ variables
- Set `include_all=False` to filter to entailed formulas only

## Troubleshooting

### Missing Dependencies

```bash
pip install networkx matplotlib pyvis pandas numpy
```

### Graph Too Large

Reduce `max_depth` or set `include_all=False`.

### HTML File Won't Open

Some browsers block local file access. Try:

- Firefox (usually works)
- `python -m http.server` and browse to <http://localhost:8000>

## Contributing

This is an educational/research tool. Contributions welcome:

- Additional inference rules (conjunction intro, biconditional elim, etc.)
- CNF/NNF normalization
- Resolution proofs
- Natural deduction trees
- Sequent calculus support

## License

MIT License - See repository for details.

## References

- Propositional logic: <https://en.wikipedia.org/wiki/Propositional_calculus>
- Natural deduction: <https://en.wikipedia.org/wiki/Natural_deduction>
- NetworkX: <https://networkx.org/>
- Pyvis: <https://pyvis.readthedocs.io/>
