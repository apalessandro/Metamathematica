# Quick Start Guide

## Installation

```bash
pip install networkx matplotlib pyvis pandas numpy
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

## Minimal Example

```python
from boolean_functions import Var, Implies, build_logic_graph, export_to_html

# Define variables and axioms
p = Var("p")
q = Var("q")
axioms = [p, Implies(p, q)]  # p, p → q

# Build the graph
g = build_logic_graph(
    var_names=["p", "q"],
    axioms=axioms,
    max_depth=2
)

# Export to interactive HTML
export_to_html(g, "my_graph.html")
print("Open my_graph.html in your browser!")
```

## Run the Demos

```bash
# Full demo suite (creates 7 HTML files)
python logic_graph_demo.py

# Quick test (creates 3 HTML files)
python test_logic_graph.py
```

## What You'll See

The HTML files show:

- **Nodes**: Boolean formulas (propositions)
- **Colors**:
  - Gold ⭐ = AXIOMS (starting point)
  - Orange = Your axioms
  - Light Green = Tautologies (always true)
  - Sky Blue = Entailed (provable from axioms)
  - Gray = Not entailed
- **Edges**: Inference rules
  - Red = Modus Ponens
  - Pink = Modus Tollens
  - Purple = Disjunctive Syllogism
  - Indigo = Hypothetical Syllogism
  - Green = Conjunction Elimination
  - Blue = Disjunction Introduction

## Interactive Controls

- **Drag** nodes to rearrange
- **Hover** over nodes for truth tables
- **Hover** over edges for rule names
- **Scroll** to zoom in/out
- **Drag background** to pan

## Formula Syntax

```python
from boolean_functions import Var, Const, Not, And, Or, Implies

# Variables
p = Var("p")
q = Var("q")

# Constants
true = Const(1)
false = Const(0)

# Operators
not_p = Not(p)           # ¬p
p_and_q = And(p, q)      # p ∧ q
p_or_q = Or(p, q)        # p ∨ q
p_implies_q = Implies(p, q)  # p → q

# Compound formulas
complex = Implies(And(p, q), Or(p, Not(q)))  # (p ∧ q) → (p ∨ ¬q)
```

## Common Axiom Sets

### Modus Ponens

```python
axioms = [Var("x"), Implies(Var("x"), Var("y"))]
# Proves: y
```

### Modus Tollens

```python
axioms = [Implies(Var("p"), Var("q")), Not(Var("q"))]
# Proves: ¬p
```

### Disjunctive Syllogism

```python
axioms = [Or(Var("p"), Var("q")), Not(Var("p"))]
# Proves: q
```

### Chain of Implications

```python
axioms = [
    Var("a"),
    Implies(Var("a"), Var("b")),
    Implies(Var("b"), Var("c"))
]
# Proves: b, c, and (a → c)
```

## Troubleshooting

**Too many formulas?**

- Reduce `max_depth` (try 1 or 2)
- Use fewer variables
- Set `include_all=False` to show only entailed formulas

**HTML file won't open?**

- Try Firefox (works best)
- Or run: `python -m http.server` and open <http://localhost:8000>

**Import errors?**

```bash
pip install networkx matplotlib pyvis pandas numpy
```

## Next Steps

See `README.md` for:

- Complete API reference
- Advanced usage patterns
- Performance tuning
- Custom axiom systems
