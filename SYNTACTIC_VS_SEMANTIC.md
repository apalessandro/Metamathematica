# Syntactic vs Semantic Distinction

## Summary of Changes

The library now clearly distinguishes between two fundamental approaches to logic:

### 1. **Syntactic Derivability** (Proof-Based)

- **Function name:** `build_syntactic_graph()`
- **Approach:** Forward-chaining with inference rules
- **What it does:** Starts with axioms and applies inference rules (Modus Ponens, Modus Tollens, etc.) to derive new formulas step-by-step
- **Key characteristic:** Shows the **proof path** - how each formula is derived
- **Formula count:** Focused and efficient (typically 20-100 formulas)
- **Use when:** You want to see **how** formulas are proven from axioms

### 2. **Semantic Entailment** (Truth-Based)

- **Function name:** `build_semantic_graph()`
- **Approach:** Exhaustive formula enumeration with truth checking
- **What it does:** Generates all possible formulas up to a depth, then checks which are true in all models where axioms are true
- **Key characteristic:** **Complete** for propositional logic - finds ALL logical consequences
- **Formula count:** Comprehensive (typically 1000s of formulas)
- **Use when:** You want to find **all** formulas that are logically entailed

## Key Relationship

```text
Syntactic ⊆ Semantic
```

- Everything **syntactically derivable** IS **semantically entailed**
- NOT everything **semantically entailed** can be **syntactically derived** with a finite set of inference rules

### Example

Given axioms: `p`, `p → q`, `q → r`

**Syntactic approach derives:**

- `p`, `q`, `r` (via Modus Ponens)
- `p ∧ q`, `p ∧ r`, `q ∧ r` (via Conjunction Introduction)
- `p → r` (via Hypothetical Syllogism)
- Total: ~20 formulas

**Semantic approach finds:**

- All of the above PLUS
- `(p ∧ q) → r` (true in all models, but not directly derivable)
- `(r ∨ p)` (tautology given axioms)
- `(p → p)`, `(q → q)`, `(r → r)` (tautologies)
- Many more complex formulas true under axioms
- Total: ~2985 formulas

## Documentation Updates

1. **Module docstring** - Explains both approaches and their distinction
2. **Function names** - Renamed to `build_syntactic_graph()` and `build_semantic_graph()`
3. **README.md** - Added comprehensive "Key Concepts" section explaining:
   - Syntactic Derivability (Proof-Based)
   - Semantic Entailment (Truth-Based)
   - Their relationship and when to use each
4. **Function docstrings** - Updated to clarify the approach each uses
5. **Examples** - Updated throughout to use new names

## When to Use Each Approach

### Use Syntactic Derivability When

- Teaching logic and proof techniques
- Understanding **how** conclusions follow from premises
- Working with many variables (scales better)
- You want focused, interpretable results
- Showing derivation paths is important

### Use Semantic Entailment When

- You need **all** logical consequences
- Exploring the complete formula space
- Finding tautologies and complex entailments
- Completeness is required
- Working with 2-3 variables only (due to exponential growth)
