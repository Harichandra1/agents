# LangGraph Graph API — working notes

*(Day 3. Originally kept as a docstring inside `GraphAPI.py`; notes belong in
markdown, not in a module that pretends to be code.)*

Three things matter in LangGraph:

1. **State**
2. **Nodes**
3. **Edges**

The underlying graph algorithm uses message passing.

## State

Consists of the schema of the graph plus reducer functions. Can be a
`TypedDict` or a Pydantic model. There can be multiple schemas.

## Reducers

If no reducer is mentioned, all updates to that key **overwrite** it.

Every reducer takes two args:

- **left arg** — the current value already stored in state for that key
- **right arg** — the update for that key returned by a node

Custom reducer:

```python
class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]
```

In some cases you may want to bypass a reducer and directly overwrite a state
value. LangGraph provides the `Overwrite` type for this.

## Nodes

Receive `state`, `config`, `runtime`. Node caching is available.

- **`Send`** — used in fan-out situations: sends a separate state to a node.
- **`Command`** — lets a node return both a state update *and* an instruction
  about where the graph should go next.
