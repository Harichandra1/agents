"""
Little more notes on Graph API overview 
Three things matter in Langgraph 
1. State 
2. Nodes
3. Edges
Underlying graph alogrithm uses message passing 

State 
It consist of schema of the graph and reducer functions 
Can be TypedDict,Pydantic model

Their can be multiple schemas

Reducers
If no reducer mention then assumed that all updates to that key should be override it 
Every reducer has two args 
left arg : the current value already stored in state for that key 
right arg: the update for that key returned by a node 
Custom reducer 
class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]


In some cases, you may want to bypass a reducer and directly overwrite a state value. LangGraph provides the Overwrite type for this purpose. Learn how to use Overwrite here.

Node : state , config,runtime
Node caching 
Send is use in fan out situations : basically to send a seprate state to a node 
Command : lets a node return both a state and an instruction about where the graph should go next. 
"""





