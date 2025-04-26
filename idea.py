# 1. Import necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from huggingface_hub import login
from reportlab.pdfgen import canvas
from IPython.display import Image
import datetime
import os
import torch
import graphviz

# 2. Authenticate Hugging Face
HUGGINGFACE_TOKEN = "your_huggingface_token"
login(token=HUGGINGFACE_TOKEN)

# 3. Load Mistral model and tokenizer
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HUGGINGFACE_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    token=HUGGINGFACE_TOKEN
)

def generate_response(prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        top_k=50,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 4. Prompt Template
prompt = PromptTemplate(
    input_variables=["original_idea", "previous_refinement"],
    template="""
You are an expert AI startup consultant.

Startup idea:
"{original_idea}"

{previous_refinement}

1. Search the market for similar solutions.
2. Identify gaps, risks, or missed opportunities.
3. Suggest creative improvements, pivots, or refinements.

Respond with a clear, structured improvement plan.
"""
)

# 5. Chain using wrapped LLM
def wrapped_llm(input_dict):
    prompt_text = prompt.format(**input_dict)
    raw_output = generate_response(prompt_text)
    return raw_output.replace(prompt_text.strip(), "").strip()

chain = RunnableLambda(wrapped_llm)


# 6. Refinement Function
refinement_history = []
def refine_idea(original_idea, previous_refinement=None):
    previous_text = f"Previous refinement:\n{previous_refinement}" if previous_refinement else ""
    inputs = {
        "original_idea": original_idea,
        "previous_refinement": previous_text
    }
    output = chain.invoke(inputs).strip()

    version = {
        "original": original_idea,
        "refined_from": previous_refinement,
        "refined_output": output
    }
    refinement_history.append(version)
    return output

# 7. LangGraph
class RefineState(BaseModel):
    original_idea: str
    previous_refinement: str = ""
    refined_output: str = ""

def process_node(state: RefineState) -> RefineState:
    output = refine_idea(state.original_idea, state.previous_refinement)
    return RefineState(
        original_idea=state.original_idea,
        previous_refinement=state.previous_refinement,
        refined_output=output
    )


# 8. LangGraph Build
builder = StateGraph(RefineState)
builder.add_node("refine", RunnableLambda(refine_node))
builder.add_node("decide", RunnableLambda(lambda x: x))
builder.set_entry_point("refine")
builder.add_edge("refine", "decide")
builder.add_conditional_edges(
    "decide",
    RunnableLambda(decision_condition),
    {
        "refine": "refine",
        "end": END
    }
)

# 9. Visualize the Graph
viz = graphviz.Digraph()
viz.attr(rankdir="LR")
viz.node("Start")
viz.node("Refine")
viz.node("Decide")
viz.node("End")
viz.edge("Start", "Refine")
viz.edge("Refine", "Decide")
viz.edge("Decide", "Refine", label="yes")
viz.edge("Decide", "End", label="no")
viz.render("refinement_graph", format="png", cleanup=True)
display(Image(filename="refinement_graph.png"))

# 10. Refinement Loop
idea = input("Enter your startup idea: ")
refined = refine_idea(idea)
print("\n First Refinement:\n", refined)

while input("\n Refine again? (yes/no): ").strip().lower() == "yes":
    refined = refine_idea(idea, previous_refinement=refined)
    print("\n Next Refinement:\n", refined)

# 11. Show Refinement History
print("\n Refinement History:")
for i, version in enumerate(refinement_history):
    print(f"\n--- Version {i+1} ---")
    print("Original:", version["original"])
    if version["refined_from"]:
        print("Refined From (truncated):", version["refined_from"][:100], "...")
    print("Refined Output:\n", version["refined_output"])

# 12. Export to PDF

def export_to_pdf(filename="refined_idea.pdf"):
    c = canvas.Canvas(filename)
    c.setFont("Helvetica", 12)
    y = 800
    for i, version in enumerate(refinement_history):
        c.drawString(50, y, f"--- Version {i+1} ---")
        y -= 20
        for line in version["refined_output"].split("\n"):
            c.drawString(60, y, line[:100])
            y -= 15
            if y < 50:
                c.showPage()
                y = 800
    c.save()
    print(f"\n Exported refinement history to {filename}")

if input("\n Export to PDF? (yes/no): ").lower() == "yes":
    filename = f"refined_idea_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    export_to_pdf(filename)
