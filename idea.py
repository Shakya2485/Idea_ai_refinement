# 1. Import necessary libraries
import os
import re
import torch
import datetime
import numpy as np
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents import initialize_agent, Tool
from langchain_core.runnables import RunnableLambda
from langchain.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraphExecutor
from langgraph.graph import StateGraph, END
from graphviz import Digraph
from IPython.display import Image, display
from pydantic import BaseModel
from IPython.display import Image
import graphviz
from reportlab.pdfgen import canvas
from random import choice

# 2. Authenticate Hugging Face 
HUGGINGFACE_TOKEN = "my_huggingface_token" #insert huggingface token here
login(token=HUGGINGFACE_TOKEN)

# 3. Load LLM 
model_id = "mistralai/Mistral-7B-Instruct-v0.1" # I chosen Mistral Instruct due to resource constraint
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HUGGINGFACE_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    token=HUGGINGFACE_TOKEN
)

def generate_response(prompt_text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        top_p=0.8,
        temperature = 0.6,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 4. Initialize FAISS for Document Retrieval (RAG)
documents = [
    "E-commerce is growing rapidly, with a market size of $3 trillion.",
    "AI in healthcare is revolutionizing diagnostics and treatment planning.",
    "Startups in fintech are raising massive amounts of funding, focusing on digital payments."
]
# documents added for example, It's for Retrieval during generation
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vectorstore directly from texts
vectorstore = FAISS.from_texts(documents, embedding_model)

# 5. Prompt Templates 
prompt_template = PromptTemplate(
    input_variables=["original_idea", "previous_refinement", "context"],
    template="""Startup idea:
"{original_idea}"

{previous_refinement}

Context:
{context}

1. Search the market for similar solutions.
2. Identify gaps, risks, or missed opportunities.
3. Suggest creative improvements, pivots, or refinements. 

Respond with a structured improvement plan.
"""
)

def wrapped_llm(input_dict):
    prompt_text = prompt_template.format(**input_dict)
    raw_output = generate_response(prompt_text)
    return raw_output.replace(prompt_text.strip(), "").strip()

chain = RunnableLambda(wrapped_llm)

# 6. Refinement Tracker and Memory Update
refinement_history = []

# 7. Core Refinement Function (with RAG and Memory)
def refine_idea_with_rag(original_idea, previous_refinement=None):
    previous_text = f"Previous refinement:\n{previous_refinement}" if previous_refinement else ""

    # Retrieve relevant documents from FAISS
    relevant_docs = vectorstore.similarity_search(original_idea, k=3)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    inputs = {
        "original_idea": original_idea,
        "previous_refinement": previous_text,
        "context": context
    }

    output = chain.invoke(inputs).strip()

    # Update memory
    version = {
        "original": original_idea,
        "refined_from": previous_refinement,
        "refined_output": output
    }
    refinement_history.append(version)

    return output

# 8. Optional Critique Agent
def critique_refinement(refined_text):
    critique_prompt = f"Evaluate this idea's feasibility and uniqueness.\n\n{refined_text}\n\nRate 1-10 and explain."
    return generate_response(critique_prompt)

# 9. LangGraph Model
class RefineState(BaseModel):
    original_idea: str
    previous_refinement: str = ""
    refined_output: str = ""
    critique_score: float = 0.0

def refine_node(state: RefineState) -> RefineState:
    output = refine_idea_with_rag(state.original_idea, state.previous_refinement)
    return RefineState(
        original_idea=state.original_idea,
        previous_refinement=state.previous_refinement,
        refined_output=output,
        critique_score=state.critique_score  # carry over
    )

def critique_node(state: RefineState) -> RefineState:
    critique = critique_refinement(state.refined_output)
    match = re.search(r"\b(\d+)\b", critique)
    score = int(match.group(1)) if match else 0

    return RefineState(
        original_idea=state.original_idea,
        previous_refinement=state.previous_refinement,
        refined_output=state.refined_output,
        critique_score=score
    )

def decision_condition(state: RefineState):
    return "refine" if state.critique_score < 8 else "end"



# 10. LangGraph Build (Updated)
builder = StateGraph(RefineState)

# Add nodes
builder.add_node("refine", RunnableLambda(refine_node))
builder.add_node("critique", RunnableLambda(critique_node))
builder.add_node("decide", RunnableLambda(lambda x: x))  # just passes state through

# Define entry point
builder.set_entry_point("refine")

# Define edges
builder.add_edge("refine", "critique")     # First refine the idea
builder.add_edge("critique", "decide")     # Then critique it

# Conditional branching from "decide"
builder.add_conditional_edges(
    "decide",
    RunnableLambda(decision_condition),    # Uses score in state
    {
        "refine": "refine",                # Loop back if score < 8
        "end": END                         # Exit if score >= 8
    }
)


# 11. Visualize LangGraph

viz = Digraph()
viz.attr(rankdir="LR")

# Nodes
viz.node("Start")
viz.node("Refine")
viz.node("Critique")
viz.node("Decide")
viz.node("End")

# Edges
viz.edge("Start", "Refine")
viz.edge("Refine", "Critique")
viz.edge("Critique", "Decide")
viz.edge("Decide", "Refine", label="score < 8")
viz.edge("Decide", "End", label="score ≥ 8")

# Render
viz.render("refinement_graph", format="png", cleanup=True)
display(Image(filename="refinement_graph.png"))


# # 12. Run Loop
# idea = input("Enter your startup idea: ")
# refined = refine_idea_with_rag(idea)
# print("\nFirst Refinement:\n", refined)
# print("\nCritique:\n", critique_refinement(refined))

# while input("\nRefine again? (yes/no): ").strip().lower() == "yes":
#     refined = refine_idea_with_rag(idea, previous_refinement=refined)
#     print("\nNext Refinement:\n", refined)
#     print("\nCritique:\n", critique_refinement(refined))

# 12. Run LangGraph Execution Loop Automatically
# Create graph executor from builder
graph = builder.compile()

# Provide initial state
initial_state = RefineState(original_idea=input("Enter your startup idea: "))

# Run the graph — it will handle refinement, critique, and decision
final_state = graph.invoke(initial_state)

# Print result
print("\nFinal Output:\n", final_state.refined_output)


# 13. Export PDF
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
    print(f"\nExported refinement history to {filename}")

if input("\nExport to PDF? (yes/no): ").lower() == "yes":
    filename = f"refined_idea_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    export_to_pdf(filename)
