from dataclasses import dataclass
from langchain_core.messages import convert_to_messages
from typing import List
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
import getpass
import os
from dotenv import load_dotenv


load_dotenv()

def get_api_key(env_var, prompt):
    if not os.getenv(env_var):
        os.environ[env_var] = getpass.getpass(prompt)

def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        if "messages" in node_update:
            messages = convert_to_messages(node_update["messages"])
            if last_message:
                messages = messages[-1:]
            for m in messages:
                pretty_print_message(m, indent=is_subgraph)
            print("\n")
            continue

        if "structured_response" in node_update:
            sr = node_update["structured_response"]
            try:
                data = sr.model_dump()
            except Exception:
                data = sr if isinstance(sr, dict) else {"response": getattr(sr, "response", str(sr)),
                                                        "sources": getattr(sr, "sources", [])}
            resp = data.get("response")
            sources = data.get("sources", [])
            indent = "\t" if is_subgraph else ""
            if resp:
                print(f"{indent}{resp}\n")
            if sources:
                print(f"{indent}Sources: {', '.join(sources)}\n")
            print("\n")


def create_mock_retriever(docs: List[str]):
    from langchain.schema import Document
    from langchain_openai.embeddings import OpenAIEmbeddings


    documents = [Document(page_content=doc) for doc in docs]
    embeddings = OpenAIEmbeddings()
    vectorstore = InMemoryVectorStore.from_documents(documents, embeddings)
    return vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "lambda_mult": 0.6}
    )
