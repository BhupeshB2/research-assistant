# main.py

from dotenv import load_dotenv
load_dotenv()

from app.chain import build_chain_with_memory

def run():
    chain = build_chain_with_memory()
    session_id = "cli-session-1"

    print("Research Assistant ready. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break

        result = chain.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}},
        )

        print(f"\nAssistant: {result.answer}")
        print(f"Sources: {', '.join(result.sources)}\n")

if __name__ == "__main__":
    run()