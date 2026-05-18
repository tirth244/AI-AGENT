import math
import os
from typing import Dict

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent



load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables.")


SAFE_GLOBALS = {
    "__builtins__": {},
    "math": math,
}


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    Example:
    - 2 + 5 * 3
    - math.sqrt(25)
    - 2 ** 10
    """

    try:
        result = eval(expression, SAFE_GLOBALS, {})
        return f"Result: {result}"

    except Exception as e:
        return f"Calculation Error: {str(e)}"


MOCK_DB: Dict[int, str] = {
    42: "Name: Alice Johnson | Role: Senior Engineer | Access: Admin",
    85: "Name: Bob Smith | Role: Data Analyst | Access: Standard",
}


@tool
def lookup_user(user_id: int) -> str:
    """
    Lookup user information by user ID.
    """

    return MOCK_DB.get(
        user_id,
        f"User ID {user_id} not found."
    )


tools = [calculator, lookup_user]

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a professional AI assistant. "
        "Use tools whenever calculations or user lookups are required."
    ),
)


def main():
    print("\nAI Assistant Started")
    print("Type 'exit' or 'quit' to stop.\n")

    chat_history = []

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break

            chat_history.append({
                "role": "user",
                "content": user_input
            })

            response = agent.invoke({
                "messages": chat_history
            })

            messages = response.get("messages", [])

            if not messages:
                print("Agent: No response generated.")
                continue

            final_message = messages[-1]

            print(f"\nAgent: {final_message.content}\n")

            # Preserve full conversation
            chat_history = messages

        except KeyboardInterrupt:
            print("\nInterrupted. Exiting...")
            break

        except Exception as e:
            print(f"\nUnexpected Error: {e}\n")


if __name__ == "__main__":
    main()