from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env file")


llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.9,
    max_tokens=150
)


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a professional live cricket commentator.

Your commentary should:
- Sound realistic and emotional
- Be short and punchy
- Use cricket terminology
- Match the intensity of the situation
- Feel like live TV commentary

Avoid long paragraphs.
"""
    ),
    (
        "human",
        """
Match Situation:
{situation}

Batsman:
{batsman_name}

Bowler:
{bowler_name}

Generate live commentary:
"""
    )
])

# Create Chain
chain = prompt | llm


# ==============================
# Main App
# ==============================
def generate_commentary():
    print("\n🏏 AI Cricket Commentary Generator")
    print("-" * 40)

    while True:
        try:
            # User Inputs
            situation = input("\n📌 Enter match situation: ").strip()
            batsman_name = input("🏏 Enter batsman name: ").strip()
            bowler_name = input("🎯 Enter bowler name: ").strip()

            # Validation
            if not situation or not batsman_name or not bowler_name:
                print("\n⚠️ All fields are required.")
                continue

            # Generate Response
            response = chain.invoke({
                "situation": situation,
                "batsman_name": batsman_name,
                "bowler_name": bowler_name
            })

            # Output
            print("\n🎙 LIVE COMMENTARY:")
            print("-" * 40)
            print(response.content)

        except KeyboardInterrupt:
            print("\n\n👋 Exiting Commentary Generator...")
            break

        except Exception as e:
            print(f"\n❌ Error: {e}")

        # Continue Option
        again = input("\nGenerate another commentary? (y/n): ").lower()

        if again != "y":
            print("\n✅ Thank you for using AI Commentary Generator")
            break


if __name__ == "__main__":
    generate_commentary()