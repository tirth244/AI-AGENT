from dotenv import load_dotenv
import os

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq



load_dotenv()


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file")

    return ChatGroq(
        groq_api_key=api_key,
        model="llama-3.1-8b-instant",
        temperature=0.7
    )


def get_prompt():
    return PromptTemplate(
        input_variables=["tone", "target", "audience"],
        template="""
You are a professional LinkedIn content writer.

Write a concise and engaging LinkedIn post.

Requirements:
- Professional tone
- Clear structure
- Use relevant emojis naturally
- Add CTA if suitable
- Include relevant hashtags

Details:
Tone: {tone}
Target: {target}
Audience: {audience}
"""
    )


def generate_post(llm, prompt, tone, target, audience):
    final_prompt = prompt.format(
        tone=tone.strip(),
        target=target.strip(),
        audience=audience.strip()
    )

    response = llm.invoke(final_prompt)

    return response.content


def main():
    print("=== LinkedIn Post Generator ===\n")

    tone = input("Tone: ")
    target = input("Target: ")
    audience = input("Audience: ")

    llm = get_llm()
    prompt = get_prompt()

    post = generate_post(
        llm,
        prompt,
        tone,
        target,
        audience
    )

    print("\nGenerated LinkedIn Post:\n")
    print(post)


if __name__ == "__main__":
    main()