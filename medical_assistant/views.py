import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from django.shortcuts import render
from ai_engine.rag_engine import medical_symptom_rag


# Load API key from the project root .env file
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env",override=True)


def home(request):
    result = None
    error = None

    if request.method == "POST":

        symptoms = request.POST.get("symptoms", "")
        duration = request.POST.get("duration", "")
        severity = request.POST.get("severity", "")
        additional_symptoms = request.POST.get(
            "additional_symptoms", ""
        )

        api_key = os.getenv("GROQ_API_KEY")
        print("API key loaded:", bool(api_key))
        print("API key length:", len(api_key) if api_key else 0)
        print("Starts with gsk_:", api_key.startswith("gsk_") if api_key else False)

        if not api_key:
            error = "Groq API key is missing. Please check your .env file."

        else:
            try:
                client = Groq(api_key=api_key)

                user_details = f"""
                Main symptoms: {symptoms}
                Duration: {duration}
                Severity: {severity}
                Additional symptoms: {additional_symptoms}
                """

                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {
                            "role": "system",
                            "content": """
                            You are a medical information assistant.

                            Explain possible general health-related reasons
                            for the symptoms in simple English.


                            Follow this response format:

                            1. Important Note
                            2. Possible Health-Related Reasons
                            3. Why These Symptoms May Occur
                            4. Recommended Next Steps
                            5. When to Seek Emergency Help

                            Use simple paragraphs and numbered lists.

                            Do not use Markdown symbols, tables, HTML tags,
                            or backslash escape characters.

                            Never provide a definite diagnosis.
                            Never prescribe medicines or dosages.
                            Recommend appropriate medical evaluation.
                            Do not claim that a condition is confirmed
                            based only on symptoms.

                        For excessive thirst and frequent urination,
                        mention that a healthcare professional may
                        recommend blood glucose testing.

                        Clearly state that this is educational
                        information, not a medical diagnosis.
                        """

                        },
                        {
                            "role": "user",
                            "content": user_details
                        }
                    ],
                    max_completion_tokens=800
                )

                # result = response.choices[0].message.content
                result= medical_symptom_rag(user_details)
                print("========== CALLING RAG FUNCTION ==========", flush=True)
                print("========== RAG FUNCTION RETURNED ==========", flush=True)
                print("RESULT:", result, flush=True)

            except Exception as e:
                error = f"Unable to generate response: {str(e)}"

    return render(request,"home.html",{
        "result": result,
        "error": error
        }
    )