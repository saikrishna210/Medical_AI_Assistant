EMERGENCY_KEYWORDS = [
    "chest pain",
    "difficulty breathing",
    "severe breathing difficulty",
    "unconscious",
    "not responding",
    "severe bleeding",
    "stroke",
    "seizure",
    "suicidal thoughts"
]


def check_emergency(user_question):
    question = user_question.lower()

    for keyword in EMERGENCY_KEYWORDS:
        if keyword in question:
            return True

    return False
