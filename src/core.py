import json

import requests

from conf import settings


def NLI(app_id: int, reviews: list[str]):
    try:
        resp = requests.post(f"{settings.PHASE_1_BASE_ENDPOINT}/phase1/filter", params={"top_k": 30},
                            json={"1": {"app_id": app_id, "reviews": reviews}}, 
                            timeout=120, 
                            headers={"X-API-Key": settings.PHASE1_API_KEY})
        resp.raise_for_status()
        phase2_input = {k: {"app_name": g["app_name"], "reviews": g["reviews"]} for k, g in resp.json().items()}
        return phase2_input["1"]["app_name"], phase2_input["1"]["reviews"]
    except requests.exceptions.HTTPError as erh:
        print(erh)
        raise

def ABSA(app_name: str, reviews: list[str]):
    try:
        resp = requests.post(f"{settings.PHASE_2_BASE_ENDPOINT}/inference",
                              json={"app_name": app_name, "reviews": reviews})
        resp.raise_for_status()
        phase3_input = resp.json()
        return phase3_input["app_name"], phase3_input["raw_prediction"], phase3_input["reviews"] 
    except requests.exceptions.HTTPError as erh:
        print(erh)
        raise

def Summarizer(app_name: str, aspects: list[dict], reviews: list[str]) -> str:
    try:
        Aspects = [f"{aspect}" for aspect in aspects]
        Reviews = [f"{review}" for review in reviews]
        formatted_reviews = "\n".join(f"- {r.strip()}" for r in Reviews)
        clean_aspects = [a.replace("Aspects:", "").strip() for a in Aspects]
        formatted_aspects = ", ".join(clean_aspects)

        content = (f"Summarize {app_name} based on Reviews:\n{formatted_reviews}\n\n"
                   f"Aspects: {formatted_aspects}")

        resp = requests.post(f"{settings.PHASE_3_BASE_ENDPOINT}/inference", 
                             json={"content": content})
        resp.raise_for_status()
        data = resp.json()
        return data
    except requests.exceptions.HTTPError as erh:
        print(erh)
        raise

def Pipeline(app_id, reviews):
    print("Processing NLI:")
    app_name, reviews = NLI(app_id, reviews)
    print("\n")
    print("ABSA:")
    app_name, aspects, reviews = ABSA(app_name, reviews)
    print("\n")
    print("Summarizer:")
    summary = Summarizer(app_name, aspects, reviews)
    print("\n")
    return summary

    