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
        if len(phase2_input["1"]["reviews"]) == 0:
            return None, None
        else: return phase2_input["1"]["app_name"], phase2_input["1"]["reviews"]
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

def EntailmentScore(premises: list[str], hypothesis: list[str]) -> float:
    try:
        resp = requests.post(
            f"{settings.PHASE_3_BASE_ENDPOINT}/entailment_score",
            json={"premises": premises, "hypothesis": hypothesis},
        )
        resp.raise_for_status()
        return resp.json()["score"]
    except requests.exceptions.HTTPError as erh:
        print(erh)
        raise

def Pipeline(app_id, reviews):
    print("=" * 60)
    print(f"[Phase 1: NLI Filter] Input reviews: {len(reviews)}")
    print("=" * 60)
    app_name, filtered_reviews = NLI(app_id, reviews)
    if app_name is None:
        print("No relevant reviews passed NLI filter.")
        return "No reviews"

    print(f"App Name: {app_name}")
    print(f"Kept Reviews ({len(filtered_reviews)}):")
    for idx, rev in enumerate(filtered_reviews, 1):
        print(f"  {idx}. {rev}")

    print("\n" + "=" * 60)
    print("[Phase 2: ABSA]")
    print("=" * 60)
    app_name, aspects, absa_reviews = ABSA(app_name, filtered_reviews)
    print(f"Extracted Aspects ({len(aspects)}):")
    for idx, aspect in enumerate(aspects, 1):
        print(f"  {idx}. {aspect}")

    print("\n" + "=" * 60)
    print("[Phase 3: Summarizer]")
    print("=" * 60)
    summary = Summarizer(app_name, aspects, absa_reviews)
    print(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print("[Phase 3: Entailment Score]")
    print("=" * 60)
    ground_text = "\n".join(f"- {str(r).strip()}" for r in absa_reviews)
    score = EntailmentScore([ground_text], [summary["msg"]])
    print(f"Entailment Score: {score:.4f}")
    print("=" * 60)
    return summary['msg']

    