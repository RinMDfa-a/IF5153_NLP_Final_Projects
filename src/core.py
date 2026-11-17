from conf import settings

import requests
import json

def NLI(func:object):
    pass

def ABSA(func:object):
    pass

def Summarizer(formatted_data: any) -> str:
    """Fungsi utama yang melakukan calling via endpoint ngrok buat calling model"""
    try:
        resp = requests.post(f"{settings.PHASE_3_BASE_ENDPOINT}/inference", json=formatted_data)
        resp.raise_for_status()
        data = resp.json()
        return data
    except requests.exceptions.HTTPError as erh:
        return erh.args[0]

def Pipeline():
    pass