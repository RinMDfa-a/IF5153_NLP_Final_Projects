import sys
from enum import Enum, auto

from core import *


class Functions(Enum):
    MENU = auto()
    DEMO_MENU = auto()
    ABSA_MENU = auto()
    NLI_MENU = auto()
    SUMMARIZE_MENU = auto()
    PIPELINE_MENU = auto()

def function_matcher(func) -> Enum.name:
    """Helper buat mengembalikan tipe fungsi untuk mempermudah routing"""
    match func:
        case f if f is menu:
            return Functions.MENU.name
        case f if f is demo_menu:
            return Functions.DEMO_MENU.name
        case f if f is NLI:
            return Functions.NLI_MENU.name
        case f if f is ABSA:
            return Functions.ABSA_MENU.name
        case f if f is Summarizer:
            return Functions.SUMMARIZE_MENU.name
        case f if f is Pipeline:
            return Functions.PIPELINE_MENU.name

def greetings() -> None: 
    print("Selamat datang pada program CLI demo tugas NLP")

def menu() -> None:
    print("Tekan angka yang tersedia untuk melakukan aksi:")
    print("1. Lakukan Demo")
    print("2. Keluar Program")

    choice = input_handler(menu)
    router(choice, menu)

def demo_menu() -> None:
    print("Tekan angka yang tersedia untuk melakukan aksi:")
    print("1. Fase 1 (NLI)")
    print("2. Fase 2 (ABSA)")
    print("3. Fase 3 (Summarizer)") 
    print("4. Pipeline keseluruhan")
    print("5. Kembali")

    choice = input_handler(demo_menu)
    router(choice, demo_menu)

def router(choice: any, func: object) -> None:
    """Helper eksekutor fungsi lanjutan setelah adanya masukkan."""
    function = function_matcher(func)

    match function:
        case f if f is Functions.MENU.name:
            if choice == 1:
                demo_menu()
            else:
                sys.exit()
        case f if f is Functions.DEMO_MENU.name:
            if choice == 1:
                sys.exit()
            if choice == 5:
                menu()
        
def action_transition() -> None:
    print("Aksi telah selesai, anda akan dikembalikan ke menu.")

def input_retryer(valid_options: list):
    """Helper buat loop masukkan input sesuai dengan pilihan yang tersedia."""
    while True:
        print(f"Pilihan yang valid: {valid_options}")
        response = input().rstrip()
        if response not in valid_options:
            continue
        else: return int(response)

def input_handler(func: object):
    """Handler error sekaligus input."""
    function = function_matcher(func)

    match function:
        case f if f is Functions.MENU.name:
            resp = input_retryer(["1", "2"])
            return resp
        case f if f is Functions.DEMO_MENU.name:
            resp = input_retryer(["1", "2", "3", "4", "5"])
            return resp
        
