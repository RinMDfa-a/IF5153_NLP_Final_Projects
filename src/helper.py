import os
import sys

from core import *
from enums import Enum, Functions


def function_matcher(func) -> Enum.name:
    """Helper buat mengembalikan tipe fungsi untuk mempermudah routing"""
    match func:
        case f if f is menu:
            return Functions.MENU.name
        case f if f is demo_menu:
            return Functions.DEMO_MENU.name
        case f if f is NLI_menu:
            return Functions.NLI_MENU.name
        case f if f is ABSA_menu:
            return Functions.ABSA_MENU.name
        case f if f is Summarizer_menu:
            return Functions.SUMMARIZE_MENU.name
        case f if f is Pipeline_menu:
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
                NLI_menu()
            elif choice == 2:
                ABSA_menu()
            elif choice == 3:
                Summarizer_menu()
            elif choice == 4:
                Pipeline_menu() 
            else:
                menu()
        
def action_transition() -> None:
    print("Aksi telah selesai, anda akan dikembalikan ke menu.")
    menu()

def input_retryer(valid_options: list):
    """Helper buat loop masukkan input sesuai dengan pilihan yang tersedia."""
    while True:
        print(f"Pilihan yang valid: {valid_options}")
        response = input().rstrip()
        if response not in valid_options:
            continue
        else: return int(response)

def file_directory_input_retryer():
    """Helper buat loop masukkan input nama direktori"""
    while True:
        file_directory = input().rstrip()
        if not os.path.exists(file_directory):
            print("file tidak ditemukan.")
            continue
        else: return file_directory

def input_handler(func: object):
    """Handler error sekaligus input."""
    function = function_matcher(func)

    match function:
        case f if f is Functions.MENU.name:
            return input_retryer(["1", "2"])
        case f if f is Functions.DEMO_MENU.name:
            return input_retryer(["1", "2", "3", "4", "5"])
        case f if f is Functions.SUMMARIZE_MENU.name:
            return input_retryer(["1", "2"])

def NLI_menu():
    pass

def ABSA_menu():
    pass

def Summarizer_menu():
    print("Tekan angka yang tersedia untuk melakukan aksi:")
    print("1. Uji coba summarizer")
    print("2. Kembali")

    choice = input_handler(Summarizer_menu)

    if choice == 1:
        print("Masukkan nama direktori (relatif dari posisi program dijalankan) dimana file masukkan untuk model summarizer \nsudah sesuai dengan format yang ditentukan.")
        directory = file_directory_input_retryer()
        data = Input_Parser(Summarizer_menu, directory=directory, prev_data=None)
        print(data)
        print(Summarizer(data))
        action_transition()
    else:
        router(choice, Summarizer_menu)

def Pipeline_menu():
    print("Tekan angka yang tersedia untuk melakukan aksi:")
    print("1. Uji coba pipeline")
    print("2. Kembali")

    choice = input_handler(Pipeline_menu)
    router(choice, Pipeline_menu)

# TODO buat parser file.txt and parser hasil output model lain
def Input_Parser(func: object, directory: str | None, prev_data: str | None):
    """Helper untuk melakukan parsing atau membaca file untuk melakukan demo per fase
    baik untuk contoh per fase atau full pipeline."""
    function = function_matcher(func)

    match function:
        case f if f is Functions.SUMMARIZE_MENU.name:
            lines = []
            with open(directory, mode="r") as file:
                for line in file:
                    lines.append(line)
            return {'content': " ".join(lines)}
        case f if f is Functions.PIPELINE_MENU.name:
            return "parser result"




    