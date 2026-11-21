import os
import sys
import json

from core import *
from enums import Functions


def function_matcher(func):
    """Helper buat mengembalikan tipe fungsi untuk mempermudah routing"""
    match func:
        case f if f is menu:
            return Functions.MENU.name
        case f if f is demo_menu:
            return Functions.DEMO_MENU.name
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
    print("1. Pipeline keseluruhan")
    print("2. Kembali")

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
            return input_retryer(["1", "2"])
        case f if f is Functions.PIPELINE_MENU.name:
                    return input_retryer(["1", "2"])

def Pipeline_menu():
    print("Tekan angka yang tersedia untuk melakukan aksi:")
    print("1. Uji coba pipeline")
    print("2. Kembali")

    choice = input_handler(Pipeline_menu)

    if choice == 1:
        print("Masukkan nama direktori (relatif dari posisi program dijalankan) dimana file masukkan untuk pipeline \nsudah sesuai dengan format yang ditentukan.")
        directory = file_directory_input_retryer()
        try: 
            app_id, reviews = Input_Parser(directory)
            print(Pipeline(app_id, reviews))
        except requests.exceptions.HTTPError as e:
            print(e)
            
        action_transition()
    else:
        router(choice, Pipeline_menu)

def Input_Parser(directory: str | None):
    with open(directory, mode="r", encoding="utf-8") as file:
        content = json.load(file)
        app_id = content["app_id"]
        reviews = content["reviews"]

    return app_id, reviews




    