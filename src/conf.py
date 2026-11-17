# import os

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=f"{Path(__file__).resolve().parent.parent}/.env", env_file_encoding="utf-8")

    # Nanti masing-masing endpoint harus punya endpoint buat handle overall pipeline atau
    # test satu per satu.

    PHASE_3_BASE_ENDPOINT: str
    NGROK_PHASE_3_TOKEN: str

settings = Settings()