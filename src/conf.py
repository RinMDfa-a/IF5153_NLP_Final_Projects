# import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="nanti dilengkapin", env_file_encoding='utf-8')

    # Nanti masing-masing endpoint harus punya endpoint buat handle overall pipeline atau
    # test satu per satu.

    PHASE_1_BASE_ENDPOINT: str
    PHASE_2_BASE_ENDPOINT: str
    PHASE_3_BASE_ENDPOINT: str

settings = Settings()