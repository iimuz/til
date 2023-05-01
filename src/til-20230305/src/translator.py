from pathlib import Path
from typing import Any

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from select_items import AvailableDeviceName, AvailableLanguage, AvailableModelName


class Translator:
    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._model: None | any = None
        self._tokenizer: None | any = None

    def load(
        self,
        model_name: AvailableModelName,
        source_language: AvailableLanguage,
        device_name: AvailableDeviceName,
    ) -> None:
        self._model = _load_model(
            model_name=model_name,
            device_name=device_name,
            cache_dir=self._cache_dir,
        )
        self._tokenizer = _load_tokenizer(
            model_name=model_name,
            source_language=source_language,
            cache_dir=self._cache_dir,
        )

    def translate(
        self, source: str, target_language: AvailableLanguage, max_length: int
    ) -> str:
        if self._model is None:
            raise ValueError("no load model")
        if self._tokenizer is None:
            raise ValueError("no load tokenizer")

        inputs = self._tokenizer(source, return_tensors="pt")
        translated_tokens = self._model.generate(
            **inputs.to(self._model.device),
            forced_bos_token_id=self._tokenizer.lang_code_to_id[target_language.value],
            max_length=max_length,
        )
        result = self._tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True
        )[0]

        return result


def _get_model_path(model_name: AvailableModelName) -> str:
    # 利用可能なモデル一覧からモデルのパスを返す.
    model_paths = {
        AvailableModelName.NLLB_200_DISTILLED_600M: "facebook/nllb-200-distilled-600M",
        AvailableModelName.NLLB_200_DISTILLED_1_3B: "facebook/nllb-200-distilled-1.3B",
        AvailableModelName.NLLB_200_1_3B: "facebook/nllb-200-1.3B",
        AvailableModelName.NLLB_200_3_3B: "facebook/nllb-200-3.3B",
    }

    return model_paths[model_name]


def _load_model(
    model_name: AvailableModelName,
    device_name: AvailableDeviceName,
    cache_dir: Path,
) -> Any:
    model_path = _get_model_path(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, cache_dir=cache_dir)
    model = model.to(device_name.value)

    return model


def _load_tokenizer(
    model_name: AvailableModelName,
    source_language: AvailableLanguage,
    cache_dir: Path,
) -> Any:
    model_path = _get_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        src_lang=source_language.value,
    )

    return tokenizer
