from select_items import AvailableDeviceName, AvailableModelName
from translator import _get_model_path


from transformers import AutoModelForSeq2SeqLM


from pathlib import Path
from typing import Any


def _load_model(
    model_name: AvailableModelName,
    device_name: AvailableDeviceName,
    cache_dir: Path,
) -> Any:
    model_path = _get_model_path(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, cache_dir=cache_dir)
    model = model.to(device_name.value)

    return model
