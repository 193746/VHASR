import argparse
import logging
import os
from pathlib import Path
from typing import Union

import torch
import yaml

# from funasr.build_utils.build_model import build_model
from build_model import build_asr_model
from funasr.models.base_model import FunASRModel
from funasr.build_utils.build_model import build_model
from funasr.models.frontend.wav_frontend import WavFrontend
from transformers import CLIPProcessor


def build_model_from_file(
        model_file: Union[Path, str] = None,
        config_file: Union[Path, str] = None,
        cmvn_file: Union[Path, str] = None,
        device: str = "cpu",
        task_name: str = "asr",
        mode: str = "paraformer",
):
    """Build model from the files.

    This method is used for inference or fine-tuning.

    Args:
        config_file: The yaml file saved when training.
        model_file: The model file saved when training.
        device: Device type, "cpu", "cuda", or "cuda:N".

    """
    if config_file is None:
        assert model_file is not None, (
            "The argument 'model_file' must be provided "
            "if the argument 'config_file' is not specified."
        )
        config_file = Path(model_file).parent / "config.yaml"
    else:
        config_file = Path(config_file)

    with config_file.open("r", encoding="utf-8") as f:
        args = yaml.safe_load(f)

    if cmvn_file is not None:
        args["cmvn_file"] = cmvn_file
    else:
        cmvn_file = Path(model_file).parent / "am.mvn"
        args["cmvn_file"] = cmvn_file
    
    clip_config_file=Path(model_file).parent / "clip_config"

    args = argparse.Namespace(**args)
    args.infer=True
    args.file_path=Path(model_file).parent
    model = build_asr_model(args)

    model.to(device)
    model_dict = dict()
    model_name_pth = None
    if model_file is not None:
        logging.info("model_file is {}".format(model_file))
        if device == "cuda":
            device = f"cuda:{torch.cuda.current_device()}"
        model_dir = os.path.dirname(model_file)
        model_name = os.path.basename(model_file)
        model_dict = torch.load(model_file, map_location=device)
        model.load_state_dict(model_dict)

    preprocessor = CLIPProcessor.from_pretrained(clip_config_file)

    return model, args, preprocessor


