## [VHASR: A Multimodal Speech Recognition System With Vision Hotwords](https://arxiv.org/abs/2410.00822)
This repository is the official implementation of VHASR. The paper has been accepted by EMNLP 2024.
### Prepare dataset
Download the image and audio data from {dataset} and place them separately in VHASR/dataset/{dataset}/image and VHASR/dataset/{dataset}/audio
```sh
VHASR/
│
└── dataset/
    ├── ADE20k/
    │   ├── image/
    |   |   ├── ADE_train_00000001.jpg
    |   |   ├── ADE_train_00000002.jpg
    |   |   └── ......
    │   ├── audio/
    |   |   ├── ade20k_train_0000000000000001_90.ogg
    |   |   ├── ade20k_train_0000000000000002_72.ogg
    |   |   └── ......
    │   └── train_data/   
    ├── COCO/
    ├── Flickr8k/
    └── OpenImages/
```
VHASR/dataset/{dataset}/train_data/{split}/img.txt and VHASR/dataset/{dataset}/train_data/{split}/wav.scp record the required image path and audio path.

The audio of Flickr8k is available at https://sls.csail.mit.edu/downloads/placesaudio/downloads.cgi.

The audios of ADE20k, COCO and OpenImages are available at https://google.github.io/localized-narratives/.

### Prepare pretrained model
Download [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32/tree/main) and place it on VHASR/pretrained-model/clip-vit-base-patch32
```sh
VHASR/
│
└── pretrained_model/
    └── clip-vit-base-patch32/
        ├── config.json
        ├── merges.txt
        ├── pytorch_model.bin
        └── ......
```

### Install packages
```sh
pip install -r requirements.txt
```

### Train
Download base model: "speech_paraformer_asr-en-16k-vocab4199-pytorch"
```sh
from modelscope import snapshot_download
snapshot_download('damo/speech_paraformer_asr-en-16k-vocab4199-pytorch',local_dir='{path_to_save_model}')
```

Copy the model file and othe config files to VHASR/pretrained_model/VHASR_base
```sh
cp -rn {path_to_save_model}/speech_paraformer_asr-en-16k-vocab4199-pytorch/* VHASR/pretrained_model/VHASR_base
```

Start training.
```sh
cd VHASR
CUDA_VISIBLE_DEVICES=1 python src/finetune.py \
--model_name "pretrained_model/VHASR_base" \
--output_dir "{dataset}_checkpoint" \
--data_path "dataset/{dataset}/train_data" \
--epoch 120 
```

Follow our paper, you can use [VHASR pretrained on SpokenCOCO](https://drive.google.com/drive/folders/1fXQhNITijB2pG1R0ove9qskkU6ivMDnU?usp=drive_link) as the base model for training.
```sh
cd VHASR
CUDA_VISIBLE_DEVICES=1 python src/finetune.py \
--model_name "pretrained_model/VHASR_pretrain" \
--output_dir "{dataset}_checkpoint" \
--data_path "dataset/{dataset}/train_data" \
--epoch 120 
```
After training, place the trained model file and other configuration files in the same folder for subsequent testing.
```sh
cd VHASR
mkdir pretrained_model/my_VHASR_{dataset}
ls pretrained_model/VHASR_base/ | grep -v model.pb | xargs -i cp -r pretrained_model/VHASR_base/{} pretrained_model/my_VHASR_{dataset}
cp {dataset}_checkpoint/valid.acc.best.pb pretrained_model/my_VHASR_{dataset}/model.pb
```

### Test
Test you own trained model. "merge_method" can select 1, 2, or 3, corresponding to $M_1$, $M_2$, and $M_3$ in the paper, respectively.
```sh
cd VHASR
CUDA_VISIBLE_DEVICES=1 python src/evaluate.py \
--model_name "pretrained_model/my_VHASR_{dataset}" \
--data_path "dataset/{dataset}/train_data" \
--merge_method 3
```

You can download [Trained VHASR](https://drive.google.com/drive/folders/1fXQhNITijB2pG1R0ove9qskkU6ivMDnU?usp=drive_link) and put the files in VHASR/pretrained_model/VHASR_{dataset}.
```sh
VHASR/
│
└── pretrained_model/
    ├── VHASR_ADE20k/
    |   ├── am.mvn
    |   ├── config.yaml
    |   ├── model.pb
    |   └── ......
    ├── VHASR_COCO/
    ├── VHASR_Flickr8k/
    └── VHASR_OpenImages/
```

Start testing.
```sh
cd VHASR
CUDA_VISIBLE_DEVICES=1 python src/evaluate.py \
--model_name "pretrained_model/VHASR_{dataset}" \
--data_path "dataset/{dataset}/train_data" \
--merge_method 3
```

### Statement
Most of the code in this repository is modified from https://github.com/modelscope/FunASR/tree/v0.8.8 

### Citation
```sh
@misc{hu2024vhasrmultimodalspeechrecognition,
      title={VHASR: A Multimodal Speech Recognition System With Vision Hotwords}, 
      author={Jiliang Hu and Zuchao Li and Ping Wang and Haojun Ai and Lefei Zhang and Hai Zhao},
      year={2024},
      eprint={2410.00822},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2410.00822}, 
}
```