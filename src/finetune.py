import os
# from modelscope.metainfo import Trainers
# from modelscope.trainers import build_trainer
# from modelscope.msdatasets.audio.asr_dataset import ASRDataset
from asr_dataset import ASRDataset
from asr_trainer import ASRTrainer

def modelscope_finetune(params):
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir, exist_ok=True)
    ds_dict = ASRDataset.load(params.data_path, namespace='speech_asr')
    asr_trainer=ASRTrainer(model=params.model,
                           data_dir=ds_dict,
                           dataset_type=params.dataset_type,
                           work_dir=params.output_dir,
                           batch_bins=params.batch_bins,
                           max_epoch=params.max_epoch,
                           lr=params.lr,
                           distributed=False)
    asr_trainer.train()

if __name__ == '__main__':
    from funasr.utils.modelscope_param import modelscope_args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='path to model')
    parser.add_argument('--output_dir', type=str, required=True, help='checkpoint path')
    parser.add_argument('--data_path', type=str, required=True, help='path to training data')
    parser.add_argument('--dataset_type', type=str, default="small")
    parser.add_argument('--batch_bins', type=int, default=4000, help='fbank frames')
    parser.add_argument('--epoch', type=int, default=100, help='training epoch')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    args = parser.parse_args()

    params = modelscope_args(model=args.model_name)
    params.output_dir = args.output_dir  
    params.data_path = args.data_path  
    params.dataset_type = args.dataset_type
    params.batch_bins = args.batch_bins
    params.max_epoch = args.epoch
    params.lr = args.lr  
    modelscope_finetune(params)