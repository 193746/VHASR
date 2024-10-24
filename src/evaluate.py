import os
# from modelscope.metainfo import Trainers
# from modelscope.trainers import build_trainer
# from modelscope.msdatasets.audio.asr_dataset import ASRDataset
from asr_trainer import ASRTrainer
from asr_dataset import ASRDataset


def modelscope_finetune(params):
    ds_dict = ASRDataset.load(params.data_path, namespace='speech_asr')
    asr_trainer=ASRTrainer(model=params.model,
                           data_dir=ds_dict,
                           dataset_type=params.dataset_type,
                           batch_bins=params.batch_bins,
                           max_epoch=params.max_epoch,
                           lr=params.lr,
                           merge_method=params.merge_method)
    asr_trainer.evaluate()

if __name__ == '__main__':

    from funasr.utils.modelscope_param import modelscope_args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='path to model')
    parser.add_argument('--data_path', type=str, required=True, help='path to training data')
    parser.add_argument('--merge_method', type=int, choices=[1, 2, 3], default=3, help='merge method')
    parser.add_argument('--dataset_type', type=str, default="small")
    parser.add_argument('--batch_bins', type=int, default=4000, help='fbank frames')
    args = parser.parse_args()

    params = modelscope_args(model=args.model_name)
    params.data_path = args.data_path  
    params.dataset_type = args.dataset_type
    params.batch_bins = args.batch_bins
    params.max_epoch = 1
    params.merge_method = args.merge_method
    modelscope_finetune(params)

