import torch

from funasr.optimizers.fairseq_adam import FairseqAdam
from funasr.optimizers.sgd import SGD


def build_optimizer(args, model):
    optim_classes = dict(
        adam=torch.optim.Adam,
        fairseq_adam=FairseqAdam,
        adamw=torch.optim.AdamW,
        sgd=SGD,
        adadelta=torch.optim.Adadelta,
        adagrad=torch.optim.Adagrad,
        adamax=torch.optim.Adamax,
        asgd=torch.optim.ASGD,
        lbfgs=torch.optim.LBFGS,
        rmsprop=torch.optim.RMSprop,
        rprop=torch.optim.Rprop,
    )

    optim_class = optim_classes.get(args.optim)
    if optim_class is None:
        raise ValueError(f"must be one of {list(optim_classes)}: {args.optim}")
    # 冻结参数不传入优化器
    optimizer = optim_class(filter(lambda p : p.requires_grad, model.parameters()), **args.optim_conf)

    optimizers = [optimizer]
    return optimizers