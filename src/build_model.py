import logging

import torch
from funasr.layers.global_mvn import GlobalMVN
from funasr.layers.utterance_mvn import UtteranceMVN
from funasr.models.ctc import CTC
from funasr.models.decoder.abs_decoder import AbsDecoder
from funasr.models.decoder.contextual_decoder import ContextualParaformerDecoder
from funasr.models.decoder.rnn_decoder import RNNDecoder
from funasr.models.decoder.sanm_decoder import ParaformerSANMDecoder, FsmnDecoderSCAMAOpt
from funasr.models.decoder.transformer_decoder import (
    DynamicConvolution2DTransformerDecoder,  # noqa: H301
)
from funasr.models.decoder.transformer_decoder import DynamicConvolutionTransformerDecoder
from funasr.models.decoder.transformer_decoder import (
    LightweightConvolution2DTransformerDecoder,  # noqa: H301
)
from funasr.models.decoder.transformer_decoder import (
    LightweightConvolutionTransformerDecoder,  # noqa: H301
)
from funasr.models.decoder.transformer_decoder import ParaformerDecoderSAN
from funasr.models.decoder.transformer_decoder import TransformerDecoder
from funasr.models.decoder.rnnt_decoder import RNNTDecoder
from funasr.models.decoder.transformer_decoder import SAAsrTransformerDecoder
from funasr.models.e2e_asr import ASRModel
from funasr.models.e2e_asr_contextual_paraformer import NeatContextualParaformer
from funasr.models.e2e_asr_mfcca import MFCCA

from funasr.models.e2e_asr_transducer import TransducerModel, UnifiedTransducerModel
from funasr.models.e2e_asr_bat import BATModel

from funasr.models.e2e_sa_asr import SAASRModel
from funasr.models.e2e_asr_paraformer import Paraformer, ParaformerOnline, ParaformerBert, BiCifParaformer, ContextualParaformer

from funasr.models.e2e_tp import TimestampPredictor
from funasr.models.e2e_uni_asr import UniASR
from funasr.models.encoder.conformer_encoder import ConformerEncoder, ConformerChunkEncoder
from funasr.models.encoder.data2vec_encoder import Data2VecEncoder
from funasr.models.encoder.mfcca_encoder import MFCCAEncoder
from funasr.models.encoder.resnet34_encoder import ResNet34Diar
from funasr.models.encoder.rnn_encoder import RNNEncoder
from funasr.models.encoder.sanm_encoder import SANMEncoder, SANMEncoderChunkOpt
from funasr.models.encoder.branchformer_encoder import BranchformerEncoder
from funasr.models.encoder.e_branchformer_encoder import EBranchformerEncoder
from funasr.models.encoder.transformer_encoder import TransformerEncoder
from funasr.models.frontend.default import DefaultFrontend
from funasr.models.frontend.default import MultiChannelFrontend
from funasr.models.frontend.fused import FusedFrontends
from funasr.models.frontend.s3prl import S3prlFrontend
from funasr.models.frontend.wav_frontend import WavFrontend
from funasr.models.frontend.windowing import SlidingWindow
from funasr.models.joint_net.joint_network import JointNetwork
from funasr.models.predictor.cif import CifPredictor, CifPredictorV2, CifPredictorV3, BATPredictor
from funasr.models.specaug.specaug import SpecAug
from funasr.models.specaug.specaug import SpecAugLFR
from funasr.modules.subsampling import Conv1dSubsampling
from funasr.torch_utils.initialize import initialize
from funasr.train.class_choices import ClassChoices
import MySanmDecoder
import MyParaformer
from vgs_utils import SpeechAdapter
from mm_utils import Merger

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
        fused=FusedFrontends,
        wav_frontend=WavFrontend,
        multichannelfrontend=MultiChannelFrontend,
    ),
    default="default",
)
specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(
        specaug=SpecAug,
        specaug_lfr=SpecAugLFR,
    ),
    default=None,
    optional=True,
)
normalize_choices = ClassChoices(
    "normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    default=None,
    optional=True,
)
model_choices = ClassChoices(
    "model",
    classes=dict(
        asr=ASRModel,
        uniasr=UniASR,
        paraformer=Paraformer,
        paraformer_online=ParaformerOnline,
        paraformer_bert=ParaformerBert,
        bicif_paraformer=BiCifParaformer,
        contextual_paraformer=ContextualParaformer,
        neatcontextual_paraformer=NeatContextualParaformer,
        mfcca=MFCCA,
        timestamp_prediction=TimestampPredictor,
        rnnt=TransducerModel,
        rnnt_unified=UnifiedTransducerModel,
        sa_asr=SAASRModel,
        bat=BATModel,
    ),
    default="asr",
)
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        rnn=RNNEncoder,
        sanm=SANMEncoder,
        sanm_chunk_opt=SANMEncoderChunkOpt,
        data2vec_encoder=Data2VecEncoder,
        branchformer=BranchformerEncoder,
        e_branchformer=EBranchformerEncoder,
        mfcca_enc=MFCCAEncoder,
        chunk_conformer=ConformerChunkEncoder,
    ),
    default="rnn",
)
asr_encoder_choices = ClassChoices(
    "asr_encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        rnn=RNNEncoder,
        sanm=SANMEncoder,
        sanm_chunk_opt=SANMEncoderChunkOpt,
        data2vec_encoder=Data2VecEncoder,
        mfcca_enc=MFCCAEncoder,
    ),
    default="rnn",
)

spk_encoder_choices = ClassChoices(
    "spk_encoder",
    classes=dict(
        resnet34_diar=ResNet34Diar,
    ),
    default="resnet34_diar",
)
encoder_choices2 = ClassChoices(
    "encoder2",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        rnn=RNNEncoder,
        sanm=SANMEncoder,
        sanm_chunk_opt=SANMEncoderChunkOpt,
    ),
    default="rnn",
)
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        transformer=TransformerDecoder,
        lightweight_conv=LightweightConvolutionTransformerDecoder,
        lightweight_conv2d=LightweightConvolution2DTransformerDecoder,
        dynamic_conv=DynamicConvolutionTransformerDecoder,
        dynamic_conv2d=DynamicConvolution2DTransformerDecoder,
        rnn=RNNDecoder,
        fsmn_scama_opt=FsmnDecoderSCAMAOpt,
        paraformer_decoder_sanm=ParaformerSANMDecoder,
        paraformer_decoder_san=ParaformerDecoderSAN,
        contextual_paraformer_decoder=ContextualParaformerDecoder,
        sa_decoder=SAAsrTransformerDecoder,
    ),
    default="rnn",
)
decoder_choices2 = ClassChoices(
    "decoder2",
    classes=dict(
        transformer=TransformerDecoder,
        lightweight_conv=LightweightConvolutionTransformerDecoder,
        lightweight_conv2d=LightweightConvolution2DTransformerDecoder,
        dynamic_conv=DynamicConvolutionTransformerDecoder,
        dynamic_conv2d=DynamicConvolution2DTransformerDecoder,
        rnn=RNNDecoder,
        fsmn_scama_opt=FsmnDecoderSCAMAOpt,
        paraformer_decoder_sanm=ParaformerSANMDecoder,
    ),
    type_check=AbsDecoder,
    default="rnn",
)
predictor_choices = ClassChoices(
    name="predictor",
    classes=dict(
        cif_predictor=CifPredictor,
        ctc_predictor=None,
        cif_predictor_v2=CifPredictorV2,
        cif_predictor_v3=CifPredictorV3,
        bat_predictor=BATPredictor,
    ),
    default="cif_predictor",
    optional=True,
)
predictor_choices2 = ClassChoices(
    name="predictor2",
    classes=dict(
        cif_predictor=CifPredictor,
        ctc_predictor=None,
        cif_predictor_v2=CifPredictorV2,
    ),
    default="cif_predictor",
    optional=True,
)
stride_conv_choices = ClassChoices(
    name="stride_conv",
    classes=dict(
        stride_conv1d=Conv1dSubsampling
    ),
    default="stride_conv1d",
    optional=True,
)
rnnt_decoder_choices = ClassChoices(
    name="rnnt_decoder",
    classes=dict(
        rnnt=RNNTDecoder,
    ),
    default="rnnt",
    optional=True,
)
joint_network_choices = ClassChoices(
    name="joint_network",
    classes=dict(
        joint_network=JointNetwork,
    ),
    default="joint_network",
    optional=True,
)

class_choices_list = [
    # --frontend and --frontend_conf
    frontend_choices,
    # --specaug and --specaug_conf
    specaug_choices,
    # --normalize and --normalize_conf
    normalize_choices,
    # --model and --model_conf
    model_choices,
    # --encoder and --encoder_conf
    encoder_choices,
    # --decoder and --decoder_conf
    decoder_choices,
    # --predictor and --predictor_conf
    predictor_choices,
    # --encoder2 and --encoder2_conf
    encoder_choices2,
    # --decoder2 and --decoder2_conf
    decoder_choices2,
    # --predictor2 and --predictor2_conf
    predictor_choices2,
    # --stride_conv and --stride_conv_conf
    stride_conv_choices,
    # --rnnt_decoder and --rnnt_decoder_conf
    rnnt_decoder_choices,
    # --joint_network and --joint_network_conf
    joint_network_choices,
    # --asr_encoder and --asr_encoder_conf
    asr_encoder_choices,
    # --spk_encoder and --spk_encoder_conf
    spk_encoder_choices,
]

def build_asr_model(args):
    # token_list
    if isinstance(args.token_list, str):
        with open(args.token_list, encoding="utf-8") as f:
            token_list = [line.rstrip() for line in f]
        args.token_list = list(token_list)
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size}")
    elif isinstance(args.token_list, (tuple, list)):
        token_list = list(args.token_list)
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size}")
    else:
        token_list = None
        vocab_size = None

    # frontend
    if hasattr(args, "input_size") and args.input_size is None:
        frontend_class = frontend_choices.get_class(args.frontend)
        if args.frontend == 'wav_frontend' or args.frontend == 'multichannelfrontend':
            frontend = frontend_class(cmvn_file=args.cmvn_file, **args.frontend_conf)
        else:
            frontend = frontend_class(**args.frontend_conf)
        input_size = frontend.output_size()
    else:
        args.frontend = None
        args.frontend_conf = {}
        frontend = None
        input_size = args.input_size if hasattr(args, "input_size") else None

    # data augmentation for spectrogram
    if args.specaug is not None:
        specaug_class = specaug_choices.get_class(args.specaug)
        specaug = specaug_class(**args.specaug_conf)
    else:
        specaug = None

    # normalization layer
    if args.normalize is not None:
        normalize_class = normalize_choices.get_class(args.normalize)
        if args.model == "mfcca":
            normalize = normalize_class(stats_file=args.cmvn_file, **args.normalize_conf)
        else:
            normalize = normalize_class(**args.normalize_conf)
    else:
        normalize = None

    # encoder
    encoder_class = encoder_choices.get_class(args.encoder)
    encoder = encoder_class(input_size=input_size, **args.encoder_conf)

    # decoder
    if hasattr(args, "decoder") and args.decoder is not None:
        # !!!additional code
        if args.decoder=="MySanmDecoder":
            decoder=MySanmDecoder.ParaformerSANMDecoder(
                vocab_size=vocab_size,
                encoder_output_size=encoder.output_size(),
                **args.decoder_conf,
            )
        else:
            decoder_class = decoder_choices.get_class(args.decoder)
            decoder = decoder_class(
                vocab_size=vocab_size,
                encoder_output_size=encoder.output_size(),
                **args.decoder_conf,
            )
    else:
        decoder = None

    # ctc
    ctc = CTC(
        odim=vocab_size, encoder_output_size=encoder.output_size(), **args.ctc_conf
    )

    if args.model in ["asr", "mfcca"]:
        model_class = model_choices.get_class(args.model)
        model = model_class(
            vocab_size=vocab_size,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            token_list=token_list,
            **args.model_conf,
        )

    elif args.model=="MyLargeParaformer":
        # predictor
        predictor_class = predictor_choices.get_class(args.predictor)
        predictor = predictor_class(**args.predictor_conf)
        bias_decoder_conf = args.bias_decoder_conf
        bias_decoder = MySanmDecoder.ParaformerSANMDecoder(
            vocab_size=vocab_size,
            encoder_output_size=512,
            **bias_decoder_conf,
        )

        from transformers import CLIPModel
        clip_encoder = CLIPModel.from_pretrained(args.vh_conf["clip_path"]).vision_model
        visual_adapter=CLIPModel.from_pretrained(args.vh_conf["clip_path"]).visual_projection
        bias_encoder = torch.nn.LSTM(args.vh_conf["encoder_input_size"],
                                    args.vh_conf["encoder_output_size"], 
                                    args.vh_conf["encoder_layer"],
                                    batch_first=True, 
                                    dropout=args.vh_conf["encoder_dropout"],
                                    bidirectional=False)
        bias_output_layer = torch.nn.Linear(args.vh_conf["linear_output_size"], vocab_size)
        speech_adapter=SpeechAdapter(args.vh_conf["speech_adapter_size"],args.vh_conf["speech_adapter_layer"])
        merger=Merger(args.vh_conf["clip_path"])
        model = MyParaformer.Paraformer(
            vocab_size=vocab_size,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            encoder=encoder,
            decoder=decoder,
            clip_encoder=clip_encoder,
            visual_adapter=visual_adapter,
            bias_encoder=bias_encoder,
            bias_decoder=bias_decoder,
            bias_output_layer=bias_output_layer,
            speech_adapter=speech_adapter,
            merger=merger,  
            ctc=ctc,
            token_list=token_list,
            predictor=predictor,
            **args.model_conf,
        )
        
    elif args.model in ["paraformer", "paraformer_online", "paraformer_bert", "bicif_paraformer",
                        "contextual_paraformer", "neatcontextual_paraformer"]:
        # predictor
        predictor_class = predictor_choices.get_class(args.predictor)
        predictor = predictor_class(**args.predictor_conf)

        model_class = model_choices.get_class(args.model)
        model = model_class(
            vocab_size=vocab_size,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            token_list=token_list,
            predictor=predictor,
            **args.model_conf,
        )
    elif args.model == "uniasr":
        # stride_conv
        stride_conv_class = stride_conv_choices.get_class(args.stride_conv)
        stride_conv = stride_conv_class(**args.stride_conv_conf, idim=input_size + encoder.output_size(),
                                        odim=input_size + encoder.output_size())
        stride_conv_output_size = stride_conv.output_size()

        # encoder2
        encoder_class2 = encoder_choices2.get_class(args.encoder2)
        encoder2 = encoder_class2(input_size=stride_conv_output_size, **args.encoder2_conf)

        # decoder2
        decoder_class2 = decoder_choices2.get_class(args.decoder2)
        decoder2 = decoder_class2(
            vocab_size=vocab_size,
            encoder_output_size=encoder2.output_size(),
            **args.decoder2_conf,
        )

        # ctc2
        ctc2 = CTC(
            odim=vocab_size, encoder_output_size=encoder2.output_size(), **args.ctc_conf
        )

        # predictor
        predictor_class = predictor_choices.get_class(args.predictor)
        predictor = predictor_class(**args.predictor_conf)

        # predictor2
        predictor_class = predictor_choices2.get_class(args.predictor2)
        predictor2 = predictor_class(**args.predictor2_conf)

        model_class = model_choices.get_class(args.model)
        model = model_class(
            vocab_size=vocab_size,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            token_list=token_list,
            predictor=predictor,
            ctc2=ctc2,
            encoder2=encoder2,
            decoder2=decoder2,
            predictor2=predictor2,
            stride_conv=stride_conv,
            **args.model_conf,
        )
    elif args.model == "timestamp_prediction":
        # predictor
        predictor_class = predictor_choices.get_class(args.predictor)
        predictor = predictor_class(**args.predictor_conf)

        model_class = model_choices.get_class(args.model)
        model = model_class(
            frontend=frontend,
            encoder=encoder,
            predictor=predictor,
            token_list=token_list,
            **args.model_conf,
        )
    elif args.model == "rnnt" or args.model == "rnnt_unified":
        # 5. Decoder
        encoder_output_size = encoder.output_size()

        rnnt_decoder_class = rnnt_decoder_choices.get_class(args.rnnt_decoder)
        decoder = rnnt_decoder_class(
            vocab_size,
            **args.rnnt_decoder_conf,
        )
        decoder_output_size = decoder.output_size

        if getattr(args, "decoder", None) is not None:
            att_decoder_class = decoder_choices.get_class(args.decoder)

            att_decoder = att_decoder_class(
                vocab_size=vocab_size,
                encoder_output_size=encoder_output_size,
                **args.decoder_conf,
            )
        else:
            att_decoder = None
        # 6. Joint Network
        joint_network = JointNetwork(
            vocab_size,
            encoder_output_size,
            decoder_output_size,
            **args.joint_network_conf,
        )

        model_class = model_choices.get_class(args.model)
        # 7. Build model
        model = model_class(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            encoder=encoder,
            decoder=decoder,
            att_decoder=att_decoder,
            joint_network=joint_network,
            **args.model_conf,
        )
    elif args.model == "bat":
        # 5. Decoder
        encoder_output_size = encoder.output_size()

        rnnt_decoder_class = rnnt_decoder_choices.get_class(args.rnnt_decoder)
        decoder = rnnt_decoder_class(
            vocab_size,
            **args.rnnt_decoder_conf,
        )
        decoder_output_size = decoder.output_size

        if getattr(args, "decoder", None) is not None:
            att_decoder_class = decoder_choices.get_class(args.decoder)

            att_decoder = att_decoder_class(
                vocab_size=vocab_size,
                encoder_output_size=encoder_output_size,
                **args.decoder_conf,
            )
        else:
            att_decoder = None
        # 6. Joint Network
        joint_network = JointNetwork(
            vocab_size,
            encoder_output_size,
            decoder_output_size,
            **args.joint_network_conf,
        )

        predictor_class = predictor_choices.get_class(args.predictor)
        predictor = predictor_class(**args.predictor_conf)

        model_class = model_choices.get_class(args.model)
        # 7. Build model
        model = model_class(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            encoder=encoder,
            decoder=decoder,
            att_decoder=att_decoder,
            joint_network=joint_network,
            predictor=predictor,
            **args.model_conf,
        )
    elif args.model == "sa_asr":
        asr_encoder_class = asr_encoder_choices.get_class(args.asr_encoder)
        asr_encoder = asr_encoder_class(input_size=input_size, **args.asr_encoder_conf)
        spk_encoder_class = spk_encoder_choices.get_class(args.spk_encoder)
        spk_encoder = spk_encoder_class(input_size=input_size, **args.spk_encoder_conf)
        decoder = decoder_class(
            vocab_size=vocab_size,
            encoder_output_size=asr_encoder.output_size(),
            **args.decoder_conf,
        )
        ctc = CTC(
            odim=vocab_size, encoder_output_size=asr_encoder.output_size(), **args.ctc_conf
        )

        model_class = model_choices.get_class(args.model)
        model = model_class(
            vocab_size=vocab_size,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            asr_encoder=asr_encoder,
            spk_encoder=spk_encoder,
            decoder=decoder,
            ctc=ctc,
            token_list=token_list,
            **args.model_conf,
        )

    else:
        raise NotImplementedError("Not supported model: {}".format(args.model))

    # initialize
    if args.init is not None:
        initialize(model, args.init)

    return model