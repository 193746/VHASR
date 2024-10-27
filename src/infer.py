import os
import torch
import torchaudio
import librosa
from build_model_from_file import build_model_from_file
from PIL import Image
from transformers import CLIPProcessor

def infer_pipeline(model_path,speech_path,image_path=None,merge_method=3):
    device="cuda" if torch.cuda.is_available() else "cpu"

    model,args,preprocessor=build_model_from_file(model_file=os.path.join(model_path,"model.pb"))
    model.eval()
    model=model.to(device)

    speech, _ = librosa.load(speech_path,sr=16000)
    speech=torch.tensor(speech).to(device)
    speech_length=torch.tensor([len(speech)]).to(device)
    speech=speech.unsqueeze(dim=0)

    if image_path:
        image=Image.open(image_path)
        image_prep=preprocessor(images=image,return_tensors="pt")["pixel_values"]
        image_prep=image_prep.to(device)
        result=model.infer(speech,speech_length,image_prep,merge_method)[0]
    else:
        result=model.infer(speech,speech_length)[0]
    return result

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='path to model')
    parser.add_argument('--speech_path', type=str, required=True, help='path to speech')
    parser.add_argument('--image_path', type=str, default=None, help='path to image')
    parser.add_argument('--merge_method', type=int, choices=[1, 2, 3], default=3, help='merge method')
    args = parser.parse_args()

    result=infer_pipeline(
        model_path=args.model_name,
        speech_path=args.speech_path,
        image_path=args.image_path,
        merge_method=args.merge_method
    )
    print(result)
