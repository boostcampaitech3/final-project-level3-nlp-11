"""
$ uvicorn main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, Form, File, UploadFile

from transformers import OFATokenizer, OFAForConditionalGeneration, AutoTokenizer
from torchvision import transforms
import torch

from typing import Union
from PIL import Image
import time

from googletrans import Translator

from utils import caption_image_beam_search


app = FastAPI()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# IC
ic_checkpoint = None
ic_decoder = None
ic_encoder = None
ic_tokenizer = None


# VQA
vqa_model = None
vqa_tokenizer = None
translator = None
mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 256
patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
    transforms.ToTensor(), 
    transforms.Normalize(mean=mean, std=std)
])

@app.on_event('startup')
def startup_event():
    global ic_checkpoint, ic_decoder, ic_encoder, ic_tokenizer, vqa_model, vqa_tokenizer, translator

    # IC
    ic_checkpoint = torch.load('./model.pth.tar', map_location=str(device))

    ic_decoder = ic_checkpoint["decoder"]
    ic_decoder = ic_decoder.to(device)
    ic_decoder.eval()
    ic_encoder = ic_checkpoint["encoder"]
    ic_encoder = ic_encoder.to(device)
    ic_encoder.eval()

    ic_tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")

    # VQA
    vqa_model = OFAForConditionalGeneration.from_pretrained('OFA-Sys/OFA-tiny')
    vqa_tokenizer = OFATokenizer.from_pretrained('OFA-Sys/OFA-tiny')

    translator = Translator()


@app.get('/api/v1/ping')
def check_ping():
    return {'ping': 'good'}


@app.post("/api/v1/ic")
def reed_root(beam: int = Form(...), file: UploadFile = File(...)):
    # beam_size default=3
    start = time.time()
    seq, decoded_seq, alphas = caption_image_beam_search(
        ic_encoder, ic_decoder, file.file, ic_tokenizer, beam
    )

    try:
        caption = str(decoded_seq)
    except:
        caption = '어떤 사진인지 잘 모르겠습니다 😥'

    return {
        'device': str(device),
        'inference_time': str(time.time() - start),
        'seq': str(seq),
        'caption': caption}


@app.post('/api/v1/vqa')
def vqa(query: Union[str, None] = Form(...), file: UploadFile = File(...)):

    start = time.time()
    translated_query = translator.translate(query, src='ko', dest='en').text
    inputs = vqa_tokenizer([translated_query], max_length=1024, truncation=True, return_tensors='pt')['input_ids']

    image = Image.open(file.file)
    patch_image = patch_resize_transform(image).unsqueeze(0)

    gen = vqa_model.generate(inputs, patch_images=patch_image, num_beams=4)
    result = vqa_tokenizer.batch_decode(gen, skip_special_tokens=True)
    
    if result[0]:
        translated_result = translator.translate(result[0], src='en', dest='ko').text
    else:
        translated_result = '잘 모르겠습니다. 죄송합니다 😥'

    return {
        'device': str(device),
        'inference_time': str(time.time() - start),
        'answer': translated_result
    }
