# 🐕 리트리버
<div align="center">
  <img src="https://shields.io/badge/python-v3.8.5-blue?logo=python" />
  <img src="https://shields.io/badge/pytorch-v1.10.2-red?logo=pytorch" />
  <img src="https://img.shields.io/badge/transformers-4.10.0-yellow" />
  <img src="https://img.shields.io/badge/Streamlit-red" />
  <img src="https://img.shields.io/badge/FastAPI-blue" />
</div>

<br />

<br />

# 1. 프로젝트 개요

🐕 리트리버는 **시각장애인을 위하여 시야를 음성으로 안내하는 서비스**입니다.
시각장애인이 기존의 보조기구로 시야를 인지할 수 없을 때, 사진을 찍어 상황에 대한 설명이 음성으로 제공(Image Captioning)되는 기능과 이미지에 대한 부가 설명이 필요하다면 질문에 대한 대답(Visual Question Answering)이 제공됩니다.

<div align="center">
  <img src="https://github.com/boostcampaitech3/final-project-level3-nlp-11/blob/main/assets/img2.png?raw=true">
</div>

음향신호기가 없는 신호등과 무신호 횡단보도는 시각장애인의 보행을 어렵게 하는 많은 요소 중 하나 입니다. 무신호 횡단보도 뿐만 아니라 도로 위의 장애물, 깨져 있는 연석 등 시각장애인이 보행을 하는데 위협이 되는 요소가 곳곳에 존재합니다. 이를 해결하고자 해당 프로젝트를 진행하였습니다.

**AI를 이용하여 사회에 긍정적인 방향으로 기여**하고, **시각장애인들의 외부 활동을 보조하기 위한 도구로서 기능**을 제공할 수 있습니다.

<br />

<br />

# 2. 사용 방법

## **Image Captioning**

**Image Captioning model**

Windows : [download](https://drive.google.com/file/d/1_sQt__98URycUtygbjHZ3Th8hpGTx806/view?usp=sharing)

```bash
# linux
$ wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_sQt__98URycUtygbjHZ3Th8hpGTx806' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_sQt__98URycUtygbjHZ3Th8hpGTx806" -O quantized_model.pth.tar && rm -rf /tmp/cookies.txt
```

<br />

<br />

## VQA(**Visual Question Answering)**

### **VQA model**

Windows : [download](https://drive.google.com/file/d/1Ttz8DLQl9ZxrwLznJwNLcutr2hhZxZOk/view?usp=sharing)

```bash
# linux
$ wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Ttz8DLQl9ZxrwLznJwNLcutr2hhZxZOk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Ttz8DLQl9ZxrwLznJwNLcutr2hhZxZOk" -O vqa_demo_epoch1.pth && rm -rf /tmp/cookies.txt
```

<br />

<br />

### **Label to Answer**

Windows : [download](https://drive.google.com/file/d/1truanDe3btDUFKBtoxKpVux7jx3cyTtI/view?usp=sharing)

```bash
# linux
$ wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1truanDe3btDUFKBtoxKpVux7jx3cyTtI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1truanDe3btDUFKBtoxKpVux7jx3cyTtI" -O trainval_label2ans.kvqa.pkl && rm -rf /tmp/cookies.txt
```

<br />

<br />

## **Deploy**

### **FastAPI**

```bash
$ uvicorn main:app --host 0.0.0.0 --port PORT
```

### **Streamlit**

```bash
$ streamlit run streamlit_deploy.py --server.port PORT --browser.serverAddress 0.0.0.0
```

<br />

<br />

# 3. 프로젝트 모델

## Image Captioning

<p align="center">
  <img src="https://github.com/boostcampaitech3/final-project-level3-nlp-11/blob/main/assets/Untitled%201.png?raw=true">
</p>

- *Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering*에서 제안한 Model을 사용했습니다.
- Bottom-Up and Top-Down Attention Model은 이미지 내 객체를 탐지할 수 있는 모델인 Faster-RCNN을 사용하여 후보영역 인 ROI를 추출하고, 이 ROI가 객체간의 상호작용을 반영하는 데 사용됩니다.
- Decoder의 첫번째 Layer인 Top-Down Attention LSTM에서는 Hidden state와 Visual eature의 평균 그리고 Word embedding 값을 Concat한 Vector를 입력으로 받아, 기존 Baseline(Show, Attend and Tell)이 수행하던 Top-down Attention 역할을 합니다.
- Top-Down Attention LSTM의 Output hidden state는 ROI Feature와 Align을 수행 후 Attention vector와 Concat 하여 최종적으로 Language LSTM Layer의 입력으로 사용되고, Language LSTM에서는 이를 바탕으로 Caption을 생성합니다.

## VQA

<p align="center">
  <img src="https://github.com/boostcampaitech3/final-project-level3-nlp-11/raw/main/assets/Untitled%202.png?raw=true">
</p>

- *VisualBERT: A Simple and Performant Baseline for Vision and Language*에서 제안한 Model을 사용했습니다.
- VisualBERT는 Faster RCNN 을 이용한 Bottom-up구조와, VilBERT를 이용한 Top-down 구조로
이미지를 질문에 대하여 세세하게 학습할 수 있습니다.
- Faster RCNN으로 추출된 ROI feature을 Question embedding과 concatenate 후 모델에 입력되어
이미지와 함께 단어의 의미를 학습하게 됩니다.
- 본 프로젝트에서는 VQA를 Multiple-Choice Task로 접근하였기 때문에 output을 Classification layer를 통하여 answer를 추출하였습니다.

<br />

<br />

# 4. 프로젝트 결과

### Image Caption

|                   Model Name |                     Tokenizer |       BLEU-4 |
| --- | --- | --- |
| Baseline(ResNet101 + LSTM with Attention) | 어절 단위 토크나이징 |         30.27 |
| Baseline(ResNet101 + LSTM with Attention) | Tokenizer(”monologg/kobigbird-bert-base”) |         33.49 |
| Baseline(ResNet101 + LSTM with Attention) | Tokenizer(”monologg/kobigbird-bert-base”) |         34.01 |
| Baseline(ResNet101 + LSTM with Attention) | Tokenizer(”monologg/kobigbird-bert-base”) |         34.17 |
| BUTD(Bottom-Up and Top-Down Attention) | Tokenizer(”monologg/kobigbird-bert-base”) |         38.73 |

  
### VQA

| Model Name | Tokenizer | ACC |
| --- | --- | --- |
| ConvNext_Large+LSTM(over 6 same responses) | Tokenizer(”klue/bert-base”) | 0.53 |
| VisualBert(with Faster RCNN)(over 6 same responses) | Tokenizer(”klue/bert-base”) | 0.58 |
| VisualBert(with Faster RCNN, additional dataset)(over 6 same responses) | Tokenizer(”klue/bert-base”) | 0.64 |

<br />

<br />

# 5. 시연 예시

실제 시연 영상은 [이곳](https://youtube.com/shorts/AVXBsLZOkvE?feature=share)에서 확인하실 수 있습니다.

<div align="center">
  <img src="https://github.com/boostcampaitech3/final-project-level3-nlp-11/blob/main/assets/img4.png?raw=true">
</div>

<br />

<br />

# 6. 프로젝트 팀 구성 및 역할

| 이름 | 역할 |
| --- | --- |
| 김덕래 | Image Captioning |
| 김용희 | VQA |
| 이두호 | Image Captioning(ClipCap) / Streamlit / FastAPI |
| 이승환 | VQA |
| 조혁준 | Image Captioning |
| 최진혁 | VQA |

<br />

<br />

# Reference

### Datasets

**Image Captioning**

- **[MSCOCO 사진](https://cocodataset.org/)**
- **[MSCOCO 한글 주석](https://aihub.or.kr/opendata/keti-data/recognition-visual/KETI-01-003)**

**VQA**

- **[SKT BRAIN - KVQA](https://sktbrain.github.io/KVQA/)**
- **[셀렉트스타 - 교차로 정보 데이터셋](https://open.selectstar.ai/data-set/wesee)**
- **[AI HUB - 생활 및 거주환경 기반 VQA](https://aihub.or.kr/aidata/34147)**

### Papers

**Image Captioning**

- ****[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)****
- ****[Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning](https://arxiv.org/abs/1612.01887)****
- ****[Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998)****

**VQA**

- ****[VQA: Visual Question Answering(2016)](https://arxiv.org/abs/1505.00468)****
- ****[VisualBERT: A Simple and Performant Baseline for Vision and Language(2019)](https://arxiv.org/abs/1908.03557)****
- **[SKTBrain / BAN-KVQA](https://github.com/SKTBrain/BAN-KVQA)**
