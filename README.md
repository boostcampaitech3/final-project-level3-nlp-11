# π λ¦¬νΈλ¦¬λ²
<div align="center">
  <img src="https://shields.io/badge/python-v3.8.5-blue?logo=python" />
  <img src="https://shields.io/badge/pytorch-v1.10.2-red?logo=pytorch" />
  <img src="https://img.shields.io/badge/transformers-4.10.0-yellow" />
  <img src="https://img.shields.io/badge/Streamlit-red" />
  <img src="https://img.shields.io/badge/FastAPI-blue" />
</div>

<br />

<br />

# 1. νλ‘μ νΈ κ°μ

π λ¦¬νΈλ¦¬λ²λ **μκ°μ₯μ μΈμ μνμ¬ μμΌλ₯Ό μμ±μΌλ‘ μλ΄νλ μλΉμ€**μλλ€.
μκ°μ₯μ μΈμ΄ κΈ°μ‘΄μ λ³΄μ‘°κΈ°κ΅¬λ‘ μμΌλ₯Ό μΈμ§ν  μ μμ λ, μ¬μ§μ μ°μ΄ μν©μ λν μ€λͺμ΄ μμ±μΌλ‘ μ κ³΅(Image Captioning)λλ κΈ°λ₯κ³Ό μ΄λ―Έμ§μ λν λΆκ° μ€λͺμ΄ νμνλ€λ©΄ μ§λ¬Έμ λν λλ΅(Visual Question Answering)μ΄ μ κ³΅λ©λλ€.

<div align="center">
  <img src="https://github.com/boostcampaitech3/final-project-level3-nlp-11/blob/main/assets/img2.png?raw=true">
</div>

μν₯μ νΈκΈ°κ° μλ μ νΈλ±κ³Ό λ¬΄μ νΈ ν‘λ¨λ³΄λλ μκ°μ₯μ μΈμ λ³΄νμ μ΄λ ΅κ² νλ λ§μ μμ μ€ νλ μλλ€. λ¬΄μ νΈ ν‘λ¨λ³΄λ λΏλ§ μλλΌ λλ‘ μμ μ₯μ λ¬Ό, κΉ¨μ Έ μλ μ°μ λ± μκ°μ₯μ μΈμ΄ λ³΄νμ νλλ° μνμ΄ λλ μμκ° κ³³κ³³μ μ‘΄μ¬ν©λλ€. μ΄λ₯Ό ν΄κ²°νκ³ μ ν΄λΉ νλ‘μ νΈλ₯Ό μ§ννμμ΅λλ€.

**AIλ₯Ό μ΄μ©νμ¬ μ¬νμ κΈμ μ μΈ λ°©ν₯μΌλ‘ κΈ°μ¬**νκ³ , **μκ°μ₯μ μΈλ€μ μΈλΆ νλμ λ³΄μ‘°νκΈ° μν λκ΅¬λ‘μ κΈ°λ₯**μ μ κ³΅ν  μ μμ΅λλ€.

<br />

<br />

# 2. μ¬μ© λ°©λ²

## **Image Captioning**

**Image Captioning model**

Windows :Β [download](https://drive.google.com/file/d/1_sQt__98URycUtygbjHZ3Th8hpGTx806/view?usp=sharing)

```bash
# linux
$ wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_sQt__98URycUtygbjHZ3Th8hpGTx806' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_sQt__98URycUtygbjHZ3Th8hpGTx806" -O quantized_model.pth.tar && rm -rf /tmp/cookies.txt
```

<br />

<br />

## VQA(**Visual Question Answering)**

### **VQA model**

Windows :Β [download](https://drive.google.com/file/d/1Ttz8DLQl9ZxrwLznJwNLcutr2hhZxZOk/view?usp=sharing)

```bash
# linux
$ wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Ttz8DLQl9ZxrwLznJwNLcutr2hhZxZOk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Ttz8DLQl9ZxrwLznJwNLcutr2hhZxZOk" -O vqa_demo_epoch1.pth && rm -rf /tmp/cookies.txt
```

<br />

<br />

### **Label to Answer**

Windows :Β [download](https://drive.google.com/file/d/1truanDe3btDUFKBtoxKpVux7jx3cyTtI/view?usp=sharing)

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

# 3. νλ‘μ νΈ λͺ¨λΈ

## Image Captioning

<p align="center">
  <img src="https://github.com/boostcampaitech3/final-project-level3-nlp-11/blob/main/assets/Untitled%201.png?raw=true">
</p>

- *Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering*μμ μ μν Modelμ μ¬μ©νμ΅λλ€.
- Bottom-Up and Top-Down Attention Modelμ μ΄λ―Έμ§ λ΄ κ°μ²΄λ₯Ό νμ§ν  μ μλ λͺ¨λΈμΈ Faster-RCNNμ μ¬μ©νμ¬ νλ³΄μμ­ μΈ ROIλ₯Ό μΆμΆνκ³ , μ΄ ROIκ° κ°μ²΄κ°μ μνΈμμ©μ λ°μνλ λ° μ¬μ©λ©λλ€.
- Decoderμ μ²«λ²μ§Έ LayerμΈ Top-Down Attention LSTMμμλ Hidden stateμ Visual eatureμ νκ·  κ·Έλ¦¬κ³  Word embedding κ°μ Concatν Vectorλ₯Ό μλ ₯μΌλ‘ λ°μ, κΈ°μ‘΄ Baseline(Show, Attend and Tell)μ΄ μννλ Top-down Attention μ­ν μ ν©λλ€.
- Top-Down Attention LSTMμ Output hidden stateλ ROI Featureμ Alignμ μν ν Attention vectorμ Concat νμ¬ μ΅μ’μ μΌλ‘ Language LSTM Layerμ μλ ₯μΌλ‘ μ¬μ©λκ³ , Language LSTMμμλ μ΄λ₯Ό λ°νμΌλ‘ Captionμ μμ±ν©λλ€.

## VQA

<p align="center">
  <img src="https://github.com/boostcampaitech3/final-project-level3-nlp-11/raw/main/assets/Untitled%202.png?raw=true">
</p>

- *VisualBERT: A Simple and Performant Baseline for Vision and Language*μμ μ μν Modelμ μ¬μ©νμ΅λλ€.
- VisualBERTλ Faster RCNN μ μ΄μ©ν Bottom-upκ΅¬μ‘°μ, VilBERTλ₯Ό μ΄μ©ν Top-down κ΅¬μ‘°λ‘
μ΄λ―Έμ§λ₯Ό μ§λ¬Έμ λνμ¬ μΈμΈνκ² νμ΅ν  μ μμ΅λλ€.
- Faster RCNNμΌλ‘ μΆμΆλ ROI featureμ Question embeddingκ³Ό concatenate ν λͺ¨λΈμ μλ ₯λμ΄
μ΄λ―Έμ§μ ν¨κ» λ¨μ΄μ μλ―Έλ₯Ό νμ΅νκ² λ©λλ€.
- λ³Έ νλ‘μ νΈμμλ VQAλ₯Ό Multiple-Choice Taskλ‘ μ κ·ΌνμκΈ° λλ¬Έμ outputμ Classification layerλ₯Ό ν΅νμ¬ answerλ₯Ό μΆμΆνμμ΅λλ€.

<br />

<br />

# 4. νλ‘μ νΈ κ²°κ³Ό

### Image Caption

|                   Model Name |                     Tokenizer |       BLEU-4 |
| --- | --- | --- |
| Baseline(ResNet101 + LSTM with Attention) | μ΄μ  λ¨μ ν ν¬λμ΄μ§ |         30.27 |
| Baseline(ResNet101 + LSTM with Attention) | Tokenizer(βmonologg/kobigbird-bert-baseβ) |         33.49 |
| Baseline(ResNet101 + LSTM with Attention) | Tokenizer(βmonologg/kobigbird-bert-baseβ) |         34.01 |
| Baseline(ResNet101 + LSTM with Attention) | Tokenizer(βmonologg/kobigbird-bert-baseβ) |         34.17 |
| BUTD(Bottom-Up and Top-Down Attention) | Tokenizer(βmonologg/kobigbird-bert-baseβ) |         38.73 |

  
### VQA

| Model Name | Tokenizer | ACC |
| --- | --- | --- |
| ConvNext_Large+LSTM(over 6 same responses) | Tokenizer(βklue/bert-baseβ) | 0.53 |
| VisualBert(with Faster RCNN)(over 6 same responses) | Tokenizer(βklue/bert-baseβ) | 0.58 |
| VisualBert(with Faster RCNN, additional dataset)(over 6 same responses) | Tokenizer(βklue/bert-baseβ) | 0.64 |

<br />

<br />

# 5. μμ° μμ

μ€μ  μμ° μμμ [μ΄κ³³](https://youtube.com/shorts/AVXBsLZOkvE?feature=share)μμ νμΈνμ€ μ μμ΅λλ€.

<div align="center">
  <img src="https://github.com/boostcampaitech3/final-project-level3-nlp-11/blob/main/assets/img4.png?raw=true">
</div>

<br />

<br />

# 6. νλ‘μ νΈ ν κ΅¬μ± λ° μ­ν 

| μ΄λ¦ | μ­ν  |
| --- | --- |
| κΉλλ | Image Captioning |
| κΉμ©ν¬ | VQA |
| μ΄λνΈ | Image Captioning(ClipCap) / Streamlit / FastAPI |
| μ΄μΉν | VQA |
| μ‘°νμ€ | Image Captioning |
| μ΅μ§ν | VQA |

<br />

<br />

# Reference

### Datasets

**Image Captioning**

- **[MSCOCO μ¬μ§](https://cocodataset.org/)**
- **[MSCOCO νκΈ μ£Όμ](https://aihub.or.kr/opendata/keti-data/recognition-visual/KETI-01-003)**

**VQA**

- **[SKT BRAIN - KVQA](https://sktbrain.github.io/KVQA/)**
- **[μλ νΈμ€ν - κ΅μ°¨λ‘ μ λ³΄ λ°μ΄ν°μ](https://open.selectstar.ai/data-set/wesee)**
- **[AI HUB - μν λ° κ±°μ£Όνκ²½ κΈ°λ° VQA](https://aihub.or.kr/aidata/34147)**

### Papers

**Image Captioning**

- ****[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)****
- ****[Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning](https://arxiv.org/abs/1612.01887)****
- ****[Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998)****

**VQA**

- ****[VQA: Visual Question Answering(2016)](https://arxiv.org/abs/1505.00468)****
- ****[VisualBERT: A Simple and Performant Baseline for Vision and Language(2019)](https://arxiv.org/abs/1908.03557)****
- **[SKTBrain / BAN-KVQA](https://github.com/SKTBrain/BAN-KVQA)**
