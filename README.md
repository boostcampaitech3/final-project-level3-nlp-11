# 시각장애인을 위한 IC & VQA

## Image Captioning

<br>

### IC model

window : [download](https://drive.google.com/file/d/1_sQt__98URycUtygbjHZ3Th8hpGTx806/view?usp=sharing)

linux

```bash
$ wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_sQt__98URycUtygbjHZ3Th8hpGTx806' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_sQt__98URycUtygbjHZ3Th8hpGTx806" -O quantized_model.pth.tar && rm -rf /tmp/cookies.txt
```

<br>

<br>

## Visual Question Answering

<br>

### VQA model

window : [download](https://drive.google.com/file/d/1Ttz8DLQl9ZxrwLznJwNLcutr2hhZxZOk/view?usp=sharing)

linux
```bash
$ wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Ttz8DLQl9ZxrwLznJwNLcutr2hhZxZOk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Ttz8DLQl9ZxrwLznJwNLcutr2hhZxZOk" -O vqa_demo_epoch1.pth && rm -rf /tmp/cookies.txt
```

<br>

<br>

### Label to Answer

window : [download](https://drive.google.com/file/d/1truanDe3btDUFKBtoxKpVux7jx3cyTtI/view?usp=sharing)

linux
```bash
$ wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1truanDe3btDUFKBtoxKpVux7jx3cyTtI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1truanDe3btDUFKBtoxKpVux7jx3cyTtI" -O trainval_label2ans.kvqa.pkl && rm -rf /tmp/cookies.txt
```

<br>

<br>

## Deploy

### FastAPI

```bash
$ uvicorn main:app --host 0.0.0.0 --port PORT
```

<br>

### Streamlit

```bash
$ streamlit run streamlit_deploy.py --server.port PORT --browser.serverAddress 0.0.0.0
```