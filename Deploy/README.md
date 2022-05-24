# Deploy

<br>

현재 Image Captioning 은 학습한 모델의 가중치를 다운받아 사용하고, Visual Question Answering 은 모델 개발 중이어서 huggingface.co 의 OFA-tiny 를 사용하였습니다.

IC 가중치 [download](https://drive.google.com/file/d/1oj76AgNxzTb1WfhsdMcIOoZXl0SKbxqa/view?usp=sharing)

<br>

## FastAPI

```bash
$ uvicorn main:app --host 0.0.0.0 --port PORT
```



<br>

## Streamlit

```bash
$ streamlit run streamlit_deploy.py --server.port PORT --browser.serverAddress 0.0.0.0
```