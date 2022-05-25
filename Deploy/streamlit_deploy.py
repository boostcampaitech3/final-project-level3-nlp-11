"""
$ streamlit run streamlit_deploy.py --server.port 30001 --browser.serverAddress 0.0.0.0
"""

import streamlit as st

from PIL import Image
import requests
import os


st.set_page_config(
    page_title='IC & VQA for hard of hearing',
)

st.title('Image Captioning & Visual Question Answering Demo')

st.text('')

URI = os.environ.get('URI')
ping = requests.get(f'{URI}/api/v1/ping').status_code

if ping == 200:
    uploaded_file = st.file_uploader('이미지를 올려주세요')
    if uploaded_file is not None:
        st.image(Image.open(uploaded_file))

        ic_res = requests.post(
            url=f'{URI}/api/v1/ic',
            data={'beam': 3},
            files={'file': uploaded_file.getvalue()}
        )

        st.text('')
        st.text(f"[Caption] {ic_res.json()['caption']}")
        st.text(f"{ic_res.json()['device']} time {ic_res.json()['inference_time'][:5]}s")
        st.text('')
        st.text('')

        query = st.text_input('질문을 적으시고 Enter 를 눌러주세요.')
        if query:
            st.text(f"[질문]: {query}")

            res = requests.post(
                url=f'{URI}/api/v1/vqa',
                data={'query': ' '+query},
                files={'file': uploaded_file.getvalue()})
            
            st.text(f"[대답]: {res.json()['answer']}")
            st.text(f"{res.json()['device']} time {res.json()['inference_time'][:5]}s")
else:
    st.error('Sorry, Server is not online.')
