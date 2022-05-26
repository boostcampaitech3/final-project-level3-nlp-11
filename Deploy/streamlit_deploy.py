"""
$ streamlit run streamlit_deploy.py --server.port 30001 --browser.serverAddress 0.0.0.0
"""

import streamlit as st
from bokeh.models import CustomJS
from bokeh.models.widgets import Button
from streamlit_bokeh_events import streamlit_bokeh_events

from dotenv import load_dotenv

from PIL import Image
import requests
import os


def file_upload_on_change():
    if 'vqa_input' in st.session_state:
        st.session_state.vqa_input = ''


load_dotenv()

st.set_page_config(
    page_title='IC & VQA for hard of hearing',
    layout='wide'
)


with st.sidebar:
    state = st.selectbox(
        'IC & VQA',
        ('Demo', 'Description', 'Logs')
    )


if state == 'Demo':
    st.title('Image Captioning & Visual Question Answering Demo')

    st.text('')
    st.text('')
    st.text('')

    URI = os.environ.get('URI')
    ping = requests.get(f'{URI}/api/v1/ping').status_code

    if ping == 200:
        acol1, _ = st.columns([15, 2])
        uploaded_file = acol1.file_uploader('이미지를 올려주세요', on_change=file_upload_on_change)

        bcol1, _, bcol3, _ = st.columns([7, 1, 7, 2])
        if uploaded_file is not None:
            bcol1.image(Image.open(uploaded_file))
            
            with bcol3.container():
                with st.spinner('무슨 사진인지 확인중입니다...'):
                    ic_res = requests.post(
                        url=f'{URI}/api/v1/ic',
                        data={'beam': 3},
                        files={'file': uploaded_file.getvalue()}
                    )

            bcol3.text('')
            try:
                caption = ic_res.json()['caption']
                device = ic_res.json()['device']
                time_str = ic_res.json()['inference_time'][:5]
                bcol3.markdown(f"#### {caption}")
                bcol3.caption(f"{device} time {time_str}s")
            except:
                caption = "무슨 사진인지 잘 모르겠습니다 😥"
                bcol3.markdown(f"무슨 사진인지 잘 모르겠습니다 😥")
            bcol3.text('')
            bcol3.text('')
            

            with bcol3.container():
                # MIC setting
                caption_tts_button = Button(label="대답 듣기", width=20)
                caption_tts_button.js_on_event("button_click", CustomJS(args=dict(caption=caption), code="""
                    var voices = window.speechSynthesis.getVoices();

                    function setVoiceList() {
                        voices = window.speechSynthesis.getVoices();
                    }

                    if (window.speechSynthesis.onvoiceschanged !== undefined){
                        window.speechSynthesis.onvoicechanged = setVoiceList;
                    }

                    if(!window.speechSynthesis){
                        alert('음성 재생을 지원하지 않는 브라우저입니다. 크롬, 파이어폭스 등의 최신 브라우저를 이용하세요.');
                        return;
                    }

                    var lang = 'ko-KR';
                    var utterThis = new SpeechSynthesisUtterance(caption);

                    var voiceFound = false;
                    for(var i = 0; i < voices.length; i++){
                        if(voices[i].lang.indexOf(lang) >= 0 || voices[i].lang.indexOf(lang.replace('-', '_')) >= 0){
                            utterThis.voice = voices[i];
                            voiceFound = true;
                        }
                    }

                    utterThis.lang = lang;
                    utterThis.pitch = 1;
                    utterThis.rate = 1;
                    window.speechSynthesis.speak(utterThis);
                    """))

                caption_result = streamlit_bokeh_events(
                    caption_tts_button,
                    key="caption_talk",
                    refresh_on_update=False,
                    override_height=45,
                    debounce_time=0)


            query = None

            bcol3.text('')
            input_type = bcol3.selectbox(
                '입력 타입을 선택해주세요. (마이크는 크롬 브라우저에서만 동작합니다)',
                ('Mic', 'Keyboard')
            )
            if input_type == 'Mic':
                with bcol3.container():
                    # MIC setting
                    stt_button = Button(label="마이크 입력", width=20)
                    stt_button.js_on_event("button_click", CustomJS(code="""
                        var recognition = new webkitSpeechRecognition();
                        recognition.continuous = false;
                        recognition.interimResults = true;
                        recognition.lang = "ko-KR";
                        recognition.maxAlternatives = 100;

                        recognition.onresult = function (e) {
                            var value = "";
                            for (var i = e.resultIndex; i < e.results.length; ++i) {
                                if (e.results[i].isFinal) {
                                    value += e.results[i][0].transcript;
                                }
                            }
                            if ( value != "") {
                                document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
                            }
                        }
                        recognition.start();
                        """))

                    result = streamlit_bokeh_events(
                        stt_button,
                        events="GET_TEXT",
                        key="listen",
                        refresh_on_update=False,
                        override_height=45,
                        debounce_time=0)
            
                    if result:
                        if "GET_TEXT" in result:
                            query = result.get("GET_TEXT")
                            if query and query[-1] != '?':
                                query += '?'
            else:
                query = bcol3.text_input('질문을 적으시고 Enter 를 눌러주세요.', key='vqa_input')

            if query:
                with bcol3.container():
                    with st.spinner('대답 준비 중입니다...'):
                        vqa_res = requests.post(
                            url=f'{URI}/api/v1/vqa',
                            data={'query': ' '+query},
                            files={'file': uploaded_file.getvalue()})

                        Q = f"[질문]: {query}"
                        answer = vqa_res.json()['answer']
                        A = f"[대답]: {vqa_res.json()['answer']}"
                        C = f"{vqa_res.json()['device']} time {vqa_res.json()['inference_time'][:5]}s"
                        bcol3.text(Q)
                        bcol3.text(A)
                        bcol3.caption(C)

                        query = None

                with bcol3.container():
                    # MIC setting
                    tts_button = Button(label="대답 듣기", width=20)
                    tts_button.js_on_event("button_click", CustomJS(args=dict(answer=answer), code="""
                        var voices = window.speechSynthesis.getVoices();

                        function setVoiceList() {
                            voices = window.speechSynthesis.getVoices();
                        }

                        if (window.speechSynthesis.onvoiceschanged !== undefined){
                            window.speechSynthesis.onvoicechanged = setVoiceList;
                        }

                        if(!window.speechSynthesis){
                            alert('음성 재생을 지원하지 않는 브라우저입니다. 크롬, 파이어폭스 등의 최신 브라우저를 이용하세요.');
                            return;
                        }

                        var lang = 'ko-KR';
                        var utterThis = new SpeechSynthesisUtterance(answer);

                        var voiceFound = false;
                        for(var i = 0; i < voices.length; i++){
                            if(voices[i].lang.indexOf(lang) >= 0 || voices[i].lang.indexOf(lang.replace('-', '_')) >= 0){
                                utterThis.voice = voices[i];
                                voiceFound = true;
                            }
                        }

                        utterThis.lang = lang;
                        utterThis.pitch = 1;
                        utterThis.rate = 1;
                        window.speechSynthesis.speak(utterThis);
                        """))

                    answer_result = streamlit_bokeh_events(
                        tts_button,
                        key="answer_talk",
                        refresh_on_update=False,
                        override_height=45,
                        debounce_time=0)
    else:
        st.error('Sorry, Server is not online.')
