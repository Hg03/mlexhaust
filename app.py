import streamlit as st
from markdownlit import mdlit
import openai
from streamlit_option_menu import option_menu
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI

st.set_page_config(layout='wide',page_title='mlexhaust')


def generate_img(prompt):
    response = openai.Image.create(
    prompt=prompt,n=1,size="256x256")
    image_url = response['data'][0]['url']
    return image_url

def home():
    mdlit('# ML [red]Exhaust[/red] 👾')
    one, two = st.columns(2)
    mdlit("### Hi 👋, Welcome to All ML enthusiasts, My name is [green] Harish Gehlot [/green] and presently I am Data Science Intern at @([violet] Katonic.ai [/violet])(https://katonic.ai). Here In this streamlit app, I am going to implement lots of projects with [blue] basic stuff [/blue] as well a [red] complicated stuff [/red]")
    mdlit('#### So what is MLExhaust ??')
    mdlit("> It's kinda Upgrading my mind [blue]time[/blue] to [blue]time[/blue] by exploring the awesomeness of usecases of [orange]Machine Learning[/orange]")
    #st.markdown("[![LinkedIn](https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg)](https://www.linkedin.com/in/harish-gehlot-5338a021a/)",unsafe_allow_html=True)
    #st.markdown("[![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/AvratanuBiswas/PubLit)",unsafe_allow_html=True)
    #st.subheader('Links')
    #mdlit(" <br> 1. @(LinkedIn)(https://www.linkedin.com/in/harish-gehlot-5338a021a/) <br> 2. @(Github)(https://github.com/Hg03) <br> 3. @(Blog)(https://mlpapers.substack.com/)")


def openai_():
    mdlit('## [green] OpenAI [/green] exploration 🔥')
    user_api_key = None
    submit_btn = None
    if 'generated' not in st.session_state:
        st.session_state.generated = []
    if 'past' not in st.session_state:
        st.session_state.past = []
    if 'input' not in st.session_state:
        st.session_state.input = ''
    if 'stored_session' not in st.session_state:
        st.session_state.stored_session = []
    with st.expander("If you have your own OpenAI API key, or mine will exhausted"):
        user_api_key = st.text_input(label = 'Your API key',placeholder='Your API key',type='password')
        submit_btn = st.button('Submit')

    
    if not submit_btn:
        openai.api_key = st.secrets['OPENAI_API_KEY']
        mdlit("> ### Let's [red]first[/red] look at the Dalle API to generate some awesome images")
        input_text_for_image = st.text_input("Put your imagination here to generate image")
        if input_text_for_image:
            if st.button('Generate Image'):
                url_img = generate_img(input_text_for_image)
                st.info(f"View your awesome image at {url_img}")
        mdlit("> ### Let's [blue]Secondly[/blue] look at ChatGPT api through which we can start a conversation")
        input_text = st.text_input("You: ","Hi, What you need to talk about?", key="input")
        llm = OpenAI(temperature=0.7,model_name="text-davinci-003")

        if "entity_memory" not in st.session_state:
            st.session_state.entity_memory = ConversationEntityMemory(llm=llm,k=10)
        
        CONVERSATION = ConversationChain(llm=llm,prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,memory=st.session_state.entity_memory)
        if input_text:
            output = CONVERSATION.run(input=input_text)
            st.session_state.past.append(input_text)
            st.session_state.generated.append(output)

        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

    else:
        openai.api_key = user_api_key
        mdlit("> ### Let's [red]first[/red] look at the Dalle API to generate some awesome images")
        input_text_for_image = st.text_input("Put your imagination here to generate image")
        if input_text_for_image:
            if st.button('Generate Image'):
                url_img = generate_img(input_text_for_image)
                st.info(f"View your awesome image at {url_img}")
        mdlit("> ### Let's [blue]Secondly[/blue] look at ChatGPT api through which we can start a conversation")
        input_text = st.text_input("You: ","Hi, What you need to talk about?", key="input")
        llm = OpenAI(temperature=0.7,openai_api_key=user_api_key,model_name="text-davinci-003")

        if "entity_memory" not in st.session_state:
            st.session_state.entity_memory = ConversationEntityMemory(llm=llm,k=10)
        
        CONVERSATION = ConversationChain(llm=llm,prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,memory=st.session_state.entity_memory)
        if input_text:
            output = CONVERSATION.run(input=input_text)
            st.session_state.past.append(input_text)
            st.session_state.generated.append(output)

        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')


    
#     openai.api_key = st.secrets['OPENAI_API_KEY']



def life_expectancy():
	mdlit("## Life Expectancy [blue]Prediction[/blue]")
        dashboard, prediction = st.tabs(['Little Analysis Dashboard','Prediction'])
	with dashboard:
		mdlit('Beautiful [red]dashboard[\red] comes here')
        with prediction:
		with st.form('prediction form'):
	        country = st.selectbox('Select your Country',[])
                submit = st.form_submit_button('Submit')	    




hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {
	              visibility: hidden;
                    }
            footer:after {
              content:'Made by Harish Gehlot'; 
              visibility: visible;
              display: block;
              position: relative;
              #background-color: purple;
              padding: 5px;
              top: 2px;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'OpenAI','Life Expectancy Prediction'], icons=['house', '0-circle','settings'], menu_icon="cast", default_index=0)
    mdlit("[blue]Social Links[/blue]")
    mdlit("- @(linkedIn)(https://www.linkedin.com/in/harish-gehlot-5338a021a/)")
    mdlit("- @(Github)(https://github.com/Hg03/mlexhaust)")

if selected == 'Home':
    home()
elif selected == 'OpenAI':
    openai_()
elif selected == 'Life Expectancy Prediction':
    life_expectancy()
	
	
