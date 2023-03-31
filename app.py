import streamlit as st
from markdownlit import mdlit
import openai
from streamlit_option_menu import option_menu
from streamlit_chat import message
st.set_page_config(layout='wide',page_title='mlexhaust')


def generate_response(prompt):
    completions = openai.Completion.create (
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = completions.choices[0].text
    return message

def generate_img(prompt):
    response = openai.Image.create(
    prompt=prompt,n=1,size="256x256")
    image_url = response['data'][0]['url']
    return image_url

def home():
    mdlit('# ML [red]Exhaust[/red] 👾')
    one, two = st.columns(2)
    mdlit("### Hi 👋, Welcome to All ML enthusiasts, My name is [green] Harish Gehlot [/green] and presently I am Data Science Intern at @([violet] Katonic.ai [/violet])(https://katonic.ai). Here In this streamlit app, I am going to implement lots of projects with [blue] basic stuff [/blue] as well a [red] complicated stuff [/red]")
    #st.markdown("[![LinkedIn](https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg)](https://www.linkedin.com/in/harish-gehlot-5338a021a/)",unsafe_allow_html=True)
    #st.markdown("[![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/AvratanuBiswas/PubLit)",unsafe_allow_html=True)
    #st.subheader('Links')
    #mdlit(" <br> 1. @(LinkedIn)(https://www.linkedin.com/in/harish-gehlot-5338a021a/) <br> 2. @(Github)(https://github.com/Hg03) <br> 3. @(Blog)(https://mlpapers.substack.com/)")


def openai_():
    mdlit('## [green] OpenAI [/green] exploration 🔥')
    mdlit("> ### Let's [red]first[/red] look at the Dalle API to generate some awesome images")

    input_text_for_image = st.text_input("Put your imagination here to generate image")
    if input_text_for_image:
        if st.button('Generate Image'):
            url_img = generate_img(input_text_for_image)
            st.info(f"View your awesome image at {url_img}")

    mdlit("> ### Let's [blue]Secondly[/blue] look at ChatGPT api through which we can start a conversation")
    openai.api_key = st.secrets['api_key']
    # if 'generated' not in st.session_state:
    #     st.session_state['generated'] = []

    # if 'past' not in st.session_state:
    #     st.session_state['past'] = []

    input_text = st.text_input("You: ","Hi, What you need to talk about?", key="input")
    # if input_text:
    #     output = generate_response(input_text)
    #     st.session_state.past.append(input_text)
    #     st.session_state.generated.append(output)

    # if st.session_state['generated']:
    #     for i in range(len(st.session_state['generated'])-1, -1, -1):
    #         message(st.session_state["generated"][i], key=str(i))
    #         message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')



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
    selected = option_menu("Main Menu", ["Home", 'OpenAI'], icons=['house', 'fire'], menu_icon="cast", default_index=1)

if selected == 'Home':
    home()
elif selected == 'OpenAI':
    openai_()
