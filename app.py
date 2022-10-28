import streamlit as st
import nltk
#nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('model/model_Zyo.h5')
import json
import random
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import streamlit.components.v1 as components


st.set_page_config(page_title="Zyo", page_icon="assets/ZyoLogo.png", layout="centered", initial_sidebar_state="auto", menu_items=None)

# Recommender System - Hotels
@st.cache(allow_output_mutation=True)
def load_data_hotel():
    df = pd.read_excel("data/datahotelZyo.xlsx")
    df['types'] = df.types.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    exploded_track_df_hotel = df.explode("types")
    return exploded_track_df_hotel

tipehotel_names = ['hotel', 'resort']
maphotel_feats = ["gymSwimmingPoolFacility", "affordable", "restaurant", "beachView", "recreation", "soloFriendly"]

exploded_track_df_hotel = load_data_hotel()

def n_neighbors_url_map(tipe, test_feat):
    tipe = tipe.lower()
    tipe_data = exploded_track_df_hotel[(exploded_track_df_hotel["types"]==tipe)]
    tipe_data = tipe_data.sort_values(by='popularity', ascending=False)[:500]

    neigh = NearestNeighbors()
    neigh.fit(tipe_data[maphotel_feats].to_numpy())

    n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(tipe_data), return_distance=False)[0]

    urls = tipe_data.iloc[n_neighbors]["url"].tolist()
    descriptions = tipe_data.iloc[n_neighbors]["description"].tolist()
    maps = tipe_data.iloc[n_neighbors][maphotel_feats].to_numpy()
    return urls, maps, descriptions

def recommend_hotels():
    st.write("Do you want me to recommend you hotels or resorts?")
    tipe = st.radio(
        "",
    tipehotel_names, index=tipehotel_names.index("hotel"))
    st.markdown("")

    st.write("What kind of facilities do you like? (slide this based on your preferences)")
    with st.container():
        col1, col2,= st.columns((2,2))
        with col2:
            beach = st.slider(
                'Beach View',
                0, 100, 50)
            recreation = st.slider(
                'Recreation Place',
                0, 100, 50)
            solofriendly = st.slider(
                'Solo-friendly',
                0, 100, 50)
        with col1:
            gym = st.slider(
                'Gym & Swimming Pool Facility',
                0, 100, 50)
            affordable = st.slider(
                'Affordable',
                0, 100, 50)
            restaurant = st.slider(
                'Restaurant',
                0, 100, 50)
            

    tracks_per_page = 6
    test_feat = [gym, affordable, restaurant, beach, recreation, solofriendly]
    urls, maps, descriptions = n_neighbors_url_map(tipe, test_feat)
    

    tracks = []
    for url,description in zip(urls, descriptions):
        track = """<div class="mapouter"><div class="gmap_canvas"><iframe class="gmap_iframe" width="100%" frameborder="0" scrolling="no" marginheight="0" marginwidth="0" src="https://maps.google.com/maps?width=400&amp;height=400&amp;hl=en&amp;q={}&amp;t=&amp;z=14&amp;ie=UTF8&amp;iwloc=B&amp;output=embed"></iframe><h4 style="color:white;font-family:Arial">{}</h4></div><style>.mapouter.gmap_canvas.gmap_iframe </style></div>
        <div><h6 style="color:#d3dfe1;font-family:Arial;text-align:justify">{}</h6></div>""".format(url,url,description)
        tracks.append(track)

    if 'previous_inputs' not in st.session_state:
        st.session_state['previous_inputs'] = [tipe] + test_feat
    
    current_inputs = [tipe] + test_feat
    if current_inputs != st.session_state['previous_inputs']:
        if 'start_track_i' in st.session_state:
            st.session_state['start_track_i'] = 0
        st.session_state['previous_inputs'] = current_inputs

    if 'start_track_i' not in st.session_state:
        st.session_state['start_track_i'] = 0
    
    st.write("This is the best place for you!")
    with st.container():
        col1, col2, col3 = st.columns([2,1,2])
        if st.button("Recommend More Hotels"):
            if st.session_state['start_track_i'] < len(tracks):
                st.session_state['start_track_i'] += tracks_per_page

        current_tracks = tracks[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
        current_maps = maps[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
        if st.session_state['start_track_i'] < len(tracks):
            for i, (track, map) in enumerate(zip(current_tracks, current_maps)):
                if i%2==0:
                    with col1:
                        components.html(
                            track,
                            height=400,
                        )
            
                else:
                    with col3:
                        components.html(
                            track,
                            height=400,
                        )
        else:
            st.write("No places left to recommend")

# ---------------------------------------------------------------- #

# Recommender System - Destinations
@st.cache(allow_output_mutation=True)
def load_data_destination():
    df = pd.read_excel("data/datadestinationZyo.xlsx")
    df['types'] = df.types.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    exploded_track_df = df.explode("types")
    return exploded_track_df

tipe_names = ['cultural', 'nature']
map_feats = ["affordable", "exotic", "unique", "culture","transportation", "soloFriendly"]

exploded_track_df = load_data_destination()

def n_neighbors_url_map_dest(tipe, test_feat):
    tipe = tipe.lower()
    tipe_data = exploded_track_df[(exploded_track_df["types"]==tipe)]
    tipe_data = tipe_data.sort_values(by='popularity', ascending=False)[:500]

    neigh = NearestNeighbors()
    neigh.fit(tipe_data[map_feats].to_numpy())

    n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(tipe_data), return_distance=False)[0]

    urls = tipe_data.iloc[n_neighbors]["url"].tolist()
    descriptions = tipe_data.iloc[n_neighbors]["description"].tolist()
    maps = tipe_data.iloc[n_neighbors][map_feats].to_numpy()
    return urls, maps, descriptions

def recommend_destination():
    st.write("What kind of places do you want me to recommend you?")
    tipe = st.radio(
        "",
    tipe_names, index=tipe_names.index("cultural"))
    st.markdown("")

    st.write("What kind of facilities do you like? (slide this based on your preferences)")
    with st.container():
        col1, col2,= st.columns((2,2))
        with col2:
            culture = st.slider(
                'Culture',
                0, 100, 50)
            transportation = st.slider(
                'Transportation',
                0, 100, 50)
            solofriendly = st.slider(
                'Solo-friendly',
                0, 100, 50)
        with col1:
            affordable = st.slider(
                'Affordable',
                0, 100, 50)
            exotic = st.slider(
                'Exotic',
                0, 100, 50)
            unique = st.slider(
                'Unique',
                0, 100, 50)
            

    tracks_per_page = 6
    test_feat = [affordable, exotic, unique, culture, transportation, solofriendly]
    urls, maps, descriptions = n_neighbors_url_map_dest(tipe, test_feat)

    tracks = []
    for url,description in zip(urls, descriptions):
        track = """<div class="mapouter"><div class="gmap_canvas"><iframe class="gmap_iframe" width="100%" frameborder="0" scrolling="no" marginheight="0" marginwidth="0" src="https://maps.google.com/maps?width=400&amp;height=400&amp;hl=en&amp;q={}&amp;t=&amp;z=14&amp;ie=UTF8&amp;iwloc=B&amp;output=embed"></iframe><h4 style="color:white;font-family:Arial">{}</h4></div><style>.mapouter.gmap_canvas.gmap_iframe </style></div>
        <div><h6 style="color:#d3dfe1;font-family:Arial;text-align:justify">{}</h6></div>""".format(url,url,description)
        tracks.append(track)

    if 'previous_inputs' not in st.session_state:
        st.session_state['previous_inputs'] = [tipe] + test_feat
    
    current_inputs = [tipe] + test_feat
    if current_inputs != st.session_state['previous_inputs']:
        if 'start_track_i' in st.session_state:
            st.session_state['start_track_i'] = 0
        st.session_state['previous_inputs'] = current_inputs

    if 'start_track_i' not in st.session_state:
        st.session_state['start_track_i'] = 0
    
    st.write("This is the best place for you!")
    with st.container():
        col1, col2, col3 = st.columns([2,1,2])
        if st.button("Recommend More Destinations"):
            if st.session_state['start_track_i'] < len(tracks):
                st.session_state['start_track_i'] += tracks_per_page

        current_tracks = tracks[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
        current_maps = maps[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
        if st.session_state['start_track_i'] < len(tracks):
            for i, (track, map) in enumerate(zip(current_tracks, current_maps)):
                if i%2==0:
                    with col1:
                        components.html(
                            track,
                            height=400,
                        )
            
                else:
                    with col3:
                        components.html(
                            track,
                            height=400,
                        )
        else:
            st.write("No places left to recommend")

# ---------------------------------------------------------------- #




# DIALOGPT CHATBOT
# --------------------------------------------------------------- #
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

@st.cache(hash_funcs={transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast: hash}, suppress_st_warning=True)
def load_data():    
 st.warning("Wait a second, i have to prepare myself to talk...")
 tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small", padding_side='left')
 model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
 return tokenizer, model

def generative_model():
    try:
        tokenizer, model = load_data()

        st.write("###### You can talk to me anything you want. I hope i can understand you :)")
        input = st.text_input('Talk anything to me:')

        if 'count' not in st.session_state or st.session_state.count == 6:
            st.session_state.count = 0 
            st.session_state.chat_history_ids = None
            st.session_state.old_response = ''
        else:
            st.session_state.count += 1

        # tokenize the new input sentence
        new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1) if st.session_state.count > 1 else new_user_input_ids

        # generate a response 
        st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=500, pad_token_id=tokenizer.eos_token_id)

        # convert the tokens to text, and then split the responses into lines
        response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        if st.session_state.old_response == response:
            bot_input_ids = new_user_input_ids
            st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id, temperature=0.6, repetition_penalty=1.3)
            response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            
        st.write(f"Zyo: {response}")
        st.session_state.old_response = response
    except:
        pass

# --------------------------------------------------------------- #



intents = json.loads(open('intents/intentsZyoLarge.json').read())
words = pickle.load(open('model/texts.pkl','rb'))
labels = pickle.load(open('model/labels.pkl','rb'))
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))
def predict_class(sentence, model):
    # filter out predictions below a threshold
    
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.50
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
        
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": labels[r[0]], "probability": str(r[1])})
        return return_list
    
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            if tag == 'bestressort_faq':
                recommend_hotels()
                result = ""
                break
            elif tag == 'besthotel_faq':
                recommend_hotels()
                result = ""
                break
            elif tag == 'just_talk':
                generative_model()
            elif tag == 'recomend_destination':
                recommend_destination()
                result = ""
                break
            result = random.choice(i['responses'])
    return result
    
def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


# STREAMLIT APP
st.markdown('<style>body{text-align:center;background-color:black;color:white;align-items:justify;display:flex;flex-direction:column;}</style>', unsafe_allow_html=True)
st.title("Zyo: Solo Travel Like a Pro!")

st.markdown("Zyo is a chatbot that will guide you to explore Indonesia, even if you are alone!")
st.image("assets/ZyoLanding.png")
message = st.text_input("You can ask me anything about Bali, or just share your feelings with me!")


ints = predict_class(message, model)
res = getResponse(ints,intents)
if res != "":
    st.success("Zyo: {}".format(res))

st.markdown('<div style="text-align: justify; font-size: 10pt"><b>Tips:</b> Dont know anything about Bali? you can ask FAQ question to Zyo! try: "How is the weather in Bali" or "What language do they speak?"</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: justify;></div>', unsafe_allow_html=True)

if st.button('Feel stuck? More tips'):
    st.image('assets/tutorial.png')