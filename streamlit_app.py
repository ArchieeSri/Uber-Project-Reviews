import streamlit as st

def Main_page():
    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")

def page2():
    st.markdown("# WordCloud ❄️")
    st.sidebar.markdown("# WordCloud ❄️")

def page3():
    st.markdown("# Analysis 🎉")
    st.sidebar.markdown("# Analysis 🎉")

def page4():
    st.markdown("# Word_Cloud_Vectorisers 🎉")
    st.sidebar.markdown("# Word_Cloud_Vectorisers 🎉")

page_names_to_funcs = {
    "Main Page": main_page,
    "WordCloud": WordCloud,
    "Analysis": Analysis,
    "Word_Cloud_Vectorisers": Word_Cloud_Vectorisers,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()