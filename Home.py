import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from utils import validate_google_play_link, get_google_play_reviews, sentiment_score, get_app_store_reviews, validate_app_store_link, app_analyzer_description


#-- setting up the main page --
st.set_page_config(page_title="Appstore sentiment Analysis", page_icon=":bar_chart:", layout="wide")

# Create a placeholder for the welcome message
welcome_placeholder = st.empty()

# Display the welcome message
welcome_placeholder.markdown(app_analyzer_description(), unsafe_allow_html=True)

# --Setting the side_bar title
st.sidebar.title('App Analyser')

#-- Reading the data--
app_link = st.sidebar.text_input("Enter App link here:")
df = None

if app_link:
    app_store_result = validate_app_store_link(app_link)
    google_play_result = validate_google_play_link(app_link)
    
    if app_store_result:
        try:
            country_code, app_name, app_id = app_store_result
            with st.spinner(f"Fetching Appstore reviews for {app_name} please hold..."):
                df = get_app_store_reviews(country_code, app_name, app_id)
                name = app_name.title()
                st.markdown(f"<p style='font-size:35px; text-align:center;'>{name} APP Dashboard</p>", unsafe_allow_html=True)
        except Exception as e:
            st.sidebar.error("Please check the link. An error occurred while fetching Appstore reviews.")
            st.stop()
    elif google_play_result:
        try:
            app_name = google_play_result
            with st.spinner(f"Fetching Playstore reviews for {app_name} please hold..."):
                df = get_google_play_reviews(app_name)
                name = app_name.split('com.')
                name = name[1].title()
                st.markdown(f"<p style='font-size:35px; text-align:center;'>{name} APP Dashboard</p>", unsafe_allow_html=True)
        except Exception as e:
            st.sidebar.error("Please check the link. An error occurred while fetching Playstore reviews.")
            st.stop()
    else:
        st.sidebar.warning('Invalid link, please check the link')


    

# ---- SIDEBAR ----
st.sidebar.header("Please Filter Here:")
if df is not None:
    # Clear the welcome message placeholder
    welcome_placeholder.empty()
    # A filter for Years
    selected_years = st.sidebar.multiselect("Select the Year:",
    options=df["Year"].unique(),
    default=df["Year"].unique())

    # A filter for reviews
    selected_review = st.sidebar.multiselect("Select Review type:",
    options=df["Sentiments"].unique(),
    default=df["Sentiments"].unique())

    # A filter for Ratings
    selected_rating = st.sidebar.multiselect("Select Rating type:",
    options=df["Ratings"].unique(),
    default=df["Ratings"].unique())

    # Setting the Title for the Main Page



    df_selection = df.query("Year == @selected_years & Sentiments == @selected_review & Ratings == @selected_rating")
    st.dataframe(df_selection)

    # Check if the dataframe is empty:
    if df_selection.empty:
        st.warning("No data available based on the current filter settings!")
        st.stop() # This will halt the app from further execution.
    

    # TOP KPI's
    total_reviews = len(df_selection)
    negative_reviews = len(df_selection[df_selection['Sentiments']=='Negative'])
    positive_reviews = len(df_selection[df_selection['Sentiments']=='Positive'])
    average_ratings = df_selection['Ratings'].mean().round(2)
    star_rating = "⭐️" * int(round(average_ratings, 0))

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("#### Total Reviews")
        st.markdown(f"<p style='font-size:20px; text-align:center;'>{total_reviews}</p>", unsafe_allow_html=True)

    with col2:
        st.markdown("#### Positive Reviews")
        st.markdown(f"<p style='font-size:20px; text-align:center;'>{positive_reviews}</p>", unsafe_allow_html=True)

    with col3:
        st.markdown("#### Negative Reviews")
        st.markdown(f"<p style='font-size:20px; text-align:center;'>{negative_reviews}</p>", unsafe_allow_html=True)

    with col4:
        st.markdown("#### Average Ratings")
        st.markdown(f"<p style='font-size:20px; text-align:center;'>{average_ratings} {star_rating}</p>", unsafe_allow_html=True)
            
    st.markdown("""---""")


    # Reviews count (doughnut chart)
    sentiment_counts = df_selection['Sentiments'].value_counts()

    fig_sentiment_counts = px.pie(
        sentiment_counts,
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution (Doughnut Chart)",
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.5,  # Set the hole parameter to create a doughnut chart
    )
    fig_sentiment_counts.update_layout(
        xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=False),  # Corrected from yaxis=(dict(showgrid=False))
    )

    # Ratings count(bar chart)
    rating_counts = df_selection['Ratings'].value_counts()
    fig_ratings_counts = px.bar(
        rating_counts,
        x="count",
        y=rating_counts.index,
        orientation="h",
        title=" Ratings Performance",  
        color_discrete_sequence=["#0083B8"] * len(rating_counts),  
        template="plotly_white",
    )

    fig_ratings_counts.update_layout(
        xaxis_visible=False,
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=False),  
    )

    left_column, right_column = st.columns(2)
    left_column.plotly_chart(fig_sentiment_counts, use_container_width=True)
    right_column.plotly_chart(fig_ratings_counts, use_container_width=True)

    st.markdown("""---""")

    st.subheader('Common Words')

    #-- Wordcloud
    df = df_selection['Reviews'].astype(str)
    # Generate WordCloud for 'Poem' entries in the 'Music' genre with stopwords removed
    wordcloud = WordCloud(stopwords=STOPWORDS, max_font_size=30).generate(' '.join(df))
    fig,ax = plt.subplots()
    ax.imshow(wordcloud)
    ax.axis('off')
    st.pyplot(fig)

    st.markdown("""---""")

    st.subheader('Top Poor Feedbacks')
    # Getting the top negative reviews
    top_negative = df_selection[df_selection['Sentiments'] == 'Negative'].sort_values(by='Ratings', ascending=True).head(10)
    top_negative_reviews_df = top_negative['Reviews'] 

    # Displaying the top negative reviews 
    st.dataframe(top_negative_reviews_df, use_container_width=True)

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)