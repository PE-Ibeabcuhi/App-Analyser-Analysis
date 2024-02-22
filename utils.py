from google_play_scraper import app, Sort, reviews
from app_store_scraper import AppStore
import pandas as pd
import numpy as np
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re


# Initiating the transformer
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Getting the sentiment from BERT LLM
@st.cache_resource
def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1

# Getting the reviews from google playstore
@st.cache_data
def get_google_play_reviews(app_id):
    try:
        # Get reviews using play_scraper
        result, continuation_token = reviews(
            app_id,
            lang='en', # sets the language to english
            country='us', # sets the country to us
            sort=Sort.NEWEST, # Sort the reviews from the first 1000
            count= 1000, # defaults to 100
            filter_score_with=None # Apply no filters
        )

        # If you pass `continuation_token` as an argument to the reviews function at this point,
        # it will crawl the items after 3 review items.

        result, _ = reviews(
            app_id,
            continuation_token = None # load the model from the beginning
        )
        # Convert the reviews to a DataFrame
        g_df = pd.DataFrame(np.array(result), columns=['review'])
        g_df2 = g_df.join(pd.DataFrame(g_df.pop('review').tolist()))

        # Extract relevant columns and rename them
        df1 = g_df2[['content', 'score', 'at']]
        df1 = df1.rename(columns={'content': 'Reviews', 'score': 'Ratings', 'at': 'Date'})

        # Add a 'Source' column
        df1['Source'] = 'Google Play'

        # Adding a year column
        df1['Year']= df1['Date'].dt.year
        
        # Adding a sentiment column
        j = df1['Reviews'].apply(lambda x: sentiment_score(x[:512]))
        df1['Sentiments'] = np.select([j.isin([1,2]), j==3, j.isin([4,5])],['Negative','Neutral','Positive'],
                                    default ='Unknown')

         #Return DataFrame                           
        return df1
    except Exception as e:
        # Catch exceptions and display a custom error message
        st.error.sidebar("Please check the link. An error occurred while fetching Google Play reviews")
        st.stop()
        

# Getting the reviews from appstores
@st.cache_data
def get_app_store_reviews(countrycode, app_name, app_id):
    try:
        # Get reviews using app_store_scraper
        a_reviews = AppStore(countrycode, app_name, app_id)
        a_reviews.review(how_many=1000)
        
        # Convert the reviews to a DataFrame
        a_df = pd.DataFrame(np.array(a_reviews.reviews), columns=['review'])
        a_df2 = a_df.join(pd.DataFrame(a_df.pop('review').tolist()))

        # Extract relevant columns and rename them
        df2 = a_df2[['review', 'rating', 'date']]
        df2 = df2.rename(columns={'review': 'Reviews', 'rating': 'Ratings', 'date': 'Date'})

        # Add a 'Source' column
        df2['Source'] = 'App Store'

        # Adding a year column
        df2['Year']= df2['Date'].dt.year
        
        # Adding a sentiment analysis
        j = df2['Reviews'].apply(lambda x: sentiment_score(x[:512]))
        df2['Sentiments'] = np.select([j.isin([1,2]), j==3, j.isin([4,5])],['Negative','Neutral','Positive'],
                                    default ='Unknown')
        
        return df2
    except Exception as e:
        # Catch the exception and display a custom error message
        st.error.sidebar("Please check the link. An error occurred while fetching Appstore reviews.")
        st.stop()
        
          # Stop further execution

# Validating the link from google play store
@st.cache_data
def validate_google_play_link(link):
    google_play_pattern = r'^https://play\.google\.com/store/apps/details\?id=[a-zA-Z0-9._]+$'
    
    if not re.match(google_play_pattern, link):
        return False  # Invalid link

    # Split the link using '=' and '&' as delimiters
    link_parts = link.split('=')[1].split('&')

    # Extract the app name (the first part after '=')
    
    app_name = link_parts[0]
    
    return app_name

# Validating the link from apple store
@st.cache_data
def validate_app_store_link(link):
    apple_store_pattern = r'^https://apps\.apple\.com/[a-z]{2}/app/.+?/id\d+$'
    
    if not re.match(apple_store_pattern, link):
        return None  # Invalid link

    # Split the link using '/'
    link_parts = link.split('/')

    # Extract components
    country_code = link_parts[3]
    app_name = link_parts[5]
    app_id = link_parts[-1].split("id")[1]

    return country_code, app_name, app_id

import streamlit as st

def app_analyzer_description():
    content = """
    # App Analyser: Unveiling the Sentiments behind Your App Choices

    Welcome to App Analyser, a web app designed to analyze your favorite apps in terms of reviews and ratings on both the Google Play Store and Apple App Store. It utilizes the BERT (Bidirectional Encoder Representations from Transformers) model to classify reviews into various sentiment categories.

    ## What Happens Inside

    The web app utilizes the [google-play-scraper](https://pypi.org/project/google-play-scraper/) and [Appstore-scraper](https://pypi.org/project/app-store-scraper/) to fetch data from the Google Play Store and Apple App Store. It has been optimized to exclude unnecessary columns, retaining only those essential for sentiment analysis. After retrieving the data, the BERT model is employed to classify reviews on a scale of 1 to 5. Ratings of 1 and 2 represent negative reviews, 3 signifies a neutral review, while 4 and 5 indicate positive reviews.

    ### Important Notice: Acknowledging Processing Limitations

    You may encounter a slow loading time during the initial run of the analyzer. This is because the web scrapers fetch reviews in small increments and generate new HTTPS requests after every 200 reviews. Additionally, the BERT model processes each individual review before classification, so more reviews mean longer preprocessing time. We are actively working to optimize this process. In the meantime, it's advisable to analyze specific apps, and if you have patience, you can analyze larger apps as well.

    ## Usage

    To use the analyzer, locate your favorite app on the Play Store or App Store, and copy the links. A typical App Store link looks like this:

    ```
    https://apps.apple.com/ie/app/1password-password-manager/id1511601750
    ```

    A typical Play Store link looks like this:

    ```
    https://play.google.com/store/apps/details?id=com.artemchep.keyguard
    ```

    For more clarity, watch the video below.

    If you're curious and want to see how the app works, here are app links for both App Store and Play Store:

    - **Appstore:** [1Password Password Manager](https://apps.apple.com/ie/app/1password-password-manager/id1511601750)
    - **Playstore:** [Keyguard](https://play.google.com/store/apps/details?id=com.artemchep.keyguard)

    It is also important to note that the app analyzes links from the specific store provided, so for apps on both Play Store and App Store, each link on both websites will be analyzed separately.

    To download the retrieved data for personal analysis, hover over the CSV table and click on the download icon.

    ## Disclaimer

    This web app is developed as a personal project for educational purposes only. Please refrain from using it for malicious or illegal activities. Use responsibly.
    """
    return content

