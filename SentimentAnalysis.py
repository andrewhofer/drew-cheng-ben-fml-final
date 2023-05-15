import nltk
#nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    score_cutoff = 0.1

    analyzer = SentimentIntensityAnalyzer()

    sentiment_scores = analyzer.polarity_scores(text)
    scr = sentiment_scores['compound']

    if scr >= score_cutoff:
        sentiment = 'Positive'
    elif scr <= -1 * score_cutoff:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    return sentiment, sentiment_scores

ex1 = "I really enjoyed the movie. The acting was excellent!"
ex2 = "We went to the store on a very average day."
ex3 = "That restaurant made me want to throw up. I hope the chef dies."
ex4 = "Apple stock is expected to rise significantly over the next few days."
ex5 = "Google CEO caught having an joyous affair."
ex6 = "Everything about today was exactly the opposite of disgusting, horrible, and gross."
sent, scores = analyze_sentiment(ex5)
print(sent)
print(scores)
