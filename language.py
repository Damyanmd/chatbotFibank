from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

stemmer_en = SnowballStemmer("english")
stemmer_de = SnowballStemmer("german")

stop_words_eng = set(stopwords.words('english'))
stop_words_ger = set(stopwords.words('german'))

# Function to process words with appropriate stemmer
def stem_word(word, lang):
    if lang == "english":
        return stemmer_en.stem(word)
    elif lang == "german":
        return stemmer_de.stem(word)
    else:
        return word

# Determine language for each pattern (simple heuristic: if word in German stopwords, it's German)
def detect_language(word):
    if word.lower() in stop_words_ger:
        return "german"
    return "english"