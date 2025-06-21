import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation

## İdeal Sıralama: Bu yüzden, en doğru sonuçlar için sıralama genellikle şöyledir: 
# Tokenization → Lowercasing → Stopword Removal → POS Tagging → Lemmatization.
## Lemmatization'ın en doğru şekilde çalışması için kelimenin POS etiketine ihtiyacı vardır.

## Konu: Haber Başlıklarından Duygu Analizi ve Topic Modeling## 

def download_nltk_data():
    """Gerekli NLTK veri paketlerini indirir."""
    packages = [
        ('corpora/stopwords', 'stopwords'),
        ('tokenizers/punkt', 'punkt'),
        ('corpora/wordnet', 'wordnet'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('sentiment/vader_lexicon', 'vader_lexicon')
    ]
    for path, package_id in packages:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"NLTK paketi indiriliyor: {package_id}")
            nltk.download(package_id)

def get_wordnet_pos(treebank_tag):
    """NLTK POS etiketlerini WordNet formatına dönüştürür."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text, lemmatizer, stop_words):
    """Bir metin üzerinde tüm ön işleme adımlarını uygular."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    pos_tagged_tokens = nltk.pos_tag(tokens)
    processed_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tagged_tokens]
    return " ".join(processed_tokens)

def perform_preprocessing(df):
    """DataFrame üzerinde metin ön işlemeyi gerçekleştirir."""
    print("--- Adım 1: Metin Ön İşleme Başlatılıyor ---")
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    df['cleaned_headline'] = df['headline'].apply(lambda text: preprocess_text(text, lemmatizer, stop_words))
    print("Ön işleme tamamlandı.")
    print("--- Örnek Temizlenmiş Başlıklar ---")
    print(df[['headline', 'cleaned_headline']].head())
    df.to_csv('processed_news.csv', index=False)
    print("\nÖn işlenmiş veriler 'processed_news.csv' dosyasına kaydedildi.")
    return df

def perform_vectorization(data):
    """Metin verilerini CountVectorizer ve TF-IDF ile sayısallaştırır."""
    print("\n\n--- Adım 2: Metinlerin Sayısallaştırılması (Vectorization) ---")
    sample_data = data.dropna().head()

    # CountVectorizer
    print("\n--- Yöntem 1: CountVectorizer (Bag-of-Words Modeli) ---")
    count_vectorizer = CountVectorizer()
    bow_matrix = count_vectorizer.fit_transform(sample_data)
    bow_df = pd.DataFrame(bow_matrix.toarray(), columns=count_vectorizer.get_feature_names_out(), index=sample_data.index)
    print("--- Örnek CountVectorizer Çıktısı ---")
    print(bow_df)

    # TF-IDF Vectorizer
    print("\n--- Yöntem 2: TF-IDF Vectorizer ---")
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sample_data)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=sample_data.index)
    print("--- Örnek TF-IDF Çıktısı ---")
    print(tfidf_df)

def perform_sentiment_analysis(df):
    """Haber başlıkları üzerinde VADER ile duygu analizi yapar."""
    print("\n\n--- Adım 3: Duygu Analizi (Sentiment Analysis) ---")
    sia = SentimentIntensityAnalyzer()
    
    def get_sentiment(text):
        scores = sia.polarity_scores(text)
        compound_score = scores['compound']
        if compound_score >= 0.05:
            return 'Pozitif'
        elif compound_score <= -0.05:
            return 'Negatif'
        else:
            return 'Nötr'
            
    df['sentiment'] = df['cleaned_headline'].apply(get_sentiment)
    print("--- Duygu Analizi Sonuç Özeti ---")
    print(df['sentiment'].value_counts())
    df.to_csv('processed_news_with_sentiment.csv', index=False)
    print("\nDuygu analizi sonuçlarını içeren veriler 'processed_news_with_sentiment.csv' dosyasına kaydedildi.")
    return df

def perform_topic_modeling(df):
    """LDA kullanarak konu modellemesi yapar."""
    print("\n\n--- Adım 4: Konu Modelleme (Topic Modeling) ---")
    # LDA için en iyi sonuçlar genellikle kelime sayımlarıyla (CountVectorizer) elde edilir.
    # Çok sık ve çok nadir kelimeleri filtreleyerek daha anlamlı konular bulmayı hedefleriz.
    lda_vectorizer = CountVectorizer(max_df=0.90, min_df=5, stop_words='english')
    doc_term_matrix = lda_vectorizer.fit_transform(df['cleaned_headline'].dropna())
    
    n_topics = 10
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_model.fit(doc_term_matrix)

    def display_topics(model, feature_names, no_top_words):
        print("\n--- LDA Konu Başlıkları ve Anahtar Kelimeleri ---")
        for topic_idx, topic in enumerate(model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
            
            # Konuları anahtar kelimelere göre yorumlayarak anlamlı bir isim atama
            topic_name = f"Konu #{topic_idx+1}" # Varsayılan isim
            if any(word in top_words for word in ["trump", "president", "obama", "white", "house", "republican", "democrat"]):
                topic_name = "ABD Politikası"
            elif any(word in top_words for word in ["woman", "parent", "kid", "life", "feel", "mother", "child"]):
                topic_name = "Yaşam & Aile"
            elif any(word in top_words for word in ["style", "beauty", "dress", "fashion", "hair"]):
                topic_name = "Moda & Güzellik"
            elif any(word in top_words for word in ["food", "recipe", "eat", "make", "home", "kitchen", "chicken"]):
                topic_name = "Yemek & Ev"
            elif any(word in top_words for word in ["travel", "world", "city", "visit", "destination"]):
                topic_name = "Seyahat & Dünya"
            elif any(word in top_words for word in ["photo", "video", "show", "star", "movie"]):
                topic_name = "Eğlence & Magazin"
            elif any(word in top_words for word in ["health", "study", "say", "people", "like"]):
                topic_name = "Sağlık & Bilim"
            
            print(f"[{topic_name}]: {' '.join(top_words)}")

    display_topics(lda_model, lda_vectorizer.get_feature_names_out(), 10)
    print("\nKonu modellemesi tamamlandı.")


def main():
    """Ana program akışı."""
    # Gerekli kütüphaneleri indir
    download_nltk_data()
    
    # Veri setini yükle
    try:
        df = pd.read_json('News_Category_Dataset_v3.json', lines=True)
    except Exception as e:
        print(f"JSON yüklenirken hata oluştu: {e}")
        return

    # Adım 1: Ön İşleme
    df_processed = perform_preprocessing(df.copy())
    
    # Adım 2: Sayısallaştırma
    perform_vectorization(df_processed['cleaned_headline'])
    
    # Adım 3: Duygu Analizi
    df_sentiment = perform_sentiment_analysis(df_processed.copy())

    # Adım 4: Konu Modelleme
    perform_topic_modeling(df_sentiment.copy())
    
    print("\n\nTüm adımlar başarıyla tamamlandı.")


if __name__ == "__main__":
    main() 