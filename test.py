import tensorflow as tf
import pickle


test_words = ['fuck me daddy', 'good game', 'hi there mister', 'melanin papi', 'cock rings are the best', 'space is so mysterious', 'monster dong in my ass', 'get a life bitch', 'absolute scum', 'wankers the lot of them', 'i miss my petal gone but not forgotten']

text_filter_model = tf.keras.models.load_model("text_profanity_model.keras")
with open("./tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)
    f.close()

checks = []
for word in test_words:
    prediction = tfidf.transform([word]).toarray()
    prediction = text_filter_model.predict(prediction)
    print(word, " : ", prediction)
    is_profane = True if prediction > 0.77 else False
    print(f"{word} = {is_profane}")
    checks.append(is_profane)
