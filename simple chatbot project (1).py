import random
import pandas as pd
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= Response =================
resposes = {
    "greeting" : ["hello! how can i help you😊","Hi there","Hi! Nice to see you ?"],
    "goodbye" : ["goodbye! have a great day😊","Bye","see you soon","Talk to you later"],
    "thanks" : ["You're welcome!","No problem","I'm Glad to help"],
    "about" : ["I am a simple ML-based chatbot"],
    "fallback": [
        "Sorry, I didn't understand that 😕",
        "Can you rephrase your question?",
        "I'm still learning, please ask something else",
        "That seems new to me 🤔" 
    ]

}


gk_df = pd.read_csv("Gk.csv")
chat_df = pd.read_csv("chatbot_data.csv")

# ================= TEXT CLEAN =================
def clean_text(text):
    return text.lower().strip()

def saved_csv(question,answer):
    file_name = "Gk.csv"
    with open(file_name,mode = "a", encoding="utf-8") as f:
        writer  = csv.writer(f)
        writer.writerow([question,answer])
        print("Data successfully saved to CSV!")

gk_df["question"] = gk_df["question"].apply(clean_text)
chat_df["text"] = chat_df["text"].apply(clean_text)

# ================= TF-IDF =================
vectorizer = TfidfVectorizer()

all_text = pd.concat([
    gk_df["question"],
    chat_df["text"]
])
vectorizer.fit(all_text)

gk_vectors = vectorizer.transform(gk_df["question"])
x = vectorizer.transform(chat_df["text"]) 

# ================= USER INPUT ===========
#
while True: 
    user_input = input("Enter the message or Type 'exit' to close the bot")
    user_input_clean = clean_text(user_input)
    
    if user_input_clean == "exit":
        print("Goobye")
        break


        
    user_vector = vectorizer.transform([user_input_clean])

        # ================= CHAT CSV (greetings etc.) =================
    chat_sim = cosine_similarity(user_vector, x)[0]
    chat_idx = chat_sim.argmax()
    chat_conf = chat_sim[chat_idx]
    print("Confidence:", round(chat_conf, 2))

    if chat_conf > 0.55:
        answer = chat_df.iloc[chat_idx]["intent"]
        bot_reply = random.choice(resposes[answer])
        print(f"  You : {user_input}")
        print(f"ChatGPT: {bot_reply}")
        continue
        


    # ================= GK CSV (knowledge) =================
    gk_sim = cosine_similarity(user_vector, gk_vectors)[0]
    best_idx = gk_sim.argmax()
    confidence = gk_sim[best_idx]

    print("Confidence:", round(confidence, 2))

    if confidence > 0.60:
        answer = gk_df.iloc[best_idx]["answer"]
        print(f"ChatGPT: {answer}")

    else:
        fallback = [
            "Sorry, I didn't understand 😕",
            "Can you rephrase your question?",
            "I'm still learning "
        ]
        print(f"ChatGPT: {random.choice(fallback)}")
        print("I don't know the answer,if you know answer write below")
        new_answer = input("Iska sahi ansewer kya hona chahiye ?")
        
        if new_answer.lower() != 'skip':
            saved_csv(user_input,new_answer)
            
        
