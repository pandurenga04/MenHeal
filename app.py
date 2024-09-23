from flask import Flask, render_template, request
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Train a simple model for demonstration purposes
data = [
    ('I am happy', 'happy'), ('I am sad', 'sad'), ('I am angry', 'angry'),
    ('I feel calm', 'calm'), ('I am bored', 'bored'), ('I am stressed', 'stressed'),
    ('I feel guilty', 'guilty')
]
texts, labels = zip(*data)
model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
model.fit(texts, labels)

# Emotion-specific advice and activities
advice_data = {
    "happy": "Keep up the positivity! Engage in activities that make you feel fulfilled.",
    "sad": "It's okay to feel sad. Try practicing mindfulness or listen to some calming music. You can also try a meditation session.",
    "angry": "Take a deep breath and try to relax. How about a short meditation or a walk outside?",
    "calm": "Great! Keep up the calmness by engaging in mindful activities like yoga or meditation.",
    "stressed": "You seem stressed. Try some guided meditation, or take a break with this mini-quiz or joke.",
    "bored": "Feeling bored? How about playing a quick quiz or reading some jokes to lighten the mood?",
    "guilty": "Don't be too hard on yourself. Reflect on the situation and practice forgiveness. Try meditation to clear your mind."
}

activities = {
    "meditation": "Here's a link to a 5-minute meditation: <a href=https://youtu.be/inpok4MKVLM?si=eFX4_eJnL0Xi7Bw9>Meditation Video</a>",
    "quiz": "Want to lighten up? Take this fun quiz:<a href=https://www.quotev.com/quiz/16736742/Which-MAGIC-Character-Are-You> Quiz Time!</a>x",
    "jokes": [
        "Why don't skeletons fight each other? Because they don't have the guts!",
        "Why did the scarecrow win an award? Because he was outstanding in his field!",
        "What do you call fake spaghetti? An impasta!",
        "I told my wife she was drawing her eyebrows too high. She looked surprised!",
        "Why don’t scientists trust atoms? Because they make up everything!"
    ]
}

# Route to handle both input and output on the same page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input = request.form['input_text']
        prediction = model.predict([user_input])[0]

        # Prepare response based on the emotion detected
        advice = advice_data.get(prediction, "Take care of yourself!")
        activities_list = []

        # Add meditation, jokes, or quiz for specific emotions
        if prediction in ["stressed", "guilty", "sad"]:
            activities_list.append(activities["meditation"])
            activities_list.append(activities["quiz"])

            # Randomly pick one joke from the list of jokes
            import random
            joke = random.choice(activities["jokes"])
            activities_list.append(joke)

        return render_template('index.html', emotion=prediction, advice=advice, activities=activities_list)
    
    return render_template('index.html', emotion=None, advice=None, activities=None)

if __name__ == '__main__':
    app.run(debug=True)
