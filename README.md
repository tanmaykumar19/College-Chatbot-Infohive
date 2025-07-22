# College-Chatbot-Infohive
InfoHive - College Chatbot (GNE)
InfoHive is an intelligent chatbot built using Python and PyTorch to provide instant information to students and staff of Guru Nanak Dev Engineering College (GNE), Ludhiana. It answers frequently asked questions about college facilities, departments, academic schedules, and more.

🚀 Features

Instant responses to general college queries

Handles greetings, goodbyes, and thank-you messages

Provides details like:

Department locations (e.g., IT block, library)

College timings

Contact information

Staff and faculty details

Easy to train and customize using intents.json

Built with Flask, PyTorch, and NLTK

📷 Demo
!(demo.png)



🛠️ Technologies Used

Python 3

PyTorch (Neural Network)

Flask (Web framework)

NLTK (Natural Language Processing)

HTML/CSS/JavaScript (for frontend)

JSON (for storing intents)

🧠 How It Works

The user types a message (e.g., "Where is the IT department?")

The message is tokenized and stemmed

A bag-of-words vector is created

The trained model predicts the intent

The bot replies with a suitable response from intents.json

📁 Project Structure

InfoHive/
├── static/
│ └── css, js, images
├── templates/
│ └── chatbot.html
├── intents.json
├── model.py
├── nltk_utils.py
├── train.py
├── chat.py
├── app.py
└── data.pth (trained model)

📦 Installation & Setup

Clone the repository

git clone https://github.com/your-username/InfoHive.git
cd InfoHive

Create a virtual environment and install dependencies

pip install -r requirements.txt

Train the chatbot model

python train.py

Run the Flask app

python app.py

Open your browser and go to http://localhost:5000

🧪 Sample Intents (intents.json)

{
"intents": [
{
"tag": "greeting",
"patterns": ["Hi", "Hello", "Good morning"],
"responses": ["Hello!", "Hi there!", "Greetings!"]
},
{
"tag": "college_timing",
"patterns": ["When does college start?", "College timing"],
"responses": ["College runs from 9:00 AM to 5:00 PM."]
}
]
}

📌 Future Improvements

Add login authentication for student-specific queries

Connect to live college database (timetable, faculty availability, etc.)

Support voice input using SpeechRecognition

Add Hindi/Punjabi language support

👨‍💻 Developed By

Tanmay Kumar
B.Tech Information Technology
Guru Nanak Dev Engineering College (GNE), Ludhiana
GitHub: https://github.com/tanmaykumar-info

📄 License

This project is licensed under the MIT License.
