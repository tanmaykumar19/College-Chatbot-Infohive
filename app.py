from flask import Flask, render_template, request, jsonify
from chat import get_response

app = Flask(__name__)

# Route to render the chatbot frontend
@app.route("/")
def index():
    return render_template("chatbot.html")

# Route to handle message predictions
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "Invalid input. 'message' key is required."}), 400

    user_message = data["message"]
    bot_response = get_response(user_message)

    return jsonify({"answer": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
