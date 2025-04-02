from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/receive_recipe', methods=['POST'])
def receive_recipe():
    try:
        data = request.json  # Get JSON data from Express.js
        print("Received data:", data)  # Debugging
        return jsonify({"message": "Recipe received successfully", "data": data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)
