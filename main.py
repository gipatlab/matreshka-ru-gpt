from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/message", methods=["POST"])
def message():
  try:
    message = request.get_json()["message"]
  except:
    return jsonify({'error': "Parameter message is required."})


  return jsonify({'message': "res"})

def main():
  app.run(host="0.0.0.0", port=8081, debug=True, threaded=True)


if __name__ == "__main__":
  main()

