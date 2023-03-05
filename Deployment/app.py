from flask import Flask
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline
from flask import jsonify, request

app = Flask(__name__)
model = ORTModelForSequenceClassification.from_pretrained(
    "checkpoint-450", from_transformers=True
)
tokenizer = AutoTokenizer.from_pretrained("checkpoint-450")
onnx_classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=-1,
    truncation=True,
    max_length=510,
)
onnx_classifier("測試", top_k=6)


@app.route("/")
def hello_world():
    return "<p>HSK3 level predictor</p>"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = str(data["text"])
    if not text:
        return jsonify({"label": "LV1", "prob": "0.99"})
    results = onnx_classifier(text, top_k=6)
    label = results[0].get("label")
    hsk3 = {
        "LABEL_0": "LV1",
        "LABEL_1": "LV2",
        "LABEL_2": "LV3",
        "LABEL_3": "LV4",
        "LABEL_4": "LV5",
        "LABEL_5": "LV6",
    }
    prob = results[0].get("score")
    postdata = {
        "label": hsk3[label],
        "prob": f"{prob:.2f}",
    }
    return jsonify(postdata)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8030, debug=True)
