from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline

model = ORTModelForSequenceClassification.from_pretrained(
    "checkpoint-450", from_transformers=True
)
tokenizer = AutoTokenizer.from_pretrained("checkpoint-450")
onnx_classifier = pipeline(
    "text-classification", model=model, tokenizer=tokenizer, device=-1
)
results = onnx_classifier("這是一篇簡單的文章。", top_k=6)
print(results)
