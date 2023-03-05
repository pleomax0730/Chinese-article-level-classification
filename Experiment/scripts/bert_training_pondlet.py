import evaluate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv(".env")

from transformers.integrations import MLflowCallback

dataset = load_dataset(
    "csv",
    data_files={
        "train": "datasets/pondlet_STB_pondlet_20220803_content_data_train.csv",
        "test": "datasets/pondlet_STB_pondlet_20220803_content_data_test.csv",
    },
)


model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def preprocess_function(examples):
    return tokenizer(examples["content"], truncation=True)


tokenized_dataset = dataset.map(preprocess_function, batched=True)

label_list = tokenized_dataset["train"].unique("labels")
label_list.sort()
num_labels = len(label_list)
label_to_id = {}
id_to_label = {}
for i, label in enumerate(label_list):
    label_to_id[label] = i
    id_to_label[i] = label


def preprocess_labels(examples):
    if label_to_id is not None and "labels" in examples:
        examples["labels"] = [label_to_id[l] for l in examples["labels"]]
    return examples


tokenized_dataset = tokenized_dataset.map(preprocess_labels, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels
)


accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

cf_arrays = []


def compute_metrics(eval_pred: EvalPrediction):
    preds = np.argmax(eval_pred.predictions, axis=1)
    results = accuracy_metric.compute(references=eval_pred.label_ids, predictions=preds)
    f1_results = f1_metric.compute(
        references=eval_pred.label_ids, predictions=preds, average="weighted"
    )
    results.update(f1_results)
    array = confusion_matrix(eval_pred.label_ids, preds)
    cf_arrays.append(array)

    report = classification_report(
        eval_pred.label_ids,
        preds,
        target_names=["Lv.0", "Lv.1", "Lv.2", "Lv.3", "Lv.4", "Lv.5"],
    )
    with open("report.txt", "w", encoding="UTF-8") as f:
        f.write(report)

    return results

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    num_train_epochs=30,
    weight_decay=0.01,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
    
)

trainer.train()

trainer.evaluate()

with open("report.txt") as f:
    print(f.read())


for array in cf_arrays:
    df_cm = (
        pd.DataFrame(
            array,
            index=["Lv.0", "Lv.1", "Lv.2", "Lv.3", "Lv.4", "Lv.5"],
            columns=["Lv.0", "Lv.1", "Lv.2", "Lv.3", "Lv.4", "Lv.5"],
        ),
    )

    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm[0], annot=True, cmap="Blues_r")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.show()
