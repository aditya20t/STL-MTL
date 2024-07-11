from multitask_data_collator import DataLoaderWithTaskname
import numpy as np
import torch
import transformers
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import json


def multitask_eval_fn(multitask_model, model_name, features_dict, batch_size=8):
    preds_dict = {}
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    multitask_model.to(device)  # Ensure the model is on the correct device

    for task_name in ["bias", "stereotype"]:
        true_list = []
        pred_list = []
        val_len = len(features_dict[task_name]["validation"])
        acc = 0.0
        for index in range(0, val_len, batch_size):
            batch = features_dict[task_name]["validation"][
                index: min(index + batch_size, val_len)
            ]["doc"]
            labels = features_dict[task_name]["validation"][
                index: min(index + batch_size, val_len)
            ]["target"]
            inputs = tokenizer(batch, max_length=512, padding=True, truncation=True)

            # Ensure all inputs are tensors and move them to the device
            inputs = {key: torch.tensor(val).to(device) if not isinstance(val, torch.Tensor) else val.to(device) for key, val in inputs.items()}
            labels = torch.tensor(labels).to(device)  # Move labels to device

            # Ensure inputs and labels are in the correct shape
            if len(inputs["input_ids"].shape) == 1:
                inputs = {key: val.unsqueeze(0) for key, val in inputs.items()}
            if len(labels.shape) == 0:
                labels = labels.unsqueeze(0)

            logits = multitask_model(task_name, **inputs)[0]

            predictions = torch.argmax(
                torch.softmax(logits, dim=1),
                dim=1,
            ).cpu().numpy()  # Move predictions to CPU and convert to numpy array
            true_list.extend(labels.cpu().numpy())  # Move labels to CPU and convert to numpy array
            pred_list.extend(predictions)
            acc += sum(predictions == labels.cpu().numpy())
        acc = acc / val_len
        print(f"Task name: {task_name}")

        print("---------------------------------Confusion Matrix------------------------------------")
        final_create_confusion_matrix = confusion_matrix(true_list, pred_list)
        final_confusion_matrix_df = pd.DataFrame(final_create_confusion_matrix)
        print(final_confusion_matrix_df)

        # Precision, Recall and F1 score calculation
        final_eval_metrics = classification_report(true_list, pred_list, output_dict=True)
        print(final_eval_metrics)

        # Convert the evaluation metrics to a DataFrame and save as CSV
        final_eval_metrics_df = pd.DataFrame(final_eval_metrics).transpose()
        final_eval_metrics_df = final_eval_metrics_df.iloc[:, :-1]
        csv_filename = f"./results/{task_name}_{model_name}.csv"
        final_eval_metrics_df.to_csv(csv_filename, index=True)

        print(f"Results saved to {csv_filename}")
        print("---------------------------------Evaluation Metrics------------------------------------")
        final_eval_metrics_df = pd.DataFrame(final_eval_metrics).transpose()
        final_eval_metrics_df = final_eval_metrics_df.iloc[:, :-1]
        print(final_eval_metrics_df)