import transformers
import torch


def save_model(model_name, multitask_model):
    print("hello I AM HERE-----------------------------------------")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    for task_name in ["bias", "stereotype"]:
        multitask_model.taskmodels_dict[task_name].config.to_json_file(
            f"./{task_name}_model/config.json"
        )
        torch.save(
            multitask_model.taskmodels_dict[task_name].state_dict(),
            f"./{task_name}_model/pytorch_model.bin",
        )
        tokenizer.save_pretrained(f"./{task_name}_model/", safe_serialization=False)