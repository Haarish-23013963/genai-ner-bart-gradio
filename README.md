## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
Build an easy-to-use application that can identify and highlight named entities (such as names, places, organizations) from user-input text, using a pre-trained language model and a simple interactive interface.

### DESIGN STEPS:

### STEP 1:
Set up access to the pre-trained NER model API by configuring authentication and endpoint URLs.

### STEP 2:
Create a function to send text input to the model API and process the model’s output to identify named entities.

### STEP 3:
Develop a Gradio interface for users to input text, display the recognized entities highlighted, and provide example inputs for testing.
### PROGRAM:
```
import os
import io
from IPython.display import Image, display, HTML
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']
API_URL = os.environ['HF_API_NER_BASE']

# Helper function
import requests, json

#Summarization endpoint
def get_completion(inputs, parameters=None,ENDPOINT_URL=os.environ['HF_API_SUMMARY_BASE']): 
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL, headers=headers,
                                data=json.dumps(data)
                               )
    return json.loads(response.content.decode("utf-8"))
import gradio as gr
def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        label = token["entity"]
        word = token["word"]

        clean_word = word.replace("##", "")

        if merged_tokens:
            prev = merged_tokens[-1]
            prev_label = prev["entity"]

            if label.endswith(prev_label.split("-")[-1]) and (label.startswith("I-") or label.startswith("B-")):
                prev["word"] += clean_word
                prev["end"] = token["end"]
                prev["score"] = (prev["score"] + token["score"]) / 2
                continue

            if word.startswith("##") and prev_label.startswith("B-"):
                prev["word"] += clean_word
                prev["end"] = token["end"]
                prev["score"] = (prev["score"] + token["score"]) / 2
                continue
            
            if prev["entity"].endswith("PER") and len(prev["word"]) == 1 and word.startswith("##"):
                prev["word"] += clean_word
                prev["end"] = token["end"]
                prev["score"] = (prev["score"] + token["score"]) / 2
                continue


        merged_tokens.append({
            "entity": label,
            "word": clean_word,
            "start": token["start"],
            "end": token["end"],
            "score": token["score"]
        })

    return merged_tokens


def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    merged_tokens = merge_tokens(output)
    return {"text": input, "entities": merged_tokens}

gr.close_all()
demo = gr.Interface(
    fn=ner,
    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
    outputs=[gr.HighlightedText(label="Text with entities")],
    title="NER with Fine-Tuned BART ",
    description="Highlights people, organizations, and locations in your text.",
    allow_flagging="never",
    examples=[
        ["My name is Haarish, I'm building DeepLearningAI and I live in Chennai"],
        ["Elon Musk founded SpaceX in the United States"]
    ]
)

demo.launch(share=True, server_port=int(os.environ.get("PORT4", 7860)))





```

### OUTPUT:
<img width="1183" height="646" alt="image" src="https://github.com/user-attachments/assets/d4cecf3c-3136-47d1-870c-3d760cdc6225" />


### RESULT:
A prototype application for Named Entity Recognition (NER) was successfully designed and developed by leveraging a fine-tuned BART model. The application was deployed using the Gradio framework, enabling easy user interaction and effective evaluation of the model’s performance.
