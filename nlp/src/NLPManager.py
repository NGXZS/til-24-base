from typing import Dict
import torch

# Load model directly
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# GOAL: from "transcript" key, GET the 3 other keys
# access "transcript key" in string form DONE
def convertStringToDict(str):
    dict = {}
    for item in str.split(', "'):
        key, value = item.split(":")

        # process key, value
        newKey = key.strip('{"')
        newValue = value.strip(' "}\n')

        # store in dict
        dict[newKey] = newValue

    # print(dict.keys())
    # print(dict.values())
    return dict

def printNewLine():
    print("*************************************")

class NLPManager:
    def __init__(self):
        # initialize the model here

        # Load pre-trained model and tokenizer (from https://huggingface.co/Intel/dynamic_tinybert)
        self.tokenizer = AutoTokenizer.from_pretrained("Intel/dynamic_tinybert")
        self.model = AutoModelForQuestionAnswering.from_pretrained("Intel/dynamic_tinybert")

    # output type is Dict[str, str]
    def qa(self, context: str) -> Dict[str, str]:
        ########## perform NLP question-answering ##########
        # Define questions
        QUESTION_HEADING = "What is the heading?"
        QUESTION_TOOL = "What is the tool?"
        QUESTION_TARGET = "What is the target?"

        # Encode questions to prepared for model
        inputs_heading = self.tokenizer.encode_plus(
            QUESTION_HEADING, context, return_tensors="pt", truncation=True
        )
        inputs_tool = self.tokenizer.encode_plus(
            QUESTION_TOOL, context, return_tensors="pt", truncation=True
        )
        inputs_target = self.tokenizer.encode_plus(
            QUESTION_TARGET, context, return_tensors="pt", truncation=True
        )

        # Get input IDs and attention mask
        input_ids_heading = inputs_heading["input_ids"]
        input_ids_tool = inputs_tool["input_ids"]
        input_ids_target = inputs_target["input_ids"]

        attention_mask_heading = inputs_heading["attention_mask"]
        attention_mask_tool = inputs_tool["attention_mask"]
        attention_mask_target = inputs_target["attention_mask"]

        # Perform question answering
        outputs_heading = self.model(input_ids_heading, attention_mask=attention_mask_heading)
        start_scores_heading = outputs_heading.start_logits
        end_scores_heading = outputs_heading.end_logits

        outputs_tool = self.model(input_ids_tool, attention_mask=attention_mask_tool)
        start_scores_tool = outputs_tool.start_logits
        end_scores_tool = outputs_tool.end_logits

        outputs_target = self.model(input_ids_target, attention_mask=attention_mask_target)
        start_scores_target = outputs_target.start_logits
        end_scores_target = outputs_target.end_logits

        # Find start and end positions of answer
        answer_start_heading = torch.argmax(start_scores_heading)
        answer_end_heading = torch.argmax(end_scores_heading)
        answer_heading = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(input_ids_heading[0][answer_start_heading:answer_end_heading])
        )

        answer_start_tool = torch.argmax(start_scores_tool)
        answer_end_tool = torch.argmax(end_scores_tool)
        answer_tool = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(input_ids_tool[0][answer_start_tool:answer_end_tool])
        )

        answer_start_target = torch.argmax(start_scores_target)
        answer_end_target = torch.argmax(end_scores_target)
        answer_target = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(input_ids_target[0][answer_start_target:answer_end_target])
        )

        return {"heading": answer_heading, "tool": answer_tool, "target": answer_target}
    
def test(): # training purposes
    NUM_OF_LINES = 5

    file = open("nlp.jsonl", "r")
    c = 0
    for line in file: # 3500 lines
        # dictObject = convertStringToDict(line)
        # context = dictObject["transcript"] # same transcipt for 3 keys
        
        # QA
        # model = NLPManager()
        # qa_dict = model.qa(context)
        # for key in qa_dict:
        #     print(key, ':', qa_dict[key], '|', dictObject[key]) # prints qa v actual value
        # printNewLine()

        # for first specified objects in nlp.jsonl only
        c += 1
        # if c == NUM_OF_LINES:
        #     break

    # file.close()
    

test()