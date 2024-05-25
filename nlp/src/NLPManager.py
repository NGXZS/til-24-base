from typing import Dict
import torch

# Load model directly
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DefaultDataCollator, TrainingArguments, Trainer
from datasets import load_dataset

class NLPManager:
    def __init__(self):
        # initialize the model here

        # Load pre-trained model and tokenizer (from https://huggingface.co/Intel/dynamic_tinybert)
        self.tokenizer = AutoTokenizer.from_pretrained("Intel/dynamic_tinybert")
        self.model = AutoModelForQuestionAnswering.from_pretrained("Intel/dynamic_tinybert")
        # print(self.model.config.max_position_embeddings) # sees max length for context = 512

    def qa(self, context: str) -> Dict[str, str]:
        ########## perform NLP question-answering ##########
        # Define questions
        QUESTION_HEADING = "What is the heading?"
        QUESTION_TOOL = "What is the tool?"
        QUESTION_TARGET = "What is the target?"
        NOT_FILLED = "not filled"
        question_dict = {"heading": QUESTION_HEADING, "tool": QUESTION_TOOL, "target": QUESTION_TARGET}
        answer_dict = {"heading": NOT_FILLED, "tool": NOT_FILLED, "target": NOT_FILLED}

        for key in question_dict : # repeat *3 for each question
            question = question_dict[key]
            # Encode questions to prepared for model
            inputs = self.tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True, max_length=512)

            # Get input IDs and attention mask
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Perform question answering
            outputs = self.model(input_ids, attention_mask=attention_mask)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits

            # Find start and end positions of answer
            answer_start = torch.argmax(start_scores)
            answer_end = torch.argmax(end_scores) + 1
            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end])
            )
            answer_dict[key] = answer

        return answer_dict

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

def getContextAndAnswers(nlp, line_num): # 1 to 3500
    NOT_FILLED = "not filled"
    context = NOT_FILLED
    actual_answers = {"heading": NOT_FILLED, "tool": NOT_FILLED, "target": NOT_FILLED}
    c = 0

    file = open("nlp.jsonl", "r")
    for line in file: # 3500 lines
        c += 1
        if c == line_num:
            dictObject = convertStringToDict(line)

            context = dictObject["transcript"] # same transcipt for 3 keys
            for key in actual_answers:
                actual_answers[key] = dictObject[key]
            break

    file.close()
    return context, actual_answers

def preprocess_function(examples):
    questions = [q.strip() for q in examples['question']]
    inputs = tokenizer(
        questions,
        examples['context'],
        max_length=128,
        truncation='only_second',
        return_offsets_mapping=True, 
        padding='max_length',
    )

    offset_mapping = inputs.pop('offset_mapping')
    answers = examples['answers']
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer['answer_start'][0]
        end_char = answer['answer_start'][0] + len(answer['text'][0])
        sequence_ids = inputs.sequence_ids(i)

        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs['start_positions'] = start_positions
    inputs['end_positions'] = end_positions
    return inputs

nlp = NLPManager()
tokenizer = nlp.tokenizer
model = nlp.model

line_num = 4
context, truth_dict = getContextAndAnswers(nlp, line_num) # for 1 line num
answer_dict = nlp.qa(context)

# compare truth v qa 
print("BEFORE")
for key in answer_dict:
    print(key, ':', answer_dict[key], '|', truth_dict[key])
printNewLine()

## training using own data
dataset_train, dataset_test = load_dataset('json', data_files='nlp.jsonl', split=['train[:3400]', 'train[3400:]'])
print(dataset_train)
print(dataset_test)

# squad = load_dataset('squad', split='train[:100]')
# squad = squad.train_test_split(test_size=0.2) # returns Dict w train, test sets

# tokenized_squad = squad.map(preprocess_function, batched=True)
# data_collator = DefaultDataCollator()

# training_args = TrainingArguments(
#     output_dir='qa_model',
#     evaluation_strategy='epoch',
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=1,
#     weight_decay=0.01,
#     push_to_hub=False,
#     )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_squad['train'],
#     eval_dataset=tokenized_squad['test'],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
# )

# trainer.train()

# model.eval() # eval mode
# with torch.no_grad():
#     new_answer_dict = nlp.qa(context)

#     # compare truth v qa 
#     print("AFTER")
#     for key in new_answer_dict:
#         print(key, ':', new_answer_dict[key], '|', answer_dict[key])
#     printNewLine()

