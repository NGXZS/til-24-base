from typing import Dict
import torch
import json
import linecache 

# Load model directly
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DefaultDataCollator, TrainingArguments, Trainer
from datasets import load_dataset

dataFilePath = 'C:/Users/sean.ng/Downloads/til-24-base/nlp/nlpNew.jsonl'
headingFilePath = 'C:/Users/sean.ng/Downloads/til-24-base/nlp/src/heading.jsonl'
toolFilePath = 'C:/Users/sean.ng/Downloads/til-24-base/nlp/src/tool.jsonl'
targetFilePath = 'C:/Users/sean.ng/Downloads/til-24-base/nlp/src/target.jsonl'
filePathList = [headingFilePath, toolFilePath, targetFilePath, dataFilePath]

class NLPManager2:
    def __init__(self):
        # initialize the model here
        self.model = torch.load('savedModel.pth')
        return
    
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

        return answer_dict # returns {"heading": str, "target": str, "tool": str}
    
def clearAllJSON():
    for filePath in filePathList:
        open(filePath, 'w').close()
    return

def printNewLine():
    print("*************************************")

def getAnswerStart(context:str, answer:str):
    return context.find(answer)

def createNewJSON():    # https://www.datacamp.com/tutorial/json-data-python
    QUESTION_HEADING = "What is the heading?"
    QUESTION_TOOL = "What is the tool?"
    QUESTION_TARGET = "What is the target?"
    questionList= [QUESTION_HEADING, QUESTION_TOOL, QUESTION_TARGET]
    keyList = ['heading', 'tool', 'target']

    with open('nlp.jsonl', 'r') as file:
        newFile = open('nlpNew.jsonl', 'a')
        # c = 0 # cap on # lines (testing)
        for line in file:
            dictObject = json.loads(line)
            context=dictObject['transcript']
            answerDict = {"text": [], "answer_start": []}

            # process dictObject
            for question in questionList:
                i = questionList.index(question)
                current_key = keyList[i]

                correct_answer = dictObject[current_key]
                answer_start = getAnswerStart(context=context, answer=correct_answer)

                answerDict["text"].append(correct_answer)
                answerDict["answer_start"].append(answer_start)
                
                # delete old keys
                del dictObject[current_key]

            # add common key
            dictObject['context'] = context
            dictObject['question'] = questionList
            dictObject['answers'] = answerDict
            # delete common key
            del dictObject['transcript']
            # update in newFile
            json.dump(dictObject, newFile)
            newFile.write('\n')

            # c += 1
            # if (c == 10):
            #     break

        newFile.close()
    return

def createQuestionJSON(dataFilePath, targetFilePath, question, questionIndex):
    
    with open(dataFilePath, 'r') as file:
        newFile = open(targetFilePath, 'a')
        # c = 0 # cap on # lines (testing)

        for line in file:
            dictObject = json.loads(line)

            # process dictObject
            dictObject['question'] = question
            dictObject['answers']['text'] = [dictObject['answers']['text'][questionIndex]]
            dictObject['answers']['answer_start'] = [dictObject['answers']['answer_start'][questionIndex]]
                
            # update in newFile
            json.dump(dictObject, newFile)
            newFile.write('\n')

            # c += 1
            # if (c == 10):
            #     break

        newFile.close()
    return

def preprocess_function(examples):
    questions = [q.strip() for q in examples['question']]
    inputs = tokenizer(
        questions,
        examples['context'],
        max_length=512, # change?
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

# Reads a line and returns a dictionary object
def getJSONFor1Line(dataFilePath, line_num): 
    # get line_num
    line = linecache.getline(filename=dataFilePath, lineno=line_num)
    
    #convert line to dict
    dictObject = json.loads(line)
    linecache.clearcache()
        
    return dictObject


nlp = NLPManager()
tokenizer = nlp.tokenizer
model = nlp.model
## reset jsonl files
clearAllJSON()
print("here")
## creates main data.jsonl
createNewJSON() 
print("created file")    

TOTAL_LINES = 3500

answerObject = {} # ground truth
qa_dict = {} # pre-trained model results {i: {heading: '', tool: '', target: ''}
new_qa_dict = {} # trained model results

for line in range(1, TOTAL_LINES):
    answerObject[line] = getJSONFor1Line(dataFilePath, line) # for 1 line num
    context = answerObject[line]['context']
    answer = answerObject[line]['answers']['text'][0] # for target only

    qa_dict[line] = nlp.qa(context)

## create 3 JSON for each question
QUESTION_HEADING = "What is the heading?"
QUESTION_TOOL = "What is the tool?"
QUESTION_TARGET = "What is the target?"
questionList = [QUESTION_HEADING, QUESTION_TOOL, QUESTION_TARGET]

for question in questionList:
    i = questionList.index(question)
    specificDataFilePath = filePathList[i]
    createQuestionJSON(dataFilePath, specificDataFilePath, question, i)

    ## training using own data
    dataset = load_dataset('json', data_files=specificDataFilePath, split='train[:100]')
    dataset = dataset.train_test_split(test_size=0.2)

    # ## training with squad dataset
    # # squad = load_dataset('squad', split='train[:100]')
    # # squad = squad.train_test_split(test_size=0.2) # returns Dict w train, test sets

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DefaultDataCollator()

    training_args = TrainingArguments(
        output_dir='qa_model',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        push_to_hub=False,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

model.eval() # eval mode
with torch.no_grad():
    for line in range(1, TOTAL_LINES):
        context = answerObject[line]['context']

        new_qa_dict[line] = nlp.qa(context) # trained model # {line: {heading: '', tool: '', target: ''}

        # compare truth v qa 
        print(f"(BEFORE v AFTER v ACTUAL) line {line}")
        i = 0
        for key in new_qa_dict[line]:
            answer = answerObject[line]['answers']['text'][i] # actual answer (for target only)
            print(key, qa_dict[line][key], new_qa_dict[line][key], answer, sep=' | ')
            i = i + 1 if (i < 2) else 0
        printNewLine()
        if (line == 10) :
            break

torch.save(model, 'savedModel.pth')