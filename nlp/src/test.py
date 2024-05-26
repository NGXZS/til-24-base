import json

def getAnswerStart(context:str, answer:str):
    return context.find(answer)

QUESTION_HEADING = "What is the heading?"
QUESTION_TOOL = "What is the tool?"
QUESTION_TARGET = "What is the target?"
    
with open('nlp.jsonl', 'r') as file:
    newFile = open('nlpNew.jsonl', 'a')
    for line in file:
        dictObject = json.loads(line)
        # process dictObject
        # add new keys
        correct_answer = dictObject['target']
        answer_start = getAnswerStart(context=dictObject['transcript'], answer=correct_answer)

        answerDict = {"text": [correct_answer], "answer_start": [answer_start]}
        dictObject['context'] = dictObject['transcript']
        dictObject['question'] = QUESTION_TARGET
        dictObject['answers'] = answerDict

        # delete old keys
        del dictObject['transcript']
        del dictObject['tool']
        del dictObject['heading']
        del dictObject['target']

        json.dump(dictObject, newFile)
        newFile.write('\n')

    newFile.close()
