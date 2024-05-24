from typing import Dict

#GOAL: from "transcript" key, GET the 3 other keys
# access "transcript key" in string form
def convertStringToDict(str):
    dict = {}
    for item in str.split(', "'):  
        key, value = item.split(':')

        # process key, value
        newKey = key.strip('{"')
        newValue = value.strip(' "}\n')
        
        # store in dict
        dict[newKey] = newValue
        
    # print(dict.keys())  
    # print(dict.values())  
    return dict
    
class NLPManager:
    def __init__(self):
        # initialize the model here
        file = open("nlp.jsonl", "r")
        c = 0
        for line in file:
            dictObject = convertStringToDict(line)
        
            for key in dictObject.keys():
                print(key + "|" + dictObject[key])
            
            print("*************")
            # for first 3 objects only
            c += 1
            if (c == 3) :
                break
  
        file.close()
        pass

    # output type is Dict[str, str]
    def qa(self, context: str) -> Dict[str, str]:
        # perform NLP question-answering
        return {"heading": "", "tool": "", "target": ""}

NLPManager()