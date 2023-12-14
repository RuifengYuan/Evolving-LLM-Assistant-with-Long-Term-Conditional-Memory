# encoding: utf-8

import json,glob,time,re,copy
from tqdm import tqdm

def load_json(path):
    new_data=[]
    with open(path,'r',encoding='UTF-8') as load_f:
        data=load_f.readlines()
        for i in data:
            new_data.append(json.loads(i))   
            
        load_f.close()
    return new_data


def call_gpt(content):
    import requests
    url = ""
    payload = json.dumps({
    "messages": content
    })

    headers = {
    'api-key': '',
    'Content-Type': 'application/json'
    }

    c=0
    tag=0
    dialogue=''
    while (tag == 0):
        try:
            completion = requests.request("POST", url, headers=headers, data=payload, timeout=100).json()
            dialogue=completion["choices"][0]["message"]["content"]
            tag = 1
        except Exception as inst:
            print(c, inst)
            time.sleep(1)
            c+=1
        if c==5:
            tag = 1
    time.sleep(1)   

    return dialogue


def one_dialogue(history):
    
    history_str=''
    for i in history:
        history_str+=i["role"]+': '+i["content"]+'/n'
        
    prompt='[Dialogue] /n'+history_str
    
    

    
    system_assistant = 'The input is a four-turn [Dialogue] between a human user and a AI assistant. The background situation is that after the [Dialogue] is finished, some time later, the user want to continue this dialogue with the assistant. \
                        Following the previous dialogue, you are reuqired to extend the dialogue to one more turn (user starts the first and then assistant make the respond). \
                        The new dialogue turn shall follow closely with the [Dialogue]. \
                        The new dialogue turn must to utilize or mention the information from the [Dialogue]. \
                        In other words, the new dialogue turn can not be independent from the [Dialogue]. The user input can not be respond without the [Dialogue] or the respond is different with or without the [Dialogue]. \
                        But do not use pronoun to represent any information in [Dialogue] for the new dialogue. As for the user in the new dialogue, do not apologize, do not compliment, do not thanks, just directly give your question or instruction.\
                        The input from the user need to start with "<user>" and the respond from the assistant need to start with "<assistant>", example: "<user> XXX <assistant> XXX", where "XXX" refer to the dialogue content.'
    #                   This can be achieved but not limited to the following ways: (1) use information in [Dialogue] as example in the respond; (2) focus on an question extended from [Dialogue] in user input;\                    
    prompt+=system_assistant
    respond=call_gpt([{"role": "user", "content" : prompt}])  
    
    return respond



    
    

def extract(string, start, end):

    start_index = string.find(start)
    end_index = string.find(end)
    middle_string = string[start_index+len(start):end_index]
    return middle_string.replace(':', '').replace('\n', '').strip(), string[end_index:]



def process(i):
    dialogue = i['dialogue']+'<end>'
    reference = i['reference']
    
    c = dialogue.count('user')
    
    history = []
    for j in range(c-1):
    
        q,dialogue = extract(dialogue, '<user>', '<assistant>')
        a,dialogue = extract(dialogue, '<assistant>', '<user>')

        history+=({"role": "user", "content": q.strip()}, {"role": "assistant", "content": a.strip()})

    q,dialogue = extract(dialogue, '<user>', '<assistant>')
    a,dialogue = extract(dialogue, '<assistant>', '<end>')
    history+=({"role": "user", "content": q.strip()}, {"role": "assistant", "content": a.strip()}) 
    
    new_dialogue = one_dialogue(history)
    
    if '<assistant>' not in new_dialogue:
        print ('not generating the target format')
        return [0, 0]
    
    new_user = new_dialogue.split('<assistant>')[0].replace('<user>', '').strip()
    
    new_respond = new_dialogue.split('<assistant>')[1].strip() 
    
    reference = [{"role": "user", "content" : new_user}, {"role": "assistant", "content" : new_respond}]
    
    return [history, reference]
    

file_path='refgpt-code-ds-en.jsonl'

data=load_json(file_path)


long_data=[]

for i in tqdm(data[1000:2000]):
    dialogue = i['dialogue']+'<end>'
    reference = i['reference']
    
    c = dialogue.count('user')

    if c == 4:
        long_data.append(i)
        
    if len(long_data) > 15:
        break
print('long_data number', len(long_data))


num=200

for i in tqdm(long_data):
    
    dialogue = i['dialogue']+'<end>'
    reference = i['reference']
    
    c = dialogue.count('user')
    
    a_list=[]
    q_list=[]
    history = []
    for j in range(c-1):
    
        q,dialogue = extract(dialogue, '<user>', '<assistant>')
        a,dialogue = extract(dialogue, '<assistant>', '<user>')

        history+=({"role": "user", "content": q.strip()}, {"role": "assistant", "content": a.strip()})

    q,dialogue = extract(dialogue, '<user>', '<assistant>')
    a,dialogue = extract(dialogue, '<assistant>', '<end>')
    history+=({"role": "user", "content": q.strip()}, {"role": "assistant", "content": a.strip()})   
    if len(history) != 8:
        continue
    
    new_dialogue = one_dialogue(history)

    if '<assistant>' not in new_dialogue:
        continue
    
    new_user = new_dialogue.split('<assistant>')[0].replace('<user>', '').strip()
    
    new_respond = new_dialogue.split('<assistant>')[1].strip() 
    
    reference = [{"role": "user", "content" : new_user}, {"role": "assistant", "content" : new_respond}]

    file_name='data_code_new/code_test_history_'+str(num)+'.txt'
    with open(file_name, 'w', encoding='utf-8') as f:
        
        json.dump(history, f,ensure_ascii=False) 
        
        f.close()       


    file_name='data_code_new/code_test_reference_'+str(num)+'.txt'
    with open(file_name, 'w', encoding='utf-8') as f:

        json.dump(reference, f,ensure_ascii=False) 
        
        f.close()  
    num+=1
    

    

    


    