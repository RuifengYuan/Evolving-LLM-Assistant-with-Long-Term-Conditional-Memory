# -*- coding: utf-8 -*-

import glob, json, openai,  time


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


def covert(d):
    
    role=d['id']
    content=d['text']

    if role == 'Speaker 1':
        role_new = 'user'
    if role == 'Speaker 2':
        role_new = 'assistant'
        
    return {"role": role_new, "content": content}
    
prompt_task='Transfer the input sentence into a question and answer pair. The generated question and answer should separated with "[sep]" \n'
#"I own a Jeep.", "I enjoy exercising at the gym.", "I have a marketing job.", "I don't eat meat.", "I am from New England.", "I like warm pants in winter."]
prompt_context="Input: I own a Jeep. \n \
                Output: What kind of car do I own? [sep] You own a Jeep. \
                Input: I enjoy exercising at the gym. \n \
                Output: Where do I excercise? [sep] You enjoy exercising at the gym. \
                Input: I have a marketing job. \n \
                Output: What job do I have? [sep] You have a marketing job. \
                Input: I don't eat meat. \n \
                Output: What kind of food I do not eat? [sep] You don't eat meat. \
                Input: I am from New England. \n \
                Output: What am I from? [sep] You are from New England. \
                "



data_s1=load_json('data_msc/test.txt')

data_s2=load_json('data_msc/test2.txt')

data_s3=load_json('data_msc/test3.txt')

data_s4=load_json('data_msc/test4.txt')


for i in range(100,500):
    
    s1=data_s1[i]
    s2=data_s2[i]
    s3=data_s3[i]    
    s4=data_s4[i] 
    
    p1 = s1["personas"][0]
    p2 = s1["personas"][1]

    
    d1 = [covert(x) for x in s1['dialog'] if x['id'] in ['Speaker 1', 'Speaker 2']]
    d2 = [covert(x) for x in s2['dialog'] if x['id'] in ['Speaker 1', 'Speaker 2']]
    d3 = [covert(x) for x in s3['dialog'] if x['id'] in ['Speaker 1', 'Speaker 2']]
    d4 = [covert(x) for x in s4['dialog'] if x['id'] in ['Speaker 1', 'Speaker 2']]
    
    ref_QA=[]
    for a in p1:
        try:
            x='Input: ' + a + '\n Output:'
            user_input = [{"role": "user", "content": prompt_task+prompt_context+x}]
            GPT_respond = call_gpt(user_input)
    
            if '[sep]' in GPT_respond:
                q, ref = GPT_respond.split('[sep]')
                print(q, ref)
            else:
                continue
            
            ref_QA+=[{"role": "user", "content": q.strip()}, {"role": "assistant", "content": ref.strip()}]
    
        except:
            continue

    history_d = [d1,d2,d3,d4]
    file_name='data_msc_new_plus/msc_test_history_'+str(i)+'.txt'
    with open(file_name, 'w', encoding='utf-8') as f:
        
        for his in history_d:
            json.dump(his, f,ensure_ascii=False) 
            f.write('\n')
        
        f.close()       


    file_name='data_msc_new_plus/msc_test_reference_'+str(i)+'.txt'
    with open(file_name, 'w', encoding='utf-8') as f:

        json.dump(ref_QA, f,ensure_ascii=False) 

        
        f.close()       

    
    
    