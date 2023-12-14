# encoding: utf-8

import json,glob,time,re
from tqdm import tqdm
import openai


def load_json(path):
    new_data=[]
    with open(path,'r',encoding='UTF-8') as load_f:
        data=load_f.readlines()
        for i in data:
            new_data.append(i)   
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

def extract(string, start, end):

    start_index = string.find(start)
    end_index = string.find(end)
    middle_string = string[start_index+len(start):end_index]
    return middle_string.replace(':', '').replace('\n', '').strip()



def process(seed):

    #generate previous dialogue
    input_seed = '[input]: ' + seed
    prompt='You need to generate a multi-turn dialogue between human and assistant in English. The dialogue should end with human saying [input].\
            The generated dialogue can be related to the topic of the [input] or just chatting. \
            Example：“<start_chat> <Human 1>:XXX <Assistant 1>：XXX <Human 2>：XXX <Assistant 2>：XXX <end_chat>”, where “XXX” refer to the actual content of the dialogue. \
                        The dialogue should follow the following plan：\
                        <start_chat> \
                        <Human 1> chat, do not mention [input] \
                        <Assistant 1> chat, do not mention [input]\
                        <Human 2> chat, do not mention [input] \
                        <Assistant 2> chat, do not mention [input]  \
                        <Human 3> chat, do not mention [input] \
                        <Assistant 3> chat, do not mention [input]  \
                        <Human 4> ' + seed + ' <end_chat>'
    
    
    dialogue = call_gpt([{"role": "user", "content":input_seed+prompt}])
    

    q1 = extract(dialogue, '<Human 1>', '<Assistant 1>')
    a1 = extract(dialogue, '<Assistant 1>', '<Human 2>')
    q2 = extract(dialogue, '<Human 2>', '<Assistant 2>')
    a2 = extract(dialogue, '<Assistant 2>', '<Human 3>')
    q3 = extract(dialogue, '<Human 3>', '<Assistant 3>')
    a3 = extract(dialogue, '<Assistant 3>', '<Human 4>')
    q4 = extract(dialogue, '<Human 4>', '<end_chat>')


    #generate a similar question with the seed 
    
    prompt_sim_question = 'Generate a similar question based on the Input. The two question should be the same type and topic but with different details.'
    in_context_sim_question =   "Input: Is there anything I can eat for a breakfast that doesn't include eggs, yet includes protein, and has roughly 700-1000 calories? \n \
                                Output: Is there anything I can eat for a lunch that doesn't include meat, yet includes protein, and has at least 1000 calories? \n \
                                Input: Brainstorm a list of possible New Year's resolutions. \n \
                                Output: Generate a list of possible Summer Vacation's resolutions. \n \
                                Input: Create a fun math question for children. \n \
                                Output: Create a physics question for children. \n \
                                Input: Write a program to compute the sum of integers from k to n. \n \
                                Output: Write a program to compute the sum of even from 0 to 10. \n \
                                Input: I am interested in playing Table tennis. \n \
                                Output: I start to find playing Football is fun. \n \
                                Input: Let's talk about the famous singer, Taylor Swift. \n \
                                Output: Let's talk about Michael Jackson. \n \
                                "  
                                
    input_sim_question = "Input: " + seed + '\n Output: '
    
    sim_question = call_gpt([{"role": "user", "content":prompt_sim_question+in_context_sim_question+input_sim_question}])
    #get the first answer of question
    

    org_answer = call_gpt([{"role": "user", "content":seed}])  
    sim_answer = call_gpt([{"role": "user", "content":sim_question}])
    
    
    #get the feedback
    input_feedback = '[Question 1]: ' + seed + '\n' + \
                     '[Answer 1]: ' + org_answer + '\n' + \
                     '[Question 2]: ' + sim_question + '\n' + \
                     '[Answer 2]: ' + sim_answer + '\n' 
                     
    prompt_feedback='Given two question-answer pairs, you need to generate one suggestion or critique that improves the quality of the answers for both QA pairs. \
                     The suggestion should be generic for both of them and do not focus on very specific content. Only generate the suggestion, do not provide the answers improved by the suggestion.'
        
    feedback = call_gpt([{"role": "user", "content":input_feedback + prompt_feedback}])
    
    #get new answer
    

    org_answer_refined = call_gpt([{"role": "user", "content":seed}, {"role": "assistant", "content":org_answer}, {"role": "user", "content":feedback+' Please generate the answer again.'}])
    sim_answer_refined = call_gpt([{"role": "user", "content":sim_question}, {"role": "assistant", "content":sim_answer}, {"role": "user", "content":feedback+' Please generate the answer again.'}])
    
    source_dia=[{"role": "user", "content":q1},{"role": "assistant", "content":a1},
                {"role": "user", "content":q2},{"role": "assistant", "content":a2},
                {"role": "user", "content":q3},{"role": "assistant", "content":a3},
                {"role": "user", "content":seed},{"role": "assistant", "content":org_answer},
                {"role": "user", "content":feedback},{"role": "assistant", "content":org_answer_refined},
                ]

    reference_dia=[{"role": "user", "content":sim_question},{"role": "assistant", "content":sim_answer_refined}]
    
    other_inf = {'seed':seed, 'feedback':feedback, 'org_answer':org_answer, 'org_answer_refined':org_answer_refined, 'sim_answer':sim_answer, 'sim_answer_refined':sim_answer_refined}
    

    return source_dia, reference_dia, other_inf
    
    
    
    
if __name__ == "__main__":
    file_path='seed.json'
    
    data=load_json(file_path)
    
    source_all=[]
    reference_all=[]
    import multiprocessing as mp
    print("start of process...")
    results=[]
    with mp.Pool(10, initargs=()) as pool:
        with tqdm(pool.imap_unordered(process, data, chunksize=1), total=len(data),desc = "building datasets...") as pbar:
            for res in pbar:
                source_dia, reference_dia, other_inf = res
                results.append((source_dia, reference_dia, other_inf))
        
    for i in range(len(results)):
        source_dia, reference_dia, other_inf = results[i]
        

        file_name='data2/refine_test_history_'+str(i+48)+'.txt'
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(source_dia, f,ensure_ascii=False)     
            f.close()    
            
        file_name='data2/refine_test_reference_'+str(i+48)+'.txt'
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(reference_dia, f,ensure_ascii=False)     
            f.close()       
            
        file_name='data2/refine_test_information_'+str(i+48)+'.txt'
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(other_inf, f,ensure_ascii=False)     
            f.close()   
