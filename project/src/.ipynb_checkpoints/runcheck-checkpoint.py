import run
import json
from run import process_datas
def has_complete_answer(questions):
    valid_answers = {'A', 'B', 'C', 'D', 'E'}

    for question in questions:
        if 'answer' not in question or question['answer'] not in valid_answers:
            return False
    return True

def filter_problems(data):
    result = []
    problem_set = set()

    for item in data:
        # print('处理的item' ,item)
        problem = item['problem']
        if problem in problem_set:
            # 找到已存在的字典
            for existing_item in result:
                if existing_item['problem'] == problem:
                    # 如果当前字典有完整答案，替换已存在的字典
                    if has_complete_answer(item['questions']):
                        existing_item['questions'] = item['questions']
                        existing_item['id'] = item['id']
                    break
        else:
            # 如果当前字典有完整答案，添加到结果列表
            if has_complete_answer(item['questions']):
                result.append(item)
                problem_set.add(problem)

    return result


def find_missing_ids(dict_list):
    # 提取所有序号
    extracted_ids = {int(d['id'][-3:]) for d in dict_list}
    import json
    data = []
    with open("round1_test_data.jsonl", encoding="utf-8") as reader:
        for line in reader:
            sample = json.loads(line)
            data.append(sample)

    # 创建0-500的序号集合
    all_ids = set(range(len(data)))

    # 找出缺失的序号
    missing_ids = all_ids - extracted_ids

    return sorted(missing_ids)

def process_and_write_unique(data, ofn):
    unique_data = {}

    for item in data:
        item_id = item['id']
        if item_id not in unique_data:
            unique_data[item_id] = item

    with open(ofn, 'w', encoding='utf-8') as writer:
        for item in unique_data.values():
            writer.write(json.dumps(item, ensure_ascii=False))
            writer.write('\n')
            
def processfile(input_file, output_file):
    # 示例process函数，它会读取input_file，执行一些处理，然后写入output_file
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        data = [json.loads(line) for line in infile]
        
        # 示例处理：假设每个数据项都包含一个'id'字段
        for item in data:
            # 模拟处理，假设将 id 字段值增加1
            item['id'] = item.get('id', 0) + 1
            
            # 输出处理后的项
            outfile.write(json.dumps(item) + '\n')
    
    # 假设 missing_ids 是通过一些逻辑计算出来的
    missing_ids = [item['id'] for item in data if item['id'] % 2 == 0]  # 示例逻辑
    
    return missing_ids            

def getsorteddata(ifn):
    import json
    data  = []


    with open(ifn, encoding="utf-8") as reader:
        for line in reader:
            sample = json.loads(line)
            data.append(sample)

    # 找出缺失的序号
    return_list = filter_problems(data)
    sorted_data = sorted(return_list, key=lambda x: int(str(x['id'])[-3:]))
    return sorted_data


def process_file(ifn,ofn):
    sorted_data = getsorteddata(ifn)
    print('sorted_data长度',len(sorted_data))
    # 示例字典列表
    dict_list = sorted_data
    missing_ids = find_missing_ids(dict_list)
    print("缺失的序号:", missing_ids)
    print("缺失的序号长度：", len(missing_ids))            

    # 从 round1_test_data.jsonl 中提取与 missing_ids 对应的条目
    missing_questions = []
    with open('round1_test_data.jsonl', 'r', encoding='utf-8') as reader:
        for line in reader:
            question = json.loads(line)
            question_id_suffix = int(question['id'][-3:])
            if question_id_suffix in missing_ids:
                missing_questions.append(question)
    print("重建问题长度合计：", len(missing_ids))        
    # 通过 process_datas 形成答案列表
    processed_answers = process_datas(missing_questions,run.MODEL_NAME)

    print("处理的问题长度：",len(processed_answers))

    # 将 processed_answers 转换为以 id 为键的字典
    processed_answers_dict = {entry['id']: entry for entry in processed_answers}

    # 初始化 merged_data 列表
    merged_data = []

    for entry in sorted_data:
        entry_id = entry['id']
        if entry_id in processed_answers_dict:
            # 如果在 processed_answers 中找到相同的 id 条目，用它替换 sorted_data 中的条目
            merged_data.append(processed_answers_dict[entry_id])
        else:
            # 否则保留原有的条目
            merged_data.append(entry)

    # 检查 processed_answers_dict 中的条目是否在 sorted_data 中
    for entry_id, processed_entry in processed_answers_dict.items():
        if not any(entry['id'] == entry_id for entry in sorted_data):
            merged_data.append(processed_entry)


    print("总长度：", len(merged_data))
    # 输出合并后的数据到文件
    print(f"=====================================================输出到文件{ofn}=============================================")    
    process_and_write_unique(merged_data,ofn)



    
    
def main(ifn,result):
    sorted_data = getsorteddata(ifn)
    print('sorted_data长度',len(sorted_data))
    # 示例字典列表
    dict_list = sorted_data
    missing_ids = find_missing_ids(dict_list)
    print("缺失的序号:", missing_ids)
    print("缺失的序号长度：", len(missing_ids))    
    

    count= 0
    ofn = f'{result}.jsonl'
    while len(missing_ids) != 0:
        count += 1
        print(f"=====================================================第{count}次迭代=============================================")
        process_file(ifn,ofn)
        
        

        # 更新输入文件名
        ifn = ofn
        
        # 处理新的文件
        sorted_data = getsorteddata(ifn)
        print(f"=====================================================第{count}次迭代结果=============================================")
        print('sorted_data长度',len(sorted_data))
        # 更新缺失的序号
        dict_list = sorted_data
        missing_ids = find_missing_ids(dict_list)
        print("缺失的序号:", missing_ids)
        print("缺失的序号长度：", len(missing_ids))
        
        # 每次迭代更新输出文件名        
        ofn = f"{result}_{count}.jsonl"        

    '''
    with open('upload.jsonl') as reader:
        for id,line in enumerate(reader):
            if(id in missing_ids):
                sample = json.loads(line)
                for question in sample['questions']:
                    question['answer'] = 'A'
                sorted_data.append(sample)
    sorted_data = sorted(sorted_data, key=lambda x: int(str(x['id'])[-3:]))
    '''
    
import datetime
filename = "../submit/submit_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

if __name__ == '__main__':
    main('upload.jsonl',filename)
