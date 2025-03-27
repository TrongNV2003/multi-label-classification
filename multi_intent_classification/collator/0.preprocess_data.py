import pandas as pd
import json
import re

def process_raw_data_excel_to_json():
    """
    Excel lấy từ db Dify, chuyển thành raw json
    """
    df = pd.read_excel('multi_intent_classification/dify_dataset/New_Query_2025_03_25_1.xlsx')

    columns = ['id', 'app_id', 'workflow_run_id', 'conversation_id', 'from_account_id', 
               'created_at', 'query', 'answer', 'workflow_inputs', 'workflow_outputs']
    df_selected = df[columns]

    def process_workflow_inputs(inputs_str):
        try:
            if pd.isna(inputs_str) or not inputs_str:
                return {'text': []}
            
            cleaned_str = re.sub(r"('\s*:\s*[^']*?)('[^']*?')", r'\1, \2', inputs_str)
            cleaned_str = cleaned_str.replace("'", '"')
            
            inputs_dict = json.loads(cleaned_str)
            
            text_value = inputs_dict['text']
            text_array = json.loads(text_value)
            
            return {'text': text_array}
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {inputs_str} - {str(e)}")
            try:
                match = re.search(r'\["(.*?)"\]', inputs_str)
                if match:
                    items = [item.strip() for item in match.group(1).split('", "')]
                    return {'text': items}
                return {'text': [inputs_str]}
            except Exception:
                return {'text': [inputs_str]}
        except Exception as e:
            print(f"Unexpected error: {inputs_str} - {str(e)}")
            return {'text': [inputs_str]}

    def process_workflow_outputs(outputs_str):
        try:
            if pd.isna(outputs_str) or not outputs_str:
                return {'intent': []}
            
            cleaned_str = outputs_str.replace("'", '"')
            cleaned_str = re.sub(r'\[(.*?)\]', lambda m: '["' + '", "'.join(
                item.strip() for item in m.group(1).split(',')) + '"]', cleaned_str)
            return json.loads(cleaned_str)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {outputs_str} - {str(e)}")
            try:
                match = re.search(r'\[(.*?)\]', outputs_str)
                if match:
                    intents = [item.strip().strip("'\"") for item in match.group(1).split(',')]
                    return {'intent': intents}
                return {'intent': [outputs_str]}
            except Exception:
                return {'intent': [outputs_str]}
        except Exception as e:
            print(f"Unexpected error: {outputs_str} - {str(e)}")
            return {'intent': [outputs_str]}

    # Áp dụng xử lý với .loc
    df_selected.loc[:, 'workflow_inputs'] = df_selected['workflow_inputs'].apply(process_workflow_inputs)
    df_selected.loc[:, 'workflow_outputs'] = df_selected['workflow_outputs'].apply(process_workflow_outputs)

    # Xử lý NaN trong các cột khác
    df_selected.fillna({
        'id': '', 'app_id': '', 'workflow_run_id': '', 'conversation_id': '', 
        'from_account_id': '', 'created_at': '', 'query': '', 'answer': ''
    }, inplace=True)

    # Chuyển thành JSON
    result = df_selected.to_dict(orient='records')

    # Hàm xử lý NaN khi dump JSON
    def json_serializer(obj):
        if pd.isna(obj):
            return None
        return obj

    # Lưu vào file với xử lý NaN
    with open('multi_intent_classification/dify_dataset/raw_data.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4, default=json_serializer)


def get_needed_data():
    with open('multi_intent_classification/dify_dataset/raw_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    needed_data = []

    for item in data:
        id = item['id']
        app_id = item['app_id']
        conversation_id = item['conversation_id']
        time = item['created_at']
        query = item['query']
        response = item["answer"]
        label_unverify = item['workflow_inputs']['text']
        label = item['workflow_outputs']['intent']

        if label and isinstance(label, list) and len(label) > 0:
            needed_data.append({
                'id': id,
                'app_id': app_id,
                'conversation_id': conversation_id,
                'created_at': time,
                'current_message': query,
                'response': response,
                'label_unverify': label_unverify,
                'label_intent': label,
            })
    
    with open('multi_intent_classification/dify_dataset/processed_data.json', 'w', encoding='utf-8') as f:
        json.dump(needed_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    get_needed_data()