import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyvi import ViTokenizer
from tqdm import tqdm

def clean_vietnamese_text(txt: str, stop_word: list) -> str:
    txt = txt.lower()
    txt = re.sub('[^A-Za-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ\s]+', ' ', txt)
    txt = txt.strip()
    for w in stop_word:
        while w in txt:
            txt = re.sub(w, '', txt)
    tokens = ViTokenizer.tokenize(txt)
    return tokens

def get_by_category(data:pd.DataFrame, category:str, column="label"):
    """
    Get rows data by a value in a column

    Parameters
    ----------
    data : DataFrame
    Column : str
    category : str

    Returns
    ----------
    A data frame containing data with the value of the column 
    is the category's respective
    """
    return  data[data[column]==category]
def log_samples(data:pd.DataFrame, column:str, category:str):
    """
    Show data by a value in a column

    data : DataFrame
    Column : str
    category : str
    """
    s = get_by_category(data, column, category)
    print(f"{len(s)} samples {category}:")
    print(s["sample"].values)

def replace_label(data:pd.DataFrame, raw_label:str, new_label:str)->pd.DataFrame:
    """
    Replace value of dataframe

    data : DataFrame
    raw_label : str
        Old value
    new_label : str
        New value
    """
    data["label"].replace(raw_label, new_label, inplace=True)
    return data

def statis(data:pd.DataFrame, column_category):
    statis_df = data[column_category].value_counts()
    return pd.DataFrame.from_dict({column_category: statis_df.index, "count": statis_df.values})

# Function to insert row in the dataframe
def Insert_row(row_number, df, row_value):
    # Starting value of upper half
    start_upper = 0
  
    # End value of upper half
    end_upper = row_number
  
    # Start value of lower half
    start_lower = row_number
  
    # End value of lower half
    end_lower = df.shape[0]
  
    # Create a list of upper_half index
    upper_half = [*range(start_upper, end_upper, 1)]
  
    # Create a list of lower_half index
    lower_half = [*range(start_lower, end_lower, 1)]
  
    # Increment the value of lower half by 1
    lower_half = [x.__add__(1) for x in lower_half]
  
    # Combine the two lists
    index_ = upper_half + lower_half
  
    # Update the index of the dataframe
    df.index = index_
  
    # Insert a row at the end
    df.loc[row_number] = row_value
   
    # Sort the index labels
    df = df.sort_index()
  
    # return the dataframe
    return df

def delete_by_category(data, category, num_delete):
    cate = get_by_category(data, category)
    id = cate.index
    drop_id = id[:num_delete]
    return pd.concat([data.drop(drop_id)], ignore_index=True)

def visualize_statis(data, column="label", size=(10,5)):
    s = statis(data, column)
    s_dict = dict(zip(s[column].values, s["count"].values))
    fig, axs = plt.subplots(1,1,figsize=size)
    name = list(s_dict.keys())
    value = list(s_dict.values())
    axs.bar(name, value, color="lightblue")
    fig.suptitle("Statis number samples of each label")
    plt.show()

def sort_dict(d:dict, by_value=True, descending=True )-> dict:
    """
    Sort a dictionary by key or value.

    Parameters
    ----------
    d : dict 
        Dictionary to sort
    by_value : bool, optional
        If this is set to False, dictionary is sorted by key. The default is True, 
        dictionary is sorted by value
    descending : bool, optional
        If this is set to False, sort ascending, others, sorting is descending
    
    Returns
    ----------
    sorted_dict: dict
        A dictionary that is sorted 

    """
    val = np.array(list(d.values()))
    keys = np.array(list(d.keys()))
    
    if by_value:
        id = np.argsort(val)
    else:
        id = np.argsort(keys)
    
    if descending:
        id = id[::-1]
    return dict(zip(keys[id], val[id]))

def clean_text(txt):
    txt = re.sub(r"\d|-|\.|\?|!|,|&|:|#|\*|/|%|\+|\"|\|", '', txt)
    txt = re.sub(r"[\(\)\{\}\[\]]",'',txt)
    while re.findall(r' [A-z] |^[A-z] | [A-z]$', txt):
        txt = re.sub(r' [A-z] |^[A-z] | [A-z]$', ' ', txt)
    while re.findall('  ', txt):
        txt = re.sub('  ', ' ', txt)
    txt = txt.strip()
    return txt

def statis_words(data: pd.DataFrame,  industry_name: str, column_sample='sample', column_industry='label'):
    """
    Statis the count appearance of each word in each industry
    """
    samples = get_by_category(data, column_industry, industry_name).dropna()
    samples = samples[column_sample].unique()
    samples = set([i.lower() for i in samples]) 
    
    token_sample_count = 0
    count_freq = {}

    for sent in tqdm(samples):
        sent = clean_text(sent)
        sent = sent.split()
        token_sample_count += len(sent)
        for word in sent:
            count_freq[word] = count_freq.get(word, 0) + 1
    count_freq = {k : round(v/token_sample_count, 5) for k,v in count_freq.items()}
    count_freq = sort_dict(count_freq)
    return list(count_freq.items())

def confusion_report(cate, cnf_matrix, labels, df_train):
    count_cate  = df_train[df_train["label"] == cate].value_counts("label").values[0]
    
    id = labels.index(cate)
    positive_predict = cnf_matrix[ : ,id]
    negative_predict = cnf_matrix[id, : ]
    TP = cnf_matrix[id,id]
    FP = np.sum(positive_predict) - TP
    FN = np.sum(negative_predict) - TP

    precision = round(TP/(TP + FP) * 100,2)
    recall = round(TP/(TP + FN)*100, 2)
    f1_score = round(2*(precision*recall)/(precision + recall), 2)
    print(f"{cate}: gồm {count_cate} mẫu trong tập train\n - TP: {TP}  - FN: {FN}  - FP: {FP}")
    print(f" - Precision: {precision}%    Recall:    {recall}%     F1-score:    {f1_score}%")
    
    arr_labels = np.array(labels)
    top_FP = arr_labels[np.argsort(positive_predict)[::-1]]
    top_FN = arr_labels[np.argsort(negative_predict)[::-1]]

    print("\nTop labels false positive:")
    for i in range(len(top_FP)):
        if top_FP[i] != cate:
            print(f" + {top_FP[i]}: {positive_predict[labels.index(top_FP[i])]}")
    print("Top labels false negative:")
    for i in range(len(top_FN)):
        if top_FN[i] != cate:
            print(f" + {top_FN[i]}: {negative_predict[labels.index(top_FN[i])]}")