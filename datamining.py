from sklearn import datasets
import string,re
import pandas as pd
import numpy as np
#import nltk
from nltk.tokenize import RegexpTokenizer
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from apyori import apriori
from statistics import mode

def strip_newsgroup_header(text):
    _before, _blankline, after = text.partition("\n\n")
    return after

_QUOTE_RE = re.compile(
    r"(writes in|writes:|wrote:|says:|said:" r"|^In article|^Quoted from|^\||^>)"
)

def strip_newsgroup_quoting(text):
    good_lines = [line for line in text.split("\n") if not _QUOTE_RE.search(line)]
    return "\n".join(good_lines)

def strip_newsgroup_footer(text):
    lines = text.strip().split("\n")
    for line_num in range(len(lines) - 1, -1, -1):
        line = lines[line_num]
        if line.strip().strip("-") == "":
            break

    if line_num > 0:
        return "\n".join(lines[:line_num])
    else:
        return text

def preprocessing(data):

    data.data = [strip_newsgroup_header(text) for text in data.data]                                # Header Removal
    data.data = [strip_newsgroup_footer(text) for text in data.data]                                # Footer Removal
    data.data = [strip_newsgroup_quoting(text) for text in data.data]                               # Quoting Removal
    data.data = [text.strip() for text in data.data]                                                # Space Removal
    data.data = [text.lower() for text in data.data]                                                # To Lower
    
    data.data = [text.translate(str.maketrans('', '', string.punctuation)) for text in data.data]   # Punctuation Removal
    data.data = [re.sub(r'\d+', '', text) for text in data.data]                                    # Number Removal
    tokenizer = RegexpTokenizer(r"\w+(?:[-'+]\w+)*|\w+")
    data.data = [tokenizer.tokenize(text) for text in data.data] 
    #data.data = [word_tokenize(text) for text in data.data]                                         # Tokenization

    lemmatizer = WordNetLemmatizer()
    data.data = [[lemmatizer.lemmatize(word) for word in text] for text in data.data]

    stop_words_set1 = stopwords.words('english')
    
    data.data = [[word for word in text if word not in stop_words_set1] for text in data.data]      # Stop Word Removal Set 1
    data.data = [[word for word in text if word not in ENGLISH_STOP_WORDS] for text in data.data]   # Stop word Removal Set 2
    data.data = [[word for word in text if len(word) > 3] for text in data.data]

    #snow_stemmer = SnowballStemmer(language = 'english')
    #data.data = [[snow_stemmer.stem(word) for word in text] for text in data.data]

    return data

def doc2vec(data):

    tfidf = TfidfVectorizer(tokenizer = lambda i:i, lowercase = False)

    tfidf_vector = tfidf.fit_transform(data.data)

    
    tfidf_df = pd.DataFrame(tfidf_vector.toarray())
    tfidf_df.columns = tfidf.get_feature_names_out()

    T_tfidf_df = tfidf_df.T
    T_tfidf_df.columns = ['doc ' + str(i) for i in range(0, len(tfidf_df))]

    for i in range (0, len(tfidf_df)):
        doc_num = "doc " + str(i)
        col = T_tfidf_df.loc[:, doc_num]
        col = col[col>0.1]
        col = col.sort_values(ascending=False)
        #cut_off = 10 if len(data.data[i]) > 10 else len(data.data[i])
        data.data[i] = list(col.index)#[:cut_off]

    return data

def association_rules(data):
    
    transactions = []
    for i in range(0, len(data.data)):
        if data.data[i]:
            transactions.append(data.data[i])

    association_rules = list((apriori(transactions, min_support = 0.0003, min_confidence = 0.8, min_length = 2, max_length = 2)))

    return association_rules

def extend(data):

    for i in range (0, len(data.data)):
        data.data[i].append(data.target_names[data.target[i]])

    return data

def rules2df(rules):

    lhs = [tuple(result[2][0][0])[0] for result in rules]
    rhs = [tuple(result[2][0][1])[0] for result in rules]
    support = [result[1] for result in rules]
    confidence = [result[2][0][2] for result in rules]
    lift = [result[2][0][3] for result in rules]

    df = pd.DataFrame(list(zip(lhs, rhs, support, confidence, lift)), columns = ['Left_Hand_Side', 'Right_Hand_Side', 'Support', 'Confidence', 'Lift'])

    return df

def evaluation(train_data, test_data):

    results_Df = pd.DataFrame(0, index = test_data.target_names, columns = ['Correct_Predictions', 'Total_Predictions', 'Precision', 'Recall'])
    for i in range (0, len(test_data.data)):
        predictions = []
        for j in range(0, len(train_data)):

            if train_data.at[j, 'Left_Hand_Side'] in test_data.data[i]:
                predictions.append(train_data.at[j, 'Right_Hand_Side'])
        if predictions:
            prediction = mode(predictions)
        else:
            continue
        if prediction == test_data.target_names[test_data.target[i]]:
            
            results_Df.at[test_data.target_names[test_data.target[i]], 'Correct_Predictions'] += 1
            results_Df.at[test_data.target_names[test_data.target[i]], 'Total_Predictions'] += 1
        else:
            results_Df.at[test_data.target_names[test_data.target[i]], 'Total_Predictions'] += 1

    targets, frequency = np.unique(test_data.target, return_counts=True)
    
    for i in range(0, len(results_Df)):

        if results_Df.at[test_data.target_names[i], 'Total_Predictions'] == 0:
            results_Df.at[test_data.target_names[i], 'Precision'] = 0
        else:
            results_Df.at[test_data.target_names[i], 'Precision'] = results_Df.at[test_data.target_names[i], 'Correct_Predictions'] / results_Df.at[test_data.target_names[i], 'Total_Predictions']

        results_Df.at[test_data.target_names[i], 'Recall'] = results_Df.at[test_data.target_names[i], 'Correct_Predictions'] / frequency[i]

    print(results_Df)
    print("Macro Average Precision:", round(results_Df['Precision'].mean(), 4))
    print("Macro Average Recall:", round(results_Df['Recall'].mean(), 4))
    
    return
#1ST PART
train_data = datasets.load_files("20news-bydate-train", encoding = 'latin-1', shuffle= False)
clean_train_data = preprocessing(train_data)
top_tfidf_train_data = doc2vec(clean_train_data)


#2ND PART
#rules = association_rules(top_tfidf_train_data)

#rules_df = rules2df(rules)
#rules_df = rules_df.sort_values(by=['Lift'])
#print(rules_df.to_string())


#3RD PART
extended_top_tfidf_train_data = extend(top_tfidf_train_data)

rules = association_rules(extended_top_tfidf_train_data)

rules_df = rules2df(rules)
#rules_df = rules_df.sort_values(by=['Lift'], ascending = False)
#print(rules_df.to_string())

for i in range (0, len(rules_df)):
    if not rules_df.at[i, 'Right_Hand_Side'] in train_data.target_names:
        rules_df.drop(i, inplace = True)   
rules_df.reset_index(inplace = True)

#4TH PART
test_data = datasets.load_files("20news-bydate-test", encoding = 'latin-1', shuffle= False)
clean_test_data = preprocessing(test_data)
top_tfidf_test_data = doc2vec(clean_test_data)

evaluation(rules_df, top_tfidf_test_data)