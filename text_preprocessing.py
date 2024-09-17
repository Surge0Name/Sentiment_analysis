import nltk
import os
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

try:
    nltk.data.find('nltk_data/corpora/stopwords')
    nltk.data.find('nltk_data/tokenizers/punkt')

except:
    print("stopword not found, downloading...")
    nltk.download('stopwords', download_dir=nltk_data_dir)
    nltk.download('punkt', download_dir=nltk_data_dir)
import re, string, json
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nlp_id.lemmatizer import Lemmatizer
from nlp_id.tokenizer import Tokenizer, PhraseTokenizer
from nlp_id.stopword import StopWord

class TextPreprocessing:
    def __init__(self) -> None:
        self.stopword_en = stopwords.words('english')
        self.stopword_id = stopwords.words('indonesian')
        with open("slang.json") as f:
            self.slang_dict = json.load(f)
        
    
    def lowercase(self, data:str):
        data = data.lower()
        return data
    
    def tokenization(self, data:str, lib= "standard"):
        if lib == "standard":
            result = word_tokenize(data)
        elif lib == "kumparan":
            token = PhraseTokenizer()
            result = token.tokenize(data)
        return result
    
    def text_cleaning(self, data:str) -> str:
        #number
        data = re.sub(r"\d+", "", data)
        #@pattern
        at_pattern = re.compile(r'@\S+')
        data = at_pattern.sub(r'', data)
        #url
        url_pattern = re.compile(r"https?://\S+|www\.\S+")
        data = url_pattern.sub(r'', data)
        #remove ellipsis
        data = re.sub(r'\s*\.+\s*', ' ', data)
        #punctuation
        data = data.translate(str.maketrans("", "", string.punctuation))
        #remove extra spaces
        data = re.sub(r'\s+', ' ', data)
        #whitespace
        data = data.strip()
        return data

    def stopword_removal(self, data, lib = "standard"):
        if lib == 'standard':
            stopwords =self.stopword_en

            resultword = [word for word in data if word not in stopwords]      
            result= ' '.join(resultword)  
        elif lib == "kumparan":
            sw = StopWord()
            result = sw.remove_stopword(data)
        return result
    
    def stemming(self, data:str):
        stemmer = PorterStemmer()
        return stemmer.stem(data)
    
    def lemmatize(self, data:str):
        lemmatizer = Lemmatizer()
        result = lemmatizer.lemmatize(data)
        return result
    
    def slang_transforming(self, data:str)->str:
        tokens = data.split()
        
        #replacing slang
        transformed_token = [
            self.slang_dict.get(token,token) for token in tokens
        ]

        #reconstruct text
        transformed_text = ' '.join(transformed_token)
        return transformed_text
    
