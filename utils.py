import csv
import numpy as np
import pandas as pd
import emoji

def read_vectors(file):
    with open(file) as f:
        f.readline()
        word_to_vec = {}
        for line in f:
            line = line.strip().split()
            word = line[0]
            print(word)
            #vector = [float(i) for i in line[1:101]]
            
            word_to_vec[word] = line[1:101]
        
    return word_to_vec

def read_csv(filename = 'data/emojify_data.csv'):
    x = []
    labels = []
    with open(filename) as f:
        Reader = csv.reader(f)
        for row in Reader:
            x.append(row[0])
            labels.append(row[1])
    return np.array(x),np.array(labels)

emoji_mapping= {"0":':heart:',"1":':baseball:',"2":':smile:',\
                "3":':disappointed:',"4":':fork_and_knife:'}

def emojize(y_pred):
     return emoji.emojize(emoji_mapping[str(y_pred)],use_aliases=True)

def create_batches(x,y,batch_size=32):
    batches_x = [] 
    batches_y =[]
    if len(x)%batch_size==0:
        num_batches =len(x)//batch_size
    else:
        num_batches = len(x)//batch_size +1
    for i in range(num_batches):
        if i+ batch_size <=len(x)-1:
            batches_x.append(x[i*batch_size:i*batch_size+batch_size])
            batches_y.append(y[i*batch_size:i*batch_size+batch_size])
        else:
            batches_x.append(x[i*batch_size:])
            batches_y.append(y[i*batch_size:])

    
    return np.array(batches_x),np.array(batches_y)

def batch_maxlen(batch_x):
    return max([len(i) for i in batch_x])


def create_sequences(batch_x):
    max_len = batch_maxlen(batch_x)
    sequences=[]
    for sentence in batch_x:
        words = sentence.strip().lower().split()
        sequence = [] 
        padding = max_len - len(words)
        padding = np.zeros((padding,100)).tolist()
        for word in words:
            embedding = word_to_vec[word]
            if embedding:
                sequence.append(embedding)
            else:
                embedding = [word_to_vec[w] for w in word_to_vec.keys() if word.startswith(w)]
                sequence.append(embedding)
        sequence += padding
        sequences.append(sequence)
    return np.array(sequences)
