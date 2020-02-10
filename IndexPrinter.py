import sys
import re

def get_positions(doc_id, word, documents):
    '''Returns all the position word occur in document'''
    positions = []
    for index,each_word in enumerate(documents[doc_id]):
        if each_word == word:
            positions.append((doc_id+1, index+1))
    return positions

def pre_process_docs(data,excluded):
    document_vocab = []
    documents = data.split('---')  # Split for each document
    for i in range(len(documents)):  # Process each documents
        documents[i] = re.sub(r'[?|$|.|!|:|)|(]', r'', documents[i])  # Remove special symbols
        documents[i] = documents[i].split()  # Convert each document in list structure
        documents[i] = [word.lower() for word in documents[i]]  # Normalise to lower case
        document_vocab.append(set(documents[i]))  # Create set of vocabulary for quick search on excluded terms
        if (document_vocab[i] & excluded):  # Erase content of document if excluded term is present
            documents[i] = []
            document_vocab[i] = []
    return documents, document_vocab

def get_inverted_index(documents, document_vocab):
    '''Method to calculate stats for all words in documents'''
    inverted_index = {}
    for doc_id, vocab in enumerate(document_vocab):
        for word in vocab:
            if word not in inverted_index:
                inverted_index[word] = {}
                inverted_index[word]['no_of_docs'] = 1
                inverted_index[word]['positions'] = get_positions(doc_id, word, documents)
                inverted_index[word]['frequency'] = len(inverted_index[word]['positions'])
            else:
                inverted_index[word]['no_of_docs'] += 1
                inverted_index[word]['positions'].extend(get_positions(doc_id, word, documents))
                inverted_index[word]['frequency'] = len(inverted_index[word]['positions'])
    return inverted_index

def print_info(inverted_index):
    all_words = list(inverted_index.keys())  # Get list of all the valid words
    all_words.sort()  # Sort all words

    # Print all stats in given format
    for each_word in all_words:
        print(each_word)
        string_positions = str(inverted_index[each_word]['positions'])
        string_positions = string_positions[1:-1]
        print(str(inverted_index[each_word]['no_of_docs']) + ',' + str(
            inverted_index[each_word]['frequency']) + ',' + string_positions.replace(" ", ""))

corpus_file = sys.argv[1] # Read argument for corpus file
excluded_terms = {excluded_term for (i,excluded_term) in enumerate(sys.argv) if i>1} #Create set of excluded terms.
with open(corpus_file, 'r') as file: #Open corpus file
    data = file.read()
'''
Description of variables used:
documents: list of pre processed documents
document_vocab: list which stores distinct vocab of each doc (For faster search and handle redundancy)
inv_ind: dictionary which store all the statistics for word
'''
documents, document_vocab = pre_process_docs(data, excluded_terms)
inv_ind = get_inverted_index(documents,document_vocab)
print_info(inv_ind)
