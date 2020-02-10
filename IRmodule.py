import sys
import re
import math
import operator
import heapq
import time

class InformationRetrievalSystem:
    def __init__(self, raw_data):

        #:param raw_data: contains raw_data from which we create IR system. Documents are separated by \n\n

        # self.documents: documents separated in by list
        # self.documents_vocab: distinct word in each doc. For faster read operations used set
        # self.corpus_word_pos: keys are all words in corpus. value is indexed mapped for vectorisation
        # self.c : cache used in next, prev etc for storing last retireval location of term e
        # self.N No of documents
        # self.Nt: No of documents containing term t. keys: terms values: No of docs containing t
        # self.ftd: term frequency in doc (key: term, value: list of freq in each doc. e.g. ftd['the'][2] gives freq of
        # 'the' in 2nd doc}

        self.documents, self.document_vocab, self.corpus_word_pos = self.pre_process_docs(raw_data)
        self.inv_ind = self.get_inverted_index(self.documents, self.document_vocab)
        self.c = {}  # This variable is used to store last index position cached for term.
        self.N = len(self.documents)  # No of documents
        self.ftd = self.get_term_frequency()  # Term frequency (Used next() method here to calculate frequency)
        self.Nt = self.get_document_frequency()  # Document frequency
        self.ld, self.lavg = self.get_document_length()

    def first(self, t):

        # t: term
        # returns first occurance of term

        return self.inv_ind[t]['positions'][0]

    def last(self, t):

        # t: term
        # Returns last occurance of term

        return self.inv_ind[t]['positions'][-1]

    def next(self, t, current):
        # t: term
        # current : current position (data structue of current tuple ---> (doc_id, position)

        # returns next occurance of t after current

        if self.inv_ind[t]['frequency'] == 0:
            return float("inf"), float("inf")

        # if last position is less than current, next does not exist
        if self.inv_ind[t]['positions'][-1] <= current:
            # Note: Python handles the comparison of tuples as per our requirements.
            # i.e. first it will match the doc id, if it is greater or less it return the boolean
            # When both id are same, it will compare the position and return the result
            return float("inf"), float("inf")

        # If first position is great than current, return first position
        if self.inv_ind[t]['positions'][0] > current:
            self.c[t] = 0
            return self.inv_ind[t]['positions'][self.c[t]]

        # Use cache-1 as low if cache is less than current otherwise set low to 0
        if t in self.c and self.c[t] > 0 and self.inv_ind[t]['positions'][self.c[t] - 1] <= current:
            low = self.c[t] - 1
        else:
            low = 0
        jump = 1

        high = low + jump

        while high < self.inv_ind[t]['frequency'] - 1 and self.inv_ind[t]['positions'][high] <= current:
            low = high
            jump = 2 * jump
            high = low + jump
        if high >= self.inv_ind[t]['frequency']:
            high = self.inv_ind[t]['frequency'] - 1  # Index in python starts with 0
        self.c[t] = self.binary_search(t, low, high, current,
                                       True)  # Passing flag to binary search to indicate next operation
        return self.inv_ind[t]['positions'][self.c[t]]

    def prev(self, t, current):
        if self.inv_ind[t]['frequency'] == 0:
            return float("-inf"), float("-inf")

        # if first position is greater than current, prev does not exist
        if self.inv_ind[t]['positions'][0] >= current:
            return float("-inf"), float("-inf")

        # If last position is less than current, return last position
        if self.inv_ind[t]['positions'][-1] < current:
            self.c[t] = self.inv_ind[t]['frequency'] - 1
            return self.inv_ind[t]['positions'][self.c[t]]

        # Use cache+1 as high if cache is greather than current
        if t in self.c and self.c[t] < self.inv_ind[t]['frequency'] - 1 and self.inv_ind[t]['positions'][
            self.c[t] - 1] >= current:
            high = self.c[t] - 1
        else:
            high = self.inv_ind[t]['frequency'] - 1  # Otherwise high is at last positions
        jump = 1

        low = high - jump

        while low >= 0 and self.inv_ind[t]['positions'][low] >= current:
            high = low
            jump = 2 * jump
            low = high - jump
        if low <= 0:
            low = 0  # Index in python starts with 0
        self.c[t] = self.binary_search(t, low, high, current, False)
        return self.inv_ind[t]['positions'][self.c[t]]

    def first_doc(self, t):
        return self.inv_ind[t]['positions'][0][0]  # Doc id of first element

    def last_doc(self, t):
        return self.inv_ind[t]['positions'][-1][0]  # Doc id of last element

    def nextDoc(self, t, current):

        # nextDoc: returns the docid of the first document after current that
        # contains the term t
        # (Used galloping search)
        if t not in self.inv_ind:
            return float("inf")
        if self.inv_ind[t]['frequency'] == 0:
            return float("inf")

        # if last position is less than current, next doc does not exist
        if self.inv_ind[t]['positions'][-1][0] <= current[0]:
            return float("inf")

        # If first position is great than current, return first document
        if self.inv_ind[t]['positions'][0][0] > current[0]:
            self.c[t] = 0
            return self.inv_ind[t]['positions'][self.c[t]][0]

        # Use cache-1 as low if cache is less than current otherwise set low to 0
        if t in self.c and self.c[t] > 0 and self.inv_ind[t]['positions'][self.c[t] - 1][0] <= current[0]:
            low = self.c[t] - 1
        else:
            low = 0
        jump = 1

        high = low + jump

        while high < self.inv_ind[t]['frequency'] - 1 and self.inv_ind[t]['positions'][high][0] <= current[0]:
            low = high
            jump = 2 * jump
            high = low + jump
        if high >= self.inv_ind[t]['frequency']:
            high = self.inv_ind[t]['frequency'] - 1  # Index in python starts with 0
        doc_id = self.binary_search_document(t, low, high, current,
                                             True)  # Passing flag to binary search to indicate next operation
        return doc_id

    def prevDoc(self, t, current):
        if t not in self.inv_ind:
            return float("-inf")
        if self.inv_ind[t]['frequency'] == 0:
            return float("-inf")

        # if first position is greater than current, prev does not exist
        if self.inv_ind[t]['positions'][0][0] >= current[0]:
            return float("-inf")

        # If last doc is less than current, return last doc
        if self.inv_ind[t]['positions'][-1][0] < current[0]:
            self.c[t] = self.inv_ind[t]['frequency'] - 1
            return self.inv_ind[t]['positions'][self.c[t]][0]

        # Use cache+1 as high if cache is greather than current
        if t in self.c and self.c[t] < self.inv_ind[t]['frequency'] - 1 and self.inv_ind[t]['positions'][
            self.c[t] + 1][0] >= current[0]:
            high = self.c[t] + 1
        else:
            high = self.inv_ind[t]['frequency'] - 1  # Otherwise high is at last positions
        jump = 1

        low = high - jump

        while low >= 0 and self.inv_ind[t]['positions'][low][0] >= current[0]:
            high = low
            jump = 2 * jump
            low = high - jump
        if low <= 0:
            low = 0  # Index in python starts with 0
        doc_id = self.binary_search_document(t, low, high, current, False)
        return doc_id

    def docRight(self, Q, u):

        # Q: is preprocessed polish query
        # e.g. "_OR _AND good dog _AND bad cat" is converted as below
        # [ 'OR', ['_AND', 'good', 'dog'], ['_AND', 'bad', 'cat']]

        if isinstance(Q, str):  # When Q is string, its term and not Query
            return self.nextDoc(Q, u)
        else:
            if Q[0] == '_AND':
                return max(self.docRight(Q[1], u), self.docRight(Q[2], u))
            if Q[0] == '_OR':
                return min(self.docRight(Q[1], u), self.docRight(Q[2], u))

    def docLeft(self, Q, u):

        # Q: is preprocessed polish query
        # e.g.
        # [ 'OR', ['_AND', 'good', 'dog'], ['_AND', 'bad', 'cat']]

        if isinstance(Q, str):  # When Q is string, its term and not Query
            return self.prevDoc(Q, u)
        else:
            if Q[0] == '_AND':
                return min(self.docLeft(Q[1], u), self.docLeft(Q[2], u))
            if Q[0] == '_OR':
                return max(self.docLeft(Q[1], u), self.docLeft(Q[2], u))

    def binary_search(self, t, low, high, current, flag):
        # same binary search can not be used for prev and next hence using flag to indicate operation
        # when flag is true, search for the next elements else prev wrt current
        while high >= low:
            if high == low + 1:  # Only two elements left in search window
                if flag:  # Want to find next
                    return low if self.inv_ind[t]['positions'][low] > current else high
                else:  # Want to find prev
                    return high if self.inv_ind[t]['positions'][high] < current else low
            mid = (low + high) // 2  # // operator rounds to whole number
            if self.inv_ind[t]['positions'][mid] == current:
                if flag:
                    return mid + 1  # Next when flag is true
                else:
                    return mid - 1  # Prev when flag is false
            elif self.inv_ind[t]['positions'][mid] < current:
                low = mid
            else:
                high = mid

    # Binary search for can not be used for nextDoc and prevDoc. Once we find the current, we need to search sequentially to get next/prev doc
    def binary_search_document(self, t, low, high, current, flag):
        # when flag is true, search for the next elements else prev wrt current
        while high >= low:
            mid = (low + high) // 2  # round to whole number
            if high == low + 1:  # Only two elements left in search window
                if flag:  # Want to find next
                    return self.inv_ind[t]['positions'][low][0] if self.inv_ind[t]['positions'][low][0] > current[
                        0] else self.inv_ind[t]['positions'][high][0]
                else:  # Want to find prev
                    return self.inv_ind[t]['positions'][high][0] if self.inv_ind[t]['positions'][high][0] < current[
                        0] else self.inv_ind[t]['positions'][low][0]
            if self.inv_ind[t]['positions'][mid][0] == current[0]:
                if flag:
                    while self.inv_ind[t]['positions'][mid][0] == self.inv_ind[t]['positions'][mid + 1][0] and mid + 1 < \
                            self.inv_ind[t]['frequency'] - 1:
                        mid = mid + 1
                    return self.inv_ind[t]['positions'][mid + 1][0]  # Next when flag is true
                else:
                    while self.inv_ind[t]['positions'][mid][0] == self.inv_ind[t]['positions'][mid - 1][0] and mid > 1:
                        mid = mid - 1
                    return self.inv_ind[t]['positions'][mid - 1][0]  # Prev when flag is false
            elif self.inv_ind[t]['positions'][mid][0] < current[0]:
                low = mid
            else:
                high = mid


    def next_solution(self, Q, position):
        v = self.docRight(Q, position)
        if v == float("inf"):
            return float("inf")
        u = self.docLeft(Q, (v + 1, 0))
        if u == v:
            return u
        else:
            return self.next_solution(Q, (v, 0))

    # Function to calculate term frequency (ftd[t][d] where t is term and d is docid)
    def get_term_frequency(self):
        ftd = {}
        for term in self.corpus_word_pos:  # For each term in corpus
            ftd[term] = [0 for i in range(self.N + 1)]  # Initialise with 0 (frequency for term)
            for docid in range(self.N + 1):
                count = 0
                current = (docid, 0)  # start with first element in docid
                next_pos = self.next(term, current)
                while next_pos < (docid + 1, 0):  # Check till last appearance of element in document
                    count += 1
                    next_pos = self.next(term, next_pos)
                ftd[term][docid] = count
        return ftd

    def get_document_length(self):
        ld = [0 for i in range(self.N+1)]
        doc_length_sum = 0
        for docid,document in enumerate(self.documents):
            ld[docid+1] = len(document)
            doc_length_sum += ld[docid+1]
        return ld,doc_length_sum/self.N

    # Calculates document frequency Nt[t] where t is term
    def get_document_frequency(self):
        Nt = {}
        for term in self.corpus_word_pos:
            count = 0
            current = (0, 0)
            next_doc = self.nextDoc(term, current)
            while (next_doc < float("inf")):
                count += 1
                next_doc = self.nextDoc(term, (next_doc, 0))
            Nt[term] = count
        return Nt

    def get_positions(self, doc_id, word, documents):
        #Returns all the positions of word occur in documents
        positions = []
        for index, each_word in enumerate(documents[doc_id]):
            if each_word == word:
                positions.append((doc_id + 1, index + 1))
        return positions

    def pre_process_docs(self, raw_data):
        document_vocab = []
        documents = raw_data.split('\n\n')  # Split for each document
        corpus_vocab = []
        corpus_word_index = {}  # Assign constant index to each word in corpus. Will be used in vector creation.
        for i in range(len(documents)):  # Process each documents
            documents[i] = re.sub(r'[?|$|.|!|:|,|)|(]', r'', documents[i])  # Remove special symbols
            documents[i] = documents[i].split()  # Convert each document in list structure
            documents[i] = [word.lower() for word in documents[i]]  # Normalise to lower case
            document_vocab.append(set(documents[i]))  # Create set of vocabulary for quick search on excluded terms
            corpus_vocab.extend(documents[i])

        corpus_vocab = (sorted(list(set(corpus_vocab))))  # Remove duplicates and sort
        for i, word in enumerate(corpus_vocab):
            corpus_word_index[word] = i
        return documents, document_vocab, corpus_word_index

    def get_inverted_index(self, documents, document_vocab):
        #Method to calculate stats for all words in documents

        inverted_index = {}
        for doc_id, vocab in enumerate(document_vocab):
            for word in vocab:
                if word not in inverted_index:
                    inverted_index[word] = {}
                    inverted_index[word]['no_of_docs'] = 1
                    inverted_index[word]['positions'] = self.get_positions(doc_id, word, documents)
                    inverted_index[word]['frequency'] = len(inverted_index[word]['positions'])
                else:
                    inverted_index[word]['no_of_docs'] += 1
                    inverted_index[word]['positions'].extend(self.get_positions(doc_id, word, documents))
                    inverted_index[word]['frequency'] = len(inverted_index[word]['positions'])
        return inverted_index