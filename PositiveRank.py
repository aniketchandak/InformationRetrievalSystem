# Author: Aniket Chandak
# SJSU ID: 013591890


import sys
import re
import math
import operator


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

    # This method will create a vector for given list of words. method is used to create query vector
    def get_query_vector(self, word_list):
        vector = [0.0 for i in range(len(self.corpus_word_pos))]
        magnitude = 0
        for term in word_list:
            if term not in self.corpus_word_pos:
                continue
            term_freq = word_list.count(term)
            tf = math.log2(term_freq) + 1
            idf = math.log2(self.N / self.Nt[term])
            vector[self.corpus_word_pos[term]] = tf * idf
            magnitude = magnitude + math.pow(tf * idf, 2)

        magnitude = math.sqrt(magnitude)
        # Normalize vector

        for i in range(len(vector)):
            if vector[i] != 0:
                vector[i] = vector[i] / magnitude
        return vector

    def get_normalized_document_vector(self, doc_id):

        # doc_id: document id
        # returns normalised document vector for document with id as doc_id

        vector = [0.0 for i in range(len(self.corpus_word_pos))]
        magnitude = 0
        for term in self.document_vocab[doc_id - 1]:  # Indexing start with 0 hence subtract 1 from docid
            term_freq = self.ftd[term][doc_id]
            if term_freq != 0:
                tf = math.log2(term_freq) + 1
                idf = math.log2(self.N / self.Nt[term])
                vector[self.corpus_word_pos[term]] = tf * idf
                magnitude = magnitude + math.pow(tf * idf, 2)

        magnitude = math.sqrt(magnitude)
        # Normalize vector

        for i in range(len(vector)):
            if vector[i] != 0:
                vector[i] = vector[i] / magnitude
        return vector

    def get_rank_cosine(self, t, k, filter=[]):

        # t: list of words in query
        # k: No of results to return
        # filter: documents to score for. If not passed, all docs will be scored.
        # returns sorted result with doc_id and score (sorted by score)

        result = []  # Result is list of list such as [x, y] where x-->docid and y--->score

        # When user passes filter based on boolean querry
        if filter != []:
            filter_length = len(filter)
            for doc_id in filter:
                doc_score = [0, 0]  # [x, y] where x is docid and y is score. Init with 0,0
                doc_score[0] = doc_id
                vec_d = self.get_normalized_document_vector(doc_id)
                vec_t = self.get_query_vector(t)
                doc_score[1] = self.sim(vec_d, vec_t)
                result.append(doc_score)
            result.sort(key=operator.itemgetter(1), reverse=True)
            # When k is more than filter length, cant return k results
            return result[:min(k, filter_length)]

        # If user do not pass filter based on boolean query, cosine will be applied on all docs
        # When user don't have filter [Not applicable for this assignment, but can be used for non boolean query]
        # Else method follows algorithm given in book for rank_cosine
        else:
            d = min([self.nextDoc(t[i], (float("-inf"), float("-inf"))) for i in range(len(t))])
            filter_length = k
            while d < float("inf") and (d in filter or filter == []):
                doc_score = [0, 0]  # [x, y] where x is docid and y is score. Init with 0,0
                doc_score[0] = d
                vec_d = self.get_normalized_document_vector(d)
                vec_t = self.get_query_vector(t)
                doc_score[1] = self.sim(vec_d, vec_t)
                result.append(doc_score)
                d = min([self.nextDoc(t[i], (d, float("-inf"))) for i in range(len(t))])
            result.sort(key=operator.itemgetter(1), reverse=True) # Sort based on scores
            print(result)
            return result[:min(k, filter_length)]

    def sim(self, vec_d, vec_q):

        # vec_d, vec_q: vectors
        # Returns similarity score by dot product on vec_d and vec_q

        result = 0.0
        for i in range(len(vec_q)):
            if vec_q[i] != 0:
                result = result + vec_d[i] * vec_q[i]
        return result

    def get_score(self, raw_query, num_results):

        # raw_query: Query in polish format in string data structure
        # num_results: No of results to return
        # returns scores

        processed_query = self.process_polish_notation(raw_query)
        raw_query = raw_query.split(" ")
        plain_query = [word.lower() for word in raw_query if word not in ['_AND', '_OR']]  # Query  without boolean
        u = float("-inf")
        doc_filter = []  # Store docid which satisfied boolean query
        while u < float("inf"):
            u = self.next_solution(processed_query, (u, 0))
            if u < float("inf"):
                doc_filter.append(u)
        if doc_filter == []:
            print("No documents match given query")
            return []
        scores = self.get_rank_cosine(plain_query, num_result, doc_filter)
        return scores

    def next_solution(self, Q, position):

        v = self.docRight(Q, position)

        if v == float("inf"):
            return float("inf")
        u = self.docLeft(Q, (v + 1, 0))
        if u == v:
            return u
        else:
            return self.next_solution(Q, (v, 0))

    def process_polish_notation(self, raw_query):

        # This method is used to process polish notation such as raw query is converted as follow
        # operator oprand1 oprand2 ------> [operator, oprand1, operator]
        # e.g. "_OR _AND good dog _AND bad cat" is converted as below
        # ['_OR', ['_AND', 'good', 'dog'], ['_AND', 'bad', 'cat']]

        raw_query = raw_query.split(" ")

        # Convert to reverse polish notation
        raw_query = raw_query[::-1]
        stack = []
        result = []
        raw_query = [word.lower() if word not in ['_AND', '_OR'] else word for word in raw_query]  # Normalise to lower
        for elem in raw_query:
            if elem in ['_AND', '_OR']:
                oprand1 = stack.pop()
                oprand2 = stack.pop()
                stack.append([elem, oprand1, oprand2])
            else:
                stack.append(elem)
        return stack.pop()

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

    def print_info(self, inverted_index):
        all_words = list(inverted_index.keys())  # Get list of all the valid words
        all_words.sort()  # Sort all words

        # Print all stats in given format
        for each_word in all_words:
            print(each_word)
            string_positions = str(inverted_index[each_word]['positions'])
            string_positions = string_positions[1:-1]
            print(str(inverted_index[each_word]['no_of_docs']) + ',' + str(
                inverted_index[each_word]['frequency']) + ',' + string_positions.replace(" ", ""))


corpus_file = sys.argv[1]  # Read argument for corpus file
num_result = int(sys.argv[2])  # Read num results to return
query = sys.argv[3]  # Read the query
with open(corpus_file, 'r') as file:  # Open corpus file
    data = file.read()

IR = InformationRetrievalSystem(data)

# Uncomment below to check inv_ind and other stats
# IR.print_info(IR.inv_ind)
# print(IR.ftd)
# print(IR.Nt)

rank_score = IR.get_score(query, num_result)
print("DocId Score")
for each in rank_score:
    print(each[0], "  ", each[1])

# Uncomment below to check cosin results
# print(IR.get_rank_cosine(["quarrel", "sir"], 5 ))
