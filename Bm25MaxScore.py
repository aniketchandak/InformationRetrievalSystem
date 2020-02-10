# Author: Aniket Chandak
# SJSU ID: 013591890


import sys
import re
import math
import operator
import heapq
import time
from IRmodule import InformationRetrievalSystem
# IRmodule is provided with submission. It has implementation from HW2 for Index creation

# Inherit inverted index from InformationRetrievalSystem class (Done in HW2)
class Bm25MaxScore(InformationRetrievalSystem):
    def get_score(self, raw_query, num_results):
        # raw_query: raw query as disjunctive
        # num_results: No of results to return
        raw_query = raw_query.split(" ")
        plain_query = [word.lower() for word in raw_query if word not in ['_AND', '_OR']]  # Query  without boolean
        u = float("-inf")
        #scores = self.get_rank_cosine(plain_query, num_result)
        scores = self.rankBM25_documentAtATime_withHeaps(plain_query, num_result)
        return scores

    def rankBM25_documentAtATime_withHeaps(self,t,k):
        results = [(0,0) for i in range(k)]  # Result with tuple (score, docid) initialised with 0,0
        heapq.heapify(results) # Heapify results (By default heapify on first term in tuple i.e. score)
        t = list(set(t)) # Remove duplicate as BM25 does not iterate through duplicate terms twice

        # terms contains tuple (nextDoc, term) instead of (term, nextDoc) from slides because python heapq works on the first element in tuple
        terms = [(self.nextDoc(t[i],(float("-inf"), float("-inf"))), t[i]) for i in range(len(t))]
        terms_max_score = [[term, self.max_bm25term_score(term)] for term in t] # Maximum possible score for each term
        terms_max_score = sorted(terms_max_score, key=lambda x: x[1]) # sort by score
        heapq.heapify(terms) # heapify terms
        terms_not_on_heap = set() # Flag of deleted terms (I have implement soft delete for terms based on MaxScore)
        # Compare nextDoc of terms[0] (nextDoc is at 0th index in tuple)
        while terms[0][0] < float('inf'):
            docid = terms[0][0] # Index 0 in tuple is docid (ie nextDoc)
            score = 0
            if terms[0][1] in terms_not_on_heap: # Ignore deleted terms
                heapq.heappop(terms)
                continue
            while(terms[0][0] == docid):
                t = terms[0][1] # Index 1 in tuple is term
                score += (math.log2(self.N/self.Nt[t])*self.get_tf_bm25(t,docid))
                term_0 = list(heapq.heappop(terms)) # Remove term temporary
                term_0[0] = self.nextDoc(t,(docid, float("-inf"))) # Calculate nextDoc for removed term
                heapq.heappush(terms, tuple(term_0)) # Add term again with nextDoc value. (Heapified automatically)
            for term in terms_not_on_heap: # Calculate contribution of score from deleted terms
                if self.nextDoc(term, (docid-1, float("-inf"))) == docid:  # If terms not on heap has docid as the scores
                    score += (math.log2(self.N / self.Nt[term]) * self.get_tf_bm25(term, docid))
            if score > results[0][0]:
                result = list(heapq.heappop(results))
                result[0] = score
                result[1] = docid
                heapq.heappush(results, tuple(result))
                terms_not_on_heap = self.max_score_filter(results[0][0], terms_max_score) # Everytime new score is added in heap, calculate terms to delete
        while(results[0][0] == 0):
            heapq.heappop(results)
        results = sorted(results, key=lambda x: x[0], reverse=True)
        return results

    # Returns maximum possible score for term
    def max_bm25term_score(self,term):
        return (math.log2(self.N/self.Nt[term]))*2.2 if term in self.Nt else 0

    # Returns TF as per BM25 formula for 'term' in document id as 'docid'
    def get_tf_bm25(self, term, docid):
        return ((self.ftd[term][docid]*2.2)/(self.ftd[term][docid]+(1.2*(0.25+(0.75*(self.ld[docid]/self.lavg)))))) \
            if term in self.Nt else 0

    # Function calculates terms which can be removed and returns set of terms which can be removed
    def max_score_filter(self, minimum_score, terms_max_score):
        terms_not_on_heap = set() # set to store deleted terms
        for item in terms_max_score:
            term = item[0]
            max_score = item[1]
            if max_score < minimum_score: # max score is less than minimum score in heap
                terms_not_on_heap.add(term)
                minimum_score -= max_score # Update minimum_score such that deleting next term will consider the max score of previous deleted terms
        return terms_not_on_heap



corpus_file = sys.argv[1]  # Read argument for corpus file
num_result = int(sys.argv[2])  # Read num results to return
query = sys.argv[3]  # Read the query
with open(corpus_file, 'r') as file:  # Open corpus file
    data = file.read()

IR = Bm25MaxScore(data)

# Uncomment below to check inv_ind and other stats
# IR.print_info(IR.inv_ind)
# print(IR.ftd)
# print(IR.Nt)


rank_score = IR.get_score(query, num_result)
print("query_id iter  docno    rank\t\tsim\t\t\trun_id")
for each in rank_score:
    print("0\t0\t",each[1],"\t1\t",each[0],"\t aniket-run")

