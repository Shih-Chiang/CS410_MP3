import numpy as np
import math
from numpy import log
from io import open


def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix

       
class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None 
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)

        self.number_of_documents = 0
        self.vocabulary_size = 0

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
        
        Update self.number_of_documents
        """
        # #############################
        # your code here
        # #############################

        file_object=open(self.documents_path, encoding='utf-8')
        for aline in file_object.readlines():
            values=aline.split()
            self.documents.append(values)
            self.number_of_documents+=1


    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """
        # #############################
        # your code here
        # #############################
        # list ordered collection
        # set unordered collection
        for i in self.documents:
            for w in i:
                if w not in self.vocabulary:
                    self.vocabulary.append(w)
                    #print(self.vocabulary)
                    #self.vocabulary_size+=1
        self.vocabulary_size=len(self.vocabulary)
        #print(self.vocabulary_size, len(self.vocabulary))


    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        # ############################
        # your code here
        # ############################
        
        def countmatrix(l, w):
            cnt=0
            for i in l:
                #print(l)
                if i==w:
                    cnt+=1
            return cnt

        #self.term_doc_matrix = np.zeros([self.number_of_documents, self.vocabulary_size], dtype=np.float)
        rows, cols = (self.number_of_documents, self.vocabulary_size) 
        self.term_doc_matrix = [[0]*cols]*rows

        for i in range(self.number_of_documents):
            for j in range(self.vocabulary_size):
                #print(i,j)
                #print(self.documents[i],self.vocabulary[j])
                #print(self.vocabulary[j])
                #print(self.term_doc_matrix[i][j])
                self.term_doc_matrix[i][j] = countmatrix(self.documents[i],self.vocabulary[j])
                


    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize! 
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        # ############################
        # your code here
        # ############################
        self.document_topic_prob=np.random.random((self.number_of_documents, number_of_topics))
        self.document_topic_prob=normalize(self.document_topic_prob)
        self.topic_word_prob=np.random.random((number_of_topics,self.vocabulary_size))
        self.topic_word_prob=normalize(self.topic_word_prob)

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """
        print("E step:")
        #self.topic_prob = [[[0]*self.number_of_documents]*self.vocabulary_size]*number_of_topics
        number_of_topics=self.topic_prob.shape[1]
        for d in range(self.number_of_documents):
            for w in range(self.vocabulary_size):
                cnt=0
                for z in range(number_of_topics):
                    #den=sum(document_topic_prob[d][z]*topic_word_prob[z][w])
                    #if (d==6):
                        #print("iteration: ", z, self.vocabulary[w])
                        #print(self.document_topic_prob[d][z], self.topic_word_prob[z][w])
                    self.topic_prob[d][z][w]=(self.document_topic_prob[d][z]*self.topic_word_prob[z][w])
                    cnt+=self.topic_prob[d][z][w]
                #if cnt == 0:
                    #for z in range(number_of_topics):
                        #self.topic_prob[d][z][w] = 0
                # if cnt == 0:
                    #print("cnt")
                    #print(cnt)
                #self.topic_prob[d,:,w] = self.topic_prob[d,:,w]/(cnt if cnt > 0 else 0.000001)
                #else:
                    #self.topic_prob[d][z][w] = self.topic_prob[d][z][w]/cnt
                if cnt>0:
                    self.topic_prob[d,:,w] = self.topic_prob[d,:,w]/cnt
        #print(self.topic_prob)
                    

        #topic_prob = (document_topic_prob*topic_word_prob)/sum()
        # ############################
        # your code here
        # ############################

        #pass    # REMOVE THIS
            

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
        print("M step:")
        
        # update P(w | z)
        
        # ############################
        # your code here
        # ############################

        for z in range(number_of_topics):
            for w in range(self.vocabulary_size):
                cnt=0
                for d in range(self.number_of_documents):
                    #self.document_topic_prob[d][z]=self.topic_prob[d][z][w]*self.term_doc_matrix[d][w]
                    cnt+=self.topic_prob[d][z][w]*self.term_doc_matrix[d][w]
                self.topic_word_prob[z][w]=cnt
        self.topic_word_prob=normalize(self.topic_word_prob)
        #print(self.topic_word_prob)

        # update P(z | d)

        # ############################
        # your code here
        # ############################
        
        #self.document_topic_prob

        for d in range(self.number_of_documents):
            for z in range(number_of_topics):
                cnt=0
                for w in range(self.vocabulary_size):
                    #self.document_topic_prob[d][z]=self.topic_prob[d][z][w]*self.term_doc_matrix[d][w]
                    cnt+=self.topic_prob[d][z][w]*self.term_doc_matrix[d][w]
                self.document_topic_prob[d][z]=cnt
            #print(d)
        self.document_topic_prob=normalize(self.document_topic_prob)
        #print(self.document_topic_prob)
        #print(self.document_topic_prob)
        #np.savetxt('doc_topic_prob2.txt',self.document_topic_prob)
        #print(self.document_topic_prob)
        #print("**")

        #pass    # REMOVE THIS


    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        # ############################
        # your code here
        # ############################
        a=0
        for d in range(self.number_of_documents):
            for w in range(self.vocabulary_size):
                cnt=0
                for z in range(number_of_topics):
                    #self.likelihoods=self.term_doc_matrix
                    cnt+=self.document_topic_prob[d][z]*self.topic_word_prob[z][w]
                # print("cnt")
                # print(cnt)
                # print("\n")
                lcnt=log(cnt if cnt > 0 else 0.000001)
                a+=self.term_doc_matrix[d][w]*lcnt
        self.likelihoods.append(a)
        return a

    def plsa(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        print ("EM iteration begins...")
        
        # build term-doc matrix
        self.build_term_doc_matrix()
        
        # Create the counter arrays.
        
        # P(z | d, w)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0
        new_likelihood=0.0

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")
            self.expectation_step()
            self.maximization_step(number_of_topics)
            new_likelihood=self.calculate_likelihood(number_of_topics)
            print(new_likelihood)
            delta=new_likelihood-current_likelihood
            if (current_likelihood!=0.0 and abs(delta)<epsilon):
                break
            current_likelihood=new_likelihood

            # ############################
            # your code here
            # ############################

            #pass    # REMOVE THIS



def main():
    documents_path = 'data/DBLP.txt'
    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    print(corpus.vocabulary)
    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))
    number_of_topics = 2
    max_iterations = 50
    epsilon = 0.001
    corpus.plsa(number_of_topics, max_iterations, epsilon)



if __name__ == '__main__':
    main()
