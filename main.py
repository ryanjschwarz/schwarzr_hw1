from lib2to3.pgen2.tokenize import tokenize
#from re import L
import pandas as pd
import numpy as np
import string

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer




#nltk.download()

#def Init_weights(embedding):
#    print('returns random weights in initial model')

def K_fold(data):
    #data should be a dataframe
    num_folds = 10 #for 10-fold validation
    num_samples = data.shape[0] #should be 1200 for training data
    fold_size = int(num_samples/num_folds) #should be 120 for training data
    num_features = 100 #play with this number
    
   # print('fold size: ' + str(fold_size))
   # print('num folds: ' + str(num_folds))
   # print('num samples: ' + str(num_samples))

    folds = []
    #start = 0
    #for i in range (start, start + num_samples/num_folds+1):
    start = 0
    end = fold_size
    
    for i in range(num_folds):
        fold = []
        for j in range(start, end):
            fold.append(data.loc[j])
            start = start+1
       
        fold = pd.DataFrame(fold)
        folds.append(fold)
        end = end + fold_size

    #returns a list of k dataframes for k-fold validation
    return folds

#def Softmax(n):
#    #basic softmax formulation using numpy
#    return (np.exp(n) / sum(np.exp(n)))

#def Sigmoid(n):
    #basic sigmoid formulation using numpy
    #return (1 / (1+ np.exp(-n)))



def Clean(n):
    #Lower case the text
    n = n.lower()

    #remove punctuation
    n = "".join([char for char in n if char not in string.punctuation])

    #tokenize cleaned data
    words = word_tokenize(n)
    
    #remove stop words
    stop_words = stopwords.words('english')
    filtered_words = [word for word in words if word not in stop_words]

    #stem words
    porter=PorterStemmer()
    stemmed_words = [porter.stem(word) for word in filtered_words]
    stemmed_words = " ".join([word for word in stemmed_words])
    #print(stemmed_words)
    #print(type(stemmed_words))
    return stemmed_words

def Clean_df(data):
    clean_data = [Clean(line) for line in data['text']]
    return clean_data 

def preprocess_test(X, final_tokens):
 
    res = np.zeros((X.shape[0], len(final_tokens)))

    for example,sentence in enumerate(X):
        token = word_tokenize(Clean(sentence))
        for word in token:
            if word in final_tokens.keys():
                #print('found word: ', word)
                res[example][final_tokens[word]]+=1
    return res


def Tokenize(data):
    #this represents a bag of stemmed words from the training data with each input given an index
    #print(type(data))
    bow = ""
    for text in data:
        bow = bow + text + " "

    tokens = set(word_tokenize(bow))

    final_tokens = {}

    #dirty way to assign an index to each token
    index = 0
    for token in tokens:
        final_tokens[token] = index
        index = index + 1
    
    #print(index)
    #print(final_tokens)

    res = np.zeros((len(data), len(final_tokens)))

    print(len(data))
    #for example,sentence in enumerate(data):
        #tok = word_tokenize(sentence)
        #for word in tok:
        #    if word in final_tokens.keys():
        #        res[example][final_tokens[word]] += 1
    
    #Keeps track of each word present in each sample from the greater BoW
    for i in range(0, len(data)):
        token = word_tokenize(data[i])
        for word in token:
            if word in final_tokens.keys():
                res[i][final_tokens[word]] += 1

    #print (final_tokens)
    
    #res represents each of the 1200 samples encoded with which words from the greater BoW
    #final_tokens represents the greater BoW from the entire dataset + their indices
    return res,final_tokens

#provides the mapping of emotioner -> corresponding values
#i.e. anger : 0, fear: 1
def label_encoding(Y,df):
    label_enc = {}
    for i,label in enumerate(np.unique(df['emotions'].values)):
        label_enc[label] = i
    #print(label_enc)
    return label_enc

#provides a one-hot encoding representation for the emotions in each row
#I.e. [1,0,0,0,0,0] for anger
def OHE(Y, n):
    #Y represents the number of samples and n is the possible number of classes (6 in this case)
    #print('n------->',n)
    #encodes all the emotion values of the data into the OHE representation
    y_ohe = np.zeros((Y.shape[0],n))
    for i,label in enumerate(Y):
        #anger = 0 = [1,0,0,0,0,0]
        y_ohe[i][label] = 1

    return y_ohe



#preprocess the test set

def LR():
    #added a random seed for simple
    #np.random.seed(12)#change this
    
    #test represents whether to use k-fold or not (disabled for submission)
    #test = True
    test = False
    #test = 'old'


    df = pd.read_csv('train.csv')
    data = df[['text', 'emotions']].values
    #np.random.shuffle(data)
    print(len(data))
    
    X = data[:,0]
    Y = data[:,1]
    #print("Y<<<<<<<<<<<<<<<", Y)

    X,final_tokens = Tokenize(Clean_df(df))
    label_enc = label_encoding(Y, df)
    X = np.hstack((X,np.ones((X.shape[0],1))))
    
    #hyperparameters - play around with these
    INSTANCES = X.shape[0] #num of samples
    FEATURES = X.shape[1] #num of tokens
    NUM_CLASSES = len(label_enc) #num of potential outcomes (6)
    LEARNING_RATE = 0.05
    BATCH_SIZE = 1 
    ITERATIONS = 100
    #ITERATIONS = 3 #for testing

    #encodes all of the answers in the emotions column
    for i,cls in enumerate(Y):
        Y[i] = label_enc[cls]
  

    w = np.random.random_sample((FEATURES, NUM_CLASSES )) #start off with random weights
    Y_OHE = OHE(Y, NUM_CLASSES)

    #helper functions
  
    def loss(X,W,Y): #Loss 
        Z = np.exp(np.matmul(X,W))
        O = np.divide(Z, np.sum(Z, axis=1).reshape(-1,1))
        return -np.sum(np.multiply(Y, np.log(O)))/O.shape[0]

    def accuracy(X,W,Y):
        Y_pred = predict(X,W)
        Y = Y.reshape(-1,1)
        #returns the number of correct predictions
        return sum(Y_pred == Y)/X.shape[0]
    

    def predict(X,W):
        Z = np.exp(np.matmul(X,W))
        prob = np.divide(Z, np.sum(Z, axis=1).reshape(-1,1))
        return np.argmax(prob, axis=1).reshape(-1,1)
    #def f1

   

    def gradient(X, W, Y_OHE, reg):
        Z = np.exp(np.matmul(X,W))
        prob = np.divide(Z, np.sum(Z, axis=1).reshape(-1,1))
        grad = np.subtract(np.matmul(X.T, prob), np.matmul(X.T, Y_OHE))/X.shape[0]

        return grad
    
    if not test: #If not using K-fold validation
        for i in range(ITERATIONS):
            print("Iteration: " + str(i))
            curr_batch = 0
            count = 0
            while curr_batch < INSTANCES:
                X_batch = X[curr_batch: min(INSTANCES, curr_batch+BATCH_SIZE)]
                Y_batch = Y_OHE[curr_batch: min(INSTANCES, curr_batch+BATCH_SIZE)]
                w-=LEARNING_RATE*gradient(X_batch, w, Y_batch, 0.001)
                curr_batch += BATCH_SIZE
                count += 1
            #print(X)
            #print(w)
            #print(Y)
            print(accuracy(X,w,Y))
           
        #print(Y.shape())
        #print("ACCURACY: " + str(loss(X,w,Y))) 
        
        #This is for the test data
        test_df = pd.read_csv('test.csv')
        X_test = test_df['text'].values
        X_test = preprocess_test(X_test, final_tokens)

        X_test = np.hstack((X_test,np.ones((X_test.shape[0],1))))
        y_preds = predict(X_test, w) 
        #print('Y PREDS____________', y_preds)
        preds = []
        for y_p in y_preds:
            if y_p == 0: preds.append("anger")
            elif y_p == 1: preds.append("fear")
            elif y_p == 2: preds.append("joy")
            elif y_p == 3: preds.append("love")
            elif y_p == 4: preds.append("sadness")
            elif y_p == 5: preds.append("surprise")

        test_df['emotions'] = preds
        test_df.to_csv('test_lg.csv', index=False)
        
    """""
    #old way of doing CV
    elif test == 'old':
        CV = 5
        units = int(INSTANCES/CV)
        X_cv = []
        Y_cv = []
        for i in range(CV):
#             print("s: " + str(i*units) + "e: " + str((i+1)*units))
            #X_CV is each of the words present in each sentence
            X_cv.append(X[i*units:(i+1)*units])
            Y_cv.append(Y[i*units:(i+1)*units])


        #print(len(X_cv))
        #print(len(X_cv[0]))
        #print(X_cv.shape)
        #print(Y_cv)
        #print(Y_cv.shape)
        print("Y ----------------", Y)
        #print(Y_cv[0].shape, Y_cv[2].shape)
        best_params = ""
        minLoss = INSTANCES
#         bow_params = {"LEARNING_RATE": [0.1, 0.2, 0.05, 0.01, 0.005], "REG_RATE": [0.0005, 0.005, 0.001, 0.05, 0.01, 0.1], "BATCH_SIZE": [1,10,100,1000], "ITERATIONS": [10,20,30]}
        bow_params = {"LEARNING_RATE": [0.5], "REG_RATE": [0.1], "BATCH_SIZE": [1], "ITERATIONS": [100]}
        for iterations in bow_params['ITERATIONS']:
            for bat_size in bow_params['BATCH_SIZE']:
                for reg_lambda in bow_params['REG_RATE']:
                    for learn_rate in bow_params['LEARNING_RATE']:
                        # cross validation
                        train_scores = []
                        test_scores = []
                        loss_value = []
                        for test_id in range(CV):
                            #print("#CV: ", test_id)
                            w = np.random.random_sample((FEATURES,NUM_CLASSES)) # (features,CLASSES)
                            X_train = np.empty((0,X.shape[1]))
                            Y_train = np.empty((0,))
                            X_test = X_cv[test_id]
                            Y_test = Y_cv[test_id]
                            #print("TRAIN 1-----------------")
                            #print(X_train)
                            #print(Y_train)
                            print("TRAIN 1-----------------")
                            for train_id in range(CV):
                                if train_id != test_id:
                                    #print(Y_cv)
                                    X_train = np.append(X_train,X_cv[train_id], axis=0)
                                    Y_train = np.append(Y_train, Y_cv[train_id], axis=0)
#                             X_train = np.array(X_train)
#                             Y_train = np.array(Y_train)
                            print("TRAIN 2-----------------")
                            #print(X_train)
                            #print(Y_train)
                            #print("TRAIN 2-----------------")
                            #print(X_train.shape)
                            #print(Y_train.shape)
#                             print(Y_train)
                            for i in range(iterations):
                                curr_bat = 0
                                ct = 0
                                while curr_bat < X_train.shape[0]:
                                    X_BAT = X_train[curr_bat: min(X_train.shape[0],curr_bat+bat_size)]
                                    Y_BAT = OHE(Y_train[curr_bat: min(X_train.shape[0],curr_bat+bat_size)], NUM_CLASSES)
                                    w -= learn_rate*gradient(X_BAT, w, Y_BAT, reg_lambda)
                                    ct += 1
                                    curr_bat += bat_size
                                #print("batch runs: ", ct)
                            loss_score = loss(X_test, w, OHE(Y_test, NUM_CLASSES))
                            train_acc = accuracy(X_train, w, Y_train)
                            test_acc = accuracy(X_test, w, Y_test)
                            print("Train Accuracy: ", train_acc)
                            print("Test Accuracy: ", test_acc)
                            print("Train Loss: ", loss(X_train, w, OHE(Y_train, NUM_CLASSES)))
                            print("Test Loss: ", loss_score)
                            train_scores.append(train_acc)
                            test_scores.append(test_acc)
                            loss_value.append(loss_score)
                        print("***************")
                        print("Params: ITER " + str(iterations) + " bat_siz: " + str(bat_size) + " reg_rate: " + str(reg_lambda) + " LR: " + str(learn_rate))
                        print("Train average: ", sum(train_scores)/len(train_scores))
                        print("Test Average: ", sum(test_scores)/len(test_scores))
                        print("Loss Average: ", sum(loss_value)/len(loss_value))
                        
                        if (sum(loss_value)/len(loss_value)) < minLoss:
                            minLoss = sum(loss_value)/len(loss_value)
                            best_params = "Params: ITER " + str(iterations) + " bat_siz: " + str(bat_size) + " reg_rate: " + str(reg_lambda) + " LR: " + str(learn_rate)
                            print("New best found")
                        print("***************")
                        print()
        """

def NN():
    #np.random.seed(12)
    #test = True
    test = False
   
    #OUTPUT = 6
    
    df = pd.read_csv('train.csv')
    data = df[['text','emotions']].values

    #np.random.shuffle(data)
    X = data[:,0]
    Y = data[:,1]

    X,final_tokens = Tokenize(Clean_df(df))
    label_enc = label_encoding(Y, df)
    #label_enc = label_enc(Y, df)

    #hyperparameters based on results from testing phase
    HIDDEN = 32
    CLASSES = len(label_enc)
    LEARNING_RATE_HIDDEN = 0.01
    LEARNING_RATE_OUTPUT = 0.01
    BATCH_SIZE = 100
    ITERATIONS = 50
    FEATURES = X.shape[1]
    INSTANCES = X.shape[0]
    REG1 = 0.01
    REG2 = 0.01 #generic regularizer terms, they affected testing very little
    
    #dirty way to assign index to each label
    for i,cls in enumerate(Y):
        Y[i] = label_enc[cls]
        
    #2781 tokens on the training set.
    #print(len(final_tokens))
    #print(final_tokens.shape)
    Y_OHE = OHE(Y, CLASSES)

    
    #print(X.shape)
    #print(Y_OHE.shape)
    
  

    #random weights
    #weights between input and hidden layer
    W1 = np.random.randn(FEATURES,HIDDEN)
    B1 = np.random.randn(1,HIDDEN) # (1,HIDDEN)

    #weights between hidden and output
    W2 = np.random.randn(HIDDEN, CLASSES) # (HIDDEN,CLASSES)
    B2 = np.random.randn(1,CLASSES) # (1,CLASSES)
    
    #ReLU layer followed by a softmax layer
    act_func_one = 'relu'
    act_func_two = 'softmax'
    
   
    def get_z(X,W,B): return (np.matmul(X,W) + B)
    
    def relu(X,W,B): return np.maximum(0,get_z(X,W,B))
        
    def softmax(X,W,B):
        Z = np.exp(get_z(X,W,B))
        return np.divide(Z, np.sum(Z, axis=1).reshape(-1,1))

    def sigmoid(X,W,B): return 1/(1 + np.exp(-(get_z(X,W,B))))
        
    def activation(a, X,W,B):
        if a == 'sigmoid': return sigmoid(X,W,B) #sigmoid performed worse on most tests
        elif a == 'softmax': return softmax(X,W,B)   
        elif a == 'relu': return relu(X,W,B)

    def der_weight(loss_func, X, W,B):
        if loss_func == 'relu':
            rel = relu(X,W,B)
            rel[rel>0] = 1
            return rel
            
        elif loss_func == 'sigmoid':
            sig = sigmoid(X,W,B)
            return np.multiply(sig,(1-sig))
        
    def predict(O): return np.argmax(O, axis=1).reshape(-1,1)        
    
    def accuracy(O, Y):
        Y_pred = predict(O)
        Y = Y.reshape(-1,1)
        return sum(Y_pred == Y)/Y.shape[0]
    
    def loss(O,Y): return -np.sum(np.multiply(Y, np.log(O)))/O.shape[0]
        
    
    def feed_forward(X,W1,B1,W2,B2, activation1):
        
        # HIDDEN LAYER
        A1 = activation(activation1, X,W1,B1)
        O = activation(act_func_two, A1, W2, B2)
        
        return A1,O
        
        
    def backPropogation(X,A1,O,W1,B1,W2,B2,Y_OHE, lr_hidden, lr_output, reg1, reg2, activ1):
        # loss function: negaetive log loss + reg1*w1^2/2 + reg2*w2^2/2
        lossDeriv = np.subtract(O, Y_OHE)/ A1.shape[0] # derivative of the loss function with respect ot z2 (shape: (instances, classes))
        temp = W2
        gradW2 = np.matmul(A1.T, lossDeriv) + reg2*W2 # derivative of loss wrt to W2 -> (hidden, instances)*(instances, classes) => (hidden, classes)
        gradB2 = np.matmul(np.ones((1,A1.shape[0])), lossDeriv) # derivative of loss wrt to B2 -> (1, instances)*(instances, classes) => (1, classes)
        
        
        
        gradW1 = np.matmul(X.T, np.multiply(np.matmul(lossDeriv,W2.T), der_weight(activ1, X,W1,B1))) + reg1*W1 # derivative of loss wrt to W1 -> (features, instances)[(instances, classes)(classes,hidden)x(instances,hidden)] => (features, hidden)
        gradB1 = np.matmul(np.ones((1,A1.shape[0])), np.multiply(np.matmul(lossDeriv,W2.T), der_weight(activ1, X,W1,B1))) # derivative of loss wrt to B2 -> (1, instances)[(instances, classes)(classes,hidden)x(instances,hidden)] => (1, hidden)
        
#         print(gradW2.shape, gradW1.shape, gradB1.shape, gradB2.shape)
        
        W2 -= lr_output*gradW2
        B2 -= lr_output*gradB2
        
        W1 -= lr_hidden*gradW1
        B1 -= lr_hidden*gradB1
        return W1,B1,W2,B2
    
    
    if not test:
        print("Acc: ")
    #     print(accuracy(O,Y))
        
        for i in range(ITERATIONS):
            print("Iteration: ", i)
            curr_bat = 0
            ct = 0
            while curr_bat < INSTANCES:
#                 print(str(curr_bat) + " " + str(curr_bat+BATCH_SIZE))
                X_BAT = X[curr_bat: curr_bat+BATCH_SIZE]
                Y_BAT = Y_OHE[curr_bat: curr_bat+BATCH_SIZE]
#                 print(X_BAT.shape)
                A1,O = feed_forward(X_BAT,W1,B1,W2,B2, act_func_one)
                W1,B1,W2,B2 = backPropogation(X_BAT,A1,O,W1,B1,W2,B2,Y_BAT, LEARNING_RATE_HIDDEN, LEARNING_RATE_OUTPUT, REG1, REG2,act_func_one)
                curr_bat += BATCH_SIZE
                ct += 1
            # print("No of batch runs: ", ct)
            A1,O = feed_forward(X,W1,B1,W2,B2, act_func_one)
            print("loss: ", loss(O,Y_OHE))
            # print(accuracy(O,Y))
            
        # testing
        test_df = pd.read_csv('test.csv')
        X_test = test_df['text'].values
        X_test = preprocess_test(X_test, final_tokens)
        A1_test,O_test = feed_forward(X_test,W1,B1,W2,B2, act_func_one)
        y_pred = predict(O_test)
        
        preds = []
        for y_p in y_pred:
            if y_p == 0: preds.append("anger")     
            elif y_p == 1: preds.append("fear")        
            elif y_p == 2: preds.append("joy")            
            elif y_p == 3: preds.append("love")          
            elif y_p == 4: preds.append("sadness") 
            else: preds.append("surprise")
        
        print(len(preds))
        test_df['emotions'] = preds
        
        test_df.to_csv('test_nn.csv', index=False)
    """""
    else:
        CV = 10
        units = int(INSTANCES/CV)
        X_cv = []
        Y_cv = []
        for i in range(CV):
            X_cv.append(X[i*units:(i+1)*units])
            Y_cv.append(Y[i*units:(i+1)*units])
        
        print(Y_cv[0].shape, Y_cv[2].shape)
        
     
        
        nn_params = {"LEARNING_RATE_OUTPUT": [0.01],"LEARNING_RATE_HIDDEN": [0.1], "REG_RATE_OUTPUT": [0.1],
                    "REG_RATE_HIDDEN": [0.5], "BATCH_SIZE": [100], "ITERATIONS": [50],
                    "act_func_one": ['relu'], "HIDDEN_LAYERS" : [32]}
#  
        for iterations in nn_params['ITERATIONS']:
            print("===============================================")
            print()
            for activ1 in nn_params['act_func_one']:
                for bat_size in nn_params['BATCH_SIZE']:
                    for reg_rate_hidden in nn_params['REG_RATE_HIDDEN']:
                        for reg_rate_output in nn_params['REG_RATE_OUTPUT']:
                            for lr_hidden in nn_params['LEARNING_RATE_HIDDEN']:
                                for lr_op in nn_params['LEARNING_RATE_OUTPUT']:
                                    for hidden in nn_params['HIDDEN_LAYERS']:
                                        train_scores = []
                                        test_scores = []
                                        loss_value = [] 
                                        for test_id in range(CV):
                                            print("#CV: ", test_id)
                                            W1 = np.random.randn(FEATURES,hidden) # (features,HIDDEN)
                                            B1 = np.random.randn(1,hidden) # (1,HIDDEN)
                                            W2 = np.random.randn(hidden, CLASSES) # (HIDDEN,CLASSES)
                                            B2 = np.random.randn(1,CLASSES) # (1,CLASSES)

                                            X_train = np.empty((0,X.shape[1]))
                                            Y_train = np.empty((0,))
                                            X_test = X_cv[test_id]
                                            Y_test = Y_cv[test_id]

                                            for train_id in range(CV):
                                                if train_id != test_id:
                                                    X_train = np.append(X_train,X_cv[train_id], axis=0)
                                                    Y_train = np.append(Y_train, Y_cv[train_id], axis=0)
                                            
                                            print(X_train.shape)
                                            print(Y_train.shape)
                                            print(X_test.shape)
                                            for i in range(iterations):
                                                curr_bat = 0
                                                ct = 0
                                                while curr_bat < X_train.shape[0]:
                                                    X_BAT = X_train[curr_bat: min(X_train.shape[0],curr_bat+bat_size)]
                                                    Y_BAT = OHE(Y_train[curr_bat: min(X_train.shape[0],curr_bat+bat_size)], CLASSES)
                                                    A1,O = feed_forward(X_BAT,W1,B1,W2,B2, activ1)
                                                    W1,B1,W2,B2 = backPropogation(X_BAT,A1,O,W1,B1,W2,B2,Y_BAT, lr_hidden, lr_op, reg_rate_hidden, reg_rate_output, activ1)
                                                    curr_bat += bat_size
                                                    ct += 1
#                                                 print("batch runs: ", ct)
                                            
                                            A1_train, O_train = feed_forward(X_train, W1,B1,W2,B2,activ1)
                                            A1_test, O_test = feed_forward(X_test, W1, B1, W2, B2, activ1)
                                            loss_score = loss(O_test, OHE(Y_test, CLASSES))
                                            train_acc = accuracy(O_train, Y_train)
                                            test_acc = accuracy(O_test, Y_test)
                                            print("Train Accuracy: ", train_acc)
                                            print("Test Accuracy: ", test_acc)
                                            print("Train Loss: ", loss(O_train, OHE(Y_train, CLASSES)))
                                            print("Test Loss: ", loss_score)
                                            train_scores.append(train_acc)
                                            test_scores.append(test_acc)
                                            loss_value.append(loss_score)
                                        print("***************")
                                        print("Params: ITER " + str(iterations) + " bat_siz: " + str(bat_size) + " reg_rate op: " + str(reg_rate_output) + " reg_rate hd: " + str(reg_rate_hidden) + " LR hid: " + str(lr_hidden)+ " LR hid: " + str(lr_op)+ " activation: " + activ1 + " Hidden layers: " + str(hidden))
                                        print("Train average: ", sum(train_scores)/len(train_scores))
                                        print("Test Average: ", sum(test_scores)/len(test_scores))
                                        print("Loss Average: ", sum(loss_value)/len(loss_value))
                                        print("ALL VALUES", train_scores)
    """    
                                        
if __name__ == '__main__':
    print ("..................Beginning of Logistic Regression................")
    LR()
    
    print ("..................End of Logistic Regression................")

    print("------------------------------------------------")

    print ("..................Beginning of Neural Network................")
    NN()
    print ("..................End of Neural Network................")