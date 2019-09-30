# Timeseries of comment volume extraction, event detection and extraction of comments for classification
##################################################################################################
def tscc(timeseries):
    '''
    A function which outputs 
    a timeseries of comment volume from 
    a list of timestamps,
    is called by 'timesseries_day'
    '''
    count = 0
    l1 = []
    l2 = []
    for i in timeseries:
        count += 1
        l1.append(i)
        if count == 1:
            small = i   # smaller of dates used to observe if day has passed
        if i - small >= 24*3600:     # 24*3600 is a day, if the difference between two timestamps is this then there is a days difference between the comments
            l2.append(len(l1))   #len(l1) gives the number of comments in a day
            count = 0    #set count back to 0, to initilise new commennts being smaller of dates used to observe if day has passed
            l1 = []    #allows for new volume of comments to be observed in day
    return l2

def timeseries_day(cat):    # subreddit data was broken into multiple files as was too large to load into a dataframe
    '''
    A function with input of a list of files
    which returns a daily timeseries of 
    comment volume
    '''
    catlist = []
    for file in cat:
        print('-----------')
        df = pd.read_csv(file)
        timeseries = df['created_utc'].values    #extract list containing timestamps of comments
        catlist += tscc(timeseries)             #call function tscc to combine timeseries of comment volume for each file - files are in                                                    time order
             
    return catlist

def event_det(comlist, window = 5, diff = 2, T = False):   
    '''
    A function with input of a daily timeseries
    of comment volume
    which returns the days (ie day 1 or 2) that corresponds to an event - 
    or position in timeseries of comment volume which is event day
    '''
    day_of_event = []
    event_numbers = []
    dic = {i:True for i in range(2*window + 1)}   # initialise sliding window
    for num, day in enumerate(comlist):
        for i in range(2*window + 1):
            dic[i] = dic.get(i+1, day)   #gives each value in dictionary the next days - except the last day where .get defaults to d
        med = [i for i in dic.values()]  # get values in window
        if True in list(dic.values()):    #skip first 10 volumes
            continue # first time this  doesnt happen we are at 11 th day  in timeseries
        median = np.median(med)  # median of values in window 
        calc = (day - median)/max([median, 1000])  # if the days comment volume is substantially ofset from the median, it is classed as event day
        if calc > diff:
            day_of_event.append(num + 1)
            event_numbers.append(day)
    
    if T is False:
        return day_of_event
    if T is True:
        return (day_of_event, event_numbers)
    
def comments_for_classif(filetype, eventdays):
    '''
    A function with input of all subreddit files and
    days of events
    - outputs a list of lists
    where each inner list is all comments from an eventday.
    Also outputs the readable dates of event days
    '''
    lists_of_comments = []
    dates = []
    day = 1  #**
    for file in filetype:
        print('files')
        count = 0    #events are found by considering position in daily timeseries
        comments = []  
        df = pd.read_csv(file)
        z1 = list(df['created_utc'])
        z2 = list(df['body'])
        
        for time, text in zip(z1, z2):   #loop through timeries and comments for forum file 
            count += 1
            if count == 1:
                small = time
            comments.append(text)    #all comments for a day , refreshed to [] once day has passed
            
            if time - small >= 24*3600:  # if a day has passed:
                if day in eventdays:   # and the day in timeseries given by ** - is in eventdays - can add all comments for a day to list of comments which is later returned 
                    lists_of_comments.append(comments)
                    x = datetime.utcfromtimestamp(small).strftime('%Y-%m-%d %H:%M:%S')  #extracting dates
                    y = x.split()
                    dates.append(y[0])
                count = 0
                comments = []
                day = day + 1  #**
                
    return (lists_of_comments, dates)  


# Functions used for sentiment classification
##############################################################################################################################

def labelsC(alist):   #functions counts the number of positive or negative labels returns ratio
    p = 0
    n = 0
    for i in alist:
        if i == 'P':
            p += 1
        elif i == 'N':
            n += 1
    try:
        ret = p/n
    except:
        ret = max(p, n)
        print(p, n)
    return ret

def classify(string, choice):
    if type(string) == float:
            string = str(string)
    if choice == 'sup':
        doc = {w.lower():True for w in string.split()}
        return classifier.classify(doc)   #bayes
    
    else:
        return demo(string)  #unsupervised classifer
    
def comment_classifier(loc, classifier = 'sup'):
    '''
    A function that inputs list of comments ie. timeseries of comments 
    given by event days
    and returns sentiment ratios timeseries
    '''
    alist = []
    for daycoms in loc: # loop through each day contains all commments for a day- day is event day
        for_labelsC = []
        if classifier != 'sup':
            print('day....')
        print(alist)
        for comment in daycoms: #consider each comment in eventday
            if classifier == 'sup':
                for_labelsC.append(classify(comment, 'sup'))  #add classification which is to be counted by function labels C
            else:
                for_labelsC.append(classify(comment, 'anything'))
        alist.append(labelsC(for_labelsC))   # adds sentiment ratio to list to be returned which will contian all sentiment ratios
    
    return alist


def demo(sentence, plot=False):   #unsupervised classifier from nltk adjusted to allow for binary classification
    """
    Basic example of sentiment classification using Liu and Hu opinion lexicon.
    This function simply counts the number of positive, negative and neutral words
    in the sentence and classifies it depending on which polarity is more represented.
    Words that do not appear in the lexicon are considered as neutral.

    :param sentence: a sentence whose polarity has to be classified.
    :param plot: if True, plot a visual representation of the sentence polarity.
    
    """
    from nltk.corpus import opinion_lexicon
    from nltk.tokenize import treebank

    tokenizer = treebank.TreebankWordTokenizer()
    pos_words = 0
    neg_words = 0
    tokenized_sent = [word.lower() for word in tokenizer.tokenize(sentence)]

    x = list(range(len(tokenized_sent)))  # x axis for the plot
    y = []

    for word in tokenized_sent:
        if word in opinion_lexicon.positive():
            pos_words += 1
            y.append(1)  # positive
        elif word in opinion_lexicon.negative():
            neg_words += 1
            y.append(-1)  # negative
        else:
            y.append(0)  # neutral

    if pos_words > neg_words:
        return 'P'
    elif pos_words < neg_words:
        return 'N'
    elif pos_words == neg_words:
        #print('neutral')
        return 'Neutral'
        #return random.choice(['P', 'N'])

    if plot == True:
        _show_plot(
            x, y, x_labels=tokenized_sent, y_labels=['Negative', 'Neutral', 'Positive']
        )
        
class NBClassifier():   #Bayes classifier 
    
    def __init__(self, priors={}, conditional_probs={}, known_vocab={}): 
        self._priors = priors
        self._c_probs = conditional_probs
        self._vocab = known_vocab
    
    def train(self, training_data):
        self._priors = class_priors(training_data)
        self._c_probs = cond_probs(training_data)
        self._vocab = known_vocabulary(training_data)
                                
    def classify(self, doc): 
        label_logprobs = {}
        highest_logprob = 1
        #Filter out the words that aren't in the known vocabulary
        doc = {w:True for w in doc if w in self._vocab}
        for label in self._priors.keys():
            prior = self._priors[label]
            probs_given_label = self._c_probs[label]
            label_logprobs[label] = reduce(lambda logprob,word: logprob + log(probs_given_label.get(word,0)), doc, log(prior))        
            if highest_logprob == 1 or label_logprobs[label] > highest_logprob:
                highest_logprob = label_logprobs[label]
        
        check = []
        for label in self._priors.keys():
            check.append(label)
            
        if label_logprobs[check[0]] == label_logprobs[check[1]]:
            print('happened')
            #return 'Neutral'
            return random.choice([check[0], check[1]])
            
        top_labels = [label for label in label_logprobs if label_logprobs[label] == highest_logprob]
        return(choice(top_labels))
    

#Functions used in testing sentiment classifiers
#################################################################################################################################

def comments_for_labels(loc):
    '''
    A functions which
    extracts sample of comments
    for each day in timeseries of comments
    '''
    alist = []
    for i in loc:
        alist2 = []
        x = random.sample(i, 1000)
        for j in x:
            alist2.append([j])
        alist.append(alist2)
        
    return alist

def self_classify(listoflist):     # went through each day separately in 15 tech events , storeing the output list separately
    '''
    A function used to 
    hand label comments, inputs
    sample of individiual day
    comments
    returns labelled comments
    '''
    count = 0                         
                                    # ie  comsfl = comments_for_labels(comst15) X1 = self_classify(comsfl[14])
    alist = []
    for x in listoflist:
        print(x)
        y = input()   # person inputs classification
        if y == 'p':
            count += 1
            x.append(y)
            alist.append(x)
        if y == 'n':
            count += 1
            x.append(y)
            alist.append(x)
        
        if count >= 50:    # if  50 positive and negative comments, all completed for day in timeseries
            break
        
    return alist   

def labels_counter(alldays):
    '''
    A function used ot combine
    all hand labelled comments
    creating hand labelled sentiment ratio
    timeseries
    '''
    ratios = []                           # example using this : y1 = labels_counter([X1, X2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15])
    for i in alldays:
        countp = 0
        countn = 0
        for j in i:
            if j[1] == 'p':
                countp += 1
            elif j[1] == 'n':
                countn += 1
            else:
                print('mis label')
        ratios.append(countp/countn)
        
    return ratios    # output plot against

class ConfusionMatrix:   # confusion matrix given by applied natural language module
    def __init__(self,predictions,goldstandard,classes=("P","N")):
        self.len = len(predictions)
        (self.c1,self.c2)=classes
        self.TP=0
        self.FP=0
        self.FN=0
        self.TN=0
        self.correct=0
        for p,g in zip(predictions,goldstandard):
            if g == p:
                self.correct += 1
              
            if g==self.c1:
                if p==self.c1:
                    self.TP+=1
                else:
                    self.FN+=1
            
            elif p==self.c1:
                self.FP+=1
            else:
                self.TN+=1
        
    def accuracy(self):
        acc = self.correct/self.len
        return acc
    
    
    def precision(self):
        p=0
        p = self.TP/(self.TP + self.FP)
        return p
    
    def recall(self):
        r=0
        r = self.TP/(self.TP + self.FN)
        return r
    
    def f1(self):
        f1=0
        p = self.precision()
        r = self.recall()
        f1 = (2 * p * r)/(p + r)
        return f1 
    

#Processing Stock Data Functions     #function: final results, given below is algorithm presented in disseration as algorithm for printing results
#################################################################################################################################
def abnormal_return(list_of_dfs):
    '''
    A function used to compute abnormal return
    , outer loop within function is that of structure no. 2
    in disseration
    '''
    dfs = []
    meanlist = []
    for i in list_of_dfs: #
        count = 0
        meanlist = []
        expectedreturn = []
        df = i
        df['return'] = 100*((df['Close'] - df['Open'])/df['Open'])
        returnlist = list(df['return'])
        for i in reversed(returnlist): # reverse list so in time order
            count += 1
            if count <= 60:   # first 60 values ignored doesnt effect results as data sets where for whole history
                meanlist.append(i)
                expectedreturn.append(0)             
            else:
                expectedreturn.append(np.mean(meanlist))   
                meanlist.append(i)   #update window to new element
                del meanlist[0]  # first element in meanlist deleted allowing for fixed window size of 60
                
                
        df['expected'] = list(reversed(expectedreturn))  #create expected return data frame column
        df['abnormal'] = df['return'] - df['expected']
        dfs.append(df)
        
    return dfs




def dfs_to_only_events(list_of_dfs, eventdays):   #effectively what is structure 4 in dissertation
    '''
    A funtion which is called by main function
    final_results, inputs list of company data structures
    reduces them to that of only event days,
    finds days where there is missing data.
    removes any company data for any missing df.
    returns company data and markings that establish
    which sentiment ratios cant be used
    '''
    dfs = []
    xvalues_correct = []
    listt3 = []
    for i in list_of_dfs:
        df = i
        add = df[df['Date'].isin(eventdays)]
        if len(add) != len(eventdays):
            print('ERROR1')
            
            
        for i in eventdays:     #finds markings - days for missing data
            if i in list(add['Date']):
                listt3.append(1)
            else:
                listt3.append(0)
        xvalues_correct.append(listt3)
        listt3=[]
        dfs.append(add)
        
    delfrom = [1 for i in range(len(eventdays))]
    for i in xvalues_correct:                  # going through each marked timespan
        if len(i) != len(eventdays):
            print('ERROR2')
        for j, k in enumerate(i):
            if k == 0:
                delfrom[j] = 0      # any missing data from any companny results in zero/marking
                
    
    print(delfrom)
    newdates = yadjust(delfrom, eventdays)    #find eventdays that can be used 
    dfs2 = []
    for i in dfs:
        df = i
        add = df[df['Date'].isin(newdates)]    #each company datset is adjusted to only include dates where each company has data from 
        dfs2.append(add)
    
    return (dfs2, delfrom)
    

def y_output(list_of_dfs):
    z = [list(i['abnormal']) for i in list_of_dfs]      
    
    a = np.array(z)
    return list(reversed(list(np.mean(a, axis=0))))   #outputs mean of abnormal returns across companies

def xadjust(markings, xvalues):   #adjusts sentiment ratios to that of usable, structure 1 in dissertation
    x_reg_new = []
    for i, j in zip(markings, xvalues):
        if i == 0:
            continue
        else:
            x_reg_new.append(j)
            
    return x_reg_new

def yadjust(markings, values):   #adjusts event days to that of usable
    y_reg_new = []
    for i, j in zip(markings, values):
        if i == 0:
            continue
        else:
            y_reg_new.append(j)
            
    return y_reg_new

def plots(model, x):
    
    res = model.resid
    plt.figure(figsize = (15, 8))
    probplot = sm.ProbPlot(res)
    fig = probplot.qqplot()
    plt.ylabel('Residual')
    plt.xlabel('Normal Percentile')
    h = plt.title('qqplot - residuals of OLS fit')
    plt.show()
    
    f, axs = plt.subplots(1,1,figsize=(15,8))
    
    res2 = stats.probplot(res, plot=plt)
    
    plt.show()
    
    
    plt.figure(figsize = (15, 8))
    plt.scatter(x, res)
 
    plt.ylabel('Residual')
    plt.xlabel('X')
    plt.title('Plot of Residuals against X')
    plt.show()
    
    
def final_results(list_of_dfs, ratiosbayes, ratiosun, eventdaylist , zing = False):     #eventdays largest to smallest
    '''
    A function that inputs list of companies, sentiment ratio timeseries
    and event days and returns summary of regressions
    This is algorithm given in dissertation
    '''
    stockdata = abnormal_return(list_of_dfs)  # compute abnormal return for all companies
    alist = []
    bayesfresh = []
    unfresh = []
    for i, j in enumerate(eventdaylist):
        dfs, cor = dfs_to_only_events(stockdata, j)  #reduces company data to that of only event days and locates dates where there is missing company data
        alist.append(dfs)
        bayesfresh.append(xadjust(cor, ratiosbayes[i]))  #adjusts sentiment ratios to that of days where no company data is missing 
        unfresh.append(xadjust(cor, ratiosun[i]))
    
    for num, lodfs in enumerate(alist):
        y = y_output(lodfs)          # function is called that outputs abnormal return response variable
        print('BAYES')
        print(len(y), len(unfresh[num]))  #uses adjusted sentiment ratio data
        est = sm.OLS(y, bayesfresh[num])   #ordinary least squares regression is performed
        est2 = est.fit()
        print(est2.summary())   #results are printed that give p values
        if zing:
            plots(est2, bayesfresh[num])   #plots of residuals etc. are given
        print('UNSUPERVISED')
        est = sm.OLS(y, unfresh[num])
        est2 = est.fit()
        print(est2.summary())
        
        
        if zing:
            plots(est2, unfresh[num])
            
# Code used for topic modelling
################################################################################################################################

def lda_clean(list_of_sents, no_dictionary=False):

    """""""""
    inputs a list of clean-tokeinized list of list 
    and gives the LDA disired format
    """""

    from gensim import corpora

    print('Preparing LDA corpus')

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(list_of_sents)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in list_of_sents]

    if no_dictionary:
        return corpus
    else:
        return corpus, dictionary

def get_training_testing(training_dir,split=1):

    filenames=os.listdir(training_dir)
    n=len(filenames)
    print("There are {} files in the training directory: {}".format(n,training_dir))
    random.seed(53)  #if you want the same random split every time
    random.shuffle(filenames)
    index=int(n*split)
    trainingfiles=filenames[:index]
    heldoutfiles=filenames[index:]
    return trainingfiles,heldoutfiles

class language_model():
    
    def __init__(self,trainingdir=TRAINING_DIR,files=[]):
        self.tokenized_files = []
        self.full_doc = []
        self.training_dir = trainingdir
        self.files = files
        self.unigram = {}
        self._processfiles()
        
        
    
                
    def _remove_stop_punc(self):
        en_stop = stopwords.words('english')
        en_stop.append("'d")
        punctuations = list(string.punctuation)
        l1 = en_stop + punctuations
        l2 = ["\'\'", "``", "n\'t", "\'s", "\'ll", "\'I", "\'m", "--"]
        l1 = l1 + l2
        self.full_doc = [[x for x in z if x.lower() not in l1] for z in self.full_doc]
    
        self.tokenized_files = [[x for x in z if x.lower() not in l1] for z in self.tokenized_files]
    
    
    def _processline(self,line):
        tokens=["__START"]+tokenize(line)+["__END"]
        previous="__END"
        for token in tokens:
            self.unigram[token]=self.unigram.get(token,0)+1
        return tokenize(line)
  

    def _processfiles(self):
        for afile in self.files:
            print("Processing {}".format(afile))
            try:
                with open(os.path.join(self.training_dir,afile), encoding="utf8") as instream:
                    self.full_doc.append(instream.read().split())
#                     for line in instream:
#                         line=line.rstrip()
#                         if len(line)>0:
#                             self.tokenized_files.append(self._processline(line))
            except UnicodeDecodeError:
                print("UnicodeDecodeError processing {}: ignoring rest of file".format(afile))
            
    def process_sentences(self, max_file=3):
        for i, afile in enumerate(self.files):
            if i > max_file:
                break
            print("Processing {}".format(afile))
            
            with open(os.path.join(self.training_dir,afile), encoding="utf8") as instream:
                for line in instream:
                    line=line.rstrip()
                    if len(line)>0:
                        try:
                            self.tokenized_files.append(self._processline(line))
                        except UnicodeDecodeError:
                            print("UnicodeDecodeError processing {}: ignoring rest of file".format(afile))

                        
            
        
    def prepare_lda_format(self):
         
        print('preparing lda bow dictionaries')
        
        # extracting bow and dictionary for model
        self.full_text, self.full_dictionary = lda_clean(self.full_doc)  
        self.sent_text, self.sent_dictionary = lda_clean(self.tokenized_files)  
       
    
    def lda_train(self, data='full_doc', topic_number=4, passes=5):
        
        text, dictionary = self.full_text, self.full_dictionary
        
        if data!='full_doc':
            text, dictionary = self.sent_text, self.sent_dictionary

        self.lda_model = models.ldamodel.LdaModel(text,
                                       num_topics=topic_number,
                                       id2word=dictionary, passes=passes)
        
        
    def vector(self, string,  data='full_doc'):
        
        text, dictionary = self.full_text, self.full_dictionary
        
        if data!='full_doc':
            dictionary = self.sent_dictionary

        # extracting bow
        bow_string = [dictionary.doc2bow(text) for text in [tokenize(string)]]

        # extracting the topic-probablity distribution
        vector_string = self.lda_model.get_document_topics(bow_string[0], minimum_probability=0)

        return vector_string
    

    
def group_coms_topic(ldaobject, comments, topics):        #can expand code for range of topic numbers, must know number of topics lda model is trained for
    '''
    This functions inputs lda object
    which allows for topic modelling, 
    as well as comments for clustering.
    Returns a list containing all comments
    clustered into n groups
    '''
    alldays = []
    for day in comments:  # each day in timeseries of comments
        groups = [[] for i in range(topics)]   #number of clustered groups depends on number of topics lda model is trained to have
        
        groupssame = 0
        for i in day: #consider each comment in day
            maxnum = 0
            alist = ldaobject.vector(i)  #obtain topic distribution for comment
            alltopicnum = []
            for j, k in enumerate(alist):
                if k[1] > maxnum:
                    maxnum = k[1]
                    listindex = j   # this gives us topic with highest probability and index for which list to add comment to 
                alltopicnum.append(k[1])

            c = 0
            for n, z in enumerate(alltopicnum):   #check if all probs are identical, if so we choose a random one
                if n == 0:
                    prev = z
                if n > 0:
                    if z == prev:
                        c += 
                    prev = z
            
            if c == (len(alltopicnum) - 1):
                listindex = random.choice(list(range(topics)))# keeps track of how many comments are randomly clustered when all topic probs are identical, this rarely happended in experiment
                groupssame += 1 #keep track of how much
                    
            
            groups[listindex].append(i)    #add comment with topic number that is list index to appropriate list
            
        alldays.append(groups)  # add clustered groups to list with entries for each day
        print(groupssame, 'out of', len(day))

    return alldays