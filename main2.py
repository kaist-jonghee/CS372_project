## Project : Text Summarization for Lectures // Name: Jonghee Jeon // StudentID: 20190574 ## 
########################### Importing Librairies ###########################
# These are for the data
import datasets
import pandas as pd
import numpy as np

# These are for preprocessing
import re
import nltk

# Finally, these are for evaluation.
import rouge

# Else...
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import math
from operator import itemgetter

stpwords = [a.lower() for a in stopwords.words('english')]

########################### Summarization Function ###########################

def resolve_pronoun(sentence):
    '''
    Resolve Pronouns inside text.
    '''
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'Novermber', 'December']
    nounphrases = ['NN', 'NNS', 'NNP', 'NNPS']
    verb = ['VB', 'VBD', 'VBN', 'VBP', 'VBZ']
    femalenames = nltk.corpus.names.words('female.txt')
    malenames = nltk.corpus.names.words('male.txt')
    
    # tokens_s = sent_tokenize(sentence)
    tokens_sw = sentence.copy()
    tokens_swp = [nltk.pos_tag(tokens) for tokens in tokens_sw]
    
    def resolve(pro, pro_offset):
        # pro.lower() only contains ['her', 'his', 'she', 'he', 'him', 'hers']
        gender_is_man = (pro.lower() in ['his', 'he', 'him'])
    
        # Get pronoun sentence/word index from tokens_swp
        pro_idx_s = pro_offset[0]
        pro_idx_w = pro_offset[1]
    
        # start finding coreference starting from the pronoun index
        idx_s, idx_w = pro_idx_s, pro_idx_w
        answer = []
    
        first = True 
    
        while(idx_s >= 0):
            # First, iter through sentences.
            sent = tokens_swp[idx_s]
            if first:
                idx_w -=1
                first = False
            else:
                idx_w = len(sent)-1
            found = False
            
            while(idx_w >= 0):
                # Secondly, iter thorugh words in such sentences.
                pair = sent[idx_w]
                if ((pair[1] in nounphrases) and (pair[0].isalpha()) and (pair[0][0].isupper()) ):
                    answer.append(pair[0])
    
    
                    # Check if it's a Name entity.
                    if (pair[0] in months) or (len(pair[0])<=2):
                        answer.pop()
                        idx_w-=1
                        continue
    
                    # Handling Gender
                    if gender_is_man:
                        if (pair[0] in femalenames):
                            answer.pop()
                            idx_w -=1
                            continue
                    else:
                        if (pair[0] in malenames):
                            answer.pop()
                            idx_w -=1
                            continue
    
    
                    # There must be a verb between the pronoun and the coreference.
                    isverb=[]
                    s = idx_s
                    w = idx_w
                    while(s<=pro_idx_s):
                        if (s < pro_idx_s):
                            while(len(tokens_swp[s])>w):
                                isverb.append(tokens_swp[s][w][1] in verb)
                                w+=1
                        if (s==pro_idx_s):
                            while(pro_idx_w>w):
                                isverb.append(tokens_swp[s][w][1] in verb)
                                w+=1
                        s += 1
                        w = 0
                    if not(True in isverb):
                        # Handling "her", "his" cases.
                        if not((pro in ['her', 'his']) and ((idx_w == (pro_idx_w-2))or(idx_w == (pro_idx_w-3)))  and (pair[1] == 'NNP')):
                            answer.pop()
                        idx_w-=1
                        continue
                    else:
                        found = True
                        break
    
                    # Handling stop condition
                    if len(answer) == 1:
                        found = True
                        break
                idx_w -=1
            if found:
                break
            idx_s -= 1
        
        
        if answer:
            return answer[0]
        else:
            return None
        
    for sent_i in range(len(tokens_sw)):
        for word_i in range(len(tokens_sw[sent_i])):
            if tokens_sw[sent_i][word_i].lower() in ['her', 'his', 'she', 'he', 'him', 'hers']:
                resolved_np = resolve(tokens_sw[sent_i][word_i], [sent_i,word_i])
                if resolved_np:
                    tokens_sw[sent_i][word_i] = resolved_np
    return tokens_sw

def sent_simil_1(sent1, sent2):
    '''
    Method based on vocabulary
    '''
    vocab1 = set(word.lower() for word in sent1 if word.isalpha())
    vocab2 = set(word.lower() for word in sent2 if word.isalpha())
    maxlen = max(len(vocab2),len(vocab2))
    intersection = vocab1.intersection(vocab2)
    if maxlen == 0:
        return 0.0
    return len(intersection)/maxlen

def sent_simil_2(sent1, sent2):
    '''
    Get the similarity score for 'sent1' and 'sent2'
    '''
    # Making sense vectors from Baseline Assumption 1.
    def change(pos):
        '''
        Change positional information for words to be changed to senses.
        Words to chagne :
        every noun, verb, adjective, adverb + delete stopwords
        '''
        pos = [(word, 'n') if p in ['NN','NNS','PRP','PRP$'] and (not word in stpwords) else (word, p) for (word,p) in pos ] # Changing nouns
        pos = [(word, 'v') if p in ['VB','VBG','VBD','VBN','VBP','VBZ'] and (not word in stpwords) else (word, p) for (word, p) in pos ] # Changing verbs
        pos = [(word, 'a') if p in ['JJ','JJR','JJS'] and (not word in stpwords) else (word, p) for (word, p) in pos ] # Changing adjectives
        pos = [(word, 'r') if p in ['RB','RBR','RBS','WRB'] and (not word in stpwords) else (word, p) for (word, p) in pos ] # Changing adverbs
        return pos
    
    # Get positional information from sent1 & sent2
    pos1 = nltk.pos_tag(sent1)
    pos2 = nltk.pos_tag(sent2)
    # change positional information for words to get senses from.
    pos1 = change(pos1)
    pos2 = change(pos2)
    # Get senses from words (Make the sense vector for each sentence)
    syn1 = [wn.synsets(word, pos)[0] for (word,pos) in pos1 if pos in ['n', 'v', 'a', 'r'] and len(wn.synsets(word, pos))>0]
    syn2 = [wn.synsets(word, pos)[0] for (word,pos) in pos2 if pos in ['n', 'v', 'a', 'r'] and len(wn.synsets(word, pos))>0]
    
    
    score = 0
    n = 0
    for syni in syn1:
        for synj in syn2:
            score+=syni.path_similarity(synj) # Calculate path_similarity for every sense in each sense vector.
            n+=1
            
    if n>0:
        return score/n
    else:
        return 0

def summarization(text, sent_simil, weights):
    '''
    The Function first segments the text into subtopics.
    Then, summarizaes at least one sentnece form each segment.
    And finally concatenates them.
    '''
    ########################### Preprocess data ###########################
    sents = sent_tokenize(text)
    words = [word_tokenize(sent) for sent in sents]
    words = resolve_pronoun(words)
    # Resolve only for words --> Resolution might go wrong, but it shouldn't affect the summary.
    
    # Detect pronouns and replace them with the corresponding noun phrase.
    
    ########################### Text Segmentation ###########################
    
    # 1. Similarity Approch.
    ##### Types of sentences #####
    segment_starts = [0] # Starting sentence indices for each segment.
    weird_sentences = [] # Weired sentences such that it doesn't match with the context, and also with the following sentences. Where the following sentences also don't matchup.
    stop_sentences = [] # Sentences that don't match with the context, but the sentences
    # Should we consider these weird sentences also as segment starts?
    
    
    window_size = 2
    window = ['']*window_size
    segment_start = [True, 0]
    
    suspect_check = 2
    
    for idx, sent in enumerate(words):
        if segment_start[0]:
            window[segment_start[1]] = sent
            segment_start[1] += 1
            if segment_start[1] == window_size:
                segment_start = [False, 0]
            continue
                
        window_simil = sent_simil(window[0], window[1])
        simil = np.mean([sent_simil(s,sent) for s in window])
        
        if window_simil*0.7 > simil:
            # Suspect for new segment.
            # 1. If the suspect has similar context with the following 2 or more sentences, claim it as a new segment.
            new_segment = True
            try:
                for i in range(suspect_check):
                    if window_simil*0.9 > sent_simil(sent, words[idx+i+1]):
                        # Discovered that it's not a start of a new segment.
                        new_segment = False
                        # Then, was the sentence just a weird one?
                        same_context = np.mean([sent_simil(s, words[idx+i+1]) for s in window])
                        if window_simil*0.7 > same_context:
                            weird_sentences.append(idx)
                        # Or was it a sentence in the previous segment that just suddenly was out of context. (stop_sentence)
                        else:
                            stop_sentences.append(idx)
                        
                        # End loop
                        break
            except:
                new_segment = False
            
            # Handle cases where the idx sentence is a start of a new segment.
            if new_segment:
                window[segment_start[1]] = sent
                segment_start[0] = True
                segment_start[1] += 1
                segment_starts.append(idx)
                
                
                
    ########################### Text Summarization ###########################
    ############# Scoring Method #############
    sentence_scores = [0]*len(words)
    # weights for each method.( range = (0,1] )
    # 1. Segment Start
    # 2. Segment End
    # 3. High Entorpy
    # 4,5. Question-Asking
    # 6. Sufficiently long Sentnece.
    # weights = [0.2, 0.7, 0.3, 0.3]
    
    
    # 1. Simply Select the First Sentence from Segment. (Why? -> Because, usually the first sentence summarizes the whole segment.)
    if weights[0]:
        for idx in segment_starts:
            sentence_scores[idx] += weights[0]
    
    
    # 2. Segment End
    if weights[1]:
        for idx in segment_starts:
            if idx == 0:
                continue
            sentence_scores[idx-1] += weights[1]
        sentence_scores[len(sents)-1] += weights[1]
    
    
    # 3. Calculate Entropy and select the Most Informative Sentences. - It's better not to use entropy...
    entropy4later = []
    # First divide the text into segments.
    segment_starts.append(len(words))
    seg_start = segment_starts.pop(0)
    for seg_end in segment_starts:
        segment = words[seg_start:seg_end] # segments divided in words.
        # Need to calculate Entropy for each sentence.
        entropies_in_sent = []
        for sentence in segment:
            # sentence contains words. Need to calculate entropy for each sentence.
            vocab = [word.lower() for word in sentence if word.isalpha() and  (word.lower() not in stpwords)]
            vocabset = set(vocab)
            # Calculate entropy for each sentence
            entropy = 0
            for word in vocabset:
                prob = vocab.count(word)/len(vocab)
                entropy += prob*math.log2(1/prob)
            entropies_in_sent.append(entropy)
        entropy4later+=entropies_in_sent
        # Add Entropy term to all of them.
        # entropies_in_sent = [entry/max(entropies_in_sent) for entry in entropies_in_sent]
        # for (idx,s) in enumerate(entropies_in_sent):
        #     if weights[2]:
        #         sentence_scores[idx+seg_start] += s*weights[2]
        
        # Find most informative sentence. (Maximum Entropy Value.)
        len4seg = 16
        maxs = sorted(entropies_in_sent, reverse=True)[:len4seg]
        indices = sorted([entropies_in_sent.index(m) for m in maxs])
        for idx in indices:
            if weights[2]:
                sentence_scores[idx+seg_start] += weights[2]
        seg_start = seg_end
            
    
    # 4,5. Question Asking - It was better to not add weight for the answer part. May be the question part is a bit more important.
    if weights[3] or weights[4]:
        asking = [i for (i,sent) in enumerate(sents) if ('?' in sent)]
        for idx in asking:
            # For Question
            sentence_scores[idx] += weights[3]
            # For Answer
            if (idx+1)<len(sents):
                sentence_scores[idx+1] += weights[4]
        
    
    # 6. Consider short sentences as NOT informative.
    if weights[5]:
        threshold4sufficient = 8
        sufficient_length = [i for (i,sent) in enumerate(words) if (len(sent)>threshold4sufficient)]
        for idx in sufficient_length:
            sentence_scores[idx] += weights[5]
    
    
    # Special process : Eliminate Weired sentences. - No Good...
    if True:
        for idx in weird_sentences:
            sentence_scores[idx] -= 0.4
    
    
    # Summary Selection (Select n highest scores. / Select sentences with score higher than threshold.)
    summary_length = max(5, int(len(sents)//5)) # Summarize the text to 20 percent. (But, at least 5 sentences long.)
    summary = ''
    maximums = sorted(list(enumerate(sentence_scores)), key = itemgetter(1), reverse = True)
    chosen = []
    for (idx,_) in sorted(maximums[:summary_length], key = itemgetter(0)):
        chosen.append(idx)
        summary += sents[idx]+' '
    summary = summary[:-1]
    
    # Since important sentences are chosen, we can use this information to make abstractive methods.
    # One possible approach is to select noun phrases from here, and construct sentences.
    # 1. Find NPs
    word_sum = [words[idx] for idx in chosen]
    pos_sum = [nltk.pos_tag(ws) for ws in word_sum]
    parse_sum = [nltk.ne_chunk(pos, binary = True) for pos in pos_sum]
    
    chosen_entropy=[]
    for idx in chosen:
        chosen_entropy.append(entropy4later[idx])
    len4seg = 2
    keywords = set()
    maxs = sorted(chosen_entropy, reverse=True)[:len4seg]
    indices = sorted([chosen_entropy.index(m) for m in maxs])
    for idx in indices:
        tree = parse_sum[idx]
        for chunk in tree:
            if hasattr(chunk, 'label'):
                nounphrase = ' '.join(c[0] for c in chunk)
                keywords.add(nounphrase)
    keywords_sent = 'KEYWORDS : ' + ', '.join(keywords)       
    
    # Using extract_rels didn't go so well...
    # for rel in nltk.sem.extract_rels('ORG','LOC', parse_sum):
    #     print(nltk.sem.rtuple(rel))
    
    
    return keywords_sent, summary






########################### Test Summarization on CNN-DailyMail Dataset ###########################
########################### Loading Dataset ###########################
data_length = 11490
# data_length = 100

dataset = datasets.load_dataset("cnn_dailymail", '3.0.0')
lst_dics = [dic for dic in dataset['test']]

########################### Keep the first N articles in order to keep it lite ###########################
dtf = pd.DataFrame(lst_dics).rename(columns = {"article":"text", "highlights":"y"})[["text","y"]].head(data_length)

weights = [0.5, 0.3, 0.1, 1,1,1]

# text = "Hi I’m Jonghee Jeon From Team eighteen. Today, I will be presenting my solo project, ‘Lecture Summarization’. And this is the index of my presentation. In the present world, new information is rapidly flowing in. And it’s hard to capture them at a glance. But fortunately, most methods for conveying information help the users to do so. For example, newspapers have ‘titles’ that summarize the whole context. However, ‘Lecturing’ doesn’t provide any. And even though lecture slides may be provided, it’s hard to know which part was emphasized during the lecture. Therefore, I performed ‘Lecture Summarization’ that summarizes subtitles of the lecture into 20% of its original size. Here is the basic idea of my algorithm. Using the systematically-organized property of lectures, I first tried to split the text by subtopics. Then, for summarization, I’ve looked at 6 aspects of the text, where some of the aspects are related to the previous segmentation. And according to this, I gave corresponding scores to every sentence. Finally, chose the top-scored sentences and concatenated them to make a summary. The details will be explained shortly. But before diving right into the algorithm, I had to preprocess the text. And thus, I’ve performed sentence tokenization, word tokenization, and finally pronoun resolution. And to be more specific for pronoun resolution, I’ve resolved these pronouns. And the following are the key methods for resolving them. And next, to the main algorithm. First, I’ll be talking about segmentation. The key concept for this was to assume that segments are grouped by semantically similar sentences. Just like how we naturally section text. Now, this slide explains how I’ve actually implemented it. The main method I’ve used is sentence similarity. First, I’ve calculated the sentence similarity of the previous 2 sentences. And this is used as an indicator of how big it needs to be, to be called similar. Then, I’ve calculated the similarity between the target sentence and the previous 2 sentences. If it was not similar as shown here, I’ve calculated the following 2 sentences. And if they were similar as shown, I’ve claimed that, the target sentence is a ‘Segment Starting Sentence’. And segmentation was done based on these sentences. Although only the ‘Segment Starting Sentence’ was explained here, I’ve declared ‘stop sentences’ and ‘weird sentences’ too using sentence similarities. Then, how did I calculate sentence similarity? By using vocabulary sets and by using synonym set path similarities. Let’s look at an example. For the first method, the similarity was calculated by dividing the number of intersecting vocabulary by the size of the bigger vocabulary. And for the second method, I’ve averaged up every possible pair’s path_similarity. And, the second similarity function was chosen to be used. Next, is summarization. For every sentence in the text, I scored them according to the 6 aspects explained before. Here, instead of adding the same score for each aspect, I’ve rated the importance of each in the range of zero to one and used it as the addition factor for the corresponding aspect. The first and second aspect is whether it’s the start or the end of the segment. These were used because usually the key points are written in the first or last part of each segment. The next aspect is entropy. Here, the higher the entropy is, the more informative it is. The fourth and fifth aspect is whether it’s a question-asking sentence, or whether it’s a question-answering sentence. I’ve included them because usually questions are used to emphasize important concepts. And lastly, I checked whether each sentence is long enough. Because usually, too short sentences aren’t informative enough. And the upper right equation shows how to calculate entropy. Finally, I’ve chosen the high-scored sentences and used the exact same sentences for the summary. And in addition to the summary, I’ve found some name phrases from the 2 most high-entropy-valued sentences and used them to be Keywords. And showed them separately from the summary. Evaluation of the generated summary was done by ROUGE scores. It checks how many n-grams match with the n-grams in the truly labeled summaries. As the main metric, I chose ROUGE-1 f1 scores since there were many other studies using the same one, thus easy to compare the performance of my algorithm. And for the dataset, it was hard to find the dataset for the exact task that I was doing. So, I chose the CNN/Daily Mail dataset, which is a dataset that summarizes news articles. I thought that it was similar in the sense that lectures and news articles both convey information. First, with the training data of the dataset, I fine-tuned the weights used in scoring for summarization and got the following result. It implied that entropy wasn’t a good aspect to determine the importance of a sentence, the first sentence in each segment usually contained more important information than the last sentence, short sentences usually aren’t informative enough, and finally, that question-asking and question-answering contain important stuff. Then, with the test data of the dataset, I got these ROUGE scores. Here, the ROUGE-1 f1-score was 30.32% which was quite astonishing for me. Because it’s in the same 30’s with the leading deep-learning-models’ f1 scores. Also, the ROUGE-2 f1 score and ROUGE-L f1 score were high at 11.65% and 28.28% respectively."
# generated_keywords, generated_summary = summarization(text, sent_simil_2, weights)

# print(generated_keywords)
# print(generated_summary)


eval_scores = np.array([0,0,0,0], dtype='float64')
eval_scoresp = np.array([0,0,0,0], dtype='float64')
eval_scoresr = np.array([0,0,0,0], dtype='float64')
for i in range(len(dtf)):
    text = dtf['text'][i]
    summary = dtf['y'][i]
    ########################### Summarization ###########################
    generated_keywords, generated_summary = summarization(text, sent_simil_2, weights)
    # print(generated_keywords)
    # print(generated_summary)
    ########################### Evaluation ###########################
    rouge_score = rouge.Rouge()
    scores = rouge_score.get_scores(summary, generated_summary, avg=True)
    score_1 = round(scores['rouge-1']['f'], 2)
    score_2 = round(scores['rouge-2']['f'], 2)
    score_L = round(scores['rouge-l']['f'], 2)
    avg = round(np.mean([score_1, score_2, score_L]), 2)
    eval_scores += np.array([score_1, score_2, score_L, avg])
    
    score_1p = round(scores['rouge-1']['p'], 2)
    score_2p = round(scores['rouge-2']['p'], 2)
    score_Lp = round(scores['rouge-l']['p'], 2)
    avgp = round(np.mean([score_1p, score_2p, score_Lp]), 2)
    eval_scoresp += np.array([score_1p, score_2p, score_Lp, avgp])
    
    score_1r = round(scores['rouge-1']['r'], 2)
    score_2r = round(scores['rouge-2']['r'], 2)
    score_Lr = round(scores['rouge-l']['r'], 2)
    avgr = round(np.mean([score_1r, score_2r, score_Lr]), 2)
    eval_scoresr += np.array([score_1r, score_2r, score_Lr, avgr])

eval_scores/=len(dtf)
eval_scoresp/=len(dtf)
eval_scoresr/=len(dtf)

print("{:<20}    {:<20}    {:<20}    {:<20}".format("Eval. Metric", 'Recall', 'Precision', "F1-Score"))
print("{:<20}    {:<20}    {:<20}    {:<20}".format("Rouge-1", eval_scoresr[0], eval_scoresp[0], eval_scores[0]))
print("{:<20}    {:<20}    {:<20}    {:<20}".format("Rouge-2", eval_scoresr[1], eval_scoresp[1], eval_scores[1]))
print("{:<20}    {:<20}    {:<20}    {:<20}".format("Rouge-L", eval_scoresr[2], eval_scoresp[2], eval_scores[2]))
print("{:<20}    {:<20}    {:<20}    {:<20}".format("Rouge-Avg", eval_scoresr[3], eval_scoresp[3], eval_scores[3]))