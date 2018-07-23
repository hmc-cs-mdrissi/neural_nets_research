# Imports
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import regex
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import multiprocessing
import warnings
import tqdm
import argparse
from functools import partial

# Main idea behind generating stories and simplified stories is that you can do preprocessing on
# simplified_stories to help you choose sentences, but still include the original sentences in the
# summary (if you want).

def preprocess_story(story, stem=True, remove_stop_words=True, remove_punctuation=True, metaparagraph_size=5):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # Split into a list of paragraphs
    paragraphs = story.split("<newline>")
    simplified_paragraphs = []
    untokenized_paragraphs = []
    par_index = 0
    
    # Loop through paragraphs
    while par_index < len(paragraphs):
        meta_paragraph = []
        
        # Combine small paragraphs into meta_paragraphs with at least some minimum number of sentences
        while par_index < len(paragraphs) and len(meta_paragraph) < metaparagraph_size:
            paragraph = paragraphs[par_index]
            
            # Split paragraph into a list of sentences
            sentences = nltk.sent_tokenize(paragraph)
            meta_paragraph += sentences
            par_index += 1
        
        meta_paragraph_unprocessed = meta_paragraph
        
        if remove_stop_words:
            meta_paragraph = [sentence.replace("<num>"," ") for sentence in meta_paragraph]
        
        # For the tokenized version, split each sentence into a list of words
        paragraph_tokenized = [nltk.word_tokenize(sentence) for sentence in meta_paragraph]
        # Extra preprocessing
        if remove_stop_words:
            paragraph_tokenized = [[word for word in sentence if word not in stop_words] for sentence in paragraph_tokenized]
        if remove_punctuation:
            paragraph_tokenized = [[regex.sub('[\p{P}\p{Sm}`]+', '', word) for word in sentence] for sentence in paragraph_tokenized]
            paragraph_tokenized = [[word for word in sentence if word != ""] for sentence in paragraph_tokenized]
        if stem:
            paragraph_tokenized = [[stemmer.stem(word) for word in sentence] for sentence in paragraph_tokenized]

        if len(meta_paragraph) < metaparagraph_size and len(untokenized_paragraphs) > 0:
            untokenized_paragraphs[-1] += meta_paragraph_unprocessed
            simplified_paragraphs[-1] += paragraph_tokenized
        else:
            if len(meta_paragraph) != 0:
                untokenized_paragraphs.append(meta_paragraph_unprocessed)
                simplified_paragraphs.append(paragraph_tokenized)

    return untokenized_paragraphs, simplified_paragraphs

# Read in data from target, breaking each story into paragraphs (and then sentences)
def load_data_as_paragraphs(file, pool, chunksize=100, stem=True, remove_stop_words=True, 
                            remove_punctuation=True, metaparagraph_size=5):
    simplified_stories = []
    stories = []
        
    # Load stories from file
    with open(file) as f:
        stories_raw = f.readlines()
    
    partial_preprocess_story = partial(preprocess_story, stem=stem, remove_stop_words=remove_stop_words, 
                                       remove_punctuation=remove_punctuation, metaparagraph_size=metaparagraph_size)

    for story, simplified_story in tqdm.tqdm(pool.imap(partial_preprocess_story, stories_raw, chunksize=chunksize)):
        stories.append(story)
        simplified_stories.append(simplified_story)

    return stories, simplified_stories

# SumBasic algorithm

# Pick the best scoring sentence that optionally contains the highest probability word.
def get_best_sentence(data, document_scores, document_index, vocab, inverse_vocab, sentence_vectorizer):    
    # Create a bag-of-words-style sentence vector
    vector_sentences = sentence_vectorizer.transform(data)
    
    # Dot the sentence vector with the document tf_idf vector
    curr_doc_scores = document_scores[document_index].transpose()
    scores = vector_sentences * curr_doc_scores
    
    # Divide each sentence's score by its length. Zero length sentences will cause a warning of divide by zero
    # to occur. This is not an issue as the infinites produced become nan after the multiplication (as they'll end up
    # multiplied by zero). As nan is considered small by argmax this doesn't cause an issue.
    lengths = 1.0 / vector_sentences.sum(axis=1)
    scores = scores.multiply(lengths)

    if scores.count_nonzero() == 0:
        return 0
        
    # Return the index of the best-scoring sentence
    best = scores.argmax(axis=0)     
    return best[0,0]

def construct_text_collection(simplified_stories, by_paragraph=False):
    # If get by paragraph, each element refers to 1 paragraph
    if by_paragraph:
        texts = [[word for sentence in paragraph for word in sentence] for story in simplified_stories for paragraph in story]
    # Otherwise each element is 1 story
    else:
        texts = [[word for paragraph in story for sentence in paragraph for word in sentence] for story in simplified_stories]
    
    return texts

def compute_all_probs(texts):
    tfidf = TfidfVectorizer(analyzer='word', tokenizer=lambda x: x,
                            preprocessor=lambda x: x,
                            norm='l1', use_idf=False, token_pattern=None)
    scores = tfidf.fit_transform(texts)
    return tfidf, scores
    
def compute_all_tfidfs(texts):
    probs = TfidfVectorizer(analyzer='word', tokenizer=lambda x: x,
                            preprocessor=lambda x: x, 
                            token_pattern=None, norm=None)
    scores = probs.fit_transform(texts)
    return probs, scores
    
def compute_all_scores(texts, tfidf=True):
    if tfidf:
        return compute_all_tfidfs(texts)
    else:
        return compute_all_probs(texts)   
    
def summarize_story(scores, vocab, feature_names, inputs): 
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    sentence_vectorizer = CountVectorizer(analyzer='word', tokenizer=lambda x: x, preprocessor=lambda x:x, 
                                          vocabulary=feature_names, token_pattern=None)
    story, simplified_story, story_index = inputs
    summary = []

    # Loop through paragraphs, adding one sentence per paragraph to the summary.
    for paragraph_index, (paragraph, simplified_paragraph) in enumerate(zip(story, simplified_story)):
        # indexing is done in a bit of a stupid way because csr matrices don't support indexing like
        # A[x][y] and instead require A[x,y].
        document_index = paragraph_index + story_index if by_paragraph else story_index
        
        # Choose sentence with best score
        next_sentence_index = get_best_sentence(simplified_paragraph, scores, document_index, vocab, feature_names, sentence_vectorizer)

        # Add it to summary
        summary.append(paragraph[next_sentence_index])
    # Join sentences into a summary
    summary_string = " <newline> ".join(summary)
    return summary_string    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Outline Generation')
    parser.add_argument('--input-file-name', help='Name of input file.')
    parser.add_argument('--output-file-name', help='Name of output file.')
    parser.add_argument('--no-stem', action='store_true', help="Don't stem words when doing SumBasic.")
    parser.add_argument('--keep-stop_words', action='store_true', help="Keep stop words when doing SumBasic.")
    parser.add_argument('--keep-punctuation', action='store_true', help="Keep punctuation when doing SumBasic.")
    parser.add_argument('--metaparagraph-size', type=int, default=5, help="Number of sentences to aim to have in each meta-paragraph. Default is 5.")
    parser.add_argument('--probs', action='store_true', help='Use probabilities instead of tfidf for SumBasic (closer to the original SumBasic).')
    parser.add_argument('--by-story_tfidf', action='store_true', help='Compute the tfidf by story instead of by paragraph. The outline will still be made up of one sentence per paragraph.')
    parser.add_argument('--chunksize', type=int, default=100, help="Size of chunks to use for each process. Default is 100.")

    args = parser.parse_args()
    stem = not args.no_stem
    remove_stop_words = not args.keep_stop_words
    remove_punctuation = not args.keep_punctuation
    tfidf = not args.probs
    by_paragraph = not args.by_story_tfidf
    print("Currently making outlines for " + args.input_file_name)
    p = multiprocessing.Pool(multiprocessing.cpu_count())

    # stories is a triply nested lists (first broken by story, then by paragraph, then by sentences)
    # simplified_stories is a quadruply nested list (broken by story, paragraph, sentence, word)
    stories, simplified_stories = load_data_as_paragraphs(args.input_file_name, p, args.chunksize, stem, 
                                                          remove_stop_words, remove_punctuation, args.metaparagraph_size)

    # Get the starting story index (i.e. starting paragraph index) for each story
    lengths = [len(story) for story in stories]
    story_indices = np.cumsum([0] + lengths[:-1])

    # TODO: If necessary, introduce other cleaning things:
    # - Deal with parens unmatched
    texts = construct_text_collection(simplified_stories, by_paragraph=by_paragraph)
    vectorizer, scores = compute_all_scores(texts, tfidf=tfidf)
    feature_names = vectorizer.get_feature_names()
    sentence_vectorizer = CountVectorizer(analyzer='word', tokenizer=lambda x: x, preprocessor=lambda x:x, 
                                          vocabulary=feature_names, token_pattern=None)

    # Don't print warnings about dividing by zero.
    inputs = zip(stories, simplified_stories, story_indices)
    partial_summarize_story = partial(summarize_story, scores, vectorizer.vocabulary_, vectorizer.get_feature_names())
    with open(args.output_file_name, 'w') as f:
        for summary in tqdm.tqdm(p.imap(partial_summarize_story, inputs, chunksize=args.chunksize)):
            f.write(summary + "\n")
