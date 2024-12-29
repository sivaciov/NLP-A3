# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import gc

nlp = spacy.load('en_core_web_sm')


class FactExample:
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis with the model tokenizer
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            # Get model predictions
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Using logits for entailment, neutral, contradiction
            entail_prob = torch.softmax(logits, dim=-1)[0, 0]  # Entailment probability

        # Clean up to avoid out-of-memory issues
        del inputs, outputs, logits
        gc.collect()

        return entail_prob.item()



class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(object):

    def __init__(self):
        self.threshold = 0.75

    def preprocess(self, text):
        # remove <s> and </s> html tags
        text = text.replace("<s>", "").replace("</s>", "")

        # remove words less than 3 characters
        text = ' '.join([word for word in text.split() if len(word) > 1])

        # make lowercase
        text = text.lower()

        # remove words that contain numbers
        #text = ' '.join([word for word in text.split() if not any(char.isdigit() for char in word)])


        # Tokenize, remove stopwords, and lemmatize
        doc = nlp(text)

        ret_tokens = []
        for token in doc:
            if not token.is_stop and not token.is_punct:
                ret_tokens.append(token.lemma_)

        return set(ret_tokens)

    def predict(self, fact: str, passages: List[dict]) -> str:
        #print(f"Fact: {fact}")

        fact_tokens = self.preprocess(fact)

        #print(f"Fact tokens: {fact_tokens}")

        max_similarity = 0
        for passage in passages:
            passage_tokens = self.preprocess(passage['text'])

            #print(f"Passage tokens: {passage_tokens}")

            if len(fact_tokens) == 0:
                continue

            similarity = len(fact_tokens & passage_tokens) / len(fact_tokens)
            max_similarity = max(max_similarity, similarity)

        return "S" if max_similarity >= self.threshold else "NS"


class EntailmentFactChecker(object):
    def __init__(self, ent_model):
        self.ent_model = ent_model

    def predict(self, fact: str, passages: List[dict]) -> str:
        max_entailment = 0.0
        for passage in passages:
            doc = nlp(passage['text'])  # Tokenize passage into sentences
            for sent in doc.sents:
                entail_prob = self.ent_model.check_entailment(sent.text, fact)
                max_entailment = max(max_entailment, entail_prob)  # Track highest entailment score

        # Final decision based on max entailment score
        return "S" if max_entailment >= 0.5 else "NS"


# OPTIONAL
class DependencyRecallThresholdFactChecker(object):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations

