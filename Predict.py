# ! This code works only in Python < 3.14, since spaCy is not compatible with Python 3.14 !


def format_texts(source_texts: list[str], repetition_texts: list[str]):
    """
    Formats source-repetition candidate pairs as follows: [source_text]***[repetition_text]
    Args:
        source_texts: List(str): List of all utterances candidate to be the source of a direct or self-repetitive pair
        repetition_texts: List(str): List of all utterances candidate to be the direct or self-repetition of a repetitive pair
    """
    return [f"{s.strip()}***{r.strip()}" for s, r in zip(source_texts, repetition_texts)]


# Predict repetitions using trained BERT model

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import torch
import numpy as np
from typing import Literal

def BERT_predict(
    source_texts,
    repetition_texts,
    language= Literal['nl','fr'],
    repetition_type= Literal['direct','self-repetition'],
    max_len: int = 256,
    return_all_probs: bool = True,
    device: Literal['cuda','cpu'] | None = None
):
    """
    Predicts repetitive vs non-repetitive character of candidate utterance pairs using a pretrained BERT classification model.
    Args:
        source_texts: List(str): List of all utterances candidate to be the source of a direct or self-repetitive pair
        repetition_texts: List(str): List of all utterances candidate to be the direct or self-repetition of a repetitive pair
        language: Language of the utterances, either 'nl' (Dutch) or 'fr' (French)
        repetition_type: Type of repetition, either 'direct' or 'self-repetition'. 
            For definitions, see https://osf.io/83bvw/overview?view_only=898ad193f8e54c62b730346238b63cf8.
        max_len: Maximum number of tokens in the source and repetitive utterance combined
        return_all_probs: Determines whether probabilities of each class are shown in the output
        device: Device to use for allocation of the model (cuda or cpu)


    """

     # Format examples as 'source_text***repetition_text'
    texts= format_texts(source_texts, repetition_texts)

    # Load model and tokenizer    
    models= {'nl': {'direct': 'NL_echo_model', 'self-repetition': 'NL_selfrep_model'},
             'fr': {'direct': 'FR_echo_model', 'self-repetition': 'FR_selfrep_model'} }
    model_load_dir= models[language][repetition_type]

    tokenizer = AutoTokenizer.from_pretrained('m0183394/AutomaticRepetitionDetection', subfolder= model_load_dir)

    model = AutoModelForSequenceClassification.from_pretrained('m0183394/AutomaticRepetitionDetection', subfolder= model_load_dir)
    model.eval()

    # Optional: send to GPU if available
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Tokenize
    enc = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    ).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

    # Format outputs
    id2label= model.config.id2label
    if return_all_probs:

        if isinstance(logits, torch.Tensor):
            # Keep computation on the same device; convert to numpy only at the end
            probs_t = torch.softmax(logits, dim=-1)
            probs = probs_t.detach().cpu().numpy()
        else:
            # logits is a NumPy array: use a numerically stable softmax
            # softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
            x = logits - logits.max(axis=-1, keepdims=True)
            exps = np.exp(x)
            probs = exps / exps.sum(axis=-1, keepdims=True)

        results = []
        for row in probs:
            results.append(
                [
                    {"label": id2label[int(i)], "prob": float(s)}
                    for i, s in enumerate(row)
                ]
            )
        output = results
    else:
        # Compute predicted class indices without outputting all probabilities
        if isinstance(logits, torch.Tensor):
            pred_ids_t = torch.argmax(logits, dim=-1)
            pred_ids = pred_ids_t.detach().cpu().numpy().tolist()
        else:
            pred_ids = np.argmax(logits, axis=-1).tolist()

        pred_labels = [id2label[int(pid)] for pid in pred_ids]
        output = pred_labels

    return output


# Predict repetitions using cosine similarities

from sentence_transformers import SentenceTransformer, util
import numpy as np
from glob import glob
import spacy
from typing import List, Optional, Literal
from itertools import product

class CosSim_predict:


    def __init__(
        self,
        source_texts: List[str],
        repetition_texts: List[str],
        sbert_model: Optional[str] = None,
        spacy_model: Optional[str]= None,
        language: str= None

    ):


        """Class to compute cosine similarities between candidate source-repetition pairs and predict repetitive vs non-repetitive class 
        based on them.

        Args:
            source_texts: List(str): List of all utterances candidate to be the source of a direct or self-repetitive pair
            repetition_texts: List(str): List of all utterances candidate to be the direct or self-repetition of a repetitive pair
            sbert_model (Optional[str]): Path or name of Sentence-BERT model for computing semantic vectors.
            spacy_model (Optional[str]): Path or name of SpaCy model for computing lexical and syntactic vectors.
            language (str): Language of the input data (language code as employed in SpaCy).
        """

        self.source_texts= source_texts
        self.repetition_texts= repetition_texts

        default_sbert_models= {'fr': SentenceTransformer('Lajavaness/sentence-camembert-large'),
                       'nl': SentenceTransformer('NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers')}

        self.language= language

        if not sbert_model:
            if language in ['nl','fr']:
                self.sbert_model= default_sbert_models[language]
            # User will be asked to specify SBERT model for other languages when initializing creation of semantic vectors 
            # or computation of semantic similarity
        else:
            self.sbert_model= sbert_model

        self.spacy_model= spacy_model if spacy_model else language + '_core_news_sm'
        self.nlp= spacy.load(self.spacy_model)





    def create_lexicon(self, data, unit: Literal['lemma', 'PoS'] = 'lemma', n_PoS: Optional[int] = 2):
        """ Converts a string into the lemmas or PoS-tags of its tokens
            Args:
                data (str): data to process
                unit (Literal['lemma', 'PoS']): chosen type of processing
                n_PoS (Optional[int]): size (n) of Part-of-Speech n-grams, defaults to 2
        """

        def extract_POS_ngrams(speech, n):
            # Extract n-grams of POS tags from the speech
            tokens = [token.pos_ for token in self.nlp(speech)]  
            ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
            if len(tokens) < n:
                ngrams = [tuple(tokens)]
            return ngrams        


        if unit== 'lemma':
            all_units= np.array([token.lemma_ for token in self.nlp(data)])
        elif unit== 'PoS':
            all_units= extract_POS_ngrams(data,n_PoS)

        return all_units


    def create_vectors_from_ling_unit(self,units_to_vectorize, all_units):
        """Creates numerical vectors from a list of linguistic units (e.g. lemma, POS n-gram),
        given a list of all units occurring in the entire file
        Args:
            units_to_vectorize (List(List)): List of utterances processed by ´create_lexicon()´
            all_units (List): Lexicon of all utterances processed by ´create_lexicon()´
        """

        # Identify all unique units across all intervals + sort for consistent dimension ordering
        unique_units= sorted(set(all_units))
        unit_index_mapping= {unit: idx for idx, unit in enumerate(unique_units)}

        # Create vectors
        vectors = np.vstack([
            np.bincount(
                [unit_index_mapping[unit] for unit in units if unit in unit_index_mapping],
                minlength=len(unique_units)
            )
            for units in units_to_vectorize
        ])

        return vectors



    def get_vectors(self, vector_type: Literal['lexical','syntactic','semantic'], n_PoS: Optional[int]=2, sbert_model= None):
        """
        Builds two np.arrays of vectors: one containing vectors of all utterances of self.source_texts, one containing those of self.rep_texts.
        Args:
            vector_type (Literal['lexical','syntactic','semantic']): Linguistic base of the vectors (lemmas, PoS-tags, sentence embeddings)
            n_PoS (Optional[int]): size (n) of Part-of-Speech n-grams, defaults to 2
            sbert_model (str): Overrides SentenceBERT model defined in class definition 
                                (if either sbert model provided, or default model selected for language 'nl' or 'fr')
        """

        if sbert_model:
            self.sbert_model= sbert_model

        type2unit= {'lexical': 'lemma', 'syntactic': 'PoS'}

        if vector_type != 'semantic':  
            total_string= ' '.join(self.source_texts + self.repetition_texts)

            all_units= self.create_lexicon(data= total_string, unit= type2unit[vector_type], n_PoS= n_PoS)
            if vector_type== 'syntactic':
                n_PoS_dummy= n_PoS
                while n_PoS_dummy >=2: # Also add encodings for n-grams with smaller n (for utterances with number of tokens < n)
                    all_units += self.create_lexicon(data= total_string, unit= type2unit[vector_type], n_PoS= n_PoS_dummy-1)
                    n_PoS_dummy -= 1
            source_units= [self.create_lexicon(data= text, unit= type2unit[vector_type], n_PoS= n_PoS) for text in self.source_texts]
            rep_units= [self.create_lexicon(data= text, unit= type2unit[vector_type], n_PoS= n_PoS) for text in self.repetition_texts]

            source_vectors= self.create_vectors_from_ling_unit(units_to_vectorize= source_units, all_units= all_units)
            rep_vectors= self.create_vectors_from_ling_unit(units_to_vectorize= rep_units, all_units= all_units)

        else:
            if not self.sbert_model:
                raise ValueError('Please specify the name of the SentenceBERT model you want to use')

            source_vectors= np.vstack([self.sbert_model.encode([utterance]) for utterance in self.source_texts])
            rep_vectors= np.vstack([self.sbert_model.encode([utterance]) for utterance in self.repetition_texts])

        def correct_data_type(vector):
            "Ensures correct data type for vectors for passing to sentence_transformers.utils.cos_sim"

            if vector.dtype != float:
                vector= vector.astype(np.float64)

            return vector

        source_vectors, rep_vectors= correct_data_type(source_vectors), correct_data_type(rep_vectors)

        return source_vectors, rep_vectors


    def get_cosine_similarities(self, vector_type: Literal['lexical','syntactic','semantic'], n_PoS: Optional[int]=2, sbert_model= None):
        """
        Calculates cosine similarities between each pair of source & repetition candidate vectors, outputted in a list of cos sim values.
        Args:
            vector_type (Literal['lexical','syntactic','semantic']): Linguistic base of the vectors (lemmas, PoS-tags, sentence embeddings)
            n_PoS (Optional[int]): size (n) of Part-of-Speech n-grams, defaults to 2
            sbert_model (str): Overrides SentenceBERT model defined in class definition 
                                (if either sbert model provided, or default model selected for language 'nl' or 'fr')
        """

        if sbert_model:
            self.sbert_model= sbert_model
        if 'semantic' in vector_type and not self.sbert_model:
            raise ValueError('Please specify the name of the SentenceBERT model you want to use')

        source_vectors, rep_vectors= self.get_vectors(vector_type=vector_type, n_PoS= n_PoS)
        cos_sims= [ float(util.cos_sim(v1, v2)) for v1, v2 in zip(source_vectors, rep_vectors) ]

        return cos_sims


    def predict_repetitions(self, repetition_type: Literal['direct','self-repetition'], 
                            vector_type: Literal['lexical','syntactic','semantic'],
                           n_PoS: Optional[int]=2, sbert_model= None, threshold: Optional[float]= None):

        """
        Predicts repetitive vs non-repetitive class for each pair of source & repetition candidates based on their cosine similarity.
        Args:
            repetition_type: Type of repetition, either 'direct' or 'self-repetition'. 
                For definitions, see https://osf.io/83bvw/overview?view_only=898ad193f8e54c62b730346238b63cf8.
            vector_type (Literal['lexical','syntactic','semantic']): Linguistic base of the vectors (lemmas, PoS-tags, sentence embeddings)
            n_PoS (Optional[int]): size (n) of Part-of-Speech n-grams, defaults to 2
            sbert_model (str): Overrides SentenceBERT model defined in class definition 
                                (if either sbert model provided, or default model selected for language 'nl' or 'fr')
            threshold (threshold: Optional[float]): float between -1 and 1 used as threshold for the cosine similarity of repetitive pairs.
                If language= 'nl' or 'fr' and (for syntactic vectors) n_PoS=2, the model uses thresholds as determined in our research
                (unless threshold is otherwise defined here).
                For other languages and/or n_PoS, threshold should be specified with this parameter.        
        """


        if sbert_model:
            self.sbert_model= sbert_model
        if 'semantic' in vector_type and not self.sbert_model:
            raise ValueError('Please specify the name of the SentenceBERT model you want to use')

        if self.language not in ['nl','fr'] or (not threshold and n_PoS != 2):
            raise ValueError(""""For languages other than 'nl' or 'fr' and/or Part-of-Speech n-grams with n != 2,
                             please provide your own theshold to evaluate cosine similarity in the ´threshold` parameter""")


        cos_sims= self.get_cosine_similarities(vector_type= vector_type)

        threshold_dict= {'lexical': {'direct': {'fr': 0.29, 'nl': 0.23}, 
                                 'self-repetition': {'fr': 0.88, 'nl': 0.92}},
            'syntactic': {'direct': {'fr': 0.23, 'nl': 0.21}, 
                                  'self-repetition': {'fr': 0.90, 'nl': 0.88}},
            'semantic': {'direct': {'fr': 0.39, 'nl': 0.37},
                                 'self-repetition': {'fr': 0.88, 'nl': 0.88}}
            }

        preds= [ cos_sim > threshold_dict[vector_type][repetition_type][self.language] for cos_sim in cos_sims]
        preds= [ 'repetitive' if pred==True else 'non-repetitive' for pred in preds]

        return preds


# Example use


#source_texts= ['ik ga naar de zee','mama','ik eet een appel']
#repetition_texts= ['ik ga naar de winkel','mama', 'jij eet een banaan']


#preds_BERT= BERT_predict(language= 'nl', repetition_type= 'self-repetition', source_texts= source_texts, repetition_texts=repetition_texts,
               #max_len=50, return_all_probs=False, device='cpu')

#print(preds_BERT)


#cossim_model= CosSim_predict(source_texts=source_texts, repetition_texts=repetition_texts,language='nl')

#preds_cossim= cossim_model.predict_repetitions(repetition_type='self-repetition',vector_type='semantic')

#print(preds_cossim)
