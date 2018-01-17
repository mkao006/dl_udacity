import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    def scoring_wrapper(model, idx):
        try:
            return model.score(*test_set.get_item_Xlengths(idx))
        except:
            return float('-inf')

    probabilities = [
        {model_key: scoring_wrapper(model_value, word_idx)
         for model_key, model_value in models.items()}
        for word_idx, word in enumerate(test_set.wordlist)]

    guesses = [max(score.keys(), key=lambda x:score[x])
               for score in probabilities]

    return probabilities, guesses
