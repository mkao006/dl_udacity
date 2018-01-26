import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states,
                                    covariance_type="diag",
                                    n_iter=1000,
                                    random_state=self.random_state,
                                    verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(
                    self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(
                    self.this_word, num_states))
            return None

    def compute_ll(self, c):
        ''' This method to compute log-likelihood handles failure.
        '''
        try:
            ll = self.base_model(c).score(self.X, self.lengths)
        except:
            ll = -1e5
        return ll


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        def bic(component):
            logL = self.compute_ll(component)
            logN = np.log(self.X.shape[0])
            n_features = self.X.shape[1]
            n_params = (component * (component - 1) +
                        2 * n_features * component)
            bic = -2 * logL + n_params * logN
            return bic

        components = list(range(self.min_n_components,
                                self.max_n_components + 1))
        best_num_components = min(components, key=bic)

        return self.base_model(best_num_components)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        components = list(range(self.min_n_components,
                                self.max_n_components + 1))

        ll = [self.compute_ll(c) for c in components]
        dic = np.matmul(np.array(ll), np.eye(len(ll)) * 2 - 1)
        best_num_components = components[np.argmax(dic)]
        return self.base_model(best_num_components)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        components = list(range(self.min_n_components,
                                self.max_n_components + 1))
        n_splits = 3
        penalisation = -1e5

        # The default of the LL is negative infinite.
        component_ll = [float('-inf')] * len(components)

        for component_idx, c in enumerate(components):
            if(len(self.sequences) < n_splits):
                break

            # Split the data with KFold
            split_method = KFold(
                random_state=self.random_state, n_splits=n_splits)
            # Instead of -Inf, we use -1e5 to penalise incase all component has
            # at least a single failure.
            cv_ll = [penalisation] * n_splits
            for split_idx, (cv_train_idx, cv_test_idx) in enumerate(split_method.split(self.sequences)):
                train_x, train_length = combine_sequences(
                    cv_train_idx, self.sequences)
                test_x, test_length = combine_sequences(
                    cv_test_idx, self.sequences)

                # Train the model on the training set
                #
                # NOTE (Michael): If the training is successful, we
                #                 can then calculate the likelihood on
                #                 the test set. However, if the
                #                 training fails, we will use the
                #                 default penalisation value.
                try:
                    hmm_model = GaussianHMM(n_components=c,
                                            covariance_type="diag",
                                            n_iter=1000,
                                            random_state=self.random_state,
                                            verbose=False).fit(train_x, train_length)

                    # compute the likelihood on the test sets and
                    # append to the cv set
                    cv_ll[split_idx] = hmm_model.score(test_x, test_length)
                except:
                    pass

            # take the average of the likelihood for the current component
            component_ll[component_idx] = np.average(cv_ll)

        if any([np.isfinite(c) for c in component_ll]):
            best_num_components = components[np.argmax(component_ll)]
        else:
            best_num_components = self.n_constant
        return self.base_model(best_num_components)
