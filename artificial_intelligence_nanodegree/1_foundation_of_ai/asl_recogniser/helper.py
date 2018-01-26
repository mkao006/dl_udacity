import numpy as np
import pandas as pd
import warnings
import datetime
import time
import matplotlib.pyplot as plt
from asl_data import SinglesData
from my_recognizer import recognize


def plot_location(location_list, legend=['left hand', 'right hand', 'nose'],
                  title='', cmap=plt.get_cmap('tab20')):
    ''' Scatter plot of list of points with mean add.
    '''
    n = len(location_list)

    handler = []
    for ind, location in enumerate(location_list):
        points = plt.scatter(x=location[0], y=location[1],
                             color=cmap.colors[ind * 2 + 1], alpha=0.2)
        # NOTE (Michael): Maybe we can try to plot the density
        # loc = np.vstack(location)
        # z = gaussian_kde(loc)(loc)
        # density = plt.contour(location[0], location[1], z)
        mean = plt.scatter(location[0].mean(), location[1].mean(),
                           s=100, color=cmap.colors[ind * 2])
        handler.append(mean)
    plt.xlabel('')
    plt.ylabel('')
    plt.gca().invert_yaxis()
    plt.legend(handler, legend[:n])
    plt.title(title)
    # plt.show()


def z_score(x):
    ''' Calculate the Z score
    '''
    return (x - x.mean()) / x.std()


def cart_2_polar(x, y):
    ''' Convert points in cartesian to polar coordinates.
    '''
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(x, y)
    return r, phi


def filled_diff(x):
    ''' Take the difference of a series then replace missing values with zero.
    '''
    return x.diff().fillna(0)


def show_errors_summary(guesses: list, test_set: SinglesData):
    """ Print WER and sentence differences in tabular form

    :param guesses: list of test item answers, ordered
    :param test_set: SinglesData object
    :return:
        nothing returned, prints error report

    WER = (S+I+D)/N  but we have no insertions or deletions for isolated words so WER = S/N
    """
    S = 0
    N = len(test_set.wordlist)
    num_test_words = len(test_set.wordlist)
    if len(guesses) != num_test_words:
        print("Size of guesses must equal number of test words ({})!".format(
            num_test_words))
    for word_id in range(num_test_words):
        if guesses[word_id] != test_set.wordlist[word_id]:
            S += 1

    print("\n**** WER = {}".format(float(S) / float(N)))
    print("Total correct: {} out of {}".format(N - S, N))


def train_all_words(data, features, model_selector):
    # Experiment here with different feature sets defined in part 1
    sequences = data.get_all_sequences()
    Xlengths = data.get_all_Xlengths()
    model_dict = {}
    for word in data.words:
        model = model_selector(sequences, Xlengths, word,
                               n_constant=3, random_state=np.random.randint(1000)).select()
        model_dict[word] = model
    return model_dict


def calculate_wer(guesses: list, test_set: SinglesData):
    """ Print WER and sentence differences in tabular form

    :param guesses: list of test item answers, ordered
    :param test_set: SinglesData object
    :return:
        nothing returned, prints error report

    WER = (S+I+D)/N  but we have no insertions or deletions for isolated words so WER = S/N
    """
    S = 0
    N = len(test_set.wordlist)
    num_test_words = len(test_set.wordlist)
    if len(guesses) != num_test_words:
        print("Size of guesses must equal number of test words ({})!".format(
            num_test_words))
    for word_id in range(num_test_words):
        if guesses[word_id] != test_set.wordlist[word_id]:
            S += 1

    return float(S) / float(N)


def train_models(asl,
                 features: dict,
                 model_selectors: list,
                 n_sample=10):

    results = list()
    start = time.time()
    for feature_name, features in features.items():
        for model_selector in model_selectors:
            info = 'currently training with "{}" features, and "{}" selector'
            print(info.format(feature_name, model_selector.__name__))
            training = asl.build_training(features)
            test_set = asl.build_test(features)

            for _ in range(n_sample):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    models = train_all_words(
                        training, features, model_selector)

                    probabilities, guesses = recognize(models, test_set)
                    current_wer = calculate_wer(guesses, test_set)

                    results.append({'model_selector': model_selector.__name__,
                                    'feature_name': feature_name,
                                    'wer': current_wer})
    end = time.time()
    print('Total Time spend: {}'.format(
        str(datetime.timedelta(seconds=end - start))))
    return pd.DataFrame(results)
