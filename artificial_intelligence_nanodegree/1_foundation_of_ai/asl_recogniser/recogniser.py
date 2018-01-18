import pandas as pd
from sklearn.preprocessing import normalize
from asl_data import AslDb
import my_model_selectors as mds
from my_recognizer import recognize
from asl_utils import show_errors
import helper as hp

asl = AslDb()  # initializes the database

# Visualise the raw data
hp.plot_location([
    [asl.df['left-x'], asl.df['left-y']],
    [asl.df['right-x'], asl.df['right-y']],
    [asl.df['nose-x'], asl.df['nose-y']]
])

########################################################################
# Calculate New Features
########################################################################

# Define new feature grnd-ry as distance of posture to nose. That is,
# use nose as reference point (0, 0).
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

# Normalise the distance
asl.df['norm-lx'] = asl.df.groupby('speaker')['left-x'].apply(hp.z_score)
asl.df['norm-ly'] = asl.df.groupby('speaker')['left-y'].apply(hp.z_score)
asl.df['norm-rx'] = asl.df.groupby('speaker')['right-x'].apply(hp.z_score)
asl.df['norm-ry'] = asl.df.groupby('speaker')['right-y'].apply(hp.z_score)


# Convert cartesian to polar coordinates
asl.df['polar-lr'], asl.df['polar-ltheta'] = hp.cart_2_polar(
    asl.df['left-x'], asl.df['left-y'])
asl.df['polar-rr'], asl.df['polar-rtheta'] = hp.cart_2_polar(
    asl.df['right-x'], asl.df['right-y'])

# Calculate frame differences
# asl.df['delta-lx'] = asl.df.groupby('speaker')['left-x'].apply(hp.filled_diff)
# asl.df['delta-ly'] = asl.df.groupby('speaker')['left-y'].apply(hp.filled_diff)
# asl.df['delta-rx'] = asl.df.groupby('speaker')['right-x'].apply(hp.filled_diff)
# asl.df['delta-ry'] = asl.df.groupby('speaker')['right-y'].apply(hp.filled_diff)

asl.df['delta-lx'] = asl.df.groupby(['video']
                                    )['left-x'].apply(hp.filled_diff)
asl.df['delta-ly'] = asl.df.groupby(['video']
                                    )['left-y'].apply(hp.filled_diff)
asl.df['delta-rx'] = asl.df.groupby(['video']
                                    )['right-x'].apply(hp.filled_diff)
asl.df['delta-ry'] = asl.df.groupby(['video']
                                    )['right-y'].apply(hp.filled_diff)


# Visualise the transformed features
hp.plt.subplot(221)
hp.plot_location([
    [asl.df['grnd-rx'], asl.df['grnd-ry']],
    [asl.df['grnd-lx'], asl.df['grnd-ly']]
])

hp.plt.subplot(222)
hp.plot_location([
    [asl.df['norm-lx'], asl.df['norm-ly']],
    [asl.df['norm-rx'], asl.df['norm-ry']]
])

hp.plt.subplot(223)
hp.plot_location([
    [asl.df['polar-lr'], asl.df['polar-ltheta']],
    [asl.df['polar-rr'], asl.df['polar-rtheta']]
])

hp.plt.subplot(224)
hp.plot_location([
    [asl.df['delta-lx'], asl.df['delta-ly']],
    [asl.df['delta-rx'], asl.df['delta-ry']]
])

hp.plt.show()

# Collect all features
features_nose = ['nose-x', 'nose-y']
features_original = ['left-x', 'left-y', 'right-x', 'right-y']
features_ground = ['grnd-rx', 'grnd-ry', 'grnd-lx', 'grnd-ly']
features_norm = ['norm-rx', 'norm-ry', 'norm-lx', 'norm-ly']
features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']
features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']


features_norm_ground = ['norm-' + f for f in features_ground]

for norm, orig in zip(features_norm_ground, features_ground):
    asl.df[norm] = asl.df.groupby('speaker')[orig].apply(hp.z_score)

features_norm_polar = ['norm-' + f for f in features_polar]

for norm, orig in zip(features_norm_polar, features_polar):
    asl.df[norm] = asl.df.groupby('speaker')[orig].apply(hp.z_score)

features_norm_delta = ['norm-' + f for f in features_delta]

for norm, orig in zip(features_norm_delta, features_delta):
    asl.df[norm] = asl.df.groupby('speaker')[orig].apply(hp.z_score)


features_custom = features_ground + \
    features_polar + features_delta


features_custom_norm = features_norm_ground + \
    features_norm_polar + features_norm_delta


features_delta_norm_ground = ['delta-norm-' + f for f in features_ground]

for trans, orig in zip(features_delta_norm_ground, features_ground):
    asl.df[trans] = asl.df.groupby('speaker')[orig].apply(
        lambda x: hp.z_score(x)).groupby('video').apply(hp.filled_diff)


hp.plt.show()

features_best = (features_norm_ground +
                 features_polar + features_delta_norm_ground)

# Nose didn't help despite used in 'not'
features_test = (features_nose + features_norm_ground +
                 features_polar + features_delta_norm_ground)

########################################################################
# Train the model
########################################################################


# TODO Choose a feature set and model selector
features = features_test  # change as needed
model_selector = mds.SelectorBIC  # change as needed

# TODO Recognize the test set and display the result with the show_errors
# method
training = asl.build_training(features)
models = hp.train_all_words(training, features, model_selector)
test_set = asl.build_test(features)
probabilities, guesses = recognize(models, test_set)
print(' '.join(features))
show_errors(guesses, test_set)
