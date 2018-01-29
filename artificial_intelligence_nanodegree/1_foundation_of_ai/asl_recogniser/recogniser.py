from asl_data import AslDb
import my_model_selectors as mds
import helper as hp

asl = AslDb()  # initializes the database

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
    asl.df['grnd-lx'], asl.df['grnd-ly'])
asl.df['polar-rr'], asl.df['polar-rtheta'] = hp.cart_2_polar(
    asl.df['grnd-rx'], asl.df['grnd-ry'])

asl.df['delta-lx'] = asl.df.groupby(['video'])['left-x'].apply(hp.filled_diff)
asl.df['delta-ly'] = asl.df.groupby(['video'])['left-y'].apply(hp.filled_diff)
asl.df['delta-rx'] = asl.df.groupby(['video'])['right-x'].apply(hp.filled_diff)
asl.df['delta-ry'] = asl.df.groupby(['video'])['right-y'].apply(hp.filled_diff)


# Collect all features
features_nose = ['nose-x', 'nose-y']
features_hand = ['left-x', 'left-y', 'right-x', 'right-y']
features_original = features_nose + features_hand
features_ground = ['grnd-rx', 'grnd-ry', 'grnd-lx', 'grnd-ly']
features_norm = ['norm-rx', 'norm-ry', 'norm-lx', 'norm-ly']
features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']
features_polar_norm_theta = ['polar-rr',
                             'polar-rtheta', 'polar-lr', 'polar-ltheta']
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


features_combined = features_ground + \
    features_polar + features_delta


features_polar_norm_theta = ['norm-polar-rr',
                             'polar-rtheta', 'norm-polar-lr', 'polar-ltheta']

features_combined_norm = features_norm_ground + \
    features_norm_polar + features_norm_delta

features_delta_norm_ground = ['delta-norm-' + f for f in features_ground]

for trans, orig in zip(features_delta_norm_ground, features_ground):
    asl.df[trans] = asl.df.groupby('speaker')[orig].apply(
        lambda x: hp.z_score(x)).groupby('video').apply(hp.filled_diff)


features_custom = (features_norm_ground +
                   features_polar + features_delta_norm_ground)

# Nose didn't help despite used in 'not'
features_test = (features_norm_ground +
                 features_norm_polar + features_delta_norm_ground)


# # Visualise the transformed features
# hp.plt.subplot(241)
# hp.plot_location([
#     [asl.df['left-x'], asl.df['left-y']],
#     [asl.df['right-x'], asl.df['right-y']],
#     [asl.df['nose-x'], asl.df['nose-y']]
# ], title='original')

# hp.plt.subplot(242)
# hp.plot_location([
#     [asl.df['grnd-rx'], asl.df['grnd-ry']],
#     [asl.df['grnd-lx'], asl.df['grnd-ly']]
# ], title='ground')

# hp.plt.subplot(243)
# hp.plot_location([
#     [asl.df['norm-grnd-rx'], asl.df['norm-grnd-ry']],
#     [asl.df['norm-grnd-lx'], asl.df['norm-grnd-ly']]
# ], title='norm-ground')

# hp.plt.subplot(244)
# hp.plot_location([
#     [asl.df['norm-lx'], asl.df['norm-ly']],
#     [asl.df['norm-rx'], asl.df['norm-ry']]
# ], title='normalised')

# hp.plt.subplot(245)
# hp.plot_location([
#     [asl.df['polar-lr'], asl.df['polar-ltheta']],
#     [asl.df['polar-rr'], asl.df['polar-rtheta']]
# ], title='polar')

# hp.plt.subplot(246)
# hp.plot_location([
#     [asl.df['norm-polar-lr'], asl.df['norm-polar-ltheta']],
#     [asl.df['norm-polar-rr'], asl.df['norm-polar-rtheta']]
# ], title='norm-polar')

# hp.plt.subplot(247)
# hp.plot_location([
#     [asl.df['delta-lx'], asl.df['delta-ly']],
#     [asl.df['delta-rx'], asl.df['delta-ry']]
# ], title='delta-x')

# hp.plt.subplot(248)
# hp.plot_location([
#     [asl.df['norm-delta-lx'], asl.df['norm-delta-ly']],
#     [asl.df['norm-delta-rx'], asl.df['norm-delta-ry']]
# ], title='Norm-delta-x')

# hp.plt.show()


########################################################################
# Train the model
########################################################################

# results = list()
# n_sample = 10
# feature_dict = {'features_original': features_original,
#                 'features_combined': features_combined,
#                 'features_combined_norm': features_combined_norm,
#                 'features_test': features_test,
#                 'features_custom': features_custom}
# start = time.time()
# for feature_name, features in feature_dict.items():
#     for model_selector in [mds.SelectorBIC, mds.SelectorDIC, mds.SelectorCV]:
#         info = 'currently training with "{}" features, and "{}" selector'
#         print(info.format(feature_name, model_selector.__name__))
#         training = asl.build_training(features)
#         test_set = asl.build_test(features)
#         # summary = list()
#         for _ in range(n_sample):
#             with warnings.catch_warnings():
#                 warnings.simplefilter('ignore')
#                 model_string = model_selector.__name__ + ':' + feature_name
#                 models = hp.train_all_words(training, features, model_selector)

#                 probabilities, guesses = recognize(models, test_set)
#                 current_wer = hp.calculate_wer(guesses, test_set)
#                 # summary.append(current_wer)

#                 # results[model_string] = summary
#                 # results.update({model_string: summary})
#                 results.append({'model_selector': model_selector.__name__,
#                                 'feature_name': feature_name,
#                                 'wer': current_wer})
# end = time.time()
# print(pd.DataFrame(results))
# print('Total Time spend: {}'.format(
#     str(datetime.timedelta(seconds=end - start))))


# features_combined_multi_norm = ['mnorm-' + f for f in features_combined]
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# multi_scaled_feautres = scaler.fit_transform(asl.df[features_combined])
# for name, values in zip(features_combined_multi_norm, multi_scaled_feautres.T):
#     asl.df[name] = values

# features_test = (features_polar + features_delta_norm_ground)


# training = asl.build_training(features_test)
# test_set = asl.build_test(features_test)
# with warnings.catch_warnings():
#     warnings.simplefilter('ignore')
#     model_string = model_selector.__name__ + ':' + feature_name
#     models = hp.train_all_words(training, features, model_selector)

#     probabilities, guesses = recognize(models, test_set)
#     current_wer = hp.calculate_wer(guesses, test_set)

feature_dict = {
    'features_original': features_original,
    'features_combined': features_combined,
    'features_combined_norm': features_combined_norm,
    'features_test': features_test,
    'features_custom': features_custom
}
# model_selector_list = [mds.SelectorBIC, mds.SelectorDIC, mds.SelectorCV]
model_selector_list = [mds.SelectorBIC, mds.SelectorDIC]
model_comparison_results = hp.train_models(asl=asl,
                                           features=feature_dict,
                                           model_selectors=model_selector_list,
                                           n_sample=1)
