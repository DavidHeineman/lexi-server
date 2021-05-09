import os
from lexi.core.en_nrr.nrr import NRR
from lexi.core.en_nrr.metrics import evaluate_ranker
from lexi.core.en_nrr.features.feature_extractor_fast import FeatureExtractorFast
from lexi.core.simplification.lexical_en import MounicaNRR, MounicaRanker
from lexi.config import RESOURCES, NRR_DIR, NRR_MODEL_PATH_TEMPLATE, NRR_PATH_TEMPLATE, RANKER_PATH_TEMPLATE, RANKER_DIR

lr = 0.0005
epochs = 100
dropout = 0.2
dim = 49 # 600 for normal extractor, 588 for fast extractor, 49 without binning

# Makes NRR directory
if not os.path.exists(NRR_DIR):
    os.makedirs(NRR_DIR)

# Makes RANKER directory
if not os.path.exists(RANKER_DIR):
    os.makedirs(RANKER_DIR)

print("Loading resources")
feat_extractor = FeatureExtractorFast(RESOURCES['en']['nrr'])
print("Loaded google n-gram with %d words" % len(feat_extractor.google_frequency.google_frequencies))

print("Extracting training data features")
train_x, train_y = feat_extractor.get_features(RESOURCES['en']['nrr']['train'], True)

print("Extracting test data features")
test_x, test_y = feat_extractor.get_features(RESOURCES['en']['nrr']['test'], False)

print("Training NRR model")
nrr = MounicaNRR('default', feat_extractor, dimensionality=dim)
nrr.train_model(train_x, train_y, epochs, lr)

print("Ranking test data substitutions using NRR")
nrr.model.set_testing()
prediction_scores = [score[0] for score in nrr.model.predict(test_x).data.numpy()]

count = -1
pred_rankings = []
for line in open(RESOURCES['en']['nrr']['test'], encoding='utf-8'):
    line = line.strip().split('\t')
    substitutes = [sub.strip().split(':')[1].strip() for sub in line[3:]]
    score_map = {}
    for sub in substitutes:
        score_map[sub] = 0.0

    for s1 in substitutes:
        for s2 in substitutes:
            if s1 != s2:
                count += 1
                score = prediction_scores[count]
                score_map[s1] += score

    pred_rankings.append(sorted(score_map.keys(), key=score_map.__getitem__))
p_at_1, pearson = evaluate_ranker(RESOURCES['en']['nrr']['test'], pred_rankings)
print("Metrics (P@1, Pearson): %f %f" % (p_at_1*100, pearson))

nrr.save()
print('Saved model to %s' % NRR_MODEL_PATH_TEMPLATE.format('default'))
print('Saved NRR wrapper object to %s' % NRR_PATH_TEMPLATE.format('default'))

rank = MounicaRanker('default', nrr)
rank.save('default')
print('Saved MounicaRanker object to %s' % RANKER_PATH_TEMPLATE.format('default'))