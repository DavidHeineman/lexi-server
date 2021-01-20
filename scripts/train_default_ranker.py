import torch
import argparse
from torch.autograd import Variable
from lexi.core.mounica.nrr.nrr import NRR
from lexi.core.mounica.nrr.metrics import evaluate_ranker
from lexi.core.mounica.nrr.config import RankerConfig
from lexi.core.mounica.nrr.features.feature_extractor_sr import FeatureExtractorSR
from lexi.config import RESOURCES, RANKER_DIR

args = RESOURCES['en']['nrr']

lr = 0.0005
epochs = 100
dropout = 0.2

print("Loading resources")
feat_extractor = FeatureExtractorSR(args)

print("Extracting training data features")
train_x, train_y = feat_extractor.get_features(args['train'], True)

print("Extracting test data features")
test_x, test_y = feat_extractor.get_features(args['test'], False)

print("Ranking test data substitutions using NRR")
nrr.set_testing()
prediction_scores = [score[0] for score in nrr.predict(test_x).data.numpy()]

count = -1
pred_rankings = []
for line in open(args['test'], encoding='utf-8'):
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

print("Evaluation")
p_at_1, pearson = evaluate_ranker(args['test'], pred_rankings)
print("Metrics (P@1, Pearson): %f %f" % (p_at_1*100, pearson))

torch.save(nrr, RANKER_DIR+'/default.bin')