import os
import torch
import argparse
from torch.autograd import Variable
from lexi.core.en_nrr.nrr import NRR
from lexi.core.en_nrr.metrics import evaluate_ranker
from lexi.core.en_nrr.features.feature_extractor_sr import FeatureExtractorSR
from lexi.config import RESOURCES, RANKER_DIR

lr = 0.0005
epochs = 100
dropout = 0.2

print("Loading resources")
feat_extractor = FeatureExtractorSR(RESOURCES['en']['nrr'])
print("Loaded google n-gram with %d words" % len(feat_extractor.google_frequency.google_frequencies))

print("Extracting training data features")
train_x, train_y = feat_extractor.get_features(RESOURCES['en']['nrr']['train'], True)

print("Extracting test data features")
test_x, test_y = feat_extractor.get_features(RESOURCES['en']['nrr']['test'], False)

print("Training NRR model")
nrr = NRR(train_x, train_y, dropout)
nrr.model.training = True
optimizer = torch.optim.Adam(nrr.model.parameters(), lr=lr)

for epoch in range(epochs):
    y_pred = nrr.model(nrr.train_x)
    loss = nrr.loss_fn(torch.cat(y_pred.unbind()), nrr.train_y)  # should be just y_pred
    if epoch % 20 == 0:
        loss_val = loss.data.cpu().numpy().tolist()
        print("MSE loss after %d iterations: %.2f" % (epoch, loss_val))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Ranking test data substitutions using NRR")
nrr.set_testing()
prediction_scores = [score[0] for score in nrr.predict(test_x).data.numpy()]

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

print("Evaluation")
p_at_1, pearson = evaluate_ranker(RESOURCES['en']['nrr']['test'], pred_rankings)
print("Metrics (P@1, Pearson): %f %f" % (p_at_1*100, pearson))

if not os.path.exists(RANKER_DIR):
    print('Saved model to %s' % (RANKER_DIR+'/default.bin'))
    os.makedirs(RANKER_DIR)

torch.save(nrr, RANKER_DIR+'/default.bin')