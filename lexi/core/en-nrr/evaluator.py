from lexi.core.mounica.nrr.features.feature_extractor_sr import FeatureExtractorSR
from lexi.core.mounica.nrr.nrr import NRR
import torch

class SingleNRR:
    
    def __init__(self, resources):
        self.feat_extractor = FeatureExtractorSR(resources)
        self.nrr = torch.load(resources['model'])
        
    def evaluate(self, sent, target, candidates):
        # Extracts features from input
        features = self.feat_extractor.get_features_single(sent, target, list(candidates))
        
        # Uses trianed input NRR to predict scores
        prediction_scores = [score[0] for score in self.nrr.predict(features).data.numpy()]

        # Sorts output prediction scores
        count = -1
        substitutes = candidates
        score_map = {}
        for sub in substitutes:
            score_map[sub] = 0.0

        for s1 in substitutes:
            for s2 in substitutes:
                if s1 != s2:
                    count += 1
                    score = prediction_scores[count]
                    score_map[s1] += score

        return sorted(score_map.keys(), key=score_map.__getitem__)