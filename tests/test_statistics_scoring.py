''' Test Stats Scoring Module '''

from bananas.statistics.random import RandomState
from bananas.statistics.scoring import ScoringFunction
from .test_profiling import ProfilingTestCase, main


# pylint: disable=missing-docstring
class TestUtils(ProfilingTestCase):

    def test_score_accuracy(self):
        labels_true = [0] * 100
        labels_probs = [[1, 0, 0]] * 50 + [[0, 1, 0]] * 25 + [[0, 0, 1]] * 25

        scoring_function = ScoringFunction.create(ScoringFunction.ACCURACY)

        score = scoring_function(labels_true[:50], labels_probs[:50])
        self.assertEqual(score, 1)

        score = scoring_function(labels_true[:75], labels_probs[:75])
        self.assertEqual(score, 50 / 75)

        score = scoring_function(labels_true[:100], labels_probs[:100])
        self.assertEqual(score, 50 / 100)

    def test_score_r2(self):
        y_len = 1024
        y_len_mid = y_len // 2
        rng = RandomState(seed=0)
        y_true = rng.randn(y_len).tolist()
        y_pred = y_true[:y_len_mid] + rng.randn(y_len_mid).tolist()

        scoring_function = ScoringFunction.create(ScoringFunction.R2)

        score = scoring_function(y_true[:y_len_mid], y_pred[:y_len_mid])
        self.assertEqual(score, 1)

        score = scoring_function(y_true, y_pred)
        self.assertLess(score, 1)

    def test_score_precision(self):
        scoring_function = ScoringFunction.create(ScoringFunction.PRECISION)

        labels_true = [0] * 100
        labels_probs = [[1, 0, 0]] * 50 + [[0, 1, 0]] * 25 + [[0, 0, 1]] * 25

        score = scoring_function(labels_true[:50], labels_probs[:50])
        self.assertAlmostEqual(score, 1 / 3)

        score = scoring_function(labels_true[:75], labels_probs[:75])
        self.assertAlmostEqual(score, 1 / 3)

        score = scoring_function(labels_true[:100], labels_probs[:100])
        self.assertAlmostEqual(score, 1 / 3)

    def test_score_recall(self):
        scoring_function = ScoringFunction.create(ScoringFunction.RECALL)

        labels_true = [0] * 100
        labels_probs = [[1, 0, 0]] * 50 + [[0, 1, 0]] * 25 + [[0, 0, 1]] * 25

        score = scoring_function(labels_true[:50], labels_probs[:50])
        self.assertAlmostEqual(score, 1 / 3)

        score = scoring_function(labels_true[:75], labels_probs[:75])
        self.assertAlmostEqual(score, 2 / 9)

        score = scoring_function(labels_true[:100], labels_probs[:100])
        self.assertAlmostEqual(score, 1 / 6)

    def test_score_f1(self):
        scoring_function = ScoringFunction.create(ScoringFunction.F1)

        labels_true = [0] * 100
        labels_probs = [[1, 0, 0]] * 50 + [[0, 1, 0]] * 25 + [[0, 0, 1]] * 25

        score = scoring_function(labels_true[:50], labels_probs[:50])
        self.assertAlmostEqual(score, 1 / 3)

        score = scoring_function(labels_true[:75], labels_probs[:75])
        self.assertAlmostEqual(score, 4 / 15)

        score = scoring_function(labels_true[:100], labels_probs[:100])
        self.assertAlmostEqual(score, 2 / 9)

    def test_score_auroc(self):
        labels_true = [0] * 100
        labels_probs = [[1, 0, 0]] * 50 + [[0, 1, 0]] * 25 + [[0, 0, 1]] * 25

        scoring_function = ScoringFunction.create(ScoringFunction.AREA_UNDER_ROC)

        score = scoring_function(labels_true[:50], labels_probs[:50])
        self.assertGreater(score, .66)

        score = scoring_function(labels_true[:75], labels_probs[:75])
        self.assertGreater(score, .5)

        score = scoring_function(labels_true[:100], labels_probs[:100])
        self.assertLess(score, .5)


main()
