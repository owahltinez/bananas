''' Test Preprocessing Module '''

from bananas.sampledata.local import load_bike, load_boston, load_california, load_titanic

from bananas.preprocessing.standard import StandardPreprocessor

from .test_profiling import ProfilingTestCase, main


# pylint: disable=missing-docstring
class TestUtils(ProfilingTestCase):

    def test_preprocess_datasets(self):
        trials = 10
        for dataset_loader in [load_bike, load_boston, load_california, load_titanic]:
            X_list, y_list, Xt_list = [], [], []
            for _ in range(trials):
                dataset, _ = dataset_loader(random_seed=0)
                preprocessor = StandardPreprocessor(
                    categorical=dataset.categorical, continuous=dataset.continuous)
                X, y = dataset.input_fn()
                X_list.append(X)
                y_list.append(y)
                Xt_list.append(preprocessor.fit(X).transform(X).tolist())

            for i in range(1, trials):
                self.assertListEqual(X_list[0], X_list[i])
                self.assertListEqual(y_list[0], y_list[i])
                self.assertListEqual(Xt_list[0], Xt_list[i])


main()
