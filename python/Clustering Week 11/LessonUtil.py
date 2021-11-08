#
# common code given to the students
#



from sklearn.preprocessing import LabelEncoder
import sklearn.metrics as metrics
import pandas as pd

class RoboTester(object):

    def __init__(self, robo_batter, path=None):

        if path is None:
            path = path_for_data('predict.csv')

        df = pd.read_csv(path)
        self.robo_batter = robo_batter

        # not necessary, for the rand score, but useful example
        encoder = LabelEncoder()
        df['code'] = encoder.fit_transform(df['a'])

        self.df = df

    def get_distributions(self):
        centers = self.robo_batter.km.cluster_centers_

        score = [{} for i in range(0, len(centers))]
        for idx, cluster_num in enumerate(self.robo_batter.labels):
            predict = cluster_num
            actual = self.df['a'][idx]

            s = score[predict]
            v = s.get(actual, 0)
            s[actual] = v + 1

        return score

    def rand_index_score(self):
        # Perfect labeling is scored 1.0:
        # either works
        return metrics.adjusted_rand_score(self.df['code'], self.robo_batter.labels)
        return metrics.adjusted_rand_score(self.df['a'], self.robo_batter.labels)
