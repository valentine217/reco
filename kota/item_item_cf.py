import pandas as pd
import math


class item_item_cf() :
    """
    define item-item based recommendation system
    """

    def __init__(self, df) :
        """
        initiate one parameter:
        df- pandas dataframe
        """
        self.df = df
        self.cols = self.df.columns.values[1 :]

    def norm(self) :
        """
        mean centered normalization
        """
        df = self.df.iloc[:, 1 :]
        df_norm = df.T - df.T.mean(axis=0)
        return df_norm.T

    def sim_func(self, df, item1, item2) :
        """
        cosine similirity function: sum(A.B)/(sqrt(sum(A^2))*sqrt(sum(A^2)))
        """
        nom = float((df[item1] * df[item2]).sum())
        denom = math.sqrt(df[item1].pow(2).sum()) * math.sqrt(df[item2].pow(2).sum())
        cos = nom / denom
        return cos


    def sim_matrix(self, target, is_norm=False) :
        """
        cosine similirity dict
        """
        dict_sim = {}
        if not is_norm :
            df = self.df
        else :
            df = self.norm()
        for col in self.cols :
            dict_sim[(target, col)] = self.sim_func(df, target, col)
        return dict_sim

    def prediction(self, user, is_norm=False) :
        """
        predict items for user
        """
        predictions = {}
        df = self.df
        df.index = df.User
        user_rated_items = [(key, value) for (key, value) in self.df.T.to_dict()[user].items() \
                            if not math.isnan(value)]
        for item in self.cols :
            values = 0
            scores = 0
            dict_sim = self.sim_matrix(item, is_norm=is_norm)
            for (key, value) in user_rated_items :
                if key == "User" :
                    pass
                else :
                    score = dict_sim[(item, str(key))]
                    if score < 0 :
                        score = 0
                    scores += score
                    values += value * score
            predictions[item] = float(values) / scores
            #print(predictions);
        return predictions

    def recom(self, n, dicts) :
        """
        recommend top n items
        """
        top_items = sorted(dicts, key=lambda x : dicts[x], reverse=True)[:n + 1]
        return top_items


if __name__ == "__main__" :
    df = pd.read_excel('Assignment 5.xlsx', sheet_name=0)
    recom = item_item_cf(df)
    # test
    target = "1: Toy Story (1995)"
    item = "1210: Star Wars: Episode VI - Return of the Jedi (1983)"
    dict_raw = recom.sim_matrix(target)
    print("Test cases:")
    print (dict_raw[(target, item)])
    dict_norm = recom.sim_matrix(target, is_norm=True)
    print(dict_norm[(target, item)], "\n")
    # most similar to Toy Story?
    print("Top 5 movies with most similirity to Toy Story:")
    print(recom.recom(5, dict_raw), "\n")
    # the highest recommended movie for user 5277?
    preds = recom.prediction(5277)
    print("Top 5 movies with highest recommendation to user 5277:")
    print(recom.recom(5, preds), "\n")
    # movie most similar to Toy Story?
    print("Top 5 movies with most similirity to Toy Story under normalization:")
    print(recom.recom(5, dict_norm), "\n")
    # the highest recommended movie for user 5277?
    preds = recom.prediction(5277, is_norm=True)
    print("Top 5 movies with highest recommendation to user 5277 under normalization:")
    print(recom.recom(5, preds), "\n")