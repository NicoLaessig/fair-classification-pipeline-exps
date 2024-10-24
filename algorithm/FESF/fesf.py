from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from numpy import delete, vstack, hstack, min, array
from numpy.random import seed as np_seed, permutation



class FESF:

    def __init__(
        self, 
        base_model="LR",
        K=200,
        seed=1102,
    ):

        self.K = K
        self.seed = seed
        np_seed(self.seed)
        self.models = []
        if base_model == "LR":
            for j in range(self.K+1):
                self.models.append(LogisticRegression(random_state=self.seed))
        elif base_model == "RF":
            for j in range(self.K+1):
                self.models.append(RandomForestClassifier(random_state=self.seed))
        elif base_model == "SVC":
            for j in range(self.K+1):
                self.models.append(SVC(random_state=self.seed))
        elif base_model == "KNC":
            for j in range(self.K+1):
                self.models.append(KNeighborsClassifier())
        else:
            raise Exception("Unknown base model.")

    def fit(self, Xl, yl, sl, Xu, su):

        np_seed(self.seed)

        self.idx_del = []
        for j in range(Xl.shape[1]):
            if (sum(Xl[:, j] == sl) == Xl.shape[0]) & \
            (sum(Xu[:, j] == su) == Xu.shape[0]):
                self.idx_del.append(j)
        Xl = delete(Xl, self.idx_del, axis=1)
        Xu = delete(Xu, self.idx_del, axis=1)

        self.models[0].fit(Xl, yl)
        yu = self.models[0].predict(Xu)

        X = vstack((Xl, Xu))
        y = hstack((yl, yu))
        s = hstack((sl, su))

        X00 = X[(s==0) & (y==0), :]
        X01 = X[(s==0) & (y==1), :]
        X10 = X[(s==1) & (y==0), :]
        X11 = X[(s==1) & (y==1), :]

        y00 = y[(s==0) & (y==0)]
        y01 = y[(s==0) & (y==1)]
        y10 = y[(s==1) & (y==0)]
        y11 = y[(s==1) & (y==1)]

        s00 = s[(s==0) & (y==0)]
        s01 = s[(s==0) & (y==1)]
        s10 = s[(s==1) & (y==0)]
        s11 = s[(s==1) & (y==1)]

        n00 = len(y00)
        n01 = len(y01)
        n10 = len(y10)
        n11 = len(y11)

        ns = min((n00, n01, n10, n11))

        X_sample = X
        y_sample = y

        for j in range(self.K):

            if ns > 0:

                b00 = permutation([True] * ns + [False] * (n00-ns))
                b01 = permutation([True] * ns + [False] * (n01-ns))
                b10 = permutation([True] * ns + [False] * (n10-ns))
                b11 = permutation([True] * ns + [False] * (n11-ns))

                X_sample = vstack(
                    (
                        X00[b00, :], 
                        X01[b01, :], 
                        X10[b10, :], 
                        X11[b11, :], 
                    )
                )

                y_sample = hstack(
                    (
                        y00[b00], 
                        y01[b01], 
                        y10[b10], 
                        y11[b11], 
                    )
                )

                s_sample = hstack(
                    (
                        s00[b00], 
                        s01[b01], 
                        s10[b10], 
                        s11[b11], 
                    )
                )

            self.models[j+1].fit(X_sample, y_sample)
        return X_sample, y_sample, s_sample

    def predict(self, Xt):

        Xt = delete(Xt, self.idx_del, axis=1)
        preds = array([0]*Xt.shape[0])
        for j in range(self.K):
            preds = preds + self.models[j+1].predict(Xt)
        
        return (preds > 0.5*self.K) + 0



        
