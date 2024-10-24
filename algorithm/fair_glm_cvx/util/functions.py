
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

def discretization(y, A, family, max_segments=100, discretization='equal_length'):
    YD = None

    if family == 'bernoulli':
        YD = y

    elif family == 'multinomial':
        YD = y.argmax(1)

    elif family == 'poisson':
        # find left limit
        y = y.astype(int)
        y_ll = np.where([np.all([y_ in y[A == a] and (y_+1) in y[A == a] and (y_+2) in y[A == a] for a in set(A)]) for y_ in np.arange(np.max(y))])[0][0]
        # find right limit
        y_rl = np.where([np.all([np.all([y__ in y[A == a] for y__ in range(y_ll, y_+1)]) for a in set(A)]) for y_ in np.arange(np.max(y))])[0][-1]
        YD = np.copy(y)
        if y_ll > y.min():
            YD[YD < y_ll] = y_ll
        if y_rl < y.max():
            YD[YD > y_rl] = y_rl

    elif family == 'normal':
        if discretization == 'equal_length':
            for n_segments in np.arange(max_segments, 0, -1):
                YD = np.digitize(y, np.linspace(y.min(), y.max(), n_segments + 1)[1:-1])
                check_valid = 0
                for a in set(A):
                    if len(set(YD[A == a])) != n_segments:
                        break
                    check_valid += 1

                if check_valid == len(set(A)):
                    break

        elif discretization == 'equal_count':
            for n_segments in np.arange(max_segments, 0, -1):
                YD = np.digitize(y, np.quantile(y, q=np.linspace(0, 1, n_segments + 1))[1:-1])

                check_valid = 0
                for a in set(A):
                    if len(set(YD[A == a])) != n_segments:
                        break
                    check_valid += 1

                if check_valid == len(set(A)):
                    break

        else:
            raise NotImplementedError(f'{family} is not a supported family!')

    if YD is None:
        raise Exception('Something went wrong with discretization')

    YD = OrdinalEncoder().fit_transform(YD.reshape(-1, 1)).flatten()
    return YD