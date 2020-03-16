import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fancyimpute import SoftImpute


def impute(data):
    row_bool = ~np.all(np.isnan(data), axis=1)
    col_bool = ~np.all(np.isnan(data), axis=0)
    data_filtered = data[row_bool, :][:, col_bool]
    data_imputed = SoftImpute().fit_transform(data_filtered)
    tmp = np.zeros([data_filtered.shape[0], data.shape[1]])
    tmp[:, col_bool] = data_imputed
    data[row_bool, :] = tmp
    data[np.isnan(data)] = 0


def test_out(filename, data):
    np.savetxt(filename + '.txt', data, fmt='%.3g')

    fig, ax = plt.subplots()
    ax.pcolor(data)
    plt.tight_layout()
    fig.savefig(filename + '.png')
    plt.close()


if __name__ == '__main__':
    year_cols = [str(year) for year in range(2000, 2016)]
    mortal_a = pd.read_csv('../out/mortal_a.csv', usecols=year_cols)
    mortal_b = pd.read_csv('../out/mortal_b.csv', usecols=year_cols)
    mortal_n = pd.read_csv('../out/mortal_n.csv', usecols=year_cols)

    mortal_a = mortal_a.values
    mortal_b = mortal_b.values
    mortal_n = mortal_n.values

    # test_out('out_a', mortal_a)
    impute(mortal_a[:, 9:])
    mortal_a[np.isnan(mortal_a)] = 0
    # test_out('out2_a', mortal_a)

    # test_out('out_b', mortal_b)
    impute(mortal_b)
    mortal_b[np.isnan(mortal_b)] = 0
    # test_out('out2_b', mortal_b)

    # test_out('out_n', mortal_n)
    impute(mortal_n)
    mortal_n[np.isnan(mortal_n)] = 0
    # test_out('out2_n', mortal_n)

    mortal_s = mortal_a + mortal_b + mortal_n
    test_out('out_s', mortal_s)
