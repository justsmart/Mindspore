import scipy.io as io
import numpy as np
import mindspore as ms


def cal(arr):
    re = 1
    for x in arr:
        re *= x
    return re


class DMVGDataset:
    def __init__(self, x, c):
        self.x = x
        self.c = c

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.c[idx]


def get_data(dataname='NH_face', view=0, pairedrate=0.1, fold=0, batch_size=32):
    if dataname == 'NH_face':
        archs = [
            [5, 9, 10, 15],
            [5, 5, 8, 10],
            [4, 8, 8, 13],
        ]
        dims = [6750, 2000, 3304]
        configs = {}
        configs['arch'] = archs[view]
        configs['dim_x'] = cal(archs[view])
        configs['dim_c'] = sum(dims) - dims[view]

        data = io.loadmat(f"./data/{dataname}.mat")
        folds = io.loadmat(f"./data/NH_face_del_{pairedrate}.mat")['folds']

        X = data['X']
        x = []
        x.append((X[0, 0] * 100).reshape(X[0, 0].shape[0], 1, X[0, 0].shape[1]))
        x.append((X[0, 1] / 100).reshape(X[0, 1].shape[0], 1, X[0, 1].shape[1]))
        x.append((X[0, 2] ** (1 / 2) * 10).reshape(X[0, 2].shape[0], 1, X[0, 2].shape[1]))

        mask = folds[0, fold]
        ind_train = mask[:, view] == 1
        ind_test = mask[:, view] == 0

        x_train = x[view][ind_train]
        c_train = []
        for i in range(len(x)):
            if i != view:
                c_train.append((x[i] * mask[:, i].reshape((-1, 1, 1)))[ind_train])
        c_train = np.concatenate(c_train, axis=-1)

        x_test = x[view][ind_test]
        c_test = []
        for i in range(len(x)):
            if i != view:
                c_test.append((x[i] * mask[:, i].reshape((-1, 1, 1)))[ind_test])
        c_test = np.concatenate(c_test, axis=-1)

        train_dataset = DMVGDataset(x_train, c_train)
        print("len(train_dataset):", len(train_dataset))
        train_dataloader = ms.dataset.GeneratorDataset(train_dataset, column_names=['data', 'condition'], shuffle=False,
                                                       num_parallel_workers=2).batch(batch_size)
        test_dataset = DMVGDataset(x_test, c_test)
        print("len(test_dataset):", len(test_dataset))
        test_dataloader = ms.dataset.GeneratorDataset(test_dataset, column_names=['data', 'condition'], shuffle=False,
                                                      num_parallel_workers=2).batch(batch_size)
        return train_dataloader, test_dataloader, configs


def get_Qdata(dataname='NH_face', view=0, pairedrate=0.1, fold=0, batch_size=64):
    dims = [6750, 2000, 3304]
    archs = [
        [5, 9, 10, 15],
        [5, 5, 8, 10],
        [4, 8, 8, 13],
    ]
    configs = {}
    configs['arch'] = archs[view]
    configs['dim_x'] = cal(archs[view])
    configs['dim_c'] = sum(dims) - dims[view]

    data = io.loadmat(f"./data/{dataname}.mat")
    X = data['X']
    x = []

    folds = io.loadmat(f"./data/NH_face_del_{pairedrate}.mat")['folds']

    mask = folds[0, fold]

    x.append((X[0, 0] * 100).reshape(X[0, 0].shape[0], 1, X[0, 0].shape[1]))
    x.append((X[0, 1] / 100).reshape(X[0, 1].shape[0], 1, X[0, 1].shape[1]))
    x.append((X[0, 2] ** (1 / 2) * 10).reshape(X[0, 2].shape[0], 1, X[0, 2].shape[1]))

    ind_test = mask[:, view] == 0
    x_test = x[view][ind_test]
    c_test = []
    for i in range(len(x)):
        if i != view:
            c_test.append((x[i] * mask[:, i].reshape((-1, 1, 1)))[ind_test])
    c_test = np.concatenate(c_test, axis=-1)
    test_dataset = DMVGDataset(x_test, c_test)
    print("len(test_dataset):", len(test_dataset))
    test_dataloader = ms.dataset.GeneratorDataset(test_dataset, column_names=['data', 'condition'], shuffle=False,
                                                  num_parallel_workers=2).batch(batch_size)

    def Q(x, mask):
        Qx = [[] for _ in range(len(x))]
        Qmask = []
        for i in range(mask.shape[0]):
            ni = mask[i].sum()
            if ni == len(dims):
                # generate Qx
                for v in range(len(x)):
                    Qx[v].append(x[v][i:i + 1].repeat(2 ** ni - 2 - ni, 0))
                # generate Qmask
                for j in range(1, int(2 ** ni - 1)):
                    if np.log2(j) == int(np.log2(j)):
                        pass
                    else:
                        tmp = [int(char) for char in reversed(bin(j)[2:])]
                        tmp = tmp + [0] * int(ni - len(tmp))
                        qmask = np.zeros([1, len(x)])
                        qmask[0, mask[i] == 1] = tmp
                        Qmask.append(qmask)
        for v in range(len(x)):
            Qx[v] = np.concatenate(Qx[v], axis=0)
        Qmask = np.concatenate(Qmask, axis=0)
        return Qx, Qmask

    Qx, Qmask = Q(x, mask)
    for v in range(len(x)):
        Qx[v] = np.concatenate([x[v], Qx[v]], axis=0)
    Qmask = np.concatenate([mask, Qmask], axis=0)

    ind_train = Qmask[:, view] == 1
    x_train = Qx[view][ind_train]
    c_train = []
    for i in range(len(x)):
        if i != view:
            c_train.append((Qx[i] * Qmask[:, i].reshape((-1, 1, 1)))[ind_train])
    c_train = np.concatenate(c_train, axis=-1)
    train_dataset = DMVGDataset(x_train, c_train)
    print("len(train_dataset):", len(train_dataset))
    train_dataloader = ms.dataset.GeneratorDataset(train_dataset, column_names=['data', 'condition'], shuffle=True,
                                                   num_parallel_workers=2).batch(batch_size)
    return train_dataloader, test_dataloader, configs
