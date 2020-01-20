#!/usr/bin/env python3
# -*-coding:utf-8 -*

import numpy as np

def mat_mat_prod(x, y):
    """Computes the product of two non-empty numpy.ndarray,
    for-loop. The two arrays must have compatible dimensions.
    Args:
    x: has to be an numpy.ndarray, a matrix of dimension m
    y: has to be an numpy.ndarray, a vector of dimension n
    Returns:
    The product of the matrices as a matrix of dimension m
    None if x or y are empty numpy.ndarray.
    None if x and y does not share compatibles dimensions.
    Raises:
    This function should not raise any Exception.
    """
    if x.size == 0 or y.size == 0 or (x.size != 0 and x.shape[1] != y.shape[0]):
        return None
    ft_prod = np.zeros((x.shape[0], y.shape[1]), dtype=int)
    # iterate through rows of X
    for i in range(len(x)):
       # iterate through columns of Y
       for j in range(len(y[0])):
           # iterate through rows of y
           for k in range(len(y)):
               ft_prod[i][j] += x[i][k] * y[k][j]
    return ft_prod

    # for abscissa, vect_x in enumerate(x):
    #     # print(f"vector_x: {vect_x}")
    #     # print(f"abscissa: {abscissa}")
    #     # print(25 * "-")
    #     ft_temp = 0
    #     # for ordinate, vect_y in enumerate(y[:, abscissa]):
    #     #     print(f"vector_y: {vect_y}")
    #     #     print(f"ordinate: {ordinate}")
    #     ft_temp = 0
    #     # print(f"vector_x.shape: {vect_x.shape}")
    #     # print(f"vector_y.shape: {y[:, abscissa].shape}")
    #     vect_y = y[:, abscissa]
    #     ordinate = 0
    #     for x_item, y_item in np.dstack((vect_x, vect_y))[0]:
    #         ft_temp += x_item * y_item
    #         ordinate += 1
    #         # ft_temp += vect_x[ordinate] * vect_y[abscissa]
    #     print(f"res: {res}")
    #     res.append(ft_temp)
    #     # ft_prod[ordinate, ordinate] = int(ft_temp)
    #     print(25 * "-")
    #     #     ft_temp = 0
    #     #     for x_item, y_item in np.dstack((vect_x, y.reshape(vect_x.shape)))[0]:
    #     #         ft_temp += x_item * y_item
    #     # ft_prod[abscissa, ordinate] = int(ft_temp)
    # ft_prod = np.array(res)
    # ft_prod.reshape(x.shape[0], y.shape[1])
    # return ft_prod
    # return mat_vec_prod(x, y)


if __name__ == '__main__':
    W = np.array([
        [-8, 8, -6, 14, 14, -9, -4],
        [2, -11, -2, -11, 14, -2, 14],
        [-13, -2, -5, 3, -8, -4, 13],
        [2, 13, -14, -15, -14, -15, 13],
        [2, -1, 12, 3, -7, -3, -6]])
    Z = np.array([
        [-6, -1, -8, 7, -8],
        [7, 4, 0, -10, -10],
        [7, -13, 2, 2, -11],
        [3, 14, 7, 7, -4],
        [-1, -3, -8, -4, -14],
        [9, -14, 9, 12, -7],
        [-9, -4, -10, -3, 6]])
    print(mat_mat_prod(W, Z))
    # array([[45, 414, -3, -202, -163],
    #        [-294, -244, -367, -79, 62],
    #        [-107, 140, 13, -115, 385],
    #        [-302, 222, -302, -412, 447],
    #        [108, -33, 118, 79, -67]])
    print(W.dot(Z))
    # array([[45, 414, -3, -202, -163],
    #        [-294, -244, -367, -79, 62],
    #        [-107, 140, 13, -115, 385],
    #        [-302, 222, -302, -412, 447],
    #        [108, -33, 118, 79, -67]])
    print(mat_mat_prod(Z, W))
    # array([[148, 78, -116, -226, -76,
    #         7, 45],
    #        [-88, -108, -30, 174, 364, 109, -42],
    #        [-126, 232, -186, 184, -51, -42, -92],
    #        [-81, -49, -227, -208, 112, -176, 390],
    #        [70,
    #         3, -60, 13, 162, 149, -110],
    #        [-207, 371, -323, 106, -261, -248, 83],
    #        [200, -53, 226, -49, -102, 156, -225]])
    print(Z.dot(W))
    # array([[148, 78, -116, -226, -76,
    #         7, 45],
    #        [-88, -108, -30, 174, 364, 109, -42],
    #        [-126, 232, -186, 184, -51, -42, -92],
    #        [-81, -49, -227, -208, 112, -176, 390],
    #        [70,
    #         3, -60, 13, 162, 149, -110],
    #        [-207, 371, -323, 106, -261, -248, 83],
    #        [200, -53, 226, -49, -102, 156, -225]])