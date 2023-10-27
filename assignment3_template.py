
import matplotlib.pyplot as plt
import numpy as np
import lifereader
import tensorflow as tf
import math
import timeit

board = lifereader.readlife('BREEDER3.LIF', 2048)

plt.figure(figsize=(20,20))
plotstart=924
plotend=1124
plt.imshow(board[plotstart:plotend,plotstart:plotend])

plt.figure(figsize=(20,20))
plt.imshow(board)

#tf.config.set_visible_devices([], 'GPU')
tf.debugging.set_log_device_placement(True)

boardtf = tf.cast(board, dtype=tf.float16)

@tf.function
def runlife(board, iters):
    # Define the 3x3 filter for convolution to compute the number of neighbors
    filter = tf.constant([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=board.dtype)
    filter = tf.reshape(filter, [3, 3, 1, 1])

    for i in range(iters):
        # Convert board to boolean for logical operations
        bool_board = tf.cast(board, dtype=tf.bool)

        # Compute the number of neighbors for each cell using convolution
        neighbors = tf.nn.conv2d(tf.reshape(board, [1, board.shape[0], board.shape[1], 1]), filter, strides=[1, 1, 1, 1], padding='SAME')
        neighbors = tf.squeeze(neighbors)

        # Compute the 'survive' and 'born' tensors based on the game rules
        survive = tf.logical_and(bool_board, tf.logical_or(tf.equal(neighbors, 2), tf.equal(neighbors, 3)))
        born = tf.logical_and(tf.logical_not(bool_board), tf.equal(neighbors, 3))

        # Update the board
        board = tf.cast(tf.logical_or(survive, born), board.dtype)

    return board




tic = timeit.default_timer()
boardresult = runlife(boardtf, 1000);
toc = timeit.default_timer();
print("Compute time: " + str(toc - tic))
result = np.cast[np.int32](boardresult);
print("Cells alive at start: " + str(np.count_nonzero(board)))
print("Cells alive at end:   " + str(np.count_nonzero(result)))
print(np.count_nonzero(result))
plt.figure(figsize=(20,20))
plt.imshow(result[plotstart:plotend,plotstart:plotend])