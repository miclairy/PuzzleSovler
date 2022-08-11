import cv2
import matplotlib.pyplot as plt
import random
import sovler

def show_piece(piece, corners):

    for point_idx in corners:
        point = piece[point_idx]
        plt.plot(point[0], point[1], 'bo', markersize=7)

    for point in piece:
        plt.plot(point[0], point[1], 'g.')

    plt.axis('equal')
    plt.show()

img_holes = cv2.imread('puzzle-image-whited.jpg')
img_pieces_1 = cv2.imread('cropped_puzzle_pieces.jpg')
img_pieces_2 = cv2.imread('cropped_puzzle_pieces_2.jpg')

puzzle_holes = sovler.puzzle_holes(img_holes)
puzzle_pieces = sovler.puzzle_pieces(img_pieces_1) + sovler.puzzle_pieces(img_pieces_2)

i = random.randint(0, len(puzzle_pieces))
print('I is ', i)
piece = puzzle_pieces[i]
corners = sovler.find_corners(piece)
show_piece(piece, corners)
