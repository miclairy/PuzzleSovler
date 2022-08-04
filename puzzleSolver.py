import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy

def puzzle_pieces(imageName):

    pieces1 = cv2.imread(imageName)
    greyPieces = cv2.cvtColor(pieces1, cv2.COLOR_BGR2GRAY)
    _, binaryPieces1 = cv2.threshold(greyPieces, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(binaryPieces1, kernel, iterations = 1)
    segmented_pieces = cv2.dilate(erosion, kernel, iterations = 1)

    contours, hierarchy = cv2.findContours(segmented_pieces, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    pieces_contours = []
    for i in range(len(contours)):
            contour = contours[i]
            if hierarchy[0,i,3] == -1:
                if cv2.contourArea(contour) > 50:
                    pieces_contours.append(contour)

    cv2.drawContours(pieces1, pieces_contours, -1, (0,255,0), 1)

    displayImage(pieces1)
    return pieces_contours


def displayImage(image):

    while True:
        small = cv2.resize(image, (0,0), fx=0.65, fy=0.65)
        cv2.imshow('Puzzle', small)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

def puzzle_holes():
    puzzle = cv2.imread('puzzle-image-whited.jpg')
    greyPuzzle = cv2.cvtColor(puzzle, cv2.COLOR_BGR2GRAY)
    bluredPuzzle = cv2.GaussianBlur(greyPuzzle, (5,5), 0)
    _, binaryPuzzle = cv2.threshold(greyPuzzle, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binaryPuzzle, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    puzzle_contours = []
    for i in range(len(contours)):
            contour = contours[i]
            if hierarchy[0,i,3] != -1:
                if cv2.contourArea(contour) > 2500:
                    puzzle_contours.append(contour)

    cv2.drawContours(puzzle, puzzle_contours, -1, (0,255,0), 1)

    displayImage(puzzle)

    return puzzle_holes

def cartesian2polar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def find_corner(piece):
    polar_piece = []
    polar_piece_rho = []

    moment = cv2.moments(piece)
    piece_center = int(moment['m10'] / moment['m00']), int(moment['m01'] / moment['m00'])
    print(piece_center)
    for point in piece:
        point_around_center_x = point[0][0] - piece_center[0]
        point_around_center_y = point[0][1] - piece_center[1]
        polar_point = cartesian2polar(point_around_center_x, point_around_center_y)
        polar_piece.append(polar_point)
        polar_piece_rho.append(polar_point[0])

    plt.axes(projection = 'polar')
    for point in polar_piece:
        plt.polar(point[1], point[0], 'g.')

    plt.polar(polar_piece[0][1], polar_piece[0][0], 'b.')

    peaks = scipy.signal.find_peaks(polar_piece_rho + polar_piece_rho, prominence=1)

    print(polar_piece_rho + polar_piece_rho)

    for peak_index in peaks[0]:
        plt.polar(polar_piece[peak_index % len(polar_piece_rho)][1], polar_piece[peak_index % len(polar_piece_rho)][0], 'rx')


    plt.show()


puzzle_holes = puzzle_holes()
puzzle_pieces1 = puzzle_pieces('cropped_puzzle_pieces.jpg')
puzzle_pieces2 = puzzle_pieces('cropped_puzzle_pieces_2.jpg')
puzzle_pieces = puzzle_pieces1 + puzzle_pieces2
i = 1
find_corner(puzzle_pieces[i])
