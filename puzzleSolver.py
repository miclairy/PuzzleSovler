import cv2
import itertools
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import random

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

    while False:
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
    return (rho, phi)

# def pairs_are_opposite(piece, pair, opposite_pair):
#     pair_a = piece[pair[0] % len(piece)]
#     pair_b = piece[pair[1] % len(piece)]
#     opposite_a = piece[opposite_pair[0] % len(piece)]
#     opposite_b = piece[opposite_pair[1] % len(piece)]
#     opposite_line = opposite_a[0] - opposite_b[0]
#     line = pair_a[0] - pair_b[0]

#     dot = np.dot(np.transpose(line), np.transpose(opposite_line))
#     angle = np.arccos(dot / (line * opposite_line))

#     return False

def squared_90_degree_error_score(angle_a, angle_b):
    return abs(np.pi / 2 - abs(angle_a - angle_b)) ** 0.5

def angle_between(angle_a, angle_b):
    diff = abs(angle_a - angle_b)
    return min(diff, 2 * math.pi - diff)

def find_corner(piece):
    num_points = len(piece)
    polar_piece = []
    polar_piece_rho = []
    polar_piece_phi = []

    moment = cv2.moments(piece)
    piece_center = int(moment['m10'] / moment['m00']), int(moment['m01'] / moment['m00'])
    for point in piece:
        point_around_center_x = point[0][0] - piece_center[0]
        point_around_center_y = point[0][1] - piece_center[1]
        polar_point = cartesian2polar(point_around_center_x, point_around_center_y)
        polar_piece.append(polar_point)
        polar_piece_rho.append(polar_point[0])
        polar_piece_phi.append(polar_point[1])

    # plt.axes(projection = 'polar')
    for point in polar_piece:
        plt.polar(point[1], point[0], 'g.')

    peaks, _ = scipy.signal.find_peaks(polar_piece_rho + polar_piece_rho, prominence=1)

    # Peaks are pulled from the doubled signal, get all peaks with index less
    # than num_points and add peaks which wrap around but don't repeat past the
    # first peak in the first copy of the signal
    min_peak = min(peaks)
    main_peaks = [p for p in peaks if p < num_points]
    extra_peaks = [p % num_points for p in peaks if p >= num_points and p % num_points < min_peak]

    deduped_peaks = extra_peaks + main_peaks

    # Get all subsets of 4 peaks
    combinations = list(itertools.combinations(deduped_peaks, 4))

    # Score each combination, lower is better
    # We expect corners to be pi/2 radians from each other, square the error
    # for successive indices and sum to score each combination
    scores = []
    for combination in combinations:
        score = 0
        for r in range(4):
            phi_a = polar_piece_phi[combination[r]]
            phi_b = polar_piece_phi[combination[(r + 1) % 4]]
            diff = angle_between(phi_a, phi_b)
            score += ((math.pi / 2) - diff) ** 2
        scores.append(score)

    # Lowest score has phis distributed closest to pi/2 radians
    best_combination_idx = scores.index(min(scores))
    best_combination = combinations[best_combination_idx]

    for point_idx in best_combination:
        p = polar_piece[point_idx]
        plt.plot(p[1], p[0], 'bo', markersize=7)

    # corner_pairs = []
    # for peak_index in peaks:
    #     point_rho, point_phi = polar_piece[peak_index % len(polar_piece_rho)]
    #     plt.plot(point_phi, point_rho, 'rx')
    #     for other_index in peaks:
    #         other_rho, other_phi = polar_piece[other_index % len(polar_piece_rho)]
    #         # if abs(other_phi - point_phi) > np.pi / 2 - 0.3 and abs(other_phi - point_phi) < np.pi / 2 + 0.3:
    #         #     # plt.plot(point_phi, point_rho, 'b.')
    #         corner_pairs.append((peak_index, other_index))

    # errors = []
    # for pair in corner_pairs:
    #     for opposite_pair in corner_pairs:
    #         pair_a_rho, pair_a_phi = polar_piece[pair[0] % len(polar_piece)]
    #         pair_b_rho, pair_b_phi = polar_piece[pair[1] % len(polar_piece)]
    #         opposite_a_rho, opposite_a_phi = polar_piece[opposite_pair[0] % len(polar_piece)]
    #         opposite_b_rho, opposite_b_phi = polar_piece[opposite_pair[1] % len(polar_piece)]

    #         # eliminate same point used in both pairs
    #         point_set = {opposite_pair[0] % len(polar_piece), opposite_pair[1] % len(polar_piece), pair[0] % len(polar_piece), pair[1] % len(polar_piece)}
    #         if len(point_set) == 4:
    #             error = 0
    #             point_angles = [opposite_a_phi, opposite_b_phi, pair_a_phi, pair_b_phi]
    #             for point in point_angles:
    #                 for other in point_angles: # todo: order the phis then dont have to check all wrap around 360 degrees
    #                     error += squared_90_degree_error_score(point, other)
    #             errors.append([error, (pair_a_rho, pair_a_phi), (pair_b_rho, pair_b_phi), (opposite_a_rho, opposite_a_phi), (opposite_b_rho, opposite_b_phi) ])
    
    # sorted_errors = sorted(errors, key=lambda x : x[0])
                
    # print(sorted_errors[0])
    
    # for point in sorted_errors[0][1:]:
    #     plt.plot(point[1], point[0], 'b.')

    plt.show()


puzzle_holes = puzzle_holes()
puzzle_pieces1 = puzzle_pieces('cropped_puzzle_pieces.jpg')
puzzle_pieces2 = puzzle_pieces('cropped_puzzle_pieces_2.jpg')
puzzle_pieces = puzzle_pieces1 + puzzle_pieces2
i = random.randint(0, len(puzzle_pieces))
print('I is ', i)
find_corner(puzzle_pieces[i])
