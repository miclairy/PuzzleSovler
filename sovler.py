import cv2
import itertools
import numpy as np
import scipy
import math

def puzzle_pieces(image):
    greyPieces = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binaryPieces1 = cv2.threshold(greyPieces, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(binaryPieces1, kernel, iterations = 1)
    segmented_pieces = cv2.dilate(erosion, kernel, iterations = 1)

    contours, hierarchy = cv2.findContours(segmented_pieces, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    pieces_contours = []
    for i in range(len(contours)):
            contour = contours[i][:,0,:]
            if hierarchy[0,i,3] == -1:
                if cv2.contourArea(contour) > 50:
                    pieces_contours.append(contour)

    # cv2.drawContours(pieces1, pieces_contours, -1, (0,255,0), 1)
    # displayImage(pieces1)

    return pieces_contours

def puzzle_holes(image):
    greyPuzzle = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binaryPuzzle = cv2.threshold(greyPuzzle, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binaryPuzzle, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    puzzle_contours = []
    for i in range(len(contours)):
            contour = contours[i]
            if hierarchy[0,i,3] != -1:
                if cv2.contourArea(contour) > 2500:
                    puzzle_contours.append(contour)

    # cv2.drawContours(puzzle, puzzle_contours, -1, (0,255,0), 1)
    # displayImage(puzzle)

    return puzzle_holes

def cartesian2polar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def angle_between(angle_a, angle_b):
    diff = abs(angle_a - angle_b)
    return min(diff, 2 * math.pi - diff)

def find_corners(piece):
    polar_piece_rho = []
    polar_piece_phi = []

    moment = cv2.moments(piece)
    piece_center = int(moment['m10'] / moment['m00']), int(moment['m01'] / moment['m00'])
    for point in piece:
        point_around_center_x = point[0] - piece_center[0]
        point_around_center_y = point[1] - piece_center[1]
        rho, phi = cartesian2polar(point_around_center_x, point_around_center_y)
        polar_piece_rho.append(rho)
        polar_piece_phi.append(phi)

    # Find peaks in doubled signal to get potential start/end peaks
    peaks, _ = scipy.signal.find_peaks(polar_piece_rho + polar_piece_rho, prominence=1)
    deduped_peaks = list({p % len(piece) for p in peaks})
    deduped_peaks.sort()

    if len(deduped_peaks) < 4:
        return ()

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

    return best_combination

def find_ins_outs(piece, corners):
    ins_outs = []
    for idx in range(len(corners)):
        a = idx
        b = (idx + 1) % len(corners)
        
        # project points to line to get distance
        # find max and min
        # may need cross product to determine side?
