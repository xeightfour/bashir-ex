import math
import os
import sys

import cv2
import numpy as np
from scipy.stats import gaussian_kde


GAUSSIAN_SIZE = 11
MIN_GRADIENT = 100
BW_CUT = 150
LEAST_CHORDS = 80
AREA_ERROR = 0.25


def mostCrowdedValue(data):
    data = np.array(data)

    kde = gaussian_kde(data, bw_method=None)

    x_grid = np.linspace(data.min(), data.max(), 1000)
    density = kde(x_grid)

    return x_grid[np.argmax(density)]


def findDisks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gaussian_1d = cv2.getGaussianKernel(GAUSSIAN_SIZE, 0) # auto variance
    gaussian_kernel = gaussian_1d @ gaussian_1d.T

    blurred = cv2.filter2D(gray, -1, gaussian_kernel)

    # Compute gradients using Sobel
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    H, W = gray.shape

    vote = np.zeros((H, W), dtype=int)
    chord_avg = np.zeros((H, W), dtype=float)

    step = 1
    max_steps = max(H, W) * 2

    for y in range(0, H, step):
        for x in range(0, W, step):
            dx = grad_x[y, x]
            dy = grad_y[y, x]

            length = np.sqrt(dx**2 + dy**2)

            if length > MIN_GRADIENT and blurred[y, x] <= BW_CUT:
                dx = math.sqrt(2) * dx / length # <:
                dy = math.sqrt(2) * dy / length

                yn, xn = y, x
                points = []

                for _ in range(max_steps):
                    if not (0 <= int(xn) < W and 0 <= int(yn) < H) or (blurred[int(yn), int(xn)] > BW_CUT):
                        break

                    y2, x2 = int(yn), int(xn)

                    vote[y2, x2] += 1

                    chord_avg[y2, x2] = (chord_avg[y2, x2] * (vote[y2, x2]-1) + math.sqrt((y2-y)**2 + (x2-x)**2)) / vote[y2, x2]

                    points.append([y2, x2])

                    # Move towards -gradient
                    xn += -dx
                    yn += -dy

    for i in range(H):
        for j in range(W):
            if vote[i, j] < LEAST_CHORDS:
                vote[i, j] = 0
            else:
                vote[i, j] = 1

    temp_grid = cv2.normalize(vote, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    temp_grid = temp_grid.astype(np.uint8)

    _, thresh = cv2.threshold(temp_grid, 250, 255, cv2.THRESH_BINARY)
    org_thresh = thresh.copy()
    thresh = cv2.filter2D(thresh, -1, gaussian_kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    radi_list = np.zeros(num_labels)

    # Finding radius for each centroid
    for cid in range(1, num_labels):
        comp_cords = np.where(labels == cid)
        points = np.column_stack((comp_cords[0], comp_cords[1]))

        print(f"--- Finding radius for component {cid} ({len(points)} pixels) ---")

        cnt = 0
        #unused = []
        for pt in points:
            if org_thresh[pt[0], pt[1]] > 250:
                radi_list[cid] += chord_avg[pt[0], pt[1]]
                #unused.append(chord_avg[pt[0], pt[1]])
                cnt += 1

        if cnt == 0:
            print("This shouldn't have happened :/")
        else:
            radi_list[cid] /= cnt

        #radi_list[cid] = mostCrowdedValue(unused)
        #print(unused)

    final_ans = 0

    # Filtering false positives
    if len(radi_list) > 1:

        _, dark_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        areas = [-1]

        for center, rad in zip(centroids[1:], radi_list[1:]):
            circle_mask = np.zeros((H, W), dtype=np.uint8)

            cint = (int(center[0]), int(center[1]))
            rint = int(rad)

            # Draw a filled white circle on the mask
            cv2.circle(circle_mask, cint, rint, 255, -1)

            darks_inside = cv2.bitwise_and(dark_mask, dark_mask, mask=circle_mask)

            area = cv2.countNonZero(darks_inside)
            areas.append(area)

        for i in range(1, len(areas)):
            if abs(1 - areas[i] / (math.pi * radi_list[i]**2)) < AREA_ERROR:
                final_ans += 1

                # Draw found circle to img
                cv2.circle(img, (int(centroids[i][0]), int(centroids[i][1])), int(radi_list[i]), (0, 225, 0), 2)
                cv2.circle(img, (int(centroids[i][0]), int(centroids[i][1])), 2, (0, 30, 225), -1)


    return img, final_ans

def showImage(img, title):
    cv2.imshow(title, img)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

def loadImage(file_name):
    if not (os.path.isfile(file_name) and os.access(file_name, os.R_OK)):
        print(f"'{file_name}' does not exist or isn't readable.")
        return None

    img = cv2.imread(file_name)
    if img is None:
        print(f"Failed to load '{file_name}'.")
        return None

    return img


def main(args):
    if len(args) == 0:
        print("No image provided!")
        return

    img = loadImage(args[0])
    if not img is None:
        output, answer = findDisks(img)
        print(f"Found [[ {answer} ]] solid circles <:")
        showImage(output, "Result")


if __name__ == "__main__":
    main(sys.argv[1:])
