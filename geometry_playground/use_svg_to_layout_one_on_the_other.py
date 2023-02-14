import cairosvg
import os
from img_strech import make_img_center, show_and_destroy_window, show_longest_distance
import cv2
import numpy as np
import glob
import shutil

DEBUG = 0


def longest_distance(contour):
    max_distance = 0
    for i in range(len(contour)):
        for j in range(i + 1, len(contour)):
            distance = np.linalg.norm(contour[i] - contour[j])
            if distance > max_distance:
                max_distance = distance
                p1, p2 = contour[i], contour[j]
    return max_distance, p1, p2


def find_correct_size(dst_size, full_file_path_png):
    full_file_path_svg = full_file_path_png[:-3] + 'svg'
    dst_size = round(dst_size, 1)
    print(f'dst_size: {dst_size}')

    dir_full_path = os.path.dirname(full_file_path_png)
    trials_directory = os.path.join(dir_full_path, 'trials')
    os.mkdir(trials_directory)

    only_filename = os.path.basename(full_file_path_png)

    img_circle = cv2.imread(full_file_path_png, cv2.IMREAD_GRAYSCALE)
    img_circle_centered = make_img_center(img_circle)

    img_inverse_colors = cv2.bitwise_not(img_circle_centered)

    kernel = np.ones((10, 10), np.uint8)
    dilation = cv2.dilate(img_inverse_colors, kernel, iterations=1)

    if DEBUG:
        show_and_destroy_window(winname="bitwise_not", mat=dilation)

    contours, _ = cv2.findContours(image=dilation, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    low = 0.5
    high = 5
    print(f'number of contours: {len(contours)}')
    if len(contours) > 1:
        diameter, p1, p2 = longest_distance(contours[1])
    else:
        diameter, p1, p2 = longest_distance(contours[0])
    diameter = round(diameter, 1)
    print(f'Diameter at the beginning is: {diameter}')

    version = 1

    while abs(diameter - dst_size) > 10:
        mid_scale = round((low + high) / 2, 2)
        filename = f'{only_filename[:-4]}_{version}.png'
        dst_file_name = os.path.join(dir_full_path, 'trials', filename)

        cairosvg.svg2png(
            url=full_file_path_svg,
            write_to=dst_file_name,
            scale=mid_scale,
            background_color='white'
        )

        img_circle = cv2.imread(dst_file_name, cv2.IMREAD_GRAYSCALE)
        img_circle_centered = make_img_center(img_circle)
        # show_and_destroy_window(winname=f'img_circle_centered modified size (Try {version})', mat=img_circle_centered)

        diameter = round(show_longest_distance(img_circle_centered), 1)

        if diameter < dst_size:  # make it larger
            low = mid_scale + 0.1
            low = round(low, 1)
        elif diameter > dst_size:  # make it smaller
            high = mid_scale - 0.1
            high = round(high, 1)

        print(f'The new diameter is: {diameter}, low = {low}, high = {high}')
        version += 1

    contours, _ = cv2.findContours(image=dilation, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    low = 0.5
    high = 5
    print(f'number of contours: {len(contours)}')
    if len(contours) > 1:
        diameter, p1, p2 = longest_distance(contours[1])
    else:
        diameter, p1, p2 = longest_distance(contours[0])

    diameter = round(diameter, 1)

    correct_image = os.path.join(dir_full_path, filename)
    shutil.copy(src=dst_file_name,
                dst=correct_image)

    shutil.rmtree(trials_directory)
    # cv2.line(img=img_inverse_colors, pt1=tuple(p1[0]), pt2=tuple(p2[0]), color=(255, 0, 0), thickness=2)
    #
    # show_and_destroy_window(winname='img_inverse_colors', mat=img_inverse_colors)

    return correct_image


def find_correct_angle():
    pass


def find_correct_size_second_approach():
    pass


if __name__ == '__main__':
    path_circle_input = '/home/gil_diy/PycharmProjects/matplotlib_examples/geometry_playground/handwritten_drawn_by_me_imgs/circle_input_svg/*.svg'
    img_path_rec = '/home/gil_diy/PycharmProjects/matplotlib_examples/geometry_playground/handwritten_drawn_by_me_imgs/rectangle/rec0.jpg'

    img_rec = cv2.imread(img_path_rec, cv2.IMREAD_GRAYSCALE)
    img_rec_centered = make_img_center(img_rec)
    distance_in_pixels = show_longest_distance(img_rec_centered)

    for idx, circle_file in enumerate(glob.glob(path_circle_input)):
        dirname = os.path.dirname(circle_file)
        file = os.path.basename(circle_file)

        img_path_circle = os.path.join(dirname, f'{file[:-4]}.png')

        cairosvg.svg2png(
            url=circle_file,
            write_to=img_path_circle,
            scale=1,
            background_color='white'
        )
        print(f'{idx}) img_path_circle = {img_path_circle}')

        file_path = find_correct_size(dst_size=distance_in_pixels,
                                      full_file_path_png=img_path_circle)

        print(f'correct image size is: {file_path}')

        find_correct_angle()

        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img_circle_centered = make_img_center(img)

        img_arr = cv2.bitwise_and(img_rec_centered, img_circle_centered)
        cv2.imshow(winname='blend', mat=img_arr)
        cv2.waitKey()
        cv2.destroyWindow("blend")
        dir = '/home/gil_diy/PycharmProjects/matplotlib_examples/geometry_playground/blend'
        cv2.imwrite(filename=os.path.join(dir, f'output{idx}.png'), img=img_arr)
    # show_and_destroy_window(winname='img_rec_centered', mat=res)
