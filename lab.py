#!/usr/bin/env python3
from json import load
import math

from PIL import Image as Image

# NO ADDITIONAL IMPORTS ALLOWED!

def get_pixel(image, x, y):
    return image['pixels'][image['width'] * x + y]

def set_pixel(image, x, y, c):
    image['pixels'][image['width'] * x + y] = c

def apply_per_pixel(image, func):
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': image['pixels'][:],
    }        
    for x in range(image['height']):
        for y in range(image['width']):
                color = get_pixel(image, x, y)
                newcolor = func(color)
                set_pixel(result, x, y, newcolor)
    return result


def inverted(image):
    return apply_per_pixel(image, lambda c: 255-c)


# HELPER FUNCTIONS

# This function creates a dictonary of coordinates relative to the origin in key value pairs like (0,1): 1.2
def coords_from_kernel(kernel):
    kernel_layers = math.floor(len(kernel)**(1/2)/2)
    x, y = (0, 0)
    xs = list(range(x - kernel_layers, x + kernel_layers + 1))
    ys = list(range(y - kernel_layers, y + kernel_layers + 1))
    coords = [(x, y) for x in xs for y in ys]
    dict = {}
    for i, coord in enumerate(coords):
        dict[(coord)] = kernel[i]
    return dict

def get_pixel_zero(image, x, y):
    # get height and width of image
    w, h = image['width'], image['height']

    # Check if coordinate is a valid coordinate if it is return color if not return 0
    if (y >= 0 and y < w) and (x >= 0 and x < h):
        return image['pixels'][x*w + y]
    else:
        return 0

def get_pixel_extend(image, x, y):
     # get height and width of image
    w, h = image['width'], image['height']

    # Check if coordinate is a valid coordinate if it is return color if not return 0
    if (y >= 0 and y < w) and (x >= 0 and x < h):
        return image['pixels'][x*w + y]
    #There are only a couple sections where the out of bounds pixel can be. So let's do all the cases for them like in the below picture

    x2, y2 = 0, 0
    #Is the pixel to the left?
    if y < 0:
        #Is the pixel in section 1?
        if x < 0:
            x2 = 0
        #Is the pixel in section 4?
        elif x < h:
            x2 = x
        #if not the first two cases, then it must be in section 6
        else:
            x2 = h - 1
    
    # Is the pixel to the right?
    elif y >= w - 1:
        # Is the pixel in section 3?
        if x < 0:
            y2 = w - 1
        # Is the pixel in section 5?
        elif x < h:
            x2 = x
            y2 = w - 1
        # if not the first two cases, then it must be in section 8
        else:
            x2 = h - 1 
            y2 = w - 1

    # If the pixel is not on the left or right then it must be on top or bottom
    # Is the pixel in section 2?
    elif x < 0:
        y2 = y
    #Only other possibility is if the point is in section 7
    else:
        x2 = h - 1
        y2 = y
    return image['pixels'][x2 * w + y2]

def get_pixel_wrap(image, x, y):
    w, h = image['width'], image['height']
    # Use mod to wrap around for x and y coordinate and return the 1D index
    return image['pixels'][(x % h) * w + y % w]

def correlate(image, kernel, boundary_behavior):
    """
    Compute the result of correlating the given image with the given kernel.
    `boundary_behavior` will one of the strings 'zero', 'extend', or 'wrap',
    and this function will treat out-of-bounds pixels as having the value zero,
    the value of the nearest edge, or the value wrapped around the other edge
    of the image, respectively.

    if boundary_behavior is not one of 'zero', 'extend', or 'wrap', return
    None.

    Otherwise, the output of this function should have the same form as a 6.101
    image (a dictionary with 'height', 'width', and 'pixels' keys), but its
    pixel values do not necessarily need to be in the range [0,255], nor do
    they need to be integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    DESCRIBE YOUR KERNEL REPRESENTATION HERE
    """
    img = {
        'height': image['height'],
        'width': image['width'],
        'pixels': image['pixels'][:]
    }
    if boundary_behavior == 'zero':
        for x in range(image['height']):
            for y in range(image['width']):
                new_val = sum([get_pixel_zero(image, x + i, y + j) * kernel[(i, j)] for i, j in kernel.keys()])
                set_pixel(img, x, y, new_val)
        return img

    if boundary_behavior == 'extend':
        for x in range(image['height']):
            for y in range(image['width']):
                new_val = sum([get_pixel_extend(image, x + i, y+ j) * kernel[(i, j)] for i, j in kernel.keys()])
                set_pixel(img, x, y, new_val)
        return img
    
    if boundary_behavior == 'wrap':
        for x in range(image['height']):
            for y in range(image['width']):
                new_val = sum([get_pixel_wrap(image, x +i , y + j) * kernel[(i, j)] for i, j in kernel.keys()])
                set_pixel(img, x, y, new_val)
        return img

def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the 'pixels' list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    for i, pixel in enumerate(image['pixels']):
        #If the pixel is not an integer then round it
        if not isinstance(pixel, int):
            image['pixels'][i] = round(pixel)
        if pixel > 255:
            image['pixels'][i] = 255
        elif pixel < 0:
            image['pixels'][i] = 0
# FILTERS

#Return a list of vlur kernel
def get_box_blur_kernel(n):
    return [1/(n*n) for i in range(1, n*n + 1)]

def blurred(image, n):
    """
    Return a new image representing the result of applying a box blur (with
    kernel size n) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)
    kernel = get_box_blur_kernel(n)
    kernel_coords = coords_from_kernel(kernel)

    # then compute the correlation of the input image with that kernel
    img = correlate(image, kernel_coords, 'extend')

    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.
    round_and_clip_image(img)
    return img

def sharpened(image, n):
    #Get blurred image
    blurred_image = blurred(image, n)
    #scale the original image
    img = {'height': image['height'],
            'width': image['width'],
            'pixels': [2 * pixel for pixel in image['pixels']]}
    #Subtract the blurred image from the scaled image
    img['pixels'] = [sum((x, -y)) for (x, y) in zip(img['pixels'], blurred_image['pixels'])]
    round_and_clip_image(img)
    return img

def edges(image):
    kx = [-1, 0, 1,
            -2, 0, 2,
            -1, 0, 1]
    ky = [-1, -2, -1,
        0,  0,  0,
        1,  2,  1]
    kx_coords = coords_from_kernel(kx)
    ky_coords = coords_from_kernel(ky)
    ox = correlate(image, kx_coords, 'extend')
    oy = correlate(image, ky_coords, 'extend')
    img = {'height': image['height'],
            'width': image['width'],
            'pixels': [round((x**2 + y**2)**(1/2)) for (x, y) in zip(ox['pixels'], oy['pixels'])]}
    round_and_clip_image(img)
    return img

# HELPER FUNCTIONS FOR LOADING AND SAVING IMAGES

def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith('RGB'):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == 'LA':
            pixels = [p[0] for p in img_data]
        elif img.mode == 'L':
            pixels = list(img_data)
        else:
            raise ValueError('Unsupported image mode: %r' % img.mode)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_greyscale_image(image, filename, mode='PNG'):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode='L', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == '__main__':
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.
    i = {'height': 3,
        'width': 3,
        'pixels': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
#     kernel = [0.00,  -0.07,   0.00,
# -0.45,   1.20,  -0.25,
#  0.00,  -0.12,   0.00]
#     kernel_coords = coords_from_kernel(kernel)
#     print(correlate(i, kernel_coords, 'zero'))
    # kernel = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # ]
    # k = coords_from_kernel(kernel)
    # image = load_greyscale_image('test_images/pigbird.png')
    # save_greyscale_image(correlate(image, k, 'wrap'), 'wrap_pigbird.png')
    img = load_greyscale_image('test_images/construct.png')
    save_greyscale_image(edges(img), 'edges_contruct.png')