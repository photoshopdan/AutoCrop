import os
import shutil
import sys
import subprocess
import glob
import numpy as np
import boto3
from PIL import Image, ImageCms, ExifTags


# Delete previous batch from Temp folder.
def temp_cleanup(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


# Send image to AWS and recieve face landmarks.
def describe(photo):
    client = boto3.client('rekognition', region_name = 'eu-west-2')

    with open(photo, 'rb') as image:
        response = client.detect_faces(Image = {'Bytes': image.read()},
                                       Attributes = ['ALL'])

    if not response['FaceDetails']:
        return 0
    elif len(response['FaceDetails']) > 1:
        return 1
    else:
        landmarks = response['FaceDetails'][0]['Landmarks']

        chin = next(i for i in landmarks
                    if i['Type'] == 'chinBottom')
        midjaw_left = next(i for i in landmarks
                           if i['Type'] == 'midJawlineLeft')
        midjaw_right = next(i for i in landmarks
                            if i['Type'] == 'midJawlineRight')
        upperjaw_left = next(i for i in landmarks
                             if i['Type'] == 'upperJawlineLeft')
        upperjaw_right = next(i for i in landmarks
                              if i['Type'] == 'upperJawlineRight')
        eye_left = next(i for i in landmarks if i['Type'] == 'eyeLeft')
        eye_right = next(i for i in landmarks if i['Type'] == 'eyeRight')

        return (chin, midjaw_left, midjaw_right,
                upperjaw_left, upperjaw_right,
                eye_left, eye_right)


# Calculate rotation correction angle using eyes.
def compute_angle(description, image_size):
    *_, le, re, = description
    w, h = image_size
    le_x, le_y, re_x, re_y = (le['X'] * w,
                              le['Y'] * h,
                              re['X'] * w,
                              re['Y'] * h)
    angle = (np.arctan2(re_x - le_x, le_y - re_y)
             * (180 / np.pi) - 90)
    
    return angle


# Convert face landmarks into crop boundary.
def compute_crop(description, image_size, expansion, ratio, angle, sc, vs, tc):
    
    # Prepare face anchor points. Convert decimal to pixel coordinates.
    c, mjl, mjr, ujl, ujr, *_ = description
    w, h = image_size
    tl, tr, bl, br = ((0, 0), (w, 0), (0, h), (w, h))

    c_x, c_y = c['X'] * w, c['Y'] * h
    mjl_x, mjl_y = mjl['X'] * w, mjl['Y'] * h
    mjr_x, mjr_y = mjr['X'] * w, mjr['Y'] * h
    ujl_x, ujl_y = ujl['X'] * w, ujl['Y'] * h
    ujr_x, ujr_y = ujr['X'] * w, ujr['Y'] * h
    tl_x, tl_y = tl
    tr_x, tr_y = tr
    bl_x, bl_y = bl
    br_x, br_y = br

    # Rotate and offset points if required.
    if angle != 0:
        rad = (angle * (np.pi / 180) * -1)
        new = []
        a, b = ((c_x, mjl_x, mjr_x, ujl_x, ujr_x, tl_x, tr_x, bl_x, br_x),
                (c_y, mjl_y, mjr_y, ujl_y, ujr_y, tl_y, tr_y, bl_y, br_y))
        for x, y in zip(a, b):
            rx = (((x - (w / 2)) * np.cos(rad))
                  - ((y - (h / 2)) * np.sin(rad))
                  + (w / 2)) + (expansion[0] / 2)
            ry = (((x - (w / 2)) * np.sin(rad))
                  + ((y - (h / 2)) * np.cos(rad))
                  + (h / 2)) + (expansion[1] / 2)
            new += rx, ry
        (c_x, c_y, mjl_x, mjl_y, mjr_x, mjr_y,
         ujl_x, ujl_y, ujr_x, ujr_y, tl_x, tl_y,
         tr_x, tr_y, bl_x, bl_y, br_x, br_y) = new

    # Centroid of face landmarks. Approximate middle of the face.
    cen_x, cen_y = ((c_x + mjl_x + mjr_x + ujl_x + ujr_x) / 5,
                    (c_y + mjl_y + mjr_y + ujl_y + ujr_y) / 5)
    
    # The mean of the distances of each point from the centroid.
    # Representative of average face size.
    size = ((np.sqrt(((c_x - cen_x) ** 2) + ((c_y - cen_y) ** 2))
            + np.sqrt(((mjl_x - cen_x) ** 2) + ((mjl_y - cen_y) ** 2))
            + np.sqrt(((mjr_x - cen_x) ** 2) + ((mjr_y - cen_y) ** 2))
            + np.sqrt(((ujl_x - cen_x) ** 2) + ((ujl_y - cen_y) ** 2))
            + np.sqrt(((ujr_x - cen_x) ** 2) + ((ujr_y - cen_y) ** 2)))
            / 5)

    # Calculate crop coordinates.
    crop_h = size * sc

    lower = int((crop_h / vs) + cen_y)
    upper = int(cen_y - (crop_h - (crop_h / vs)))
    left = int(cen_x - (((lower - upper) * ratio) / 2))
    right = int(cen_x + (((lower - upper) * ratio) / 2))

    # Check whether crop extends outside of the image. Abort if true.
    crop_corners = ((left, upper),
                    (right, upper),
                    (left, lower),
                    (right, lower))

    AB = (br_x - bl_x, br_y - bl_y)
    AC = (tl_x - bl_x, tl_y - bl_y)
    DB = (br_x - tr_x, br_y - tr_y)
    DC = (tl_x - tr_x, tl_y - tr_y)

    for c in crop_corners:
        AM = (c[0] - bl_x, c[1] - bl_y)
        DM = (c[0] - tr_x, c[1] - tr_y)

        AM_AB = np.sum(np.multiply(AM, AB))
        AM_AC = np.sum(np.multiply(AM, AC))
        DM_DC = np.sum(np.multiply(DM, DC))
        if tc.casefold() == 'n':
            DM_DB = np.sum(np.multiply(DM, DB))
        else:
            DM_DB = 1
        if any([AM_AB <= 0, # Left.
                AM_AC <= 0, # Lower.
                DM_DC <= 0, # Right.
                DM_DB <= 0]): # Upper.
            return None
        else:
            pass
        
    return (left, upper, right, lower)


# Crop variables.
scale_constant = float(open('crop_variables\\scale_constant.txt').read())
vertical_shift = float(open('crop_variables\\vertical_shift.txt').read())
output_ratio = 2 / 3 # Ratio when resize after crop not selected.

# Functional variables.
folder = sys.argv[1:]
temp_folder = 'Temp\\'
if not os.path.exists(temp_folder):
    os.mkdir(temp_folder)
temp_folder_abs = os.path.abspath(temp_folder)
temp_longedge = 800
profile = ImageCms.ImageCmsProfile('ICC Profile/sRGB Color Space Profile.icm')
img_profile = profile.tobytes()

# Delete last backup.
temp_cleanup(temp_folder)

# Obtain parameters from user.

# Confirm whether resize required. Ask for pixel dimensions if true.
while True:
    resample_img = input('Resize image after crop? (y/n): ')
    if resample_img.casefold() in ('y', 'n'):
        break
    else:
        print('You wot m8?\n')
print()

if resample_img.casefold() == 'y':
    while True:
        fault_string = 'That\'s not a dimension.\n'
        output_dimensions = input('Please enter the required dimensions '
                                  + 'in pixels, (w x h): ')
        output_dimensions = (output_dimensions.replace(' ', '')).casefold()
        crosses = [a for a in output_dimensions if a == 'x']
        if len(crosses) != 1:
            print(fault_string)
        else:
            invalid = []
            for a in output_dimensions:
                if not a.isdigit() and a != 'x':
                    invalid += a
            if invalid:
                print(fault_string)
            else:
                output_w, output_h = output_dimensions.split('x', 1)
                try:
                    output_w, output_h = int(output_w), int(output_h)
                except ValueError:
                    print(fault_string)
                else:
                    if 20 <= output_w and output_h <= 6000:
                        output_ratio = output_w / output_h
                        break
                    else:
                        print('I can\'t make it that size. Are you crazy?\n')
    print()
    # Apply correction factor to different portrait aspect ratios.
    if output_ratio <= 1:
        scale_constant = (output_ratio ** (-0.65 * output_ratio)) * 5.2
        vertical_shift = (output_ratio ** (0.7 * output_ratio)) * 2.4

# Confirm whether rotation correction is required.
while True:
    rotate_img = input('Correct head rotation? (y/n): ')
    if rotate_img.casefold() in ('y', 'n'):
        break
    else:
        print('Errr... Excuse me?\n')
print()

# Ask whether crop should be allowed to extend over top of frame.
# White space will be added if true.
while True:
    top_crop = input('Allow crop to extend beyond top of frame? (y/n): ')
    if rotate_img.casefold() in ('y', 'n'):
        break
    else:
        print('I\'m sorry, what?\n')
print()

# Load image, crop and save.
for file_path in glob.glob(f'{folder[0]}/**/*.jpg', recursive = True):
    file_directory, file_name = os.path.split(file_path)
    relative_file_path = os.path.relpath(file_path, start = folder[0])
    temp_directory = os.path.join(temp_folder,
                                  os.path.dirname(relative_file_path))
    try:
        with Image.open(file_path) as im:
            try:
                exif = dict(im._getexif().items())
            except AttributeError:
                pass
            img = im
            img.load()
    except(IOError, SyntaxError, IndexError):
        print(f'Problem loading {file_name}.')
        continue

    # Correct for exif orientation.
    img_size = img.size
    if img_size[0] >= img_size[1]:
        if exif[274] == 3:
            img = img.transpose(method = Image.ROTATE_180)
        elif exif[274] == 6:
            img = img.transpose(method = Image.ROTATE_270)
        elif exif[274] == 8:
            img = img.transpose(method = Image.ROTATE_90)
        img_size = img.size
    
    # Create temporary file, find the landmarks and overwrite the file.
    temp_file_name = os.path.join(temp_folder, relative_file_path)
    temp_dimensions = (temp_longedge,
                       int(temp_longedge
                           / (max(img_size) / min(img_size))))
    temp_img = img.resize((temp_dimensions[img_size.index(max(img_size))],
                           temp_dimensions[img_size.index(min(img_size))]),
                          resample = Image.HAMMING,
                          reducing_gap = 2.0)
    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)
    temp_img.save(temp_file_name,
                  quality = 75)

    landmarks = describe(temp_file_name)
    if landmarks == 0:
        print(f'No face detected in {file_name}')
        continue
    if landmarks == 1:
        print(f'{file_name} has more than one face.')
        continue

    shutil.move(file_path, temp_file_name)

    # Compute the rotation, if rotation is required by the user.
    theta = 0
    expansion = (0, 0)
    if rotate_img.casefold() == 'y':
        theta = compute_angle(landmarks, img_size)
        img = img.rotate(angle = theta,
                         resample = Image.BILINEAR,
                         expand = True,
                         fillcolor = '#ffffff')
        expansion = tuple(np.subtract(img.size, img_size))
        
    # Crop the image.
    crop = compute_crop(landmarks,
                        img_size,
                        expansion,
                        output_ratio,
                        theta,
                        scale_constant,
                        vertical_shift,
                        top_crop)

    if crop == None:
        error_directory = os.path.join(file_directory,
                                       'Crop failed/')
        if not os.path.exists(error_directory):
            os.mkdir(error_directory)
        shutil.move(temp_file_name, os.path.join(error_directory, file_name))
        print(f'{file_name} could not be cropped.')
    else:
        cropped = Image.new('RGB',
                            (crop[2] - crop[0], crop[3] - crop[1]),
                            color = '#ffffff')
        cropped.paste(img, (-crop[0], -crop[1]))

        # Resample the image if required.
        if resample_img.casefold() == 'y':
            cropped = cropped.resize((output_w, output_h),
                                     resample = Image.BICUBIC)
        
        # Save the final image.
        cropped.save(file_path,
                     quality = 95,
                     dpi = (300, 300),
                     icc_profile = img_profile)

        print(f'{file_name} cropped and saved.')

# Batch copy the metadata from the backup files to the cropped files.
print('\nReinstating metadata.')
try:
    subprocess.run('exiftool -ext JPG -tagsfromfile '
                   + f'"{temp_folder_abs}/%d/%f.%e" -all:all '
                   + '--IFD0:Orientation --ThumbnailImage '
                   + '-overwrite_original -r .',
                   shell = False, stdout = subprocess.DEVNULL,
                   stderr = subprocess.STDOUT, cwd = folder[0])
    print('\nMetadata successfully added.')
except:
    print('\nThere was a problem reinstating the metadata')

input('\nFolder complete. Press enter to exit.')


# Add sharpening to compensate for rotation resampling?
# GPU acceleration to improve speed of rotation and resizing?
# Fail on images which have already been cropped.

