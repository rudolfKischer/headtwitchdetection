
from argparse import ArgumentParser
from datetime import datetime
from collections import deque
import tqdm
import numpy as np

# import os
from cv2 import (
    VideoCapture,
    VideoWriter,
    VideoWriter_fourcc,
    __version__ as cv2_version,
    addWeighted,
    cvtColor,
    COLOR_BGR2GRAY,
    COLOR_GRAY2BGR,
    threshold,
    THRESH_BINARY,
    THRESH_BINARY_INV,
    THRESH_MASK,
    bitwise_or,
    bitwise_and,
    normalize,
    NORM_MINMAX,
    CV_8U

)
import cv2
from os import path
from config import *

print(f"Using OpenCV version {cv2_version}")


# we want to extract motion from the video, and produce a new video with the motion
# to do this we will take the input video, extract the frames.
# we will duplicate the frames, invert the colors, and make it half transparent
# then shift the time by delta t, and add the frames back together


def construct_args_parser():
  parser = ArgumentParser(description="Extract motion from a video")
  parser.add_argument("input_video", help="The video to extract motion from")
  parser.add_argument("output_video", help="The video to save the motion to")
  parser.add_argument("--delta_t", help="The time shift between frames", default=1)
  parser.add_argument("--stabilize", help="Stabilize the video", action="store_true")
  return parser

parser = construct_args_parser()
args = parser.parse_args()


def extract_motion_frame(current_frame, frame_buffer):
  shifted_frame = frame_buffer.popleft()
  inverted = 255 - shifted_frame
  motion_extracted = addWeighted(current_frame, 0.5, inverted, 0.5, 0)
  return motion_extracted

def threshold_frame(motion_extracted, upper_thresh_value, lower_thresh_value):
  gray = cvtColor(motion_extracted, COLOR_BGR2GRAY)
  _, upper_thresh = threshold(gray, upper_thresh_value, 255, THRESH_BINARY)
  _, lower_thresh = threshold(gray, lower_thresh_value, 255, THRESH_BINARY_INV)
  threshed = bitwise_or(upper_thresh, lower_thresh)
  # invert the image
  threshed = 255 - threshed
  # convert the thresholded image back to color
  gray_color = cvtColor(threshed, COLOR_GRAY2BGR)
  return gray_color

def frame_distance(frame1, frame2):
    # Ensure the frames have the same shape
    if frame1.shape != frame2.shape:
        raise ValueError("Frames must have the same shape")

    # Calculate the difference in each color channel
    diff = cv2.absdiff(frame1, frame2)

    # # Calculate squared difference
    # diff_squared = np.square(diff, dtype=np.float32)

    # # Sum the squared differences across color channels
    # sum_diff_squared = np.sum(diff_squared, axis=2, keepdims=True)
    # sum_diff_squared = diff_squared[:, :, 0] + diff_squared[:, :, 1] + diff_squared[:, :, 2]

    # # Take the square root to get Euclidean distance
    # dist = np.sqrt(diff_squared)

    # convert to black and white
    # dist = cv2.cvtColor(dist, cv2.COLOR_BGR2GRAY)

    # # Normalize the distance to be in the range [0, 255]
    # dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)

    # # Convert back to uint8
    # dist = np.uint8(dist)

    # convert to black and white
    # dist = cv2.cvtColor(diff_squared, cv2.COLOR_BGR2GRAY)
    # convert to color
    # dist = cv2.cvtColor(dist, cv2.COLOR_GRAY2BGR)

    # Replicate the channel to get a 3-channel image
    # dist_3_channel = cv2.merge([dist, dist, dist])

    return diff

def min_threshold(frame, threshold):
  # if a pixel value is less than the threshold set it to 0
  # otherwise set it to the pixel value
  return np.where(frame < threshold, 0, frame)

def min_threshold_bin(frame, threshold):

  # if a pixel value is greater than the threshold set it to 255
  ret, thresh_frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
  return thresh_frame

def mask_frame(frame, mask):
  # the mask is a color image, with only monochrome values
  # we want the mask to act as a window, using the white pixels as an alpha value for the frame
  # the black pixels remain black
  # basically if its black in the mask, it should be black in the frame
  # if its white in the mask, it should be the same as the frame\
  if len(mask.shape) == 3 and mask.shape[2] == 3:
    mask = cvtColor(mask, COLOR_BGR2GRAY)
  alpha_mask = mask.astype(np.float32) / 255
  alpha_mask_3d = cv2.merge([alpha_mask, alpha_mask, alpha_mask])

  blended = frame * alpha_mask_3d

  black_background = np.zeros(frame.shape, dtype=np.uint8) * (1 - alpha_mask_3d)
  result = blended + black_background
  result = np.clip(result, 0, 255).astype(np.uint8)
  return result

def gaussian_blur(frame, kernel_size):
  return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)


params = cv2.SimpleBlobDetector_Params()
params.filterByColor = True
params.blobColor = 0
params.filterByArea = True
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False
params.minArea = 10
params.maxArea = 1000

detector = cv2.SimpleBlobDetector_create()

# def draw_blobs_bounding(frame):
#   # convert to grayscale
#   gray = cvtColor(frame, COLOR_BGR2GRAY)
#   gray = cv2.bitwise_not(gray)
#   keypoints = detector.detect(gray)
#   frame = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#   for keypoint in keypoints:
#     x = int(keypoint.pt[0])
#     y = int(keypoint.pt[1])
#     s = int(keypoint.size)
#     cv2.rectangle(frame, (x - s, y - s), (x + s, y + s), (0, 0, 255), 2)

#   return frame

import subprocess
def stabilize_video(input_path, output_path):
    # Step 1: Run vidstabdetect with FFmpeg to analyze the video
    detect_command = [
        "ffmpeg",
        "-i", input_path,
        "-vf", "vidstabdetect=shakiness=10:accuracy=40:result=my_transforms.trf",
        "-f", "null", "-"
    ]
    subprocess.run(detect_command)

    # Step 2: Run vidstabtransform to stabilize the video
    transform_command = [
        "ffmpeg",
        "-i", input_path,
        "-vf", "vidstabtransform=input=my_transforms.trf",
        output_path
    ]
    subprocess.run(transform_command)

def get_blobs_bounding(frame):
  gray = cvtColor(frame, COLOR_BGR2GRAY)
  contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
  min_area = 2000
  bounding_boxes = [box for box in bounding_boxes if box[2] * box[3] > min_area]
  # give each bounding box a life start time
  for i in range(len(bounding_boxes)):
    # add life start to end of box
    box = bounding_boxes[i]
    bounding_boxes[i] = (box[0], box[1], box[2], box[3], datetime.now())
  return bounding_boxes

def draw_boxes(frame, boxes, color=(0, 0, 255)):
  for box in boxes:
    x, y, w, h, lifeStart = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
  return frame

def draw_blobs_bounding(frame):
  # convert to grayscale
  # get contours
  # draw bounding boxes around the contours
  gray = cvtColor(frame, COLOR_BGR2GRAY)

  contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  # remove small contours (by pixel area)
  min_area = 1500
  contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]

  # remove
  bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
  # we want to merge ones that overlap and are close together
  # we can do this by sorting the bounding boxes by x and y
  # sorted_x = sorted(bounding_boxes, key=lambda box: box[0])
  # we can then merge the boxes that overlap
  # we can do this by checking if the next box overlaps with the current box
  # we need to check that it overlaps in both x and y
  # include a padding radius
  # note that for a box to overlap, it must be to the right of the current box
  # and note that we need to compare the x and y overlap and check that both of them overlap
  # note that we may need to merge multiple boxes
  # so we keep an index of the start of the merge, and once we find a box that does not overlap
  # we add all the boxes from the start of the merge to the current index to the merged list
  # them we find the max and min x and y of the merged boxes, and create a new box
  # if we fail to merge, and the current merge index is the same as the current index
  # then w
  merged_boxes = bounding_boxes
  # padding = 5

  # current_merge_min_x = sorted_x[0][0]
  # current_merge_max_x = sorted_x[0][0] + sorted_x[0][2]
  # current_merge_min_y = sorted_x[0][1]
  # current_merge_max_y = sorted_x[0][1] + sorted_x[0][3]
  # for i in range(1, len(sorted_x)):
  #   box = sorted_x[i]
  #   min_x = box[0]
  #   max_x = box[0] + box[2]
  #   min_y = box[1]
  #   max_y = box[1] + box[3]
  #   if min_x < current_merge_max_x + padding and min_y < current_merge_max_y + padding or (max_x > current_merge_min_x - padding and max_y > current_merge_min_y - padding):
  #     # the box overlaps with the current merge
  #     # update the current merge
  #     current_merge_max_x = max_x
  #     current_merge_max_y = max_y
  #   else:
  #     # the box does not overlap with the current merge
  #     # add the current merge to the merged list
  #     merged.append((current_merge_min_x, current_merge_min_y, current_merge_max_x - current_merge_min_x, current_merge_max_y - current_merge_min_y))
  #     current_merge_min_x = min_x
  #     current_merge_max_x = max_x
  #     current_merge_min_y = min_y
  #     current_merge_max_y = max_y
  
  # remove the boxes that are too small
  min_area = 1500
  merged_boxes = [box for box in merged_boxes if box[2] * box[3] > min_area]

  # draw the merged boxes
  for box in merged_boxes:
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
  return frame

def edges(frame):
  gray = cvtColor(frame, COLOR_BGR2GRAY)
  edges = cv2.Canny(gray, 100, 200)
  # convert back to color
  edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
  return edges

def get_contours(frame):
  gray = cvtColor(frame, COLOR_BGR2GRAY)
  contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  # draw the contours
  frame = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
  # back to color
  return frame

def get_k_means_segmentation(frame, k):
  # convert to float32
  Z = frame.reshape((-1, 3))
  Z = np.float32(Z)
  # define criteria, number of clusters(K) and apply kmeans()
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  ret, label, center = cv2.kmeans(Z, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
  # Now convert back into uint8, and make original image
  center = np.uint8(center)
  res = center[label.flatten()]
  res2 = res.reshape((frame.shape))
  return res2

def match_bounding_boxes(bounding_boxes, prev_bounding_boxes, k=None):
  # pair all bounding boxes with the previous bounding boxes
  # we want a a maximal matching
  # each box can only be paired with one other box

  # if there are more current bounding boxes than previous bounding boxes
  # then we just keep the current bounding boxes

  # if there are less current bounding boxes than previous bounding boxes
  # then we pair them, and remove all the previous bounding boxes that are paired

  # the unpaired get added to the current bounding boxes

  # the idea is that if we fail to get a bounding box on the current frame, its better to keep the previous bounding box
  # than to add a new one

  # the way we will do this is calculate the amount of overlap between the bounding boxes

  # loop over the current bounding boxes
  # if any overlap with  eachother, merge them
  for box1 in bounding_boxes:
    for box2 in bounding_boxes:
      if box1 == box2:
        continue
      overlap_x = min(box1[0] + box1[2], box2[0] + box2[2]) - max(box1[0], box2[0])
      overlap_y = min(box1[1] + box1[3], box2[1] + box2[3]) - max(box1[1], box2[1])
      overlap_area = overlap_x * overlap_y
      if overlap_area > 0:
        # merge the boxes
        min_x = min(box1[0], box2[0])
        min_y = min(box1[1], box2[1])
        max_x = max(box1[0] + box1[2], box2[0] + box2[2])
        max_y = max(box1[1] + box1[3], box2[1] + box2[3])
        new_box = (min_x, min_y, max_x - min_x, max_y - min_y, box1[4])
        bounding_boxes.remove(box1)
        bounding_boxes.remove(box2)
        bounding_boxes.append(new_box)
        break


  if len(prev_bounding_boxes) == 0:
    return bounding_boxes
  if len(bounding_boxes) > len(prev_bounding_boxes) and k is not None and len(bounding_boxes) < k:
    return bounding_boxes
  box_pairings = []
  for box in bounding_boxes:
    most_overlap = 0
    most_overlap_box = None

    for prev_box in prev_bounding_boxes:
      # calculate the overlap
      # if the overlap is greater than a certain threshold, then we pair them

      overlap_x = min(box[0] + box[2], prev_box[0] + prev_box[2]) - max(box[0], prev_box[0])
      overlap_y = min(box[1] + box[3], prev_box[1] + prev_box[3]) - max(box[1], prev_box[1])
      overlap_area = overlap_x * overlap_y 
      box_area = box[2] * box[3]
      box_center = (box[0] + box[2] / 2, box[1] + box[3] / 2)
      prev_box_center = (prev_box[0] + prev_box[2] / 2, prev_box[1] + prev_box[3] / 2)
      distance = np.sqrt((box_center[0] - prev_box_center[0]) ** 2 + (box_center[1] - prev_box_center[1]) ** 2)

      prev_box_area = prev_box[2] * prev_box[3]
      score = 0
      if (box_area + prev_box_area - overlap_area) != 0:
          score = overlap_area / (box_area + prev_box_area - overlap_area) 
      if score > most_overlap:
        most_overlap = score
        most_overlap_box = prev_box
      
    
    if most_overlap_box is not None:
      new_box = (box[0], box[1], box[2], box[3], most_overlap_box[4])
      box_pairings.append((new_box, most_overlap))
      # remove the most_overlap_box from the prev_bounding_boxes
      prev_bounding_boxes.remove(most_overlap_box)

      # update life time of the paired box and set it to the previous box life start

  
  # add the remaining prev_bounding_boxes to the current bounding boxes
  # sort the box pairings by the score
  box_pairings.sort(key=lambda pair: pair[1], reverse=True)
  # print(f'len of bounding_boxes: {len(bounding_boxes)}')

  if k is not None:
    if len(box_pairings) > k:
        box_pairings = box_pairings[:k]
        bounding_boxes = [pair[0] for pair in box_pairings]
    elif len(box_pairings) < k:
        # sort the prev_bounding_boxes by the life time
        prev_bounding_boxes.sort(key=lambda box: (datetime.now() - box[4]).total_seconds())
        needed_boxes = k - len(box_pairings)
        new_boxes = prev_bounding_boxes[:needed_boxes]
        bounding_boxes = [pair[0] for pair in box_pairings]
        bounding_boxes.extend(new_boxes)
    else:
        # len(box_pairings) == k
        bounding_boxes = [pair[0] for pair in box_pairings]
    # print(f'len of prev_bounding_boxes: {len(prev_bounding_boxes)}')
    # print(f'len of new_bounding_boxes: {len(bounding_boxes)}')


  return bounding_boxes
  






  















def extract_motion(input_video, output_video, delta_t):
  # unpack the video into frames, and then create a new video to write to
  cap = VideoCapture(input_video)
  fourcc = VideoWriter_fourcc(*'mp4v')
  fps = cap.get(5)
  width = int(cap.get(3))
  height = int(cap.get(4))

  out = VideoWriter(output_video, fourcc, fps, (width, height))

  frame_shift = int(fps * delta_t)
  frame_buffer_1 = deque(maxlen=frame_shift)
  # create a new video with all the frames shifted

  # add tqdm to show progress for the while loop

  frame_count = int(cap.get(7))

  bit_mask_buffer_length = frame_shift * 2

  bit_masks_buffer = deque(maxlen=bit_mask_buffer_length)


  prev_bounding_boxes = []
  for _ in tqdm.tqdm(range(frame_count)):

    ret, frame = cap.read()
    if not ret:
      break
    

    frame_buffer_1.append(frame)

    
    if len(frame_buffer_1) < frame_shift:
      continue




    motion_frame = frame_distance(frame, frame_buffer_1.popleft())
    current_motion_bit_mask = min_threshold_bin(motion_frame, 10)
    bit_masks_buffer.append(current_motion_bit_mask)

    # pop off the oldest bit mask if the buffer is full
    if len(bit_masks_buffer) == bit_mask_buffer_length:
      bit_masks_buffer.popleft()

    # bitwise or all of the bit masks together in the buffer
    frame_bit_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for bit_mask in bit_masks_buffer:
      frame_bit_mask = cv2.bitwise_or(frame_bit_mask, bit_mask)

    black_background = np.zeros(frame.shape, dtype=np.uint8)
    bounding_boxes = get_blobs_bounding(frame_bit_mask)
    # draw the new boxes
    blue_out = draw_boxes(black_background, bounding_boxes, color=(255, 0, 0))

    bounding_boxes = match_bounding_boxes(bounding_boxes, prev_bounding_boxes, k=12)
    prev_bounding_boxes = bounding_boxes

    out_frame = draw_boxes(black_background, bounding_boxes)
    # add the frame to out frame
    out_frame = cv2.add(out_frame, frame)
    out_frame = cv2.add(out_frame, blue_out)
    out.write(out_frame)

  
  cap.release()
  out.release()









def main():

  # check if the input data folder exists
  if not path.exists(INPUT_FOLDER):
    print(f"Input data folder {INPUT_FOLDER} does not exist")
    return
  
  # check if the output data folder exists
  if not path.exists(OUTPUT_FOLDER):
    print(f"Output data folder {OUTPUT_FOLDER} does not exist")
    return
  input_path = path.join(INPUT_FOLDER, args.input_video)
  # add date time stamp to the output video
  # split the output_video_name on the file type and add the date time stamp
  out_name, extension = path.splitext(args.output_video)
  output_file_name_time_stamped = f"{out_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}{extension}"
  output_path = path.join(OUTPUT_FOLDER, output_file_name_time_stamped)
  print(output_path)
  if args.stabilize:
    stabilized_video_path = path.join(OUTPUT_FOLDER, f"stabilized_{output_file_name_time_stamped}")
    stabilize_video(input_path, stabilized_video_path)
    # change input to the stabilized video
    input_path = stabilized_video_path
  extract_motion(input_path, output_path, 0.3)

if __name__ == "__main__":
  main()