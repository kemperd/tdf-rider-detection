import cv2
from ultralytics import YOLO
import supervision as sv
import easyocr
import re
from collections import Counter
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('video')
parser.add_argument('-w', '--write', action='store_true', help='Write to files instead of screen')
args = parser.parse_args()
VIDEO = args.video

FONT = cv2.FONT_HERSHEY_SIMPLEX

model = YOLO("best.pt")    # Finetuned teams model on yolov8l

reader = easyocr.Reader(['fr'])

print('Available classes: ', model.names)

results = model.track(source=VIDEO, conf=0.4, iou=0.5, show=False, stream=True, agnostic_nms=True, verbose=False)

box_annotator = sv.BoxAnnotator(
    thickness=1,
    text_thickness=1,
    text_scale=0.5,
    text_padding=2,
)

rider_temp_dict = {}      # Temporary storage for all OCR results
rider_dict = {}           # Final rider number per tracker

df_riders = pd.read_csv('riders.csv')
df_riders = df_riders.set_index('Number', drop=False)
df_riders['Rider'] = df_riders['Rider'].str.strip()
df_riders['Group'] = None
df_riders['Tracker'] = None
valid_riders = df_riders.index.tolist()

cumulative_leaders = []
cumulative_leaders_framecount = 0
cumulative_leaders_count = {}
leaders_cooldown = 0

curr_group = None

def get_race_info_detections(image):
    global curr_group

    height = image.shape[0]
    width = image.shape[1]

    # Crop the bottom part containing race info and OCR that specifically
    img = image[700:height,0:width]

    result = reader.readtext(img, width_ths=1.2)

    # Dictionary of detected groups
    groups = {}

    for detection in result:
        text = detection[1]
        confidence = detection[2]

        if confidence > 0.1:
            if(text == 'Tête de la course'):
                # Only set the detected group here, do not initialize a rider list.
                # It may occur that 'Tête de la course' is displayed without a rider list to indicate
                # which camera the viewer is looking at. We do not want to reset the rider list in
                # those situations.
                curr_group = 'leading'
            else:
                if curr_group is not None:
                    # Extract rider number from string
                    #num = re.findall(r'(\d{1,3})\s', text)
                    # Check for digit of length 1-3 followed by string
                    num = re.findall(r'(\d{1,3})\s+\w+', text)
                    if len(num) == 1:
                        rider_num = int(num[0])
                        #print('detected rider_num: ', rider_num)
                        if rider_num in valid_riders:
                            rider_name = df_riders.loc[rider_num, 'Rider']

                            # Init empty list of riders to prevent error
                            if 'leading' not in groups:
                                groups[curr_group] = []

                            groups[curr_group].append(rider_num)
    return groups

def extract_leaders_from_groups(groups):
    global cumulative_leaders_framecount
    global cumulative_leaders
    global cumulative_leaders_count
    global leaders_cooldown

    #if 'leading' in groups and 0 <= cumulative_leaders_framecount <= 50:
    if leaders_cooldown == 0:

        if 'leading' in groups:
            # Accumulate 50 detections when leaders are shown to counter misdetections.
            leaders = groups['leading']

            #print('found leaders: ', leaders)

            # Clear the cumulative counters on first frame
            if cumulative_leaders_framecount == 0:
                cumulative_leaders = []
                cumulative_leaders_count = {}
            else:
                cumulative_leaders = cumulative_leaders + leaders
                cumulative_leaders_count = Counter(cumulative_leaders)

            print(cumulative_leaders_framecount)

            # If detection threshold of 50 frames have reached, we know enough to determine the leaders from the OCR results
            if cumulative_leaders_framecount == 50:
                # Clear the group assignments in the DataFrame as those are leading for the rider OSD
                df_riders.loc[df_riders.Group == 'leading', 'Group'] = ''

                # After 50 frames, remove the riders OCRed for less than 50%, as these are assumed to be misdetections
                cumulative_leaders_count = {key:val for key, val in cumulative_leaders_count.items() if val > 25}

                #print(cumulative_leaders_count)

                # At this point the keys of cumulative_leaders_count contain the leaders detected properly
                leaders = cumulative_leaders_count.keys()
                for leader in leaders:
                    df_riders.loc[leader, 'Group'] = 'leading'

                # Initiate cooldown counter for number of frames to wait until next leaders detection
                leaders_cooldown = 700

            cumulative_leaders_framecount += 1
        else:
            # Reset counter when leader information is no longer shown, this allows new detections when these are displayed again
            cumulative_leaders_framecount = 0

    else:
        leaders_cooldown -= 1


def print_leaders(frame):
    leading_df = df_riders[df_riders.Group == 'leading']
    if len(leading_df) > 0:
        cv2.putText(frame, 'Leaders:', (50,40), FONT, 0.75, (0,255,255), 2, cv2.LINE_4)
        y = 80
        for index, row in leading_df.iterrows():
            #print(row)
            str = '{num:3d} {rider} [{team}]'.format(num=row['Number'], rider=row['Rider'], team=row['Team_code'])
            cv2.putText(frame, str, (50,y), FONT, 0.75, (0,255,255), 2, cv2.LINE_4)
            y += 40


def ocr_race_number_single_detection(xyxy, mask, confidence, class_id, tracker_id):
    global rider_temp_dict
    global rider_dict

    # Loop over the frame detections and pass them to the OCR reader
    #for xyxy, mask, confidence, class_id, tracker_id in detections:
    if tracker_id is not None:
        if tracker_id not in rider_temp_dict:
            print('tracker id {} not found in riders dict, adding'.format(tracker_id))
            rider_temp_dict[tracker_id] = ['9999']

        x1 = int(xyxy[0])
        y1 = int(xyxy[1])
        x2 = int(xyxy[2])
        y2 = int(xyxy[3])
        cropped_image = frame[y1:y2,x1:x2]
        #cv2.imwrite('test_images/test.png', cropped_image)

        ocr_result = reader.readtext(cropped_image)
        for ocr_detection in ocr_result:
            top_left = tuple(ocr_detection[0][0])
            bottom_right = tuple(ocr_detection[0][2])
            top_left = [int(i) for i in top_left]
            bottom_right = [int(i) for i in bottom_right]
            ocr_text = ocr_detection[1]
            ocr_confidence = ocr_detection[2]

            # Accumulate OCR results for tracker in rider_temp_dict
            if ocr_text.isnumeric() and not tracker_id in rider_dict.keys():
                #print('Found text {} for object {} - conf {}'.format(ocr_text, tracker_id, ocr_confidence))
                rider_temp_dict[tracker_id].append(ocr_text)

    # At this point, rider_temp_dict will map a tracker_id to a list containing the OCR results for that tracker.
    # As the OCR results are quite inaccurate the results list will contain many different results.
    # We accumulate 100 OCR results and decide on the one that has occured the most
    to_delete = []
    for i, (tracker, ocr_results_list) in enumerate(rider_temp_dict.items()):
        print('{} ocr results for tracker {} (guessing {}): {}'.format(len(ocr_results_list), tracker, max(set(rider_temp_dict[tracker]), key = rider_temp_dict[tracker].count), ocr_results_list))
        if len(ocr_results_list) >= 100:
            # Move rider name into final results dict
            detected_rider = max(set(rider_temp_dict[tracker]), key = rider_temp_dict[tracker].count)
            print('Detected final rider {} for tracker {}'.format(detected_rider, tracker))

            # Check if the final detection really exists (it may still be incorrect)
            if int(detected_rider) in df_riders.index:
                retrieved_rider = df_riders.loc[int(detected_rider), 'Rider']
                print('Retrieved {}'.format(retrieved_rider))
                rider_dict[tracker] = df_riders.loc[int(detected_rider), 'Rider']
                to_delete.append(tracker)

    # Remove from temp dict as this will no longer be filled
    for key in to_delete: rider_temp_dict.pop(key)


framenum = 0

for result in results:
    frame = result.orig_img

    detections = sv.Detections.from_yolov8(result)

    if result.boxes.id is not None:
        detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

    detections = detections[(detections.class_id != 22)]    # remove persons from detections

    # Loop over the trackers to print the team labels on the riders
    labels = []
    for xyxy, mask, confidence, class_id, tracker_id in detections:
        if tracker_id is not None:
            if tracker_id in rider_dict.keys():
                labels.append(f"{rider_dict[tracker_id]}")
            else:
                # Now also leverage the team detection results.
                # If a team has been detected of which there is only one single rider in the group, we can print the rider name based on this.
                team_from_yolo = model.model.names[class_id][0:3]

                #print('Detected team {} for tracker {}'.format(team_from_yolo, tracker_id))

                rider = df_riders[(df_riders.Team_code == team_from_yolo) & (df_riders.Group == 'leading')]
                if len(rider) == 1:
                    # If there is only one rider in the group of a specific team, we can just look him up from the race details
                    labels.append(f"{rider['Rider'].values[0]}")
                #elif len(rider) > 1:
                    # If there are more riders from one team, perform OCR on the rider to extract his number
                    # TODO: Disabled for now due to poor results
                #    print('{} riders detected from team {}, performing OCR'.format(len(rider), team_from_yolo))
                #    labels.append('OCR HERE')
                #    ocr_race_number_single_detection(xyxy, mask, confidence, class_id, tracker_id)
                else:
                    labels.append(f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}")

    frame = box_annotator.annotate(
        scene=frame,
        detections=detections,
        labels=labels
    )

    # OCR the bottom part to extract leaders info
    groups = get_race_info_detections(frame)
    extract_leaders_from_groups(groups)
    print_leaders(frame)

    if args.write == True:
        # Write to files
        cv2.imwrite('output_images/img_{}.jpg'.format(framenum), frame)
        framenum += 1
    else:
        cv2.imshow("yolov8", frame)

    if (cv2.waitKey(30) & 0xFF == ord('q')):
        break
