import cv2
print(cv2.__version__)
vidcap = cv2.VideoCapture('./advance_lane_finding/project_video.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
    success,image = vidcap.read()
    print(str(count), 'Read a new frame: ', success)
    cv2.imwrite("./advance_lane_finding/project_vid2imgs/project_img_" + str(count).zfill(5) + ".jpg", image)     # save frame as JPEG file
    count += 1


import cv2
print(cv2.__version__)
vidcap = cv2.VideoCapture('./advance_lane_finding/challenge_video.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
    success,image = vidcap.read()
    print(str(count), 'Read a new frame: ', success)
    cv2.imwrite("./advance_lane_finding/challenge_vid2imgs/challenge_img_" + str(count).zfill(5) + ".jpg", image)     # save frame as JPEG file
    count += 1


import cv2
print(cv2.__version__)
vidcap = cv2.VideoCapture('./advance_lane_finding/harder_challenge_video.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
    success,image = vidcap.read()
    print(str(count), 'Read a new frame: ', success)
    cv2.imwrite("./advance_lane_finding/harder_vid2imgs/harder_img_" + str(count).zfill(5) + ".jpg", image)     # save frame as JPEG file
    count += 1





