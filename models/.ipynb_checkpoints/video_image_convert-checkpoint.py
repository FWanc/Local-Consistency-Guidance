import cv2
import os
import sys
import numpy as np


### 从video转到image list
### 暂时没有返回帧率，要是能返回帧率就更好了
### 暂时没有管声音，要是能带着声音那就更好了 
def video2imglist(video_path, max_frame_num = None):
    trt_img_list = []
    cap = cv2.VideoCapture(video_path)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    count = 0
    # Start converting the video
    # print('Converting video to frames (img_list) : '+str(video_path))
    # print('There are '+str(video_length)+' frames.')
    while cap.isOpened():
        if max_frame_num == None: pass
        else:
            if count>= max_frame_num:  
                cap.release()
                break
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            break
        # Write the results back to output location.
        trt_img_list.append(frame)
        count = count + 1
        # print('Dealing with the '+str(count)+' frame.')
        # If there are no more frames left
        if (count > (video_length-1)):
            # Release the feed
            cap.release()
            break
    return trt_img_list


def imglist2video(imglist, video_path, fps):
    vout_ = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (imglist[0].shape[1], imglist[0].shape[0]))
    for frame_i in range(len(imglist)):
        vout_.write(imglist[frame_i])
    vout_.release()
    return None

def select_frames_uniformly(input_imglist, frame_num):
    out_imglist = []
    for i in range(frame_num):
        selected_id = int(len(input_imglist)*i/frame_num)
        if selected_id<0: selected_id = 0
        if selected_id>=len(input_imglist): selected_id = len(input_imglist) - 1
        out_imglist.append(input_imglist[selected_id])
    return out_imglist, input_imglist
    

### 将两个视频进行拼接
def merge_videos(video_a_path, video_b_path, output_video_path):
    # 打开视频A和视频B
    imgs_a = video2imglist(video_a_path)
    imgs_b = video2imglist(video_b_path)
    imgs_c = []
    for iii in range(len(imgs_a)):
        imgs_c.append(np.concatenate((imgs_a[iii],imgs_b[iii]),axis=1))
    imglist2video(imgs_c, output_video_path, 15)
    print("Merge and save video complete.")