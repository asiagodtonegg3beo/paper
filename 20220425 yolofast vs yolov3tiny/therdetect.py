import yaml
import sys, os
import ctypes
import numpy as np
import time
import cv2
import requests
from requests.adapters import HTTPAdapter
import simplejson as json

import thermalDetect.runDetect as detect
from timeit import default_timer as timer
#---NCS 2
from argparse import ArgumentParser
try:
    from armv7l.openvino.inference_engine import IENetwork, IEPlugin
except:
    from openvino.inference_engine import IENetwork, IEPlugin


from pylepton_lepton3.pylepton.Lepton3 import Lepton3
from pylepton_lepton3.pylepton.Lepton import Lepton
#head = 0xFF
#udpCliSock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
#---
import event_global_val
event_global_val.about_global_event()
event_global_val.per_15_in_bed = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
import event
import track_bar
import mouse_save_track_bar_value
import yolo_bbox
ff = yolo_bbox.Yolo_bbox()
macB = ff.GetMACB()
import psutil
try:
    addr = psutil.net_if_addrs()['wlan0'][3].address
except:
    try:
        addr = psutil.net_if_addrs()['wlan0'][2].address
    except:
        try:
            addr = psutil.net_if_addrs()['wlan0'][1].address
        except:
            addr = psutil.net_if_addrs()['wlan0'][0].address
macC = addr[11::]
# keyA = -56704 只能在第48行，要在固定的行位不能動到
keyA = -56704
# keyA = -56704 只能在第48行，要在固定的行位不能動到




classes_para = len(list(event.ObjInformation(10).continuous_action_count.keys()))

gg = event.ObjInformation(classes_para)
macA = gg.GetMACA()
keyC = ff.keyC

import os
DIR = '/home/pi/.sysfile'
how_much_file = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
#print(how_much_file)

keyB = gg.keyB

#parameter
with open('./thermalDetect/post_on_internet_parameters.yml') as file:
    parameters = yaml.load(file, Loader=yaml.FullLoader)
    # print(parameters)
    website = parameters["website"]
    account = parameters["account_name"] #="50E0"
    passward = parameters["client_secret"] #="3ef2b09aabea55a81873b1f63f2ab16bd87034437117fe496a1b564bb5b0a988"
    machine_number = parameters["mac_address"] #="C009"
    moving_bed_alert_email = parameters["moving_bed_alert_email"]

# '''
# source /opt/intel/openvino/bin/setupvars.sh;cd home/pi/ITRI_Thermal
# python3 main_detect.py
# python3 main_detect.py -o output # 記得還要再開啟REC...
# python3 main_detect.py -v video -i /home/pi/ITRI_Thermal/output.avi # 輸入一定要完整路徑
# 即時錄製結果：
# python3 main_detect.py -v video -i /home/pi/ITRI_Thermal/output.avi -r 1
# '''

mode = -1
isSetValue = 0
final_a = keyA
# isReadFromVideo = True
isOffSetMode = True
#=== offset trackbar ===
if(isOffSetMode == True):
    offset_value1 = 0
    offset_value2 = 0
    def nothing(x):
        pass
    cv2.namedWindow('offset')
#=== offset trackbar ===End

cv2.namedWindow('main_windows', cv2.WINDOW_NORMAL)
cv2.moveWindow('main_windows', 0, 0)
cv2.resizeWindow('main_windows', 640,480)

post_leave = {"in":1, "mid":1, "out":1}
post_fall = 1

class CollectCount_class():
    def __init__(self):
        self.collect_count = {'stand':10000, 'sit':10000, 'lie_down':10000, 'miss_lie_down':10000}

#上傳圖片，修改自官方Example
# pip3 install imgurpython
from imgurpython import ImgurClient
from datetime import datetime

# '''
# # <超過imgur上傳限制被鎖了>
# def upload(client_data, local_img_file, album , status, name = 'test-name!' ,title = 'test-title' ):
#     # status 0:normal/1:leave/2:fall
#     STATUS_LIST = ["normal", "leave", "fall"]
#     config = {
#         'album': album,
#         'name': name,
#         'title': title,
#         'description': f"{machine_number}_{STATUS_LIST[status]}_{str(datetime.now()).replace(' ', '_')[0:-7]}"
#     }

#     # print("Uploading image... ")
#     image = client_data.upload_from_path(local_img_file, config=config, anon=False)
#     # print("Done")

#     return image
# '''

import pyimgur

# 清空所有計數 歸零所有計數(b)
def clean_ObjInformation_all(obj_information, classes_para):
    # # Yolo_bbox
    # self.PYBoxResultList = []
    # self.pre_PYBoxResultList = []
    # self.flag = 0
    # self.situation = "Unknown"
    # self.jump_event = 0
    # self.use_pre_bbox = 0
    # self.which_case = "Unknown"
    # self.how_many_continue_frames_no_one_in_it = 0

    obj_information.iou = 0
    obj_information.interArea = 0
    obj_information.pre_interArea = 0
    obj_information.yoloArea = 0
    obj_information.state = "Unknown"
    obj_information.pre_state = "Unknown"
    obj_information.set_continuous_action_count_to_zero()
    event.set_as_of_the_present_state_action_count_all_to_zero_func(obj_information, classes_para)
    obj_information.candidate =[]
    obj_information.statusANDmotion_for_post_on_Internet = {
        "status_safe": 1,         # 安全
        "status_wanna_leave": 0,  # 想離床
        "status_leave": 0,        # 離床
        "status_fall_or_sit": 0,  # 跌倒或坐
        "status_sedentary": 0,    # 久坐
        "status_wheelchair": 0,   # 輪椅
        "motion": 0
    }
    # motion 0:stand/1:sit/2:lie_down
    obj_information.yolo_bbox_center_with_80_upper_space = np.array([0, 0])
    obj_information.sit_for_leave_first_frame_center = np.array([-1, -1])
    obj_information.someon_vanish_inside_the_bed = 0
    obj_information.this_man_wanna_leave_bed_once_through_GetOverRatio = 0
    obj_information.this_man_not_really_wanna_leave_bed = 0


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-d", "--device", help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
                                                Sample will look for a suitable plugin for device specified (CPU by default)", default="MYRIAD", type=str)
    parser.add_argument("-i", "--input", help="input", default="000", type=str)
    parser.add_argument("-o", "--output", help="output", default="000", type=str)
    parser.add_argument("-v", "--video", help="video", default="cam", type=str)
    parser.add_argument("-r", "--record_event", help="record_event", default="0", type=str)
    return parser

def state(event):
   return{
         0 : "leave",
         1 : "fall",
         2 : "in",
   }[event]

from io import BufferedReader, BytesIO
import threading
from queue import Queue

def postimg(obj_information, img_c3_ColorMap_added_upper):

    # #-----------------------------------------------------------------------------------------------------------------
    # status 0:代表狀態沒有被激發/1:代表狀態被激發
    # motion 0:stand/1:sit/2:lie_down
    obj_information = obj_information
    status_safe = obj_information.statusANDmotion_for_post_on_Internet["status_safe"] # 安全
    status_wanna_leave = obj_information.statusANDmotion_for_post_on_Internet["status_wanna_leave"] # 想離床
    status_leave = obj_information.statusANDmotion_for_post_on_Internet["status_leave"] # 離床
    status_fall_or_sit = obj_information.statusANDmotion_for_post_on_Internet["status_fall_or_sit"] # 跌倒或坐
    status_sedentary = obj_information.statusANDmotion_for_post_on_Internet["status_sedentary"] # 久坐
    status_wheelchair = obj_information.statusANDmotion_for_post_on_Internet["status_wheelchair"] # 輪椅
    motion = obj_information.statusANDmotion_for_post_on_Internet["motion"]

    data = {
        "no" : account,
        "client_secret" : passward,
        "mac_address" : machine_number,
        # "status_safe" : status_safe,         # 安全
        "status_wanna_leave" : status_wanna_leave,  # 想離床
        "status_leave" : status_leave,        # 離床
        "status_fall_or_sit" : status_fall_or_sit,  # 跌倒或坐
        # "status_sedentary" : status_sedentary,    # 久坐
        # "status_wheelchair" : status_wheelchair,   # 輪椅
        "motion" : motion,
        'position' : json.dumps([{
            'co_x' : 126,
            'co_y' : 0,
            'In_x' : 279,
            'In_y' : 472
            }]),
        'layout' : json.dumps([{'co_x' : 126,
                     'co_y' : 0,
                     'In_x' : 279,
                     'In_y' : 472
                     }]),
    }


    img_c3_ColorMap_added_upper = img_c3_ColorMap_added_upper
    # path = '....jpg'
    img = img_c3_ColorMap_added_upper	#我们使用读取图片的方式表示已经存在在内存的图像
    ret, img_encode = cv2.imencode('.jpg', img)
    str_encode = img_encode.tostring()		#将array转化为二进制类型
    f4 = BytesIO(str_encode)		#转化为_io.BytesIO类型
    f4.name = 'alert.jpg'		#名称赋值
    f5 = BufferedReader(f4)		#转化为_io.BufferedReader类型
    # print(f5)
    files = {'image_filename':f5}#open('alert.jpg','rb')}
    # print(open('alert.jpg','rb'))

    # requests.post
    try:
        response = requests.post(website, data=data, files=files, verify=False, timeout=2)
        # 'https://itricarecloud.ouorange.com/api/v1/camera/put'
        # 設定子執行緒
        '''
        # kwargs = {'files':files, 'verify':False} #X
        thre_0 = threading.Thread(target = requests.post, args=(website, data), kwargs={'files':files, 'verify':False})
        thre_0.start()
        '''
    #     requests.raise_for_status()
    # except requests.exceptions.HTTPError as a :
    #     print('HTTP error:', str(a))
    # except requests.exceptions.Timeout as b :
    #     print('timeout error:', str(b))
    # except requests.exceptions.ConnectionError as c :
    #     print('Connection error:', str(c))
    except:
        pass
        #print (response.text)

def post_fall_and_leave_on_imgur(obj_information, img_c3_ColorMap_added_upper):
    # cv2.imwrite('./alert.jpg', img_c3_ColorMap_added_upper)
    obj_information = obj_information
    img_c3_ColorMap_added_upper = img_c3_ColorMap_added_upper
    # just for debug
    # '''
    # # <超過imgur上傳限制被鎖了>
    # client_id ='ba34c39ad74157a'#'c36d865c65dd83a'
    # client_secret = '34e100855471e537aa20a85eb146abb33afa4ce1'#'5b8cfd3cd04d9397d44bf5bc9a9c62741601b6a1'
    # access_token = "dc1f73f448c0b4fe43dc45b23b038c59d1f13f1e"#"902149d953c4b446083476c63761e81750710935"
    # refresh_token = "f6f61f0f25c2f8acfd884813f73ae80c0cfb010e"#"0f0abedcc9da8e062b4596d4e23673493bb85bdd"
    # #<blockquote class="imgur-embed-pub" lang="en" data-id="a/l7YovFO"><a href="//imgur.com/a/l7YovFO">View post on imgur.com</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>
    # album = "l7YovFO"
    # local_img_file = "alert.jpg"
    # '''

    # pip3 install pyimgur
    # status 0:normal/1:leave/2:fall
    # motion 0:stand/1:sit/2:lie_down
    # 為了不影響日榮的資料搜集模式，在此沿用舊的，並建立一個新舊的轉換
    status_safe = obj_information.statusANDmotion_for_post_on_Internet["status_safe"] # 安全
    status_wanna_leave = obj_information.statusANDmotion_for_post_on_Internet["status_wanna_leave"] # 想離床
    status_leave = obj_information.statusANDmotion_for_post_on_Internet["status_leave"] # 離床
    status_fall_or_sit = obj_information.statusANDmotion_for_post_on_Internet["status_fall_or_sit"] # 跌倒或坐
    if(status_safe == 1):
        status = 0
    elif(status_wanna_leave + status_leave >= 1): #status_safe == 0
        status = 1
    else:
        pass
    if(status_fall_or_sit == 1):
        status = 2

    # # 測試：
    # print('status_safe: ', status_safe)
    # print('status_wanna_leave', status_wanna_leave)
    # print('status_leave', status_leave)
    # print('status_fall_or_sit', status_fall_or_sit)
    # print('--------------------------------------')

    motion = obj_information.statusANDmotion_for_post_on_Internet["motion"]
    STATUS_LIST = ["normal", "leave", "fall"]
    MOTION_LIST = list(obj_information.continuous_action_count.keys()) #["stand", "sit", "lie_down"]
    CLIENT_ID = "ba34c39ad74157a"
    PATH = "alert.jpg" #A Filepath to an image on your computer"
    # title = f"{machine_number}_{STATUS_LIST[status]}_{str(datetime.now()).replace(' ', '_')[0:-7]}"
    title = str(datetime.now())

    # post on google
    payload = {
        'entry.1858390567': str(datetime.now()).replace(' ', '/')[0:-7], #time
        'entry.574140101': str(machine_number), #who 代碼
        'entry.509304770': str(STATUS_LIST[status]), #Alert
        'entry.1905779135': str(obj_information.state), #state
        'entry.1150324881':str(MOTION_LIST[motion]), #motion
        'entry.1968827322': "wait_for_post_on_Imgur_and_get_the_url" #str(uploaded_image.link) #site
        }
    post_on_url = 'https://docs.google.com/forms/u/0/d/e/1FAIpQLSdKXKdpO1Nfk78DiB6f0c814J90iJF8nrvA9h5tYM0EHhdb6g/formResponse'

    # path = '....jpg'
    img_imgur = img_c3_ColorMap_added_upper	#我们使用读取图片的方式表示已经存在在内存的图像
    ret, img_encode = cv2.imencode('.jpg', img_imgur)
    str_encode = img_encode.tostring()		#将array转化为二进制类型
    f4_imgur = BytesIO(str_encode)		#转化为_io.BytesIO类型
    f4_imgur.name = 'alert.jpg'		#名称赋值
    f5_imgur = BufferedReader(f4_imgur)		#转化为_io.BufferedReader类型
    # print(type(str(f5_imgur)))

    try:
        global post_leave
        global post_fall
        if(status in [1]):
            post_fall = 1
            if(obj_information.state == "in" and post_leave["in"] == 1):
                post_leave["in"] = 0
                # cv2.imwrite('./alert.jpg', img_c3_ColorMap_added_upper)
                ### post action
                # '''
                # # <超過imgur上傳限制被鎖了>
                # client = ImgurClient(client_id, client_secret, access_token, refresh_token)
                # image = upload(client, local_img_file, album, status)
                # # print(f"圖片網址: {image['link']}")
                # '''
                im = pyimgur.Imgur(CLIENT_ID)
                uploaded_image = im.upload_image_dir(f5_imgur, title=title)
                # '''
                # q = Queue()
                # # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                # thre_leave_in_img = threading.Thread(target = im.upload_image_dir, args = (f5_imgur, title, q))
                # thre_leave_in_img.start()
                # uploaded_image = q.get()
                # # print(uploaded_image)
                # '''

                # 將檔案加入 POST 請求中
                payload['entry.1968827322'] = str(uploaded_image.link) #site
                r = requests.post(post_on_url, data = payload)
                # '''
                # thre_leave_in_google = threading.Thread(target = requests.post, args = (post_on_url, payload))
                # thre_leave_in_google.start()
                # '''

                # post_leave["in"] = 0
            elif(obj_information.state == "mid" and post_leave["mid"] == 1):
                post_leave["mid"] = 0
                # cv2.imwrite('./alert.jpg', img_c3_ColorMap_added_upper)
                ### post action
                # '''
                # # <超過imgur上傳限制被鎖了>
                # client = ImgurClient(client_id, client_secret, access_token, refresh_token)
                # image = upload(client, local_img_file, album, status)
                # # print(f"圖片網址: {image['link']}")
                # '''
                im = pyimgur.Imgur(CLIENT_ID)
                uploaded_image = im.upload_image_dir(f5_imgur, title=title)
                # '''
                # q = Queue()
                # # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                # thre_leave_mid_img = threading.Thread(target = im.upload_image_dir, args = (f5_imgur, title, q))
                # thre_leave_mid_img.start()
                # uploaded_image = q.get()
                # # print(uploaded_image)
                # '''

                # 將檔案加入 POST 請求中
                payload['entry.1968827322'] = str(uploaded_image.link) #site
                r = requests.post(post_on_url, data = payload)
                # '''
                # thre_leave_mid_google = threading.Thread(target = requests.post, args = (post_on_url, payload))
                # thre_leave_mid_google.start()
                # '''

                # post_leave["mid"] = 0
            elif(obj_information.state == "out" and post_leave["out"] == 1):
                post_leave["out"] = 0
                # cv2.imwrite('./alert.jpg', img_c3_ColorMap_added_upper)
                ### post action
                # '''
                # # <超過imgur上傳限制被鎖了>
                # client = ImgurClient(client_id, client_secret, access_token, refresh_token)
                # image = upload(client, local_img_file, album, status)
                # # print(f"圖片網址: {image['link']}")
                # '''
                im = pyimgur.Imgur(CLIENT_ID)
                uploaded_image = im.upload_image_dir(f5_imgur, title=title)
                # '''
                # q = Queue()
                # # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                # thre_leave_out_img = threading.Thread(target = im.upload_image_dir, args = (f5_imgur, title, q))
                # thre_leave_out_img.start()
                # uploaded_image = q.get()
                # '''

                # 將檔案加入 POST 請求中
                payload['entry.1968827322'] = str(uploaded_image.link) #site
                r = requests.post(post_on_url, data = payload)
                # '''
                # thre_leave_out_google = threading.Thread(target = requests.post, args = (post_on_url, payload))
                # thre_leave_out_google.start()
                # '''

                # post_leave["out"] = 0
            else:
                pass #nothing

        if(status in [2] and post_fall in [1]): #fall與normal之間的關係
            post_fall = 0
            # cv2.imwrite('./alert.jpg', img_c3_ColorMap_added_upper)
            ### post action
            # '''
            # # <超過imgur上傳限制被鎖了>
            # client = ImgurClient(client_id, client_secret, access_token, refresh_token)
            # image = upload(client, local_img_file, album, status)
            # # print(f"圖片網址: {image['link']}")
            # '''
            im = pyimgur.Imgur(CLIENT_ID)
            uploaded_image = im.upload_image_dir(f5_imgur, title=title)
            # '''
            # q = Queue()
            # # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            # thre_fall_img = threading.Thread(target = im.upload_image_dir, args = (f5_imgur, title, q))
            # thre_fall_img.start()
            # uploaded_image = q.get()
            # '''

            # 將檔案加入 POST 請求中
            payload['entry.1968827322'] = str(uploaded_image.link) #site
            r = requests.post(post_on_url, data = payload)
            # '''
            # thre_fall_google = threading.Thread(target = requests.post, args = (post_on_url, payload))
            # thre_fall_google.start()
            # '''

            # global post_fall
            # post_fall = 0

        if(status in [0]): #normal
            # global post_leave
            post_leave = {"in":1, "mid":1, "out":1}
            # global post_fall
            post_fall = 1
    except:
        pass

    # '''
    # entry.1858390567:
    # entry.574140101:
    # entry.509304770:
    # entry.1905779135:
    # entry.1968827322:

    # entry.1858390567:
    # entry.574140101:
    # entry.509304770:
    # entry.1905779135:
    # entry.1150324881: action
    # entry.1968827322:
    # '''

def post_on_imgur_with_action(collect_freq, collect_count, yolo_bbox_for_event_PYBoxResultList, obj_information, someone_is_using_pillow, img_c1_Origin):
    # pip3 install pyimgur

    # status 0:normal/1:leave/2:fall
    # motion 0:stand/1:sit/2:lie_down
    # 為了不影響日榮的資料搜集模式，在此沿用舊的，並建立一個新舊的轉換
    status_safe = obj_information.statusANDmotion_for_post_on_Internet["status_safe"] # 安全
    status_wanna_leave = obj_information.statusANDmotion_for_post_on_Internet["status_wanna_leave"] # 想離床
    status_leave = obj_information.statusANDmotion_for_post_on_Internet["status_leave"] # 離床
    status_fall_or_sit = obj_information.statusANDmotion_for_post_on_Internet["status_fall_or_sit"] # 跌倒或坐
    if(status_safe == 1):
        status = 0
    elif(status_wanna_leave + status_leave >= 1): #status_safe == 0
        status = 1
    else:
        pass
    if(status_fall_or_sit == 1):
        status = 2

    motion = obj_information.statusANDmotion_for_post_on_Internet["motion"]
    STATUS_LIST = ["normal", "leave", "fall"]
    MOTION_LIST = list(obj_information.continuous_action_count.keys()) #["stand", "sit", "lie_down"]
    CLIENT_ID = "ba34c39ad74157a"
    PATH = "alert.jpg" #A Filepath to an image on your computer"
    # title = f"{machine_number}_{STATUS_LIST[status]}_{str(datetime.now()).replace(' ', '_')[0:-7]}"
    title = str(datetime.now())

    # post on google
    payload_stand = {
        'entry.2131809544': str(datetime.now()).replace(' ', '-')[0:-7], #time
        'entry.646322256': str(MOTION_LIST[motion]), #motion
        'entry.2016566131': str(machine_number), #who
        'entry.1135947400': str(obj_information.state), #state
        'entry.322983397': "wait_for_post_on_Imgur_and_get_the_url" #str(uploaded_image.link) #site
        }
    post_on_url_stand = 'https://docs.google.com/forms/u/0/d/e/1FAIpQLSdvtoGLGF7wJdYMX-Ixhm7kSjsKqV3GDIbupvRV_51RwoKv6Q/formResponse'

    payload_sit = {
        'entry.569559685': str(datetime.now()).replace(' ', '-')[0:-7], #time
        'entry.814750832': str(MOTION_LIST[motion]), #motion
        'entry.1180300649': str(machine_number), #who
        'entry.190508539': str(obj_information.state), #state
        'entry.606887032': "wait_for_post_on_Imgur_and_get_the_url" #str(uploaded_image.link) #site
        }
    post_on_url_sit = 'https://docs.google.com/forms/u/0/d/e/1FAIpQLScmwy2woNUlX9G3Tkn9AzdOQeoIIVFswi5ee1KGbKojXSMA2A/formResponse'

    payload_lie_down = {
        'entry.1939030774': str(datetime.now()).replace(' ', '-')[0:-7], #time
        'entry.1791932805': str(MOTION_LIST[motion]), #motion
        'entry.162372186': str(machine_number), #who
        'entry.365777722': str(obj_information.state), #state
        'entry.89863013': "wait_for_post_on_Imgur_and_get_the_url" #str(uploaded_image.link) #site
        }
    post_on_url_lie_down = 'https://docs.google.com/forms/u/0/d/e/1FAIpQLSfMvIgmF3rrhMU1ls10_Z_6468oHl5R59n01FJL5e6Uuz5EVg/formResponse'

    payload_miss_lie_down = {
        'entry.1294189703': str(datetime.now()).replace(' ', '-')[0:-7], #time
        'entry.970691033': 'miss', #motion
        'entry.1231099004': str(machine_number), #who
        'entry.1599952374': str(obj_information.state), #state
        'entry.501841278': "wait_for_post_on_Imgur_and_get_the_url" #str(uploaded_image.link) #site
        }
    post_on_url_miss_lie_down = 'https://docs.google.com/forms/u/0/d/e/1FAIpQLSfcM8vkFjk4I8ZHE1l-olxnLEjMOGBxfXkGIoZLB_j7prOFmA/formResponse'


    # path = '....jpg'
    img_imgur = img_c1_Origin #我们使用读取图片的方式表示已经存在在内存的图像
    ret, img_encode = cv2.imencode('.jpg', img_imgur)
    str_encode = img_encode.tostring()		#将array转化为二进制类型
    f4_imgur = BytesIO(str_encode)		#转化为_io.BytesIO类型
    f4_imgur.name = 'thermalImage.jpg'		#名称赋值
    f5_imgur = BufferedReader(f4_imgur)		#转化为_io.BufferedReader类型
    # print(type(str(f5_imgur)))
    # print('1')

    if(len(yolo_bbox_for_event_PYBoxResultList) == 1):
        # print('2')
        if(motion == 0):
            # print('3')
            # print('@@@@@@@@@@@@@@@@@@@@@@@@')
            # print(collect_count['stand'], collect_freq['stand']*2.7)
            if(collect_count['stand'] >= collect_freq['stand']*2.7):
                # print('4')
                collect_count['stand'] = 0
                im = pyimgur.Imgur(CLIENT_ID)
                uploaded_image = im.upload_image_dir(f5_imgur, title=title)
                # 將檔案加入 POST 請求中
                payload_stand['entry.322983397'] = str(uploaded_image.link) #site
                r = requests.post(post_on_url_stand, data = payload_stand)
            else:
                pass
        elif(motion == 1):
            if(collect_count['sit'] >= collect_freq['sit']*2.7):
                collect_count['sit'] = 0
                im = pyimgur.Imgur(CLIENT_ID)
                uploaded_image = im.upload_image_dir(f5_imgur, title=title)
                # 將檔案加入 POST 請求中
                payload_sit['entry.606887032'] = str(uploaded_image.link) #site
                r = requests.post(post_on_url_sit, data = payload_sit)
            else:
                pass
        else: # motion == 2
            if(collect_count['lie_down'] >= collect_freq['lie_down']*2.7):
                collect_count['lie_down'] = 0
                im = pyimgur.Imgur(CLIENT_ID)
                uploaded_image = im.upload_image_dir(f5_imgur, title=title)
                # 將檔案加入 POST 請求中
                payload_lie_down['entry.89863013'] = str(uploaded_image.link) #site
                r = requests.post(post_on_url_lie_down, data = payload_lie_down)
            else:
                pass
    else: # len(yolo_bbox_for_event_PYBoxResultList) != 1
        if(someone_is_using_pillow == 1):
            if(collect_count['miss_lie_down'] >= collect_freq['miss_lie_down']*2.7):
                collect_count['miss_lie_down'] = 0
                im = pyimgur.Imgur(CLIENT_ID)
                uploaded_image = im.upload_image_dir(f5_imgur, title=title)
                # 將檔案加入 POST 請求中
                payload_miss_lie_down['entry.501841278'] = str(uploaded_image.link) #site
                r = requests.post(post_on_url_miss_lie_down, data = payload_miss_lie_down)
            else:
                pass
        else: # someone_is_using_pillow == 0
            pass

# 監測看看床有沒有被偷移動 抓偷移床 偷移動床 偷嚕床
def post_on_for_bed_moving(moving_mean_x_cent, moving_mean_y_cent, moving_mean_w, moving_mean_h, x_drift, y_drift, moving_mean_IoU, moving_mean_IoM, img_c3_Origin):
    # pip3 install pyimgur

    CLIENT_ID = "7a58e208a374687"
    PATH = "alert.jpg" #A Filepath to an image on your computer"
    # title = f"{machine_number}_{STATUS_LIST[status]}_{str(datetime.now()).replace(' ', '_')[0:-7]}"
    title = str(datetime.now())

    # post on google
    payload_stand = {
        'entry.910144108': str(moving_bed_alert_email), #受通知信箱
        'entry.1095486256': str(datetime.now()).replace(' ', '-')[0:-7], #time 時間
        'entry.1650959402': str(machine_number), # who 代碼
        'entry.1742683407': str(moving_mean_IoM), # IoM
        'entry.1413404381': "wait_for_post_on_Imgur_and_get_the_url" #str(uploaded_image.link) #site
        }
    post_on_url_stand = 'https://docs.google.com/forms/u/0/d/e/1FAIpQLSelOy5U0k6RmbBMRQBNdEDxFFSlPTH0YnVxhQP7o2dTQmXW0g/formResponse'

    # path = '....jpg'
    img_imgur = img_c3_Origin #我们使用读取图片的方式表示已经存在在内存的图像
    ret, img_encode = cv2.imencode('.jpg', img_imgur)
    str_encode = img_encode.tostring()		#将array转化为二进制类型
    f4_imgur = BytesIO(str_encode)		#转化为_io.BytesIO类型
    f4_imgur.name = 'thermalImage.jpg'		#名称赋值
    f5_imgur = BufferedReader(f4_imgur)		#转化为_io.BufferedReader类型


    im = pyimgur.Imgur(CLIENT_ID)
    uploaded_image = im.upload_image_dir(f5_imgur, title=title)
    # 將檔案加入 POST 請求中
    payload_stand['entry.1413404381'] = str(uploaded_image.link) #site
    r = requests.post(post_on_url_stand, data = payload_stand)



    # entry.910144108: 受通知信箱
    # entry.1095486256: 時間
    # entry.1650959402: 代碼
    # entry.1742683407: IoM
    # entry.1413404381: site


#detect from thermal camera
def detectFromThlCam(input_blob, exec_net, macAB, keyABC_A, BED_t, DETECT_REGION_t, PILLOW_t, FALL_THRES_t, SitWheelchairTooLong_t, ON_BED_THRES_t, LEAVE_THRES_t, \
    SMART_PILLOW_t, BODY_HOT_THRES_t, HEAD_PILLOW_DENSITY_t, SHOW_DETIAL_t, CURRENT_FPS_t, TEMP_LEAVE_t, ENOUGH_LIE_DOWN_t, \
        SIT_TIRED_t, OpenGetOverRatio_t, GetOverRatio_t, ShutDownOutside_t, sit_side_alarm_t, sit_back_alarm_t, SLEEP_THRES_t, RESET_TIME_t, StillInBed_t, \
            COLLECT_FREQ_t, RECORD_t, RecCutTime_t, camera_high_quality_version_t, many_bboxes_t, fall_distance_t, fall_degree_t, no_wheelchair_t, wheelchair_time_too_long_t, \
            wheelchair_distance_too_far_t, wheelchair_distance_too_close_t, bed_moving_t, moving_mean_IoM_THRES_t, extra_bed_xyxy01_t, SEND_FREQ, post_to_google_excel,warn_alert_long,need_check_repeat,check_repeat_time_range): # 參數引入

    #print(":".join((keyABC_A[0:2], keyABC_A[2:], keyB[0:2], keyB[2:], keyC[0:2], keyC[2:])) == macAB+macC)
    ####################################
    try:
        if(":".join((keyABC_A[0:2], keyABC_A[2:], keyB[0:2], keyB[2:], keyC[0:2], keyC[2:])) == macAB+macC):
    ####################################
            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # track_bar
            # print('1111111111111111111111111111111111')
            # 在無限迴圈之前產生拉霸的值
            control = track_bar.CONTROL(BED_t, DETECT_REGION_t, PILLOW_t, FALL_THRES_t, SitWheelchairTooLong_t, ON_BED_THRES_t, LEAVE_THRES_t, SMART_PILLOW_t, BODY_HOT_THRES_t, \
                HEAD_PILLOW_DENSITY_t, SHOW_DETIAL_t, CURRENT_FPS_t, TEMP_LEAVE_t, ENOUGH_LIE_DOWN_t, \
                    SIT_TIRED_t, OpenGetOverRatio_t, GetOverRatio_t, ShutDownOutside_t, sit_side_alarm_t, sit_back_alarm_t, SLEEP_THRES_t, how_much_file, RESET_TIME_t, StillInBed_t, \
                        COLLECT_FREQ_t, RECORD_t, RecCutTime_t, camera_high_quality_version_t, many_bboxes_t, fall_distance_t, fall_degree_t, no_wheelchair_t, wheelchair_time_too_long_t, \
                        wheelchair_distance_too_far_t, wheelchair_distance_too_close_t, bed_moving_t, moving_mean_IoM_THRES_t, extra_bed_xyxy01_t, SEND_FREQ, post_to_google_excel,warn_alert_long,need_check_repeat,check_repeat_time_range) # 參數引入 初始值
            # print('2222222222222222222222222222222222')
            control.track_bar()
            # mouse_save_track_bar_value
            # 滑鼠點擊視窗即時改變記憶體中的值
            mouse_save = mouse_save_track_bar_value.Mouse_Save_Parameters()
            mouse_save.mouse_save_parameters()
            # print(mouse_save.flag_mouse_save)
    #####################################
        else: # upLoad
            idx = str(0)

            ori_MAC = ":".join((keyABC_A[0:2], keyABC_A[2:], keyB[0:2], keyB[2:], keyC[0:2], keyC[2:]))
            new_MAC = macAB+macC
            payload = {
                'entry.885823784': '{}'.format(ori_MAC), #ori_MAC
                'entry.1742594334': '{}'.format(new_MAC), #new_MAC
                'entry.1867958386': 'they failed' #account
                # 'entry.129964452': '{}'.format(idx), #passward
                }
            # 將檔案加入 POST 請求中
            r = requests.post('https://docs.google.com/forms/u/0/d/e/1FAIpQLSdZUQ6Ipe12tPM8-VaeSR7mQlaFcUe8be9fnPihrVwQe0Z6Ng/formResponse', data = payload)
    except:
        pass
    #####################################

    count = 0
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ?"
    prev_time = timer()
    global isOffSetMode
    if(isOffSetMode == True):
        cv2.createTrackbar('Gray_Value_add','offset',int(-10),50,nothing)
        cv2.createTrackbar('Gray_Value_subtract','offset',0,50,nothing)

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    obj_information = event.ObjInformation(classes_para) # 生成跨檔案的全域變數
    yolo_bbox_for_event = yolo_bbox.Yolo_bbox() # 生成跨檔案的全域變數
    # pre_PYBoxResultList = [detect.PYBoxResult()]

    # 生成含上方警示圖區全黑的3通道畫布(其y方向大小視tracking id個數而異)
    pre_img_c3_ColorMap = np.zeros([480+80, 640, 3], dtype='uint8')
    ####################################
    try:
        if(":".join((keyABC_A[0:2], keyABC_A[2:], keyB[0:2], keyB[2:], keyC[0:2], keyC[2:])) == macAB+macC):
    ####################################
            xxx = True
    #####################################
        else: # upLoad
            idx = str(0)

            ori_MAC = ":".join((keyABC_A[0:2], keyABC_A[2:], keyB[0:2], keyB[2:], keyC[0:2], keyC[2:]))
            new_MAC = macAB+macC
            payload = {
                'entry.885823784': '{}'.format(ori_MAC), #ori_MAC
                'entry.1742594334': '{}'.format(new_MAC), #new_MAC
                'entry.1867958386': 'they failed' #account
                # 'entry.129964452': '{}'.format(idx), #passward
                }
            # 將檔案加入 POST 請求中
            r = requests.post('https://docs.google.com/forms/u/0/d/e/1FAIpQLSdZUQ6Ipe12tPM8-VaeSR7mQlaFcUe8be9fnPihrVwQe0Z6Ng/formResponse', data = payload)
    except:
        xxx = False
        pass
    #####################################

    collectcount_obj = CollectCount_class()

    # 鏡頭參數上界
    if(isReadFromVideo != 'video'):
        pass
    else: #if(isReadFromVideo == 'video'):

        videoPath = VIDEOPATH

        camera = cv2.VideoCapture(videoPath)
        if not camera.isOpened():
            raise IOError("Couldn't open video")


        video_frames = 0
        #(320,240)(640,480)(800,600)(1024,768)(1280,960)(1600,1200)(2048,1536)(2592,1944)(3264,2448)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"XVID"))  #*"MJPG"
        camera.set(cv2.CAP_PROP_FPS, 30)

        video_frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))

        video_FourCC    = int(camera.get(cv2.CAP_PROP_FOURCC))
        video_fps       = camera.get(cv2.CAP_PROP_FPS)
        video_size      = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        video_frames    = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))

        # codec = decode_fourcc(video_FourCC)
        # 鏡頭參數下界

    #🎥🎞📹🎬
    wanna_write_out_event_video = int(record_event)
    if(wanna_write_out_event_video):
        # print(videoPath.split('/')[-1].split('.')[0])
        write_video_fourcc_2 = cv2.VideoWriter_fourcc(*"XVID")
        ori_file_name = videoPath.split('/')[-1].split('.')[0]
        output_time = str(datetime.now()).replace(' ', '-')[0:-7]
        print(ori_file_name, output_time)
        out_2 = cv2.VideoWriter("{}_{}.avi".format(ori_file_name, output_time), write_video_fourcc_2, 4.0, (640, 480))




    while xxx:
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #initial parameters
        # 預設的隨機初始值
        if(mouse_save.flag_mouse_save == 2):

            #parameter
            # 參數引入上界 從yml檔
            with open('./thermalDetect/track_bar_parameters_ini.yml') as file:
                parameters = yaml.load(file, Loader=yaml.FullLoader)
                BED_ini = [parameters["bed_region"]["ul_x"], parameters["bed_region"]["ul_y"], parameters["bed_region"]["br_x"], parameters["bed_region"]["br_y"]]
                DETECT_REGION_ini = [parameters["detect_region"]["ul_x"], parameters["detect_region"]["ul_y"], parameters["detect_region"]["br_x"], parameters["detect_region"]["br_y"]]
                PILLOW_ini = [parameters["pillow_region"]["ul_x"], parameters["pillow_region"]["ul_y"], parameters["pillow_region"]["br_x"], parameters["pillow_region"]["br_y"]]
                FALL_THRES_ini = parameters["event_threshold"]["at_least_how_many_sec_sit_and_lie_down_could_make_fall_alert_happen"]
                SitWheelchairTooLong_ini = parameters["event_threshold"]["SitWheelchairTooLong"]
                ON_BED_THRES_ini = parameters["event_threshold"]["at_least_how_many_sec_sit_and_lie_down_could_make_on_bed_event_happen"]
                LEAVE_THRES_ini = parameters["event_threshold"]["at_least_how_many_sec_could_make_leave_event_happen_after_on_bed_event_happened"]
                SMART_PILLOW_ini = parameters["shout_down_leave_alert_threshold"]["smart_pillow"]
                BODY_HOT_THRES_ini = parameters["shout_down_leave_alert_threshold"]["at_least_how_many_gray_scale_value_in_pillow_represent_body_hot"]
                HEAD_PILLOW_DENSITY_ini = parameters["shout_down_leave_alert_threshold"]["at_least_how_many_density_of_head_in_pillow"]
                SHOW_DETIAL_ini = parameters["show_detial"]
                CURRENT_FPS_ini = parameters["current_fps"]
                TEMP_LEAVE_ini = parameters["temp_leave"]
                ENOUGH_LIE_DOWN_ini = parameters["enough_lie_down"]
                SIT_TIRED_ini = parameters["sit_tired"]
                OpenGetOverRatio_ini = parameters["OpenGetOverRatio"]
                GetOverRatio_ini = parameters["GetOverRatio"]
                ShutDownOutside_ini = parameters["ShutDownOutside"]
                sit_side_alarm_ini = parameters["sit_side_alarm"]
                sit_back_alarm_ini = parameters["sit_back_alarm"]
                SLEEP_THRES_ini = parameters["sleep_thres"]
                RESET_TIME_ini = parameters["when_to_reset_without_anyone"]
                StillInBed_ini = parameters["StillInBed"]
                RECORD_ini = parameters["REC..."]
                RecCutTime_ini = parameters["RecCutTime"]
                camera_high_quality_version_ini = parameters["high_quality"]
                fall_distance_ini = parameters["fall_distance"]
                fall_degree_ini = parameters["fall_degree"]
                many_bboxes_ini = parameters["many_bboxes"]
                bed_moving_ini = parameters["bed_moving"]
                moving_mean_IoM_THRES_ini = parameters["moving_mean_IoM_THRES"]
                extra_bed_xyxy01_ini = [parameters["extra_bed_x1"], parameters["extra_bed_y1"], parameters["extra_bed_x2"], parameters["extra_bed_y2"], parameters["extra_bed"]]
                # wheelchair
                no_wheelchair_ini = parameters["wheelchair"]["no_wheelchair"]
                wheelchair_time_too_long_ini = parameters["wheelchair"]["wheelchair_time_too_long"]
                wheelchair_distance_too_far_ini = parameters["wheelchair"]["wheelchair_distance_too_far"]
                wheelchair_distance_too_close_ini = parameters["wheelchair"]["wheelchair_distance_too_close"]

                COLLECT_FREQ_ini = {'stand':480, 'sit':480, 'lie_down':480, 'miss_lie_down':480} # 480 sec is about 8 min
                COLLECT_FREQ_ini["stand"] = parameters["collect_freq"]["stand"]
                COLLECT_FREQ_ini["sit"] = parameters["collect_freq"]["sit"]
                COLLECT_FREQ_ini["lie_down"] = parameters["collect_freq"]["lie_down"]
                COLLECT_FREQ_ini["miss_lie_down"] = parameters["collect_freq"]["miss_lie_down"]

                SEND_FREQ_ini=parameters["SEND_FREQ"]
                post_to_google_excel_ini=parameters["post_to_google_excel"]

                warn_alert_long_ini=parameters["warn_alert_long"]
                need_check_repeat_ini=parameters["need_check_repeat"]
                check_repeat_time_range_ini=parameters["check_repeat_time_range"]
            # # initial_parameters(control)
            # BED_ini = [207, 0, 518, 472]
            # DETECT_REGION_ini = [0, 0, 639, 479]
            # PILLOW_ini = [243, 361, 483, 460]
            # FALL_THRES_ini, SitWheelchairTooLong_ini, ON_BED_THRES_ini, LEAVE_THRES_ini, SMART_PILLOW_ini, BODY_HOT_THRES_ini, HEAD_PILLOW_DENSITY_ini, SHOW_DETIAL_ini = 5, 1, 3, 5, 1, 10, 8, 0
            # CURRENT_FPS_ini = 25
            # TEMP_LEAVE_ini = 24
            # ENOUGH_LIE_DOWN_ini = 5
            # SIT_TIRED_ini = 3
            # OpenGetOverRatio_ini = 0
            # GetOverRatio_ini = 10
            # ShutDownOutside_ini = 0
            # sit_side_alarm_ini = 1
            # sit_back_alarm_ini = 0
            # SLEEP_THRES_ini = 30
            # RESET_TIME_ini = 10
            # StillInBed_ini = 4
            # RECORD_ini = 0
            # camera_high_quality_version_ini = camera_high_quality_version_t
            # many_bboxes_ini = 3
            # fall_distance_ini = 25
            # fall_degree_ini = 25
            # # wheelchair
            # no_wheelchair_ini = 10
            # wheelchair_time_too_long_ini = 6
            # wheelchair_distance_too_far_ini = 10
            # wheelchair_distance_too_close_ini = 3
            # # collect_freq
            # COLLECT_FREQ_ini = {'stand':480, 'sit':480, 'lie_down':480, 'miss_lie_down':480} # 480 sec is about 8 min
            # bed_moving_ini = 1
            # moving_mean_IoM_THRES_ini = 70
            # extra_bed_xyxy01_ini = [100, 200, 300, 400, 0]
            ####################################
            try:
                if(":".join((keyABC_A[0:2], keyABC_A[2:], keyB[0:2], keyB[2:], keyC[0:2], keyC[2:])) == macAB+macC):
            ####################################
                    control = track_bar.CONTROL(BED_ini, DETECT_REGION_ini, PILLOW_ini, FALL_THRES_ini, SitWheelchairTooLong_ini, ON_BED_THRES_ini, LEAVE_THRES_ini, SMART_PILLOW_ini, BODY_HOT_THRES_ini, \
                        HEAD_PILLOW_DENSITY_ini, SHOW_DETIAL_ini, CURRENT_FPS_ini, TEMP_LEAVE_ini, ENOUGH_LIE_DOWN_ini, \
                            SIT_TIRED_ini, OpenGetOverRatio_ini, GetOverRatio_ini, ShutDownOutside_ini, sit_side_alarm_ini, sit_back_alarm_ini, SLEEP_THRES_ini, how_much_file, RESET_TIME_ini, StillInBed_ini, \
                                COLLECT_FREQ_ini, RECORD_ini, RecCutTime_ini, camera_high_quality_version_ini, many_bboxes_ini, fall_distance_ini, fall_degree_ini, no_wheelchair_ini, wheelchair_time_too_long_ini, \
                                wheelchair_distance_too_far_ini, wheelchair_distance_too_close_ini, bed_moving_ini, moving_mean_IoM_THRES_ini, extra_bed_xyxy01_ini,SEND_FREQ_ini,post_to_google_excel_ini,warn_alert_long_ini,need_check_repeat_ini,check_repeat_time_range_ini)
                    control.track_bar()
                    mouse_save.flag_mouse_save = 0
            #####################################
                else: # upLoad
                    idx = str(0)

                    ori_MAC = ":".join((keyABC_A[0:2], keyABC_A[2:], keyB[0:2], keyB[2:], keyC[0:2], keyC[2:]))
                    new_MAC = macAB+macC
                    payload = {
                        'entry.885823784': '{}'.format(ori_MAC), #ori_MAC
                        'entry.1742594334': '{}'.format(new_MAC), #new_MAC
                        'entry.1867958386': 'they failed' #account
                        # 'entry.129964452': '{}'.format(idx), #passward
                        }
                    # 將檔案加入 POST 請求中
                    r = requests.post('https://docs.google.com/forms/u/0/d/e/1FAIpQLSdZUQ6Ipe12tPM8-VaeSR7mQlaFcUe8be9fnPihrVwQe0Z6Ng/formResponse', data = payload)
            except:
                pass
            #####################################


        # 從鏡頭讀圖進來前處理一下的上界
        TimeFrameStart = time.time()

        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        if(control.pre_RECORD == 0 and control.RECORD == 1):
            control.prepare_REC = 1
            print('    ~ ~ ~ 開啟錄影功能 ~ ~ ~')
        elif(control.pre_RECORD == 1 and control.RECORD == 0):
            control.prepare_REC = 0
            control.release_video_write_out = 1
        else:
            pass

        if(control.prepare_REC == 1):
            write_video_fourcc = cv2.VideoWriter_fourcc(*"XVID")
            # out = cv2.VideoWriter("output.avi", write_video_fourcc, 18.0, (video_size))
            #---------------------------------------------------------------------speed
            # print('@@@@@@@@@@@@@@@@@@@@')

            # print("OUTPUTFILE：", OUTPUTFILE)
            output_time = str(datetime.now()).replace(' ', ',').replace('-', '')[0:-7]
            print(output_time) # 20211021,15:48:05

            s = output_time

            comma = ','
            comma_place = [pos for pos, char in enumerate(s) if char == comma]
            # print(comma_place) # [8]
            date_str = s[0:comma_place[0]]
            # print(date_str) # 20211021
            colon = ':'
            colon_place = [pos for pos, char in enumerate(s) if char == colon]
            # print(colon_place) # [11, 14]
            time_str = s[comma_place[0]+1:colon_place[1]].replace(':', '')
            # print(time_str)
            date_str
            time_str
            machine_number
            pieces = [machine_number, date_str, time_str]
            OUTPUTFILE_NAME_use_time = '_'.join(pieces)
            print("OUTPUTFILE：", OUTPUTFILE_NAME_use_time)

            out = cv2.VideoWriter("{}.avi".format(OUTPUTFILE_NAME_use_time), write_video_fourcc, 4.0, (640, 480))
            # out = cv2.VideoWriter("output.avi", write_video_fourcc, 18.0, (640, 480+80))
            # global prepare_REC
            control.prepare_REC = 0

            starttime = datetime.now()


        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        # print(isReadFromVideo)
        if (isReadFromVideo != 'video'): # ini is cam
            #image from thermal camera is uint8 C1
            # camera_high_quality_version = control.camera_high_quality_version
            img_c1_Origin = capture(flip_v = options.flip_v, device = options.device, camera_high_quality_version=camera_high_quality_version_t) #control.camera_high_quality_version)
            if img_c1_Origin.size == 0:
                raise IOError("Couldn't read image from thermal camera!!")
            WRITE_VIDEO = control.RECORD
            control.pre_RECORD = control.RECORD
            if(WRITE_VIDEO == 1):
                img_c1_Origin_3channel = cv2.cvtColor(img_c1_Origin, cv2.COLOR_GRAY2RGB)
                out.write(img_c1_Origin_3channel)
                print('影片錄製中...')
                # print(img_c1_Origin.shape)

                # 該區塊只會在影片錄製中...發生
                endtime = datetime.now()
                cont_rec_time_sec = (endtime - starttime).seconds
                print(cont_rec_time_sec)
                if(cont_rec_time_sec >= control.RecCutTime*60*60):
                    out.release()
                    print('    ~ ~ ~ 短暫關閉錄影功能 ~ ~ ~')
                    print('    ~ ~ ~ 緊接著再開啟錄影功能 ~ ~ ~')
                    control.prepare_REC = 1

            else: #if(WRITE_VIDEO == 0):
                if(control.release_video_write_out == 1):
                    out.release()
                    print('    ~ ~ ~ 關閉錄影功能 ~ ~ ~')
                    control.release_video_write_out = 0
            #cv2.namedWindow("C1_ORIGIN", cv2.WINDOW_NORMAL)
            #cv2.imshow("C1_ORIGIN", img_c1_Origin)
        else: #if(isReadFromVideo == 'video'):
            res, img = camera.read()
            img = cv2.resize(img, dsize=(640,480), interpolation = cv2.INTER_CUBIC)
            # img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_c1_Origin = img
            # # print(img.shape)
            # cv2.imshow('My Image', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


        # #image from thermal camera is uint8 C1
        # img_c1_Origin = capture(flip_v = options.flip_v, device = options.device)
        # if img_c1_Origin.size == 0:
        #     raise IOError("Couldn't read image from thermal camera!!")
        # #cv2.namedWindow("C1_ORIGIN", cv2.WINDOW_NORMAL)
        # #cv2.imshow("C1_ORIGIN", img_c1_Origin)

        #print("BBB");

        colorMapStart = time.time()
        #------color map transformatiom
        height= img_c1_Origin.shape[0]
        width= img_c1_Origin.shape[1]
        nchannels = 1

        #c1 to c3 clor map
        #img_c3_ColorMap = cv2.applyColorMap(img_c1_Origin, cv2.COLORMAP_RAINBOW)
        #img_c3_ColorMap = cv2.cvtColor(img_c3_ColorMap, cv2.COLOR_BGR2RGB)
        img_c3_ColorMap = cv2.applyColorMap(img_c1_Origin, cv2.COLORMAP_JET)
        img_c3_ColorMap_postimg = img_c3_ColorMap.copy()
        # print('HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH')
        # print(img_c1_Origin.shape)

        # img_c3_ColorMap_copy = cv2.copyMakeBorder(img_c3_ColorMap,0,0,0,0,cv2.BORDER_REPLICATE)
        # #cv2.imwrite('event.jpg', img_c3_ColorMap_copy)

        #img_c3_ColorMap = cv2.applyColorMap(img_c1_Origin, cv2.COLORMAP_HSV)

        #------run inference
        if(camera_high_quality_version_t == 0): #control.camera_high_quality_version == 0): #2.5
            DETECTION_THRESHOLD_quality_decides = 0.235
        else:  #3.5
            DETECTION_THRESHOLD_quality_decides = 0.4

        # 從鏡頭讀圖進來前處理一下的下界


        # yolov3tiny預測出來的結果
        # import thermalDetect.runDetect as detect
        results = detect.run_inference(input_blob, exec_net, img_c3_ColorMap, DETECTION_THRESHOLD_quality_decides)
        PYBoxResultList = detect.thermalDetect_py(results)

        # detect_region = {'ulx':0, 'uly':0, 'brx':640, 'bry':480}
        # detect_region['ulx']
        # detect_region['uly']
        # detect_region['brx']
        # detect_region['bry']

        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        # print(PYBoxResultList)
        # print(len(PYBoxResultList))

        LABELS = list(obj_information.continuous_action_count.keys())
        # 留下有在偵測區內的bbox的上界
        detect_region_bbox = [control.XA_DETECT_REGION, control.YA_DETECT_REGION+80, control.XB_DETECT_REGION, control.YB_DETECT_REGION+80]
        BED_for_first_detect = (control.XA_BED, control.YA_BED+80, control.XB_BED, control.YB_BED+80)
        valid_bbox_list = []
        too_many_in_bed_wait_for_select = []
        IoM_for_2_person_in_bed = 0
        for i in range(len(PYBoxResultList)):
            yolobox_ulbr_inORout = [PYBoxResultList[i].x, PYBoxResultList[i].y+80, PYBoxResultList[i].x+PYBoxResultList[i].w, PYBoxResultList[i].y+80+PYBoxResultList[i].h] # the height is with 80 pixel upper space

            iou, inter_Area, yoloArea_inORout = event.bb_intersection_over_union(yolobox_ulbr_inORout, detect_region_bbox)
            # print(inter_Area, pillowArea)
            if(yoloArea_inORout > 0):
                if(inter_Area/yoloArea_inORout < 1/2):
                    cv2.rectangle(img_c3_ColorMap, (PYBoxResultList[i].x, PYBoxResultList[i].y), (PYBoxResultList[i].x+PYBoxResultList[i].w, PYBoxResultList[i].y+PYBoxResultList[i].h), (0, 0, 0), 2, 8, 0)
                    # 在少了80 pixel upper bar的圖片上畫一條對角線
                    cv2.line(img_c3_ColorMap, (PYBoxResultList[i].x, PYBoxResultList[i].y), (PYBoxResultList[i].x+PYBoxResultList[i].w, PYBoxResultList[i].y+PYBoxResultList[i].h), (0, 0, 0), 2, 8, 0)
                    # 在少了80 pixel upper bar的圖片上畫一條對角線
                    cv2.line(img_c3_ColorMap, (PYBoxResultList[i].x+PYBoxResultList[i].w, PYBoxResultList[i].y), (PYBoxResultList[i].x, PYBoxResultList[i].y+PYBoxResultList[i].h), (0, 0, 0), 2, 8, 0)

                    # wait_for_del_list.append(i)
                else: #(inter_Area/yoloArea_inORout >= 1/2) # 有在偵測區內的所有人
                    IoM_for_2_person_in_bed = obj_information.calculateIoM(yolobox_ulbr_inORout, BED_for_first_detect)
                    if(IoM_for_2_person_in_bed >= 0.9):
                        too_many_in_bed_wait_for_select.append(PYBoxResultList[i])
                    else:
                        valid_bbox_list.append(PYBoxResultList[i])
                        label = PYBoxResultList[i].obj_id #class_id
                        label_text = LABELS[label]
                        box_color, box_thickness = (0, 0, 255), 2
                        cv2.rectangle(img_c3_ColorMap, (PYBoxResultList[i].x, PYBoxResultList[i].y), (PYBoxResultList[i].x+PYBoxResultList[i].w, PYBoxResultList[i].y+PYBoxResultList[i].h), box_color, box_thickness, 8, 0)

                        cv2.putText(img_c3_ColorMap, label_text, (int((PYBoxResultList[i].x+PYBoxResultList[i].x+PYBoxResultList[i].w)/2-20), PYBoxResultList[i].y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 9, cv2.LINE_AA)
                        cv2.putText(img_c3_ColorMap, label_text, (int((PYBoxResultList[i].x+PYBoxResultList[i].x+PYBoxResultList[i].w)/2-20), PYBoxResultList[i].y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 7, cv2.LINE_AA)
                        cv2.putText(img_c3_ColorMap, label_text, (int((PYBoxResultList[i].x+PYBoxResultList[i].x+PYBoxResultList[i].w)/2-20), PYBoxResultList[i].y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 5, cv2.LINE_AA)
                        cv2.putText(img_c3_ColorMap, label_text, (int((PYBoxResultList[i].x+PYBoxResultList[i].x+PYBoxResultList[i].w)/2-20), PYBoxResultList[i].y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(img_c3_ColorMap, label_text, (int((PYBoxResultList[i].x+PYBoxResultList[i].x+PYBoxResultList[i].w)/2-20), PYBoxResultList[i].y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        PYBoxResultList = valid_bbox_list

        # 處理床內的框，一個就直接處理，兩個就比一下（假設最多就只會同時出現兩個框在床內）
        if(len(too_many_in_bed_wait_for_select) >= 1):
            if(len(too_many_in_bed_wait_for_select) == 1):
                i = 0
            else: #假設頂多只會碰到床內有兩個人的情況
                if(too_many_in_bed_wait_for_select[0].y+too_many_in_bed_wait_for_select[0].h+80 > too_many_in_bed_wait_for_select[1].y+too_many_in_bed_wait_for_select[1].h+80): #2
                    i = 0
                else:
                    i = 1
            # 在這變化出來的話會混淆，因為有補幀的關係，所以不知道這一個會不會被拿來用，故要搬到yolofilter之後才行
            # label = too_many_in_bed_wait_for_select[i].obj_id #class_id
            # label_text = LABELS[label]
            # box_color, box_thickness = (0, 0, 255), 2
            # cv2.rectangle(img_c3_ColorMap, (too_many_in_bed_wait_for_select[i].x, too_many_in_bed_wait_for_select[i].y), (too_many_in_bed_wait_for_select[i].x+too_many_in_bed_wait_for_select[i].w, too_many_in_bed_wait_for_select[i].y+too_many_in_bed_wait_for_select[i].h), box_color, box_thickness, 8, 0)

            # cv2.putText(img_c3_ColorMap, label_text, (int((too_many_in_bed_wait_for_select[i].x+too_many_in_bed_wait_for_select[i].x+too_many_in_bed_wait_for_select[i].w)/2-20), too_many_in_bed_wait_for_select[i].y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 9, cv2.LINE_AA)
            # cv2.putText(img_c3_ColorMap, label_text, (int((too_many_in_bed_wait_for_select[i].x+too_many_in_bed_wait_for_select[i].x+too_many_in_bed_wait_for_select[i].w)/2-20), too_many_in_bed_wait_for_select[i].y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 7, cv2.LINE_AA)
            # cv2.putText(img_c3_ColorMap, label_text, (int((too_many_in_bed_wait_for_select[i].x+too_many_in_bed_wait_for_select[i].x+too_many_in_bed_wait_for_select[i].w)/2-20), too_many_in_bed_wait_for_select[i].y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 5, cv2.LINE_AA)
            # cv2.putText(img_c3_ColorMap, label_text, (int((too_many_in_bed_wait_for_select[i].x+too_many_in_bed_wait_for_select[i].x+too_many_in_bed_wait_for_select[i].w)/2-20), too_many_in_bed_wait_for_select[i].y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            # cv2.putText(img_c3_ColorMap, label_text, (int((too_many_in_bed_wait_for_select[i].x+too_many_in_bed_wait_for_select[i].x+too_many_in_bed_wait_for_select[i].w)/2-20), too_many_in_bed_wait_for_select[i].y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            PYBoxResultList.insert( 0, too_many_in_bed_wait_for_select[i])

        # 留下有在偵測區內的bbox的下界

        # 獨立於事件的偷移床發報區的上界
        # 床的區域
        BED_for_alert_moving = (control.XA_BED, control.YA_BED+80, control.XB_BED, control.YB_BED+80)
        cv2.rectangle(img_c3_ColorMap, (control.XA_BED, control.YA_BED), (control.XB_BED, control.YB_BED), (0, 255, 255), 2, 8, 0)
        # 偷移床的排除區
        extra_bed_with_80 = [control.extra_bed_x1, control.extra_bed_y1+80, control.extra_bed_x2, control.extra_bed_y2+80]
        the_max_iou_ones_iou = 0
        for i in range(len(PYBoxResultList)):
            yolobox_ulbr = [PYBoxResultList[i].x, PYBoxResultList[i].y+80, PYBoxResultList[i].x+PYBoxResultList[i].w, PYBoxResultList[i].y+80+PYBoxResultList[i].h] # the height is with 80 pixel upper space
            iou, inter_Area, yoloArea_inORout = event.bb_intersection_over_union(yolobox_ulbr, BED_for_alert_moving)
            # print('iou: ', iou)
            if(iou > the_max_iou_ones_iou):
                the_max_iou_one = i
                the_max_iou_ones_iou = iou

        # print('@@@@@@@')
        # print(the_max_iou_ones_iou)
        if(the_max_iou_ones_iou > 0):
            i = the_max_iou_one
            yolobox_ulbr = [PYBoxResultList[i].x, PYBoxResultList[i].y+80, PYBoxResultList[i].x+PYBoxResultList[i].w, PYBoxResultList[i].y+80+PYBoxResultList[i].h] # the height is with 80 pixel upper space
            iou, inter_Area, yoloArea_inORout = event.bb_intersection_over_union(yolobox_ulbr, extra_bed_with_80)
            if(control.extra_bed == 1):
                cv2.rectangle(img_c3_ColorMap, (extra_bed_with_80[0], extra_bed_with_80[1]-80), (extra_bed_with_80[2], extra_bed_with_80[3]-80), (0, 0, 0), 3)
                extra_bed_with_80_height = abs(extra_bed_with_80[3]-extra_bed_with_80[1])
                extra_bed_with_80_width = abs(extra_bed_with_80[2]-extra_bed_with_80[0])
                extra_bed_with_80_area = max(1, (extra_bed_with_80_height*extra_bed_with_80_width))
                this_one_not_in_extra_bed = (inter_Area/extra_bed_with_80_area < 1/2)
            else:
                this_one_not_in_extra_bed = 'True'
            if(this_one_not_in_extra_bed):
                action_label = PYBoxResultList[i].obj_id
                if(action_label == 2): #躺
                    # 監測看看床有沒有被偷移動 抓偷移床 偷移動床 偷嚕床 人一天差不多就躺8個小時
                    # 儲存連續8分（大約1440幀）的中心點
                    # print('#################')
                    # 拔掉離現在最遠的那一幀
                    yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'] = np.delete(yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'], 0, axis=0)
                    # np.array([(yolobox_ulbr[0]+yolobox_ulbr[2])/2, (yolobox_ulbr[1]+yolobox_ulbr[3])/2]) #[x, y]
                    # 接上當前這一幀躺的資訊
                    bbox_center_x, bbox_center_y = (yolobox_ulbr[0]+yolobox_ulbr[2])/2, (yolobox_ulbr[1]+yolobox_ulbr[3])/2
                    bbox_width = abs(yolobox_ulbr[2]-yolobox_ulbr[0])
                    bbox_heigh = abs(yolobox_ulbr[3]-yolobox_ulbr[1])
                    bbox_IoU = iou
                    bbox_IoM = obj_information.calculateIoM(yolobox_ulbr, BED_for_alert_moving)
                    six_elements_list = [bbox_center_x, bbox_center_y, bbox_width, bbox_heigh, bbox_IoU, bbox_IoM]
                    wait_for_expand_dims = np.array(six_elements_list)
                    wait_for_concat = np.expand_dims(wait_for_expand_dims, axis=0)
                    yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'] = np.concatenate((yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'], wait_for_concat), axis=0)
                    # print('#################')
                    yolo_bbox_for_event.bed_moving_cent['count'] += 1
            else:
                # extra_bed_with_80 = [control.extra_bed_x1, control.extra_bed_y1+80, control.extra_bed_x2, control.extra_bed_y2+80]
                # 畫出extra_bed的X
                # 在少了80 pixel upper bar的圖片上畫一條對角線
                cv2.line(img_c3_ColorMap, (extra_bed_with_80[0], extra_bed_with_80[1]-80), (extra_bed_with_80[2], extra_bed_with_80[3]-80), (0, 0, 0), 2, 8, 0)
                # 在少了80 pixel upper bar的圖片上畫一條對角線
                cv2.line(img_c3_ColorMap, (extra_bed_with_80[2], extra_bed_with_80[1]-80), (extra_bed_with_80[0], extra_bed_with_80[3]-80), (0, 0, 0), 2, 8, 0)
            # print('@@@@@@@@@')
            # print('bed_moving_cent_how_many_frames: ', yolo_bbox_for_event.bed_moving_cent['count'])
            if(yolo_bbox_for_event.bed_moving_cent['count'] >= 1440):
                yolo_bbox_for_event.bed_moving_cent['count'] = 0
                # 計算x中心的平均
                invalid_guys_for_row_index = np.argwhere(yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'][:, 0] == -1)  #用[:, 0]取，still 2 維
                # print(invalid_guys_for_row_index)  #[[0].....[1434]]
                # print(invalid_guys_for_row_index[-1][0], np.shape(yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'])[0]-1)
                if(invalid_guys_for_row_index.shape == (0, 1)):
                    cor_for_x = 0
                elif(invalid_guys_for_row_index[-1][0] == np.shape(yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'])[0]-1):
                    cor_for_x_the_latest_invalid_one = invalid_guys_for_row_index[-1][0] #[[1434]]
                    cor_for_x = cor_for_x_the_latest_invalid_one
                else:
                    cor_for_x_the_latest_invalid_one = invalid_guys_for_row_index[-1][0] #[[1434 ]]
                    cor_for_x = cor_for_x_the_latest_invalid_one + 1
                # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                # print(cor_for_x)
                moving_mean_x_cent = int(np.mean(yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'][cor_for_x:], axis=0)[0])
                moving_mean_y_cent = int(np.mean(yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'][cor_for_x:], axis=0)[1])
                moving_mean_w = int(np.mean(yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'][cor_for_x:], axis=0)[2])
                moving_mean_h = int(np.mean(yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'][cor_for_x:], axis=0)[3])
                moving_mean_IoU = np.mean(yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'][cor_for_x:], axis=0)[4]
                moving_mean_IoM = np.mean(yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'][cor_for_x:], axis=0)[5]
                # 計算中心點偏移
                BED_for_cent = (control.XA_BED, control.YA_BED+80, control.XB_BED, control.YB_BED+80)
                x_drift = int(abs((BED_for_cent[0]+BED_for_cent[2])/2 - moving_mean_x_cent))
                y_drift = int(abs((BED_for_cent[1]+BED_for_cent[3])/2 - moving_mean_y_cent))
                if(control.bed_moving == 1):
                    if(moving_mean_IoM < control.moving_mean_IoM_THRES/100): #預設0.7
                        # 畫出平均躺的框
                        print('疑似偷移床！IoM is ', moving_mean_IoM)
                        moving_mean_ul_x = int(max(moving_mean_x_cent-moving_mean_w/2, 0))
                        moving_mean_ul_y = int(max(moving_mean_y_cent-moving_mean_h/2, 0))
                        moving_mean_br_x = int(min(moving_mean_x_cent+moving_mean_w/2, 640))
                        moving_mean_br_y = int(min(moving_mean_y_cent+moving_mean_h/2, 480))
                        moving_mean_cent_x = max(int((moving_mean_ul_x+moving_mean_ul_x)/2), 0)
                        moving_mean_cent_y = int((moving_mean_ul_y+moving_mean_br_y)/2)
                        cv2.rectangle(img_c3_ColorMap, (moving_mean_ul_x, moving_mean_ul_y-80), (moving_mean_br_x, moving_mean_br_y-80), (255, 0, 0), 2, 10, 0)
                        cv2.putText(img_c3_ColorMap, text=' average', org=(moving_mean_cent_x, moving_mean_cent_y-80), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale=1.1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                        cv2.putText(img_c3_ColorMap, text='lie_down', org=(moving_mean_cent_x, min(480, moving_mean_cent_y+22-80)), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale=1.1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                        cv2.putText(img_c3_ColorMap, text=' average', org=(moving_mean_cent_x, moving_mean_cent_y-80), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale=1.1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                        cv2.putText(img_c3_ColorMap, text='lie_down', org=(moving_mean_cent_x, min(480, moving_mean_cent_y+22-80)), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale=1.1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                        ther_4 = threading.Thread(target = post_on_for_bed_moving, args=(moving_mean_x_cent, moving_mean_y_cent-80, moving_mean_w, moving_mean_h, x_drift, y_drift, moving_mean_IoU, moving_mean_IoM, img_c3_ColorMap))
                        ther_4.start()
            # 獨立於事件的偷移床發報區的下界



        # 點擊滑鼠右鍵即時寫入參數到yml檔
        # mouse_save_track_bar_parameter
        if mouse_save.flag_mouse_save == 1:
            with open('./thermalDetect/track_bar_parameters.yml', 'w') as file:
                yaml.dump({
                    "bed_region":{
                        "ul_x": control.XA_BED,
                        "ul_y": control.YA_BED,
                        "br_x": control.XB_BED,
                        "br_y": control.YB_BED},
                    "detect_region":{
                        "ul_x": control.XA_DETECT_REGION,
                        "ul_y": control.YA_DETECT_REGION,
                        "br_x": control.XB_DETECT_REGION,
                        "br_y": control.YB_DETECT_REGION},
                    "pillow_region":{
                        "ul_x": control.XA_PILLOW,
                        "ul_y": control.YA_PILLOW,
                        "br_x": control.XB_PILLOW,
                        "br_y": control.YB_PILLOW},
                    "event_threshold":{
                        "at_least_how_many_sec_sit_and_lie_down_could_make_fall_alert_happen": control.FALL_THRES,
                        "SitWheelchairTooLong": control.SitWheelchairTooLong,
                        "at_least_how_many_sec_sit_and_lie_down_could_make_on_bed_event_happen": control.ON_BED_THRES,
                        "at_least_how_many_sec_could_make_leave_event_happen_after_on_bed_event_happened": control.LEAVE_THRES},
                    "shout_down_leave_alert_threshold":{
                        "smart_pillow": control.SMART_PILLOW,
                        "at_least_how_many_gray_scale_value_in_pillow_represent_body_hot": control.BODY_HOT_THRES,
                        "at_least_how_many_density_of_head_in_pillow": control.HEAD_PILLOW_DENSITY},
                    "show_detial": control.SHOW_DETIAL,
                    "current_fps": control.CURRENT_FPS,
                    "temp_leave": control.TEMP_LEAVE,
                    "enough_lie_down": control.ENOUGH_LIE_DOWN,
                    "sit_tired": control.SIT_TIRED,
                    "OpenGetOverRatio": control.OpenGetOverRatio,
                    "GetOverRatio": control.GetOverRatio,
                    "ShutDownOutside": control.ShutDownOutside,
                    "sit_side_alarm": control.sit_side_alarm,
                    "sit_back_alarm": control.sit_back_alarm,
                    "sleep_thres": control.SLEEP_THRES,
                    "when_to_reset_without_anyone": control.RESET_TIME,
                    "StillInBed": control.StillInBed,
                    "REC...": control.RECORD,
                    "RecCutTime": control.RecCutTime,
                    "high_quality": camera_high_quality_version_t, #control.camera_high_quality_version,
                    "many_bboxes": control.many_bboxes,
                    "fall_distance": control.fall_distance,
                    "fall_degree": control.fall_degree,
                    "bed_moving": control.bed_moving,
                    "moving_mean_IoM_THRES": control.moving_mean_IoM_THRES,
                    # control.extra_bed control.extra_bed_x1 control.extra_bed_y1 control.extra_bed_x2 control.extra_bed_y2
                    "extra_bed": control.extra_bed,
                    "extra_bed_x1": control.extra_bed_x1,
                    "extra_bed_y1": control.extra_bed_y1,
                    "extra_bed_x2": control.extra_bed_x2,
                    "extra_bed_y2": control.extra_bed_y2,
                    # wheelchair
                    "wheelchair":{
                        "no_wheelchair": control.no_wheelchair, #10,
                        "wheelchair_time_too_long": control.wheelchair_time_too_long, #20,
                        "wheelchair_distance_too_far": control.wheelchair_distance_too_far, #45,
                        "wheelchair_distance_too_close": control.wheelchair_distance_too_close}, #20},
                    "collect_freq":{
                        "stand": control.COLLECT_FREQ["stand"],
                        "sit": control.COLLECT_FREQ["sit"],
                        "lie_down": control.COLLECT_FREQ["lie_down"],
                        "miss_lie_down": control.COLLECT_FREQ["miss_lie_down"]},
                    "SEND_FREQ": control.SEND_FREQ,
                    "post_to_google_excel": control.post_to_google_excel,
                    "warn_alert_long": control.warn_alert_long,
                    "need_check_repeat": control.need_check_repeat,
                    "check_repeat_time_range": control.check_repeat_time_range
                    }, file, sort_keys=False)
            mouse_save.flag_mouse_save = 0


        # img_c1_Origin is 640*480 without upper space 80 height
        BED_p = (control.XA_BED, control.YA_BED+80, control.XB_BED, control.YB_BED+80)

        # 開啟枕頭功能的上界
        # 預設關閉離床警報的功能被關閉也就是預設開啟離床警報(不只離床還有跌倒等所有的警報)
        # "shutdown mask" at any time
        shutdown_the_leave_alert = "switch_off"
        if(control.SMART_PILLOW == 0):
            # 手動
            PILLOW__ = (control.XA_PILLOW, control.YA_PILLOW+80, control.XB_PILLOW, control.YB_PILLOW+80)
            BODY_HOT_THRES = 170#control.BODY_HOT_THRES #= 255/2
            HEAD_PILLOW_DENSITY = control.HEAD_PILLOW_DENSITY/10 #= 8/10
        else: #smart_pillow == 1
            # 自動 smart pillow
            if(obj_information.pillow_area[0] == -1):
                PILLOW__ = BED_p[0]+20, int(BED_p[3]-(BED_p[3]-BED_p[1])/3-20), BED_p[2]-20, int(BED_p[3]-(BED_p[3]-BED_p[1])/3-20+100)
                BODY_HOT_THRES = 170
                HEAD_PILLOW_DENSITY = control.HEAD_PILLOW_DENSITY/10 #0.8%
            else:
                PILLOW__ = obj_information.pillow_area
                # 確保枕頭always都在床內
                if(abs(PILLOW__[2]-PILLOW__[0]) >= abs(BED_p[2]-BED_p[0])):
                    PILLOW__ = BED_p[0]+20, int(BED_p[3]-(BED_p[3]-BED_p[1])/3-20), BED_p[2]-20, int(BED_p[3]-(BED_p[3]-BED_p[1])/3-20+100)
                BODY_HOT_THRES = obj_information.pillow_BODY_HOT_THRES
                HEAD_PILLOW_DENSITY = control.HEAD_PILLOW_DENSITY/10 #0.8%

        # 枕頭內有沒有人
        pillow_total_area = (PILLOW__[2]-PILLOW__[0])*(PILLOW__[3]-PILLOW__[1])  #23760
        pillow_crop = img_c1_Origin[PILLOW__[1]-80:PILLOW__[3]-80, PILLOW__[0]:PILLOW__[2]]
        head_in_pillow_crop = np.sum(pillow_crop > BODY_HOT_THRES)
        if(pillow_total_area <= 0):
            pillow_total_area = 0.00001
        head_over_pillow_density = head_in_pillow_crop/pillow_total_area
        need_extra_count = 0
        if(head_over_pillow_density*100 > HEAD_PILLOW_DENSITY): # create a mask for shutdown the leave alert
            need_extra_count = 1
            # 不是枕頭一開啟就開始計數，還要分成兩種情況下去計數，否則會重複計數
            # 況1：有人在床裡面，但是yolo漏抓沒有抓到人(a)
            # 況2：有人在床裡面，yolo有抓到人
            shutdown_the_leave_alert = "switch_on"
        # 開啟枕頭功能的下界

        # fixed the error of yolov3 output
        # 過濾yolov3tiny吐出來的結果只留下正在被tracking的那一個bbox
        # 吐出來的結果只有2種而已分別為1個bbox或0個bbox
        yolo_bbox_for_event.yolo_bbox_filter(PYBoxResultList, obj_information, control, classes_para)
        # yolo_bbox_for_event.PYBoxResultList在過濾完成之後只會出現0個bbox或1個


        if(len(yolo_bbox_for_event.PYBoxResultList) != 0):
            too_many_in_bed_wait_for_select = [yolo_bbox_for_event.PYBoxResultList[0]]
            i = 0
            label = too_many_in_bed_wait_for_select[i].obj_id #class_id
            label_text = LABELS[label]
            box_color, box_thickness = (0, 0, 255), 2
            cv2.rectangle(img_c3_ColorMap, (too_many_in_bed_wait_for_select[i].x, too_many_in_bed_wait_for_select[i].y), (too_many_in_bed_wait_for_select[i].x+too_many_in_bed_wait_for_select[i].w, too_many_in_bed_wait_for_select[i].y+too_many_in_bed_wait_for_select[i].h), box_color, box_thickness, 8, 0)

            cv2.putText(img_c3_ColorMap, label_text, (int((too_many_in_bed_wait_for_select[i].x+too_many_in_bed_wait_for_select[i].x+too_many_in_bed_wait_for_select[i].w)/2-20), too_many_in_bed_wait_for_select[i].y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 9, cv2.LINE_AA)
            cv2.putText(img_c3_ColorMap, label_text, (int((too_many_in_bed_wait_for_select[i].x+too_many_in_bed_wait_for_select[i].x+too_many_in_bed_wait_for_select[i].w)/2-20), too_many_in_bed_wait_for_select[i].y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 7, cv2.LINE_AA)
            cv2.putText(img_c3_ColorMap, label_text, (int((too_many_in_bed_wait_for_select[i].x+too_many_in_bed_wait_for_select[i].x+too_many_in_bed_wait_for_select[i].w)/2-20), too_many_in_bed_wait_for_select[i].y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(img_c3_ColorMap, label_text, (int((too_many_in_bed_wait_for_select[i].x+too_many_in_bed_wait_for_select[i].x+too_many_in_bed_wait_for_select[i].w)/2-20), too_many_in_bed_wait_for_select[i].y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img_c3_ColorMap, label_text, (int((too_many_in_bed_wait_for_select[i].x+too_many_in_bed_wait_for_select[i].x+too_many_in_bed_wait_for_select[i].w)/2-20), too_many_in_bed_wait_for_select[i].y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


        # 出現many_bboxes的情況多久之後就清空計數
        # print('control.many_bboxes', control.many_bboxes)
        if(control.many_bboxes < 60):
            if(yolo_bbox_for_event.how_many_continue_frames_more_then_one['every_frames'] > control.many_bboxes*2.8): #此處不能用">="
                # 清空所有計數 歸零所有計數(a)
                # RESET
                clean_ObjInformation_all(obj_information, classes_para)
                print('CLEAN ALL')
        # print('-------')
        # print(obj_information.as_of_the_present_state_action_count["out"]["count_for_leave_bed_inside_bed"])
        # print(yolo_bbox_for_event.how_many_continue_frames_more_then_one['every_frames'])

        # 一個正在被tracking的框都沒有 zero yolo_bbox
        if(yolo_bbox_for_event.jump_event == 1): # only include the empty one

            # 況1：有人在床裡面，但是yolo漏抓沒有抓到人(b)
            # 此時就是枕頭發揮所有計數調配的重要時機！！
            if(need_extra_count == 1):
                # COUNT["in"]["count_for_leave_bed_inside_bed"][2]
                # COUNT["in"]["lie_for_sit"][2]
                # 若["in"]["lie_for_sit"][2]答標則COUNT["in"]["lie_again"][2]
                # 在此時枕頭已經被觸發的情況之下只計數lie_down也就是[2]，其他廣義的躺一律不計數，因為本來就是一幀一個人只計數一種動作。⬇
                if(obj_information.as_of_the_present_state_action_count["in"]["count_for_leave_bed_inside_bed"][2] <= 1000):
                    obj_information.as_of_the_present_state_action_count["in"]["count_for_leave_bed_inside_bed"][2] += 1
                if(obj_information.as_of_the_present_state_action_count["in"]["lie_for_sit"][2] <= 1000):
                    obj_information.as_of_the_present_state_action_count["in"]["lie_for_sit"][2] += 1
                # 此時床內有人，且與此同時必須延續剛剛『廣義的躺』的所以計數：
                # [0]=stand, [1]=sit_side, [2]=lie_down, [3]=find_leg, [4]=sit_wheelchair, [5]=sit_back, [6]=sit_static_chair
                #『廣義的躺』:（先將截至目前為止所蒐集到的『廣義的躺』存放在此處，等歸0完之後再回補）ps. sit_side暫時不回補，沒有為什麼。
                lie_down_of_general_lie_down_in_bed_for_leave = obj_information.as_of_the_present_state_action_count["in"]["count_for_leave_bed_inside_bed"][2]
                find_leg_of_general_lie_down_in_bed_for_leave = obj_information.as_of_the_present_state_action_count["in"]["count_for_leave_bed_inside_bed"][3]
                sit_back_of_general_lie_down_in_bed_for_leave = obj_information.as_of_the_present_state_action_count["in"]["count_for_leave_bed_inside_bed"][5]
                lie_down_of_general_lie_down_in_bed_for_wanna_leave = obj_information.as_of_the_present_state_action_count["in"]["lie_for_sit"][2]
                find_leg_of_general_lie_down_in_bed_for_wanna_leave = obj_information.as_of_the_present_state_action_count["in"]["lie_for_sit"][3]
                sit_back_of_general_lie_down_in_bed_for_wanna_leave = obj_information.as_of_the_present_state_action_count["in"]["lie_for_sit"][5]

                '''#重要的歸0時機，從其他地方剛回床後的躺，或延續已經躺一陣子了的躺的上界⬇
                '''
                motion_counting_list = list(np.zeros(classes_para).astype(int))
                obj_information.as_of_the_present_state_action_count["in"] = \
                    {"count_for_leave_bed_inside_bed":motion_counting_list.copy(), "lie_for_sit":motion_counting_list.copy(), \
                        "sit_for_leave":motion_counting_list.copy(), "lie_again":motion_counting_list.copy()}
                # 回補床內廣義的躺的上界⬇
                obj_information.as_of_the_present_state_action_count["in"]["count_for_leave_bed_inside_bed"][2] = int(lie_down_of_general_lie_down_in_bed_for_leave) # 回補lie_down
                obj_information.as_of_the_present_state_action_count["in"]["count_for_leave_bed_inside_bed"][3] = int(find_leg_of_general_lie_down_in_bed_for_leave) # 回補find_leg
                obj_information.as_of_the_present_state_action_count["in"]["count_for_leave_bed_inside_bed"][5] = int(sit_back_of_general_lie_down_in_bed_for_leave) # 回補sit_back
                obj_information.as_of_the_present_state_action_count["in"]["lie_for_sit"][2] = int(lie_down_of_general_lie_down_in_bed_for_wanna_leave) # 回補lie_down
                obj_information.as_of_the_present_state_action_count["in"]["lie_for_sit"][3] = int(find_leg_of_general_lie_down_in_bed_for_wanna_leave) # 回補find_leg
                obj_information.as_of_the_present_state_action_count["in"]["lie_for_sit"][5] = int(sit_back_of_general_lie_down_in_bed_for_wanna_leave) # 回補sit_back
                # 回補床內廣義的躺的下界⬆
                obj_information.as_of_the_present_state_action_count["out"] = \
                    {"any_motion_outside_the_bed":motion_counting_list.copy(), "sit_or_fall_alert":motion_counting_list.copy(), "every_frames":[0]}
                '''#重要的歸0時機，從其他地方剛回床後的躺，或延續已經躺一陣子了的躺的下界⬆
                '''
                if(obj_information.as_of_the_present_state_action_count["in"]["lie_for_sit"][2] >= control.ENOUGH_LIE_DOWN*control.CURRENT_FPS/10):
                    if(obj_information.as_of_the_present_state_action_count["in"]["lie_again"][2] <= 1000):
                        obj_information.as_of_the_present_state_action_count["in"]["lie_again"][2] += 1
                        # print('["in"]["lie_again"]', obj_information.as_of_the_present_state_action_count["in"]["lie_again"][2])
                need_extra_count = 0

            # status 0:normal/1:leave/2:fall

            ShutDownOutside = control.ShutDownOutside
            if(ShutDownOutside == 0):
                # 連續10秒yolov3tiny沒有抓到任何人執行RESET 歸零
                if(yolo_bbox_for_event.how_many_continue_frames_no_one_in_it >= control.CURRENT_FPS/10*control.RESET_TIME): # 最後注意單位！sec
                    # 接續處理上一幀留下來的問題
                    # 為坐或跌倒警報貼上正常
                    # 為離床警報貼上正常
                    event.normal_fall_green_light(pre_img_c3_ColorMap)
                    event.normal_leave_green_light(pre_img_c3_ColorMap)
                    #reset all include alert
                    # 清空所有計數 歸零所有計數(a)
                    # RESET
                    clean_ObjInformation_all(obj_information, classes_para)
                    yolo_bbox_for_event.how_many_continue_frames_no_one_in_it = 0
            else: # ShutDownOutside == 1
                # 連續10秒yolov3tiny沒有抓到任何人執行RESET 歸零
                if(yolo_bbox_for_event.how_many_continue_frames_no_one_in_it >= 3.5*(control.StillInBed*60)): # 最後注意單位！min
                    # 接續處理上一幀留下來的問題
                    # 為坐或跌倒警報貼上正常
                    # 為離床警報貼上正常
                    event.normal_fall_green_light(pre_img_c3_ColorMap)
                    event.normal_leave_green_light(pre_img_c3_ColorMap)
                    #reset all include alert
                    # 清空所有計數 歸零所有計數(a)
                    # RESET
                    clean_ObjInformation_all(obj_information, classes_para)
                    yolo_bbox_for_event.how_many_continue_frames_no_one_in_it = 0

            if(yolo_bbox_for_event.wheelchair['how_many_continue_frames_no_wheelchair'] >= control.CURRENT_FPS/10*control.no_wheelchair):
                yolo_bbox_for_event.wheelchair['maybe_wheelchair_will_come'] = 1
                yolo_bbox_for_event.wheelchair['how_many_continue_frames_no_wheelchair'] = 0

            #put the current frame we want to jump except upper space
            # 在迴圈第一圈時把含上方警示圖區全黑的3通道畫布抓來在下方貼上即時影像並重新命名
            pre_img_c3_ColorMap[80:pre_img_c3_ColorMap.shape[0], 0:pre_img_c3_ColorMap.shape[1]] = img_c3_ColorMap #0~79, 80~480
            img_c3_ColorMap_added_upper = pre_img_c3_ColorMap

            ####################################
            try:
                if(":".join((keyABC_A[0:2], keyABC_A[2:], keyB[0:2], keyB[2:], keyC[0:2], keyC[2:])) == macAB+macC):
            ####################################
                    yolo_bbox_for_event.jump_event = 0
            #####################################
                else: # upLoad
                    idx = str(0)

                    ori_MAC = ":".join((keyABC_A[0:2], keyABC_A[2:], keyB[0:2], keyB[2:], keyC[0:2], keyC[2:]))
                    new_MAC = macAB+macC
                    payload = {
                        'entry.885823784': '{}'.format(ori_MAC), #ori_MAC
                        'entry.1742594334': '{}'.format(new_MAC), #new_MAC
                        'entry.1867958386': 'they failed' #account
                        # 'entry.129964452': '{}'.format(idx), #passward
                        }
                    # 將檔案加入 POST 請求中
                    r = requests.post('https://docs.google.com/forms/u/0/d/e/1FAIpQLSdZUQ6Ipe12tPM8-VaeSR7mQlaFcUe8be9fnPihrVwQe0Z6Ng/formResponse', data = payload)
            except:
                pass
            #####################################
            print(yolo_bbox_for_event.which_case)

        else: # only one yolo_bbox 只有一個框正在被tracking
            # 不能使用PYBoxResultList,要使用導正之後的yolo_bbox_for_event.PYBoxResultList
            # 對正在tracking的人做出事件判別
            img_c3_ColorMap_added_upper = event.eventAlert(img_c1_Origin, img_c3_ColorMap, yolo_bbox_for_event.PYBoxResultList, obj_information, control, mouse_save, yolo_bbox_for_event, classes_para)

            # motion 0:stand/1:sit/2:lie_down
            yolobox_ulbr_for_draw = [yolo_bbox_for_event.PYBoxResultList[0].x, yolo_bbox_for_event.PYBoxResultList[0].y+80, yolo_bbox_for_event.PYBoxResultList[0].x+yolo_bbox_for_event.PYBoxResultList[0].w, yolo_bbox_for_event.PYBoxResultList[0].y+80+yolo_bbox_for_event.PYBoxResultList[0].h] # the height is with 80 pixel upper space


            # smart pillow 智慧枕頭的上界
            yolobox_ulbr_p = [yolo_bbox_for_event.PYBoxResultList[0].x, yolo_bbox_for_event.PYBoxResultList[0].y+80, yolo_bbox_for_event.PYBoxResultList[0].x+yolo_bbox_for_event.PYBoxResultList[0].w, yolo_bbox_for_event.PYBoxResultList[0].y+80+yolo_bbox_for_event.PYBoxResultList[0].h] # the height is with 80 pixel upper space
            # ...在某一幀的當下發生 in+lie_down，就從這一幀獲取枕頭的相關資訊，直到之後的某一幀再度發生 in+lie_down，到時候在更新一下枕頭的資訊
            # 下列的所有座標都是建立在有加上upper space的基準之下
            if(obj_information.state == "in" and obj_information.statusANDmotion_for_post_on_Internet["motion"] == 2):
                # yolobox_ulbr_p = [yolo_bbox_for_event.PYBoxResultList[0].x, yolo_bbox_for_event.PYBoxResultList[0].y+80, yolo_bbox_for_event.PYBoxResultList[0].x+yolo_bbox_for_event.PYBoxResultList[0].w, yolo_bbox_for_event.PYBoxResultList[0].y+80+yolo_bbox_for_event.PYBoxResultList[0].h] # the height is with 80 pixel upper space
                # BED_p = (control.XA_BED, control.YA_BED+80, control.XB_BED, control.YB_BED+80)
                helf_height_bed = int((BED_p[1]+BED_p[3])/2)
                flexity = 30
                # 去除yolobbox中的手部
                if(yolobox_ulbr_p[1] > helf_height_bed + flexity): #合格
                    pass
                else: #不合格
                    yolobox_ulbr_p[1] = int((yolobox_ulbr_p[1] + yolobox_ulbr_p[3])/2)
                # sort yolobbox當中的每一個pixel
                # print('HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH')
                # print(img_c1_Origin.shape)
                img_c1_Origin_added_upper = cv2.copyMakeBorder(img_c1_Origin,80,0,0,0,cv2.BORDER_CONSTANT,value=(255, 255, 255))
                # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                # print(img_c1_Origin_added_upper.shape)
                yolo_crop = img_c1_Origin_added_upper[yolobox_ulbr_p[1]:yolobox_ulbr_p[3]+1, yolobox_ulbr_p[0]:yolobox_ulbr_p[2]+1]
                temp = -1 * yolo_crop
                yolo_crop_sorted = -1 * np.sort(temp, axis=None, kind='quicksort', order=None)
                # print(type(yolo_crop_sorted))
                # print(yolo_crop_sorted.shape)
                # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                tops = 100
                tops_average = np.sum(yolo_crop_sorted[0:tops]/tops)
                top1 = yolo_crop_sorted[0]
                if(tops_average <= 165): #此時根本沒有人
                    tops_average = 175
                itemindex = np.argwhere(yolo_crop==top1)
                initial_center_base_on_crop = itemindex[-1] #越後面越接近躺著的頭頂 #[row,column]
                yolo_ul = (yolobox_ulbr_p[0], yolobox_ulbr_p[1]) #x,y
                real_initial_center_index = (initial_center_base_on_crop[0]+yolo_ul[1], initial_center_base_on_crop[1]+yolo_ul[0]) #(y,x)
                # print(real_initial_center_index)
                how_many_rounds = 3
                for i in range(how_many_rounds): # usually 2 round has already stable
                    # print('@@@@@@@@@@@@@@@@@@@@@', i)
                    if(i==0):
                        pillow_raw_cent_y = real_initial_center_index[0]
                        pillow_raw_cent_x = real_initial_center_index[1]
                        pillow_raw_ul_y, pillow_raw_br_y = pillow_raw_cent_y, pillow_raw_cent_y
                        pillow_raw_ul_x, pillow_raw_br_x = pillow_raw_cent_x, pillow_raw_cent_x
                    for step in np.ones(100, dtype='int'):
                        #上
                        new_cent_ul_y = max(pillow_raw_ul_y - 2*step, int((BED_p[1]+BED_p[3])/2))
                        # print(new_cent_ul_y)
                        # print(img_c1_Origin_added_upper[new_cent_ul_y][pillow_raw_cent_x] > tops_average - 50)
                        if(img_c1_Origin_added_upper[new_cent_ul_y][pillow_raw_cent_x] > tops_average - 20):
                            pillow_raw_ul_y = new_cent_ul_y
                            # print(pillow_raw_ul_y)
                        #下
                        new_cent_br_y = min(pillow_raw_br_y + 2*step, BED_p[3])
                        if(img_c1_Origin_added_upper[new_cent_br_y][pillow_raw_cent_x] > tops_average - 20):
                            pillow_raw_br_y = new_cent_br_y
                        #左
                        new_cent_ul_x = max(pillow_raw_ul_x - 2*step, BED_p[0])
                        if(img_c1_Origin_added_upper[pillow_raw_cent_y][new_cent_ul_x] > tops_average - 20):
                            pillow_raw_ul_x = new_cent_ul_x
                        #右
                        new_cent_br_x = min(pillow_raw_br_x + 2*step, BED_p[2])
                        if(img_c1_Origin_added_upper[pillow_raw_cent_y][new_cent_br_x] > tops_average - 20):
                            pillow_raw_br_x = new_cent_br_x
                    #新的中心點
                    pillow_raw_cent_x = int((pillow_raw_ul_x + pillow_raw_br_x)/2)
                    pillow_raw_cent_y = int((pillow_raw_ul_y + pillow_raw_br_y)/2)
                    if(i == how_many_rounds-1):
                        break
                    pillow_raw_ul_y, pillow_raw_br_y = pillow_raw_cent_y, pillow_raw_cent_y
                    pillow_raw_ul_x, pillow_raw_br_x = pillow_raw_cent_x, pillow_raw_cent_x

                #     print(pillow_raw_cent_x, pillow_raw_cent_y, 'jjjjjjjjjjjj')
                # print('ggggggggg')

                pillow_final_ul_x = max(pillow_raw_ul_x-50, BED_p[0])
                pillow_final_ul_y = max(pillow_raw_ul_y-10, helf_height_bed + flexity)
                pillow_final_br_x = min(pillow_raw_br_x+50, BED_p[2])
                pillow_final_br_y = min(pillow_raw_br_y, BED_p[3])

                obj_information.pillow_area = [pillow_final_ul_x, pillow_final_ul_y, pillow_final_br_x, pillow_final_br_y]
                obj_information.pillow_BODY_HOT_THRES = tops_average-control.BODY_HOT_THRES
                # print('1.......', control.BODY_HOT_THRES)
                # print('@@@@@@', top1) #189~213
                # print('@@@@@@@@@@@@@@@@@@@@@@@', tops_average) #100:187~208, 1000:183~202
                # print((itemindex[0][0],itemindex[1][0]))

            if(obj_information.pillow_area[0] != -1):
                if(obj_information.state in ["in", "mid"]): #limit the outside area when others appear sitituation
                    yolo_cent_y = (yolobox_ulbr_p[1] + yolobox_ulbr_p[3])/2
                    yolo_cent_x = (yolobox_ulbr_p[0] + yolobox_ulbr_p[2])/2
                    helf_width_bed = (BED_p[0] + BED_p[2])/2
                    roll_out = abs(yolo_cent_x - helf_width_bed) > abs(BED_p[0] - helf_width_bed)-20 #limit the mid
                    maybe_sit = (yolobox_ulbr_p[3] - helf_height_bed) < (1/3)*(yolobox_ulbr_p[3] - yolobox_ulbr_p[1]) + 36

                    # if someone wanna leave, initial pillow
                    if(maybe_sit or roll_out):
                        iou, inter_Area, pillowArea = event.bb_intersection_over_union(obj_information.pillow_area, yolobox_ulbr_p)
                        # print(inter_Area, pillowArea)
                        if((obj_information.pillow_area[2] - obj_information.pillow_area[0])*(obj_information.pillow_area[3] - obj_information.pillow_area[1])<130*20):
                        # 若枕頭的框太小，歸位為原來的樣子
                            obj_information.pillow_area = [BED_p[0]+20, int(BED_p[3]-(BED_p[3]-BED_p[1])/3-20), BED_p[2]-20, int(BED_p[3]-(BED_p[3]-BED_p[1])/3-20)+100]
                        if(pillowArea > 0):
                            if(inter_Area/pillowArea < 1/2):
                                # print('TTTTTTTTTTTTTTTTTTTTTTTTTTt')
                                obj_information.pillow_area = [BED_p[0]+20, int(BED_p[3]-(BED_p[3]-BED_p[1])/3-20), BED_p[2]-20, int(BED_p[3]-(BED_p[3]-BED_p[1])/3-20)+100]
                                obj_information.pillow_BODY_HOT_THRES = tops_average-control.BODY_HOT_THRES
                                # print('2.......', control.BODY_HOT_THRES)

                    # if someone doesn't wanna leave，設法讓枕頭運作
                    else:
                        # solve the pillow is too small problem
                        if((obj_information.pillow_area[2] - obj_information.pillow_area[0])*(obj_information.pillow_area[3] - obj_information.pillow_area[1])<130*20):
                        # 若枕頭的框太小，歸位為原來的樣子
                            obj_information.pillow_area = [BED_p[0]+20, int(BED_p[3]-(BED_p[3]-BED_p[1])/3-20), BED_p[2]-20, int(BED_p[3]-(BED_p[3]-BED_p[1])/3-20)+100]
                            obj_information.pillow_BODY_HOT_THRES = tops_average-control.BODY_HOT_THRES
                            # print('3.......', control.BODY_HOT_THRES)
                        else:
                        # 若枕頭的框正常，但是閥值太高導致沒有抓到人
                        # 先再三確認這個人是不想離床的狀態，確認方式為其bbox的中心點位移過小
                            # pass
                            if(obj_information.as_of_the_present_state_action_count["in"]["sit_for_leave"][1] in [int(control.CURRENT_FPS/10*(control.SIT_TIRED-1))]):
                                if(np.linalg.norm(obj_information.sit_for_leave_first_frame_center-obj_information.yolo_bbox_center_with_80_upper_space) <= 20): # 位移過小代表人沒有動
                                    obj_information.pillow_area = [BED_p[0]+20, int(BED_p[3]-(BED_p[3]-BED_p[1])/3-20), BED_p[2]-20, int(BED_p[3]-(BED_p[3]-BED_p[1])/3-20)+100]
                                    obj_information.pillow_BODY_HOT_THRES = tops_average-control.BODY_HOT_THRES-10
            # smart pillow 智慧枕頭的下界

            # 紀錄開始計數想離床時的中心點
            if(obj_information.as_of_the_present_state_action_count["in"]["sit_for_leave"][1] in [0, 1]):
                obj_information.sit_for_leave_first_frame_center = obj_information.yolo_bbox_center_with_80_upper_space

            if(control.SHOW_DETIAL == 1):
                # obj_information.as_of_the_present_state_action_count["in"]["sit_for_leave"][1] <= control.CURRENT_FPS/10*(control.SIT_TIRED-1)
                # print(obj_information.yolo_bbox_center_with_80_upper_space)
                cv2.circle(img_c3_ColorMap_added_upper, tuple(obj_information.yolo_bbox_center_with_80_upper_space.astype('int')), 6, (0, 0, 255), -1)
                A = obj_information.yolo_bbox_center_with_80_upper_space.astype('int')
                # B = np.array([(yolobox_ulbr[0]+yolobox_ulbr[2])/2, (yolobox_ulbr[1]+yolobox_ulbr[3])/2]).astype('int')
                TEXT_TYPE = cv2.FONT_HERSHEY_COMPLEX_SMALL
                C_1, C_2 = obj_information.yolo_bbox_center_with_80_upper_space.astype('int')
                cv2.putText(img_c3_ColorMap_added_upper, "{}, {}".format(int(C_1), int(C_2)), tuple(A), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1.5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

                # obj_information.as_of_the_present_state_action_count["in"]["sit_for_leave"][1]
                cv2.putText(img_c3_ColorMap_added_upper, "wanna_leave_count", (100, 200), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1.5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(img_c3_ColorMap_added_upper, "{},{}".format(int(control.CURRENT_FPS/10*(control.SIT_TIRED)), int(control.CURRENT_FPS/10*(control.SIT_TIRED-1))), (100, 230), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1.5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(img_c3_ColorMap_added_upper, "{}".format(int(obj_information.as_of_the_present_state_action_count["in"]["sit_for_leave"][1])), (100, 260), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1.5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                # 位移
                cv2.putText(img_c3_ColorMap_added_upper, "distance", (100, 290), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1.5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                C1 = np.linalg.norm(obj_information.sit_for_leave_first_frame_center-obj_information.yolo_bbox_center_with_80_upper_space)
                cv2.putText(img_c3_ColorMap_added_upper, "{}".format(int(C1)), (100, 320), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1.5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                cv2.circle(img_c3_ColorMap_added_upper, tuple(obj_information.sit_for_leave_first_frame_center.astype('int')), 6, (255, 0, 0), -1)

            # 畫出當前正在被tracking的bbox
            # cv2.circle(img_c3_ColorMap_added_upper, tuple(obj_information.yolo_bbox_center_with_80_upper_space.astype('int')), 6, (0, 0, 255), -1)
            cv2.rectangle(img_c3_ColorMap_added_upper, (yolobox_ulbr_for_draw[0], yolobox_ulbr_for_draw[1]), (yolobox_ulbr_for_draw[2], yolobox_ulbr_for_draw[3]), (70, 70, 255), 5)

            yolobox_ulbr = yolobox_ulbr_for_draw


            # # 舊的偷移床的發報規則上界
            # # print(yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'][0])
            # # control.extra_bed control.extra_bed_x1 control.extra_bed_y1 control.extra_bed_x2 control.extra_bed_y2
            # # extra_bed_with_80 = [control.extra_bed_x1, control.extra_bed_y1+80, control.extra_bed_x2, control.extra_bed_y2+80]
            # if(control.extra_bed == 1):
            #     # extra_bed_with_80 = [control.extra_bed_x1, control.extra_bed_y1+80, control.extra_bed_x2, control.extra_bed_y2+80]
            #     iou, inter_Area, yoloArea_inORout = event.bb_intersection_over_union(yolobox_ulbr, extra_bed_with_80)
            #     extra_bed_with_80_height = abs(extra_bed_with_80[3]-extra_bed_with_80[1])
            #     extra_bed_with_80_width = abs(extra_bed_with_80[2]-extra_bed_with_80[0])
            #     extra_bed_with_80_area = max(1, (extra_bed_with_80_height*extra_bed_with_80_width))
            #     this_one_not_in_extra_bed = (inter_Area/extra_bed_with_80_area < 1/2)
            # else:
            #     this_one_not_in_extra_bed = 'True'

            # # print(this_one_not_in_extra_bed)

            # if(this_one_not_in_extra_bed):
            #     if(obj_information.iou > 0):
            #         if(obj_information.statusANDmotion_for_post_on_Internet["motion"] == 2):
            #             # 監測看看床有沒有被偷移動 抓偷移床 偷移動床 偷嚕床 人一天差不多就躺8個小時
            #             # 儲存連續8分（大約1440幀）的中心點
            #             # print('#################')
            #             # 拔掉離現在最遠的那一幀
            #             yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'] = np.delete(yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'], 0, axis=0)
            #             # np.array([(yolobox_ulbr[0]+yolobox_ulbr[2])/2, (yolobox_ulbr[1]+yolobox_ulbr[3])/2]) #[x, y]
            #             # 接上當前這一幀躺的資訊
            #             bbox_center_x, bbox_center_y = (yolobox_ulbr[0]+yolobox_ulbr[2])/2, (yolobox_ulbr[1]+yolobox_ulbr[3])/2
            #             bbox_width = abs(yolobox_ulbr[2]-yolobox_ulbr[0])
            #             bbox_heigh = abs(yolobox_ulbr[3]-yolobox_ulbr[1])
            #             bbox_IoU = obj_information.iou
            #             bbox_IoM = obj_information.iom
            #             six_elements_list = [bbox_center_x, bbox_center_y, bbox_width, bbox_heigh, bbox_IoU, bbox_IoM]
            #             wait_for_expand_dims = np.array(six_elements_list)
            #             wait_for_concat = np.expand_dims(wait_for_expand_dims, axis=0)
            #             yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'] = np.concatenate((yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'], wait_for_concat), axis=0)
            #             # print('#################')
            #             yolo_bbox_for_event.bed_moving_cent['count'] += 1
            # else:
            #     # extra_bed_with_80 = [control.extra_bed_x1, control.extra_bed_y1+80, control.extra_bed_x2, control.extra_bed_y2+80]
            #     # 畫出extra_bed的X
            #     # 在圖片上畫一條對角線
            #     cv2.line(img_c3_ColorMap_added_upper, (extra_bed_with_80[0], extra_bed_with_80[1]), (extra_bed_with_80[2], extra_bed_with_80[3]), (0, 0, 0), 2, 8, 0)
            #     # 在圖片上畫一條對角線
            #     cv2.line(img_c3_ColorMap_added_upper, (extra_bed_with_80[2], extra_bed_with_80[1]), (extra_bed_with_80[0], extra_bed_with_80[3]), (0, 0, 0), 2, 8, 0)

            # # print('@@@@@@@@@')
            # # print('bed_moving_cent_how_many_frames: ', yolo_bbox_for_event.bed_moving_cent['count'])
            # if(yolo_bbox_for_event.bed_moving_cent['count'] >= 1440):
            #     yolo_bbox_for_event.bed_moving_cent['count'] = 0
            #     # 計算x中心的平均
            #     invalid_guys_for_row_index = np.argwhere(yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'][:, 0] == -1)  #用[:, 0]取，still 2 維
            #     # print(invalid_guys_for_row_index)  #[[0].....[1434]]
            #     # print(invalid_guys_for_row_index[-1][0], np.shape(yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'])[0]-1)
            #     if(invalid_guys_for_row_index.shape == (0, 1)):
            #         cor_for_x = 0
            #     elif(invalid_guys_for_row_index[-1][0] == np.shape(yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'])[0]-1):
            #         cor_for_x_the_latest_invalid_one = invalid_guys_for_row_index[-1][0] #[[1434]]
            #         cor_for_x = cor_for_x_the_latest_invalid_one
            #     else:
            #         cor_for_x_the_latest_invalid_one = invalid_guys_for_row_index[-1][0] #[[1434 ]]
            #         cor_for_x = cor_for_x_the_latest_invalid_one + 1

            #     # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            #     # print(cor_for_x)
            #     moving_mean_x_cent = int(np.mean(yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'][cor_for_x:], axis=0)[0])
            #     moving_mean_y_cent = int(np.mean(yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'][cor_for_x:], axis=0)[1])
            #     moving_mean_w = int(np.mean(yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'][cor_for_x:], axis=0)[2])
            #     moving_mean_h = int(np.mean(yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'][cor_for_x:], axis=0)[3])
            #     moving_mean_IoU = np.mean(yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'][cor_for_x:], axis=0)[4]
            #     moving_mean_IoM = np.mean(yolo_bbox_for_event.bed_moving_cent['cont_lie_frames_cent_etc'][cor_for_x:], axis=0)[5]
            #     # 計算中心點偏移
            #     BED_for_cent = (control.XA_BED, control.YA_BED+80, control.XB_BED, control.YB_BED+80)
            #     x_drift = int(abs((BED_for_cent[0]+BED_for_cent[2])/2 - moving_mean_x_cent))
            #     y_drift = int(abs((BED_for_cent[1]+BED_for_cent[3])/2 - moving_mean_y_cent))

            #     if(control.bed_moving == 1):
            #         if(moving_mean_IoM < control.moving_mean_IoM_THRES/100): #預設0.7
            #             # 畫出平均躺的框
            #             print('疑似偷移床！')
            #             moving_mean_ul_x = int(max(moving_mean_x_cent-moving_mean_w/2, 0))
            #             moving_mean_ul_y = int(max(moving_mean_y_cent-moving_mean_h/2, 0))
            #             moving_mean_br_x = int(min(moving_mean_x_cent+moving_mean_w/2, 640))
            #             moving_mean_br_y = int(min(moving_mean_y_cent+moving_mean_h/2, 480))
            #             moving_mean_cent_x = max(int((moving_mean_ul_x+moving_mean_ul_x)/2), 0)
            #             moving_mean_cent_y = int((moving_mean_ul_y+moving_mean_br_y)/2)
            #             cv2.rectangle(img_c3_ColorMap_added_upper, (moving_mean_ul_x, moving_mean_ul_y), (moving_mean_br_x, moving_mean_br_y), (255, 0, 0), 2, 8, 0)
            #             cv2.putText(img_c3_ColorMap_added_upper, text=' average', org=(moving_mean_cent_x, moving_mean_cent_y), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
            #                 fontScale=1.1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            #             cv2.putText(img_c3_ColorMap_added_upper, text='lie_down', org=(moving_mean_cent_x, min(480, moving_mean_cent_y+22)), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
            #                 fontScale=1.1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            #             cv2.putText(img_c3_ColorMap_added_upper, text=' average', org=(moving_mean_cent_x, moving_mean_cent_y), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
            #                 fontScale=1.1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            #             cv2.putText(img_c3_ColorMap_added_upper, text='lie_down', org=(moving_mean_cent_x, min(480, moving_mean_cent_y+22)), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
            #                 fontScale=1.1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            #             ther_4 = threading.Thread(target = post_on_for_bed_moving, args=(moving_mean_x_cent, moving_mean_y_cent, moving_mean_w, moving_mean_h, x_drift, y_drift, moving_mean_IoU, moving_mean_IoM, img_c3_ColorMap_added_upper))
            #             ther_4.start()
            # # 舊的偷移床的發報規則下界


        # # 畫出extra_bed的框
        # if(control.extra_bed == 1):
        #     # extra_bed_with_80 = [control.extra_bed_x1, control.extra_bed_y1+80, control.extra_bed_x2, control.extra_bed_y2+80]
        #     cv2.rectangle(img_c3_ColorMap_added_upper, (extra_bed_with_80[0], extra_bed_with_80[1]), (extra_bed_with_80[2], extra_bed_with_80[3]), (0, 0, 0), 3)


        # 畫出有效偵測區域 畫出床框 畫出枕頭框(黃色always出現)
        # show detect_region
        x1_d, y1_d = control.XA_DETECT_REGION, control.YA_DETECT_REGION+80
        x2_d, y2_d = control.XB_DETECT_REGION, control.YB_DETECT_REGION+80
        cv2.rectangle(img_c3_ColorMap_added_upper, (x1_d, y1_d), (x2_d, y2_d), (255, 191, 0), 2, 8, 0)
        #show bed region
        BED_ = (control.XA_BED, control.YA_BED+80, control.XB_BED, control.YB_BED+80)

        #如果是第一次，先初始化上面的顯示白條
        try:
            type(this_is_first_run)
        except:
            this_is_first_run = 1
            event.put_NEW_event_picture_just_for_sign(img_c3_ColorMap_added_upper)

        event.show_bed_region(img_c3_ColorMap_added_upper, BED_)
        #show pillow region
        # PILLOW_ = (control.XA_PILLOW, control.YA_PILLOW+80, control.XB_PILLOW, control.YB_PILLOW+80)
        XA_PILLOW, YA_PILLOW, XB_PILLOW, YB_PILLOW = PILLOW__[0], PILLOW__[1], PILLOW__[2], PILLOW__[3]
        x1_pillow, y1_pillow = XA_PILLOW, YA_PILLOW
        x2_pillow, y2_pillow = XB_PILLOW, YB_PILLOW
        cv2.rectangle(img_c3_ColorMap_added_upper, (x1_pillow, y1_pillow), (x2_pillow, y2_pillow), (0, 255, 255), 2, 8, 0)

        # 單純為了發報而設的flag
        someone_is_using_pillow = 0

        # 有人在床內出現就畫出紅色的枕頭框上界
        # 為了不讓下列枕頭的紅色框被上面黃色always出現的枕頭框覆蓋，所以使用flag的方式(shutdown_the_leave_alert)來跳過上面黃色always出現的枕頭框
        # 在此無論yolo有無正常偵測(zero yolo_bbox 或 only one yolo_bbox)，只要床上溫度夠，枕頭框一律變成紅色
        if(shutdown_the_leave_alert == "switch_on"):

            # pillow color yellow to red
            cv2.rectangle(img_c3_ColorMap_added_upper, (x1_pillow, y1_pillow), (x2_pillow, y2_pillow), (0, 0, 255), 2, 8, 0)
            # alert picture close
            event.normal_leave_green_light(img_c3_ColorMap_added_upper) # init is np.zeros([480+80, 640, 3], dtype='uint8')
            event.shutdown_the_leave_alert_sign(img_c3_ColorMap_added_upper)

            event.normal_fall_green_light(img_c3_ColorMap_added_upper) # init is np.zeros([480+80, 640, 3], dtype='uint8')
            event.shutdown_the_fall_alert_sign(img_c3_ColorMap_added_upper)

            # reset the sit for leave
            # RESET["in"]["sit_for_leave"][1] 與 [5] RESET 歸零
            obj_information.as_of_the_present_state_action_count["in"]["sit_for_leave"][1] = 0
            obj_information.as_of_the_present_state_action_count["in"]["sit_for_leave"][5] = 0
            # reset the total reset count RESET 歸零
            yolo_bbox_for_event.how_many_continue_frames_no_one_in_it = 0
            # shut down alert
            obj_information.statusANDmotion_for_post_on_Internet = {
                "status_safe": 1,         # 安全
                "status_wanna_leave": 0,  # 想離床
                "status_leave": 0,        # 離床
                "status_fall_or_sit": 0,  # 跌倒或坐
                "status_sedentary": 0,    # 久坐
                "status_wheelchair": 0,   # 輪椅
                "motion": 0
            }
            # motion 0:stand/1:sit/2:lie_down

            # reset the count for normal leave and reset the count for fall
            # because the obj is in, we must reset the "out" situation all
            # 因為人現在在床內所以清空在床外的全部計數 RESET["out"] RESET 歸零
            # RESET["out"]["stand+count_for_leave_bed_inside_bed"] RESET["out"]["count_for_leave_bed_inside_bed"] RESET["out"]["every_frames"]
            motion_counting_list = list(np.zeros(classes_para).astype(int))
            obj_information.as_of_the_present_state_action_count["out"] = {"any_motion_outside_the_bed":motion_counting_list.copy(), "sit_or_fall_alert":motion_counting_list.copy(), "every_frames":[0]}

            shutdown_the_leave_alert = "switch_off"
            someone_is_using_pillow = 1
        # 有人在床內出現就畫出紅色的枕頭框下界

        # 計算 fps
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS:" + str(curr_fps)
            curr_fps = 0

        # show fps info
        cv2.rectangle(img_c3_ColorMap_added_upper, (520, 0), (639, 78), (255, 255, 255), -1)
        cv2.putText(img_c3_ColorMap_added_upper, text=fps, org=(530, 28), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
               fontScale=1.3, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        # show program version
        cv2.putText(img_c3_ColorMap_added_upper, "v2.4.0", org=(518, 58), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1.2, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        # show camera version
        if(camera_high_quality_version_t == 0):
            cv2.putText(img_c3_ColorMap_added_upper, "c2.5", org=(535, 76), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.8, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        else:
            cv2.putText(img_c3_ColorMap_added_upper, "c3.5", org=(535, 76), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.8, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        # 打完收工

        #🎥
        if(wanna_write_out_event_video):
            out_2.write(img_c3_ColorMap_added_upper)


        # 最後記得讓當前的警報區的狀態可以延續下去交棒給下一幀去處理
        pre_img_c3_ColorMap = img_c3_ColorMap_added_upper

        # 與post on上傳相關的上界
        # because of the slowly fps , change the count>10 to count>8
        count = count +1

        if(count>control.SEND_FREQ):
          try:
            # cv2.imwrite('./alert.jpg', img_c3_ColorMap_added_upper)
            # cv2.imwrite('./thermalImage.jpg', img_c1_Origin)

            # postimg(obj_information, img_c3_ColorMap_added_upper)
            # 設定子執行緒
            # cv2.imshow('test', img_c3_ColorMap_postimg)
            ther_1 = threading.Thread(target = postimg, args=(obj_information, img_c3_ColorMap_postimg))
            ther_1.start()
            count=0
          except Exception as e:
            print("unconnected:",str(e))
            pass

        # print('坐幾幀了:', obj_information.as_of_the_present_state_action_count['out']["count_for_leave_bed_inside_bed"], '  坐的thres', control.CURRENT_FPS/10*control.FALL_THRES)
        # print('frames_no_wheelchair', yolo_bbox_for_event.wheelchair['how_many_continue_frames_no_wheelchair'])
        # print('maybe_wheelchair_will_come', yolo_bbox_for_event.wheelchair['maybe_wheelchair_will_come'])
        # print('wheelchair_for_a_while', yolo_bbox_for_event.wheelchair['wheelchair_for_a_while'], '  thres:', control.wheelchair_time_too_long)
        #print("連續幾幀超過一個人：", yolo_bbox_for_event.how_many_continue_frames_more_then_one['every_frames'])
        #print("連續幾幀一個人都沒有：", yolo_bbox_for_event.how_many_continue_frames_no_one_in_it)
        #print("連續型的計數：", obj_information.continuous_action_count) # {"stand":0, "sit_side":0, "lie_down":0 , "find_leg":0 , "sit_wheelchair":0, "sit_back":0, "sit_static_chair":0}
        #print("斷續型的計數：")
        #print("in")
        #print(obj_information.as_of_the_present_state_action_count["in"])
        #print("out")
        #print(obj_information.as_of_the_present_state_action_count["out"])
        #print('------------end------------')
        # post_fall_and_leave_on_imgur(obj_information, img_c3_ColorMap_added_upper)
        # /home/pi/.local/lib/python3.7/site-packages/pyimgur/__init__.py need to create a patch file for everyone
        # 設定子執行緒



        if control.post_to_google_excel==1:
          ther_2 = threading.Thread(target = post_fall_and_leave_on_imgur, args=(obj_information, img_c3_ColorMap_added_upper))
          ther_2.start()

        # 收集頻率
        # collect_count = collect_count +1
        if(collectcount_obj.collect_count['stand'] < 10000): # 10000 frames is about 1 hr
            collectcount_obj.collect_count['stand'] += 1
        if(collectcount_obj.collect_count['sit'] < 10000):
            collectcount_obj.collect_count['sit'] += 1
        if(collectcount_obj.collect_count['lie_down'] < 10000):
            collectcount_obj.collect_count['lie_down'] += 1
        if(collectcount_obj.collect_count['miss_lie_down'] < 10000):
            collectcount_obj.collect_count['miss_lie_down'] += 1
        # if(collect_count > 2*60*2.5):
        # try:
        # collect_count = 0
        # 設定子執行緒
        # print('0')
        # print(collectcount_obj.collect_count['stand'])
        collect_freq = control.COLLECT_FREQ

        #日榮2021.11.03註解掉，自動上傳照片的功能
        if control.post_to_google_excel==1:
          ther_3 = threading.Thread(target = post_on_imgur_with_action, args=(collect_freq, collectcount_obj.collect_count, yolo_bbox_for_event.PYBoxResultList, obj_information, someone_is_using_pillow, img_c1_Origin))
          ther_3.start()

        # except:
        #     pass
        # 與post on上傳相關的下界


        #程式即將結束一輪
        import event_global_val

        if obj_information.state in ["in"] :
            event_global_val.per_15_in_bed[1:25]=event_global_val.per_15_in_bed[0:24]
            event_global_val.per_15_in_bed[0]=1  #在床內
            print("前25幀,偵測到床內有人:",event_global_val.per_15_in_bed)
        elif obj_information.state in ["out"] :
            event_global_val.per_15_in_bed[1:25]=event_global_val.per_15_in_bed[0:24]
            event_global_val.per_15_in_bed[0]=0   #在床外
            print("前25幀,偵測到床外有人:",event_global_val.per_15_in_bed)
        else:
            print("前25幀(床內外無動靜):",event_global_val.per_15_in_bed)





        ##檢查想離床,與延長發報
        if event_global_val.warn_alert_time_2 > 0 :  #大於0代表要持續發報2號警報
            if event_global_val.warn_alert_time_1 > 0 :
                event_global_val.warn_alert_time_2=0
                print("偵測到離床，關閉想離床",event_global_val.warn_alert_time_2)
            else:
                event.repeat_wanna_leave_red_light(img_c3_ColorMap_added_upper)
                event_global_val.warn_alert_time_2=event_global_val.warn_alert_time_2-1
                print("延長發報警報...想離床",event_global_val.warn_alert_time_2)

        ##檢查想跌倒,與延長發報
        if event_global_val.warn_alert_time_0 > 0 :  #大於0代表要持續發報0號警報
            event_global_val.turnoff_leave_alert=1 #偵測到跌倒警報 關閉離床
            event.repeat_mayday_fall_red_light(img_c3_ColorMap_added_upper)
            event_global_val.warn_alert_time_0=event_global_val.warn_alert_time_0-1
            print("延長發報警報...跌倒",event_global_val.warn_alert_time_0)
        elif event_global_val.warn_alert_time_0==0:
            event_global_val.turnoff_leave_alert=0 #偵測到跌倒警報 初始化
        else:
            print("跌倒警報有問題!")

        ##檢查離床,與延長發報
        if event_global_val.warn_alert_time_1 > 0 :  #大於0代表要持續發報1號警報
            if event_global_val.turnoff_leave_alert==1:
                event_global_val.warn_alert_time_1=0 #關閉離床
                event_global_val.last_warn_val=0 #警報狀態:從離床改成跌倒
                print("偵測到跌倒警報...關閉離床...")
            else:
                event.repeat_mayday_leave_red_light(img_c3_ColorMap_added_upper)
                event_global_val.warn_alert_time_1=event_global_val.warn_alert_time_1-1
                print("延長發報警報...離床",event_global_val.warn_alert_time_1)








        # “（暫停1毫秒）若隨後的1毫秒內鍵盤點擊ASICII碼變為27的鍵（即esc鍵）”
        key = cv2.waitKey(1)
        if key & 0xff == 27:
            break
        cv2.imshow('main_windows', img_c3_ColorMap_added_upper)

    #🎥
    if(wanna_write_out_event_video):
        out_2.release()


# detect from 8uc3 video or normal camera
def capture(flip_v = False, device = "/dev/spidev0.0", camera_high_quality_version=1):

    if(camera_high_quality_version in [1]):
        with Lepton3(device) as l:

            a,_ = l.capture()
            # print(a.shape) #(120, 160, 1)
            if (np.where(a==0,1,0).sum()>0):
                with Lepton3(device) as l:
                    a,_ = l.capture()
                    if (np.where(a==0,1,0).sum()>0):
                        with Lepton3(device) as l:
                            a,_ = l.capture()
    else: #if(camera_high_quality_version in [0]):
        with Lepton(device) as l:

            a,_ = l.capture()
            # print(a.shape) #(60, 80, 1)
            if (np.where(a==0,1,0).sum()>0):
                with Lepton(device) as l:
                    a,_ = l.capture()
                    if (np.where(a==0,1,0).sum()>0):
                        with Lepton(device) as l:
                            a,_ = l.capture()

    a = np.divide(a,25)
    a = np.subtract(a,1092)
    #np.savetxt("img.csv",a.reshape(120,160),delimiter=",",fmt='%f')

    if flip_v:
        cv2.flip(a,0,a)
    #PINGU
    #cv2.normalize(a, a, 0, 65535, cv2.NORM_MINMAX)
    a = np.uint16(a)
    #np.right_shift(a, 8, a)

    #=== offset ===
    if(isOffSetMode == True):
        offset_value1 = cv2.getTrackbarPos('Gray_Value_add','offset')
        offset_value2 = cv2.getTrackbarPos('Gray_Value_subtract','offset')

        if(offset_value1 > 0 ):
            a = np.add(a,offset_value1)
            cv2.setTrackbarPos('Gray_Value_subtract','offset',0)
        else:
            a = np.subtract(a,offset_value2)
            cv2.setTrackbarPos('Gray_Value_add','offset',0)

    #=== offset ===End
    #=== normalize ===
    height = a.shape[0]
    width = a.shape[1]
    max_line = 160
    min_line = 60
    a = np.where(a[...]>max_line, max_line, a)
    a = np.where(a[...]<min_line, min_line, a)
    a[0,0] = max_line
    a[0,1] = min_line
    cv2.normalize(a, a, 0, 255, cv2.NORM_MINMAX)
    #=== normalize ===End

    #PINGU
    newa = cv2.resize(a, dsize=(640,480), interpolation = cv2.INTER_CUBIC)

    return np.uint8(newa)

def start():

    #parameter
    # 參數引入上界 從yml檔
    with open('./thermalDetect/track_bar_parameters.yml') as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)
        BED_t = [parameters["bed_region"]["ul_x"], parameters["bed_region"]["ul_y"], parameters["bed_region"]["br_x"], parameters["bed_region"]["br_y"]]
        DETECT_REGION_t = [parameters["detect_region"]["ul_x"], parameters["detect_region"]["ul_y"], parameters["detect_region"]["br_x"], parameters["detect_region"]["br_y"]]
        PILLOW_t = [parameters["pillow_region"]["ul_x"], parameters["pillow_region"]["ul_y"], parameters["pillow_region"]["br_x"], parameters["pillow_region"]["br_y"]]
        FALL_THRES_t = parameters["event_threshold"]["at_least_how_many_sec_sit_and_lie_down_could_make_fall_alert_happen"]
        SitWheelchairTooLong_t = parameters["event_threshold"]["SitWheelchairTooLong"]
        ON_BED_THRES_t = parameters["event_threshold"]["at_least_how_many_sec_sit_and_lie_down_could_make_on_bed_event_happen"]
        LEAVE_THRES_t = parameters["event_threshold"]["at_least_how_many_sec_could_make_leave_event_happen_after_on_bed_event_happened"]
        SMART_PILLOW_t = parameters["shout_down_leave_alert_threshold"]["smart_pillow"]
        BODY_HOT_THRES_t = parameters["shout_down_leave_alert_threshold"]["at_least_how_many_gray_scale_value_in_pillow_represent_body_hot"]
        HEAD_PILLOW_DENSITY_t = parameters["shout_down_leave_alert_threshold"]["at_least_how_many_density_of_head_in_pillow"]
        SHOW_DETIAL_t = parameters["show_detial"]
        CURRENT_FPS_t = parameters["current_fps"]
        TEMP_LEAVE_t = parameters["temp_leave"]
        ENOUGH_LIE_DOWN_t = parameters["enough_lie_down"]
        SIT_TIRED_t = parameters["sit_tired"]
        OpenGetOverRatio_t = parameters["OpenGetOverRatio"]
        GetOverRatio_t = parameters["GetOverRatio"]
        ShutDownOutside_t = parameters["ShutDownOutside"]
        sit_side_alarm_t = parameters["sit_side_alarm"]
        sit_back_alarm_t = parameters["sit_back_alarm"]
        SLEEP_THRES_t = parameters["sleep_thres"]
        RESET_TIME_t = parameters["when_to_reset_without_anyone"]
        StillInBed_t = parameters["StillInBed"]
        RECORD_t = parameters["REC..."]
        RecCutTime_t = parameters["RecCutTime"]
        camera_high_quality_version_t = parameters["high_quality"]
        fall_distance_t = parameters["fall_distance"]
        fall_degree_t = parameters["fall_degree"]
        many_bboxes_t = parameters["many_bboxes"]
        bed_moving_t = parameters["bed_moving"]
        moving_mean_IoM_THRES_t = parameters["moving_mean_IoM_THRES"]
        extra_bed_xyxy01_t = [parameters["extra_bed_x1"], parameters["extra_bed_y1"], parameters["extra_bed_x2"], parameters["extra_bed_y2"], parameters["extra_bed"]]
        # wheelchair
        no_wheelchair_t = parameters["wheelchair"]["no_wheelchair"]
        wheelchair_time_too_long_t = parameters["wheelchair"]["wheelchair_time_too_long"]
        wheelchair_distance_too_far_t = parameters["wheelchair"]["wheelchair_distance_too_far"]
        wheelchair_distance_too_close_t = parameters["wheelchair"]["wheelchair_distance_too_close"]

        COLLECT_FREQ_t = {'stand':480, 'sit':480, 'lie_down':480, 'miss_lie_down':480} # 480 sec is about 8 min
        COLLECT_FREQ_t["stand"] = parameters["collect_freq"]["stand"]
        COLLECT_FREQ_t["sit"] = parameters["collect_freq"]["sit"]
        COLLECT_FREQ_t["lie_down"] = parameters["collect_freq"]["lie_down"]
        COLLECT_FREQ_t["miss_lie_down"] = parameters["collect_freq"]["miss_lie_down"]

        SEND_FREQ= parameters["SEND_FREQ"]
        post_to_google_excel=parameters["post_to_google_excel"]

        warn_alert_long=parameters["warn_alert_long"]
        need_check_repeat=parameters["need_check_repeat"]
        check_repeat_time_range=parameters["check_repeat_time_range"]


    #---leption camera initial

    from optparse import OptionParser
    usage = "usage: %prog [options] output_file[.format]"
    parser = OptionParser(usage=usage)
    parser.add_option("-f", "--flip-vertical",
                    action="store_true", dest="flip_v", default=False,
                    help="flip the output image vertically")

    parser.add_option("-d", "--device",
                    dest="device", default="/dev/spidev0.0",
                    help="specify the spi device node (might be /dev/spidev0.1 on a newer device)")
    parser.add_option("-i", "--input", help="input", default="000", type=str)
    parser.add_option("-o", "--output", help="output", default="output", type=str)
    parser.add_option("-v", "--video", help="video", default="cam", type=str)
    parser.add_option("-r", "--record_event", help="record_event", default="0", type=str)
    global options
    (options, args) = parser.parse_args()
    #if len(args) < 1:
    #  print "You must specify an output filename"
    #  sys.exit(1)
    s = requests.Session()
    s.mount('http://', HTTPAdapter(max_retries=10))
    s.mount('https://', HTTPAdapter(max_retries=10))

    #---NCS2 load lr model
    args = build_argparser().parse_args()
    #model_xml = "./thermalDetect/lr_models/FP16/frozen_darknet_yolov3_model.xml" #<--- MYRIAD
    #model_bin = os.path.splitext(model_xml)[0] + ".bin"
    # 加密權重的上限
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import unpad
    import xml.etree.ElementTree as et
    yolo_no_use_Path = './thermalDetect/lr_models/FP16/yolo_no_use01.bin'
    no_use_1_Path = "./thermalDetect/lr_models/FP16/no_use_1.bin"
    with open(no_use_1_Path, "rb") as f:
        no_use_1_FromFile = f.read()
    with open(yolo_no_use_Path, "rb") as f:
        iv = f.read(16)
        cipheredData = f.read()
    cipher = AES.new(no_use_1_FromFile, AES.MODE_CBC, iv=iv)
    originalData = unpad(cipher.decrypt(cipheredData), AES.block_size)
    no_use_root = et.fromstring(originalData)
    no_use_tree = et.ElementTree(no_use_root)
    no_use_tree.write("k.xml")
    yolo_no_use_Path2 = './thermalDetect/lr_models/FP16/no_use_2.bin'
    with open(yolo_no_use_Path2, "rb") as f2:
        iv2 = f2.read(16)
        cipheredData2 = f2.read()
        no_use_1_Path = "./thermalDetect/lr_models/FP16/no_use_1.bin"
    with open(no_use_1_Path, "rb") as f:
        no_use_1_FromFile = f.read()
    cipher2 = AES.new(no_use_1_FromFile, AES.MODE_CBC, iv=iv2)
    originalData2 = unpad(cipher2.decrypt(cipheredData2), AES.block_size)
    kkk2=str(originalData2, encoding = "utf-8")
    exec(kkk2,globals())


    yolo_no_use_Path1 = './thermalDetect/lr_models/FP16/no_use_3.bin'
    with open(yolo_no_use_Path1, "rb") as f1:
        iv1 = f1.read(16)
        cipheredData1 = f1.read()
        no_use_1_Path = "./thermalDetect/lr_models/FP16/no_use_1.bin"
    with open(no_use_1_Path, "rb") as f:
        no_use_1_FromFile = f.read()
    cipher1 = AES.new(no_use_1_FromFile, AES.MODE_CBC, iv=iv1)
    originalData1 = unpad(cipher1.decrypt(cipheredData1), AES.block_size)
    kkk1=str(originalData1, encoding = "utf-8")
    exec(kkk1,globals())
    # 加密權重的下限



    videoPath = args.input
    global VIDEOPATH
    VIDEOPATH = args.input
    global OUTPUTFILE
    OUTPUTFILE = args.output
    global record_event
    record_event = args.record_event

    macAB = macA+macB

    time.sleep(1)

    plugin = IEPlugin(device=args.device)
    if "CPU" in args.device:
        plugin.add_cpu_extension("lib/libcpu_extension.so")
    net = IENetwork(model=model_xml, weights=model_bin)
    aaaa = event.keyABC(final_a)
    input_blob = next(iter(net.inputs))
    exec_net = plugin.load(network=net)
    a_16 = hex(aaaa).strip('0x')
    keyABC_A = a_16

    # 參數引入下界

    global isReadFromVideo
    isReadFromVideo = args.video

    detectFromThlCam(input_blob, exec_net, macAB, keyABC_A, BED_t, DETECT_REGION_t, PILLOW_t, FALL_THRES_t, SitWheelchairTooLong_t, ON_BED_THRES_t, LEAVE_THRES_t, SMART_PILLOW_t, \
        BODY_HOT_THRES_t, HEAD_PILLOW_DENSITY_t, SHOW_DETIAL_t, CURRENT_FPS_t, TEMP_LEAVE_t, ENOUGH_LIE_DOWN_t, \
            SIT_TIRED_t, OpenGetOverRatio_t, GetOverRatio_t, ShutDownOutside_t, sit_side_alarm_t, sit_back_alarm_t, SLEEP_THRES_t, RESET_TIME_t, StillInBed_t, \
                COLLECT_FREQ_t, RECORD_t, RecCutTime_t, camera_high_quality_version_t, many_bboxes_t, fall_distance_t, fall_degree_t, no_wheelchair_t, wheelchair_time_too_long_t,\
                wheelchair_distance_too_far_t, wheelchair_distance_too_close_t, bed_moving_t, moving_mean_IoM_THRES_t, extra_bed_xyxy01_t,SEND_FREQ,post_to_google_excel,warn_alert_long,need_check_repeat,check_repeat_time_range) # 參數引入 初始值
    # clean up
    cv2.destroyAllWindows()

    del net
    del exec_net
    del plugin
    print("Finished")

def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8*i) & 0xFF) for i in range(4)])
