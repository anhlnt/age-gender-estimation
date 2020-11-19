import requests
import json
from datetime import datetime
import time
import numpy as np
import os
import copy

def get_devicekey():
    with open('DEVICE_KEY') as f:
        return f.read()

# POST_URL = 'https://project.vtmlab.com/facerecognition/api/ai_analysis'
POST_URL = 'http://localhost/api/ai_analysis'
DEVICE_KEY = get_devicekey()

def clear_key(data):
    result = []
    for value in data.values():
        try:
            value.pop("ages")
        except:
            pass
        value["device_key"] = DEVICE_KEY
        result.append(value)
    return result


def get_data(folder, start_line=1, interval=5):
    if not os.path.exists(folder):
        os.mkdir(folder)
    i = 1
    data = {}
    sleep_time = 1
    start_line_tmp = start_line

    while True:
        day = datetime.fromtimestamp(time.time()).strftime("%Y%m%d")
        data_file = folder + day + '.log'
        with open(data_file, "r") as f:
            print("Read file: ", data_file)
            start = time.time()
            while True:
                current = time.time()
                day1 = datetime.fromtimestamp(current).strftime("%Y%m%d")
                second = datetime.fromtimestamp(current).strftime("%H%M%S")
                if int(day1) > int(day) and int(second) > 3 * interval:
                    i = 1
                    start_line_tmp = 1
                    break
                print("Reading line ", i)
                if time.time() - start > interval:
                    data_tmp = copy.deepcopy(data)
                    post_data = clear_key(data)
                    if len(post_data) > 0:
                        res = post(POST_URL, post_data)
                        if res["status"]:
                            post_log(folder + "post.log", data_file, start_line_tmp, i - 1)
                            start_line_tmp = i
                            data = {}
                        else:
                            time.sleep(sleep_time)
                            start = time.time()
                            if len(data) > 100:
                                data = {}
                            else:
                                data = data_tmp
                            continue
                    start = time.time()
                if i >= start_line:
                    line = f.readline().strip()
                    # print(line)
                    if line == "":
                        print("End file")
                        time.sleep(sleep_time)
                        continue
                    else:
                        row_data = json.loads(line)
                        for user in row_data:
                            if not str(user["user_id"]) in data:
                                data[str(user["user_id"])] = user
                                data[str(user["user_id"])]["ages"] = [user["user_age"]]
                            else:
                                data[str(user["user_id"])]["ages"].append(user["user_age"])
                                data[str(user["user_id"])]["user_age"] = np.mean(data[str(user["user_id"])]["ages"])
                                data[str(user["user_id"])]["user_gender"] = user["user_gender"]
                                data[str(user["user_id"])]["end_time"] = user["end_time"]
                        # data = json.loads(line)
                else:
                    if f.readline().strip() == "":
                        print("End file")
                        time.sleep(sleep_time)
                        continue
                i += 1


def post(url, data):
    print('Start post .......')
    print(json.dumps(data, indent=2))
    try:
        response = requests.post(
            url,
            json.dumps(data),
            headers={'Content-Type': 'application/json'})
        print(json.dumps(response.json(), indent=2))
        return response.json()
    except:
        print("Error")
    return {"status": False}

def post_log(logfile, read_file, start_line, end_line):
    if not os.path.exists(logfile):
        with open(logfile, "w") as f:
            f.write("Time File StartLine EndLine\n")
    with open(logfile, "a") as f:
        f.write("{} {} {} {}\n".format(datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S"), read_file, start_line, end_line))



def main():
    url = 'https://project.vtmlab.com/facerecognition/api/ai_analysis'
    data = [{
        "device_key": "KkWND17U-yc2EOeRl_lAytmY4ydrTPYfQa231eozP0k",
        "user_id": 161,
        "user_age": 66,
        "user_gender": 0,
        "start_time": 1603170791,
        "end_time": 1603170810
    }]
    res = post(url, data)
    print(json.dumps(res.json(), indent=2))

if __name__ == "__main__":
    # main()
    get_data('result/')