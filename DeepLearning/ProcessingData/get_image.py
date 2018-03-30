#-*- coding:utf-8 -*-
#onlyzs1023@gmail.com 2016/11/21
import urllib.request
from urllib.parse import quote
import httplib2
import json
import os

API_KEY = "AIzaSyAtb-v5zknbTZkHnp2tzE3XKK27JRgkCes"
CUSTOM_SEARCH_ENGINE = "008121411857115255438:xh-uenz8ojw"

def getImageUrl(search_item, total_num):
 img_list = []
 i = 0
 while i < total_num:
  query_img = "https://www.googleapis.com/customsearch/v1?key=" + API_KEY + "&cx=" + CUSTOM_SEARCH_ENGINE + "&num=" + str(10 if(total_num-i)>10 else (total_num-i)) + "&start=" + str(i+1) + "&q=" + quote(search_item) + "&searchType=image"
  print (query_img)
  res = urllib.request.urlopen(query_img)
  data = json.loads(res.read().decode('utf-8'))
  for j in range(len(data["items"])):
   img_list.append(data["items"][j]["link"])
  i=i+10
 return img_list

def getImage(search_item, img_list):
 opener = urllib.request.build_opener()
 http = httplib2.Http(".cache")
 for i in range(len(img_list)):
  try:
   fn, ext = os.path.splitext(img_list[i])
   print(img_list[i])
   response, content = http.request(img_list[i])
   with open(search_item+str(i)+ext, 'wb') as f:
    f.write(content)
  except:
   print("failed to download images.")
   continue

if __name__ == "__main__":
 img_list = getImageUrl("秋元康", 100)
 print(img_list)
 getImage("yasushi", img_list)

 # img_list2 = getImageUrl("和田まあや", 80)
 # print(img_list2)
 # getImage("mayax", img_list2)
