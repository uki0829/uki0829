import requests
import os
from urllib.parse import urlparse
import time
import random
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 请将此处替换为您的 Unsplash API 密钥
UNSPLASH_ACCESS_KEY = "6oQbvEfd_GZLUTjJe7kyVk7hJGkmTK8CFec7-ozEFXI"  # 确保这里是正确的API密钥
DOWNLOADED_IDS_FILE = 'downloaded_image_ids.json'

DOG_SEARCH_TERMS = [
    "dog", "puppy", "canine", "hound", "pooch", "mutt", "pup",
    "doggy", "furry friend", "four-legged companion"
]

def load_downloaded_ids():
    if os.path.exists(DOWNLOADED_IDS_FILE):
        with open(DOWNLOADED_IDS_FILE, 'r') as f:
            return set(json.load(f))
    return set()

def save_downloaded_ids(ids):
    with open(DOWNLOADED_IDS_FILE, 'w') as f:
        json.dump(list(ids), f)

def get_dog_images(page, per_page):
    url = "https://api.unsplash.com/search/photos"
    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
    params = {
        "query": random.choice(DOG_SEARCH_TERMS),
        "page": page,
        "per_page": per_page,
        "orientation": "landscape"
    }
    
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()["results"]

def download_dog_images(num_images=50, save_dir='dog_images', per_page=30):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    downloaded_count = 0
    downloaded_ids = load_downloaded_ids()
    page = 1
    
    while downloaded_count < num_images:
        try:
            images = get_dog_images(page, per_page)
            if not images:
                logging.warning(f"第 {page} 页没有返回任何图片，尝试下一页")
                page += 1
                continue
            
            for image in images:
                image_id = image['id']
                if image_id in downloaded_ids:
                    logging.info(f"跳过已下载的图片: {image_id}")
                    continue
                
                image_url = image['urls']['regular']
                
                logging.info(f"正在下载图片: {image_url}")
                image_response = requests.get(image_url)
                image_response.raise_for_status()
                
                file_extension = os.path.splitext(urlparse(image_url).path)[1]
                if not file_extension:
                    file_extension = '.jpg'
                
                timestamp = int(time.time() * 1000)
                random_suffix = random.randint(1000, 9999)
                file_name = f"dog_{timestamp}_{random_suffix}{file_extension}"
                file_path = os.path.join(save_dir, file_name)
                
                with open(file_path, 'wb') as file:
                    file.write(image_response.content)
                
                logging.info(f"已下载并保存: {file_name}")
                downloaded_count += 1
                downloaded_ids.add(image_id)
                
                if downloaded_count >= num_images:
                    break
            
            logging.info(f"已下载 {downloaded_count} 张图片")
            save_downloaded_ids(downloaded_ids)
            
            page += 1
            time.sleep(1)  # 避免超过 API 限制
        
        except requests.RequestException as e:
            logging.error(f"下载图片时出错: {e}")
            break
        except Exception as e:
            logging.error(f"发生未预期的错误: {e}")
            break

if __name__ == "__main__":
    logging.info(f"使用的 API 密钥: {UNSPLASH_ACCESS_KEY[:5]}...{UNSPLASH_ACCESS_KEY[-5:]}")
    download_dog_images(100)  # 尝试下载100张图片