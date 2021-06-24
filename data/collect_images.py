import re
import os
import requests
import shutil
import mechanicalsoup

from data import constants


def download_image(url):
    filename = re.findall(constants.FILE_NAME_REGEX, url)
    if len(filename) == 0:
        return False

    filename = constants.IMAGES_DIR + filename[0][0]
    
    if os.path.isfile(filename):
        return False

    r = requests.get(url, stream = True)

    # Check if the image was retrieved successfully
    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True
        
        # Open a local file with wb ( write binary ) permission.
        with open(filename,'wb') as f:
            shutil.copyfileobj(r.raw, f)
            
        print('Image sucessfully Downloaded: ',filename)
        return True


def collect():
    browser = mechanicalsoup.StatefulBrowser(user_agent='MechanicalSoup')
    browser.open(constants.START_URL)
    links = browser.links(url_regex=constants.FULL_ART_REGEX)
    image_count = 0
    for link in links:
        print(link['href'])
        browser.follow_link(link)
        images = browser.page.find_all("img")
        # print(len(images))
        for img in images:
            if download_image(img['src']):
                image_count += 1

    print(image_count)

if __name__ == '__main__':
    collect()