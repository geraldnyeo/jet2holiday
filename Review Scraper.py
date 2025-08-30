from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

options = Options()
options.add_argument('--lang=en')
options.add_argument("accept-language=en-ID,en")

chrome_options = Options()
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.265 Safari/537.36"
)

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

driver.get("https://www.google.com/maps/place/KFC/@1.3362861,103.584531,12z/data=!3m1!5s0x31da0f99944dd34b:0x377b6b1bf9233d94!4m10!1m2!2m1!1sKFC!3m6!1s0x31da0f99961bf53f:0x7b2d2b18eecedb7b!8m2!3d1.3424642!4d103.6926777!15sCgNLRkMiA4gBAVoFIgNrZmOSARRmYXN0X2Zvb2RfcmVzdGF1cmFudKoBTQoNL2cvMTFiN3FfbDE4NAoIL20vMDliNnQQASoHIgNrZmMoADIeEAEiGiNheWE3bpgriRg8yZAdF9ooXHh1o5q9efIiMgcQAiIDa2Zj4AEA!16s%2Fg%2F12hmqk83l?entry=ttu&g_ep=EgoyMDI1MDgxOS4wIKXMDSoASAFQAw%3D%3D")
time.sleep(10)

driver.find_element(By.XPATH, "//span[contains(text(), 'More reviews')]").click()
time.sleep(3)

reviews, seen = [], set()

while len(reviews) < 10:
    for c in driver.find_elements(By.CSS_SELECTOR, "div.jftiEf"):
        user_el = c.find_elements(By.CLASS_NAME, "d4r55")
        review_el = c.find_elements(By.CLASS_NAME, "wiI7pd")
        rating_el = c.find_elements(By.CLASS_NAME, "kvMYJc")
        date_el = c.find_elements(By.CLASS_NAME, "rsqaWe")

        if not review_el or not review_el[0].text.strip():
            continue

        user = user_el[0].text
        review = review_el[0].text.strip()
        rating = rating_el[0].get_attribute('aria-label').split()[0]
        date = date_el[0].text

        if (user, review) not in seen:
            seen.add((user, review))
            reviews.append({'user': user, 'review': review, 'rating': rating, 'date': date})

        try:
            driver.execute_script("arguments[0].scrollIntoView();", driver.find_elements(By.CSS_SELECTOR, "div.jftiEf")[-1])
        except:
            break

        time.sleep(2)

print(reviews)

driver.quit()
