from __future__ import print_function
from builtins import input
import base64
from Crypto.Cipher import AES
from getpass import getpass
from naoqi import ALProxy
import sys
from time import sleep
import twitter
from warnings import warn

def encrypt(key_32char, list_of_strings):
    result = []
    for msg in list_of_strings:
        cipher = AES.new(key_32char)
        result.append(base64.b64encode(cipher.encrypt(msg.rjust(128))))
        print(result[-1])
    return result

def decrypt(key_32char, list_of_strings):
    result = []
    for msg in list_of_strings:
        cipher = AES.new(key_32char)
        result.append(cipher.decrypt(base64.b64decode(msg)).strip(" "))
    return result

def prompt_32char_key(prompt="Enter decryption key:"):
    key_32char = getpass(prompt).rjust(32)
    return key_32char


encrypted_tokens = [
  'MqKK9JdkqtiuVdh2CQEgbDKiivSXZKrYrlXYdgkBIGwyoor0l2Sq2K5V2HYJASBsMqKK9JdkqtiuVdh2CQEgbDKiivSXZKrYrlXYdgkBIGwyoor0l2Sq2K5V2HYJASBsgnBEy0V62X79+ZvqT9TAzmLSOJqcpVtmtm5Y6Bw66ks=',
  'MqKK9JdkqtiuVdh2CQEgbDKiivSXZKrYrlXYdgkBIGwyoor0l2Sq2K5V2HYJASBsMqKK9JdkqtiuVdh2CQEgbLaDpuHG1PZDxvuVfAUIRt64mDIVsqJYXlng3N9aBcGSaIpHaUz1+2PClJ168M0yPYP8i+imMsa9O6OaZxpuRu0=',
  'MqKK9JdkqtiuVdh2CQEgbDKiivSXZKrYrlXYdgkBIGwyoor0l2Sq2K5V2HYJASBsMqKK9JdkqtiuVdh2CQEgbJgqGvLzFy+Kp1jsdQ221HTHU0Dg7kF1HA+z/vvA6jDxwtvpz5UZ0nPTPPlds5gMHD+NFW73Io8WJ35n9ugAeZA=',
  'MqKK9JdkqtiuVdh2CQEgbDKiivSXZKrYrlXYdgkBIGwyoor0l2Sq2K5V2HYJASBsMqKK9JdkqtiuVdh2CQEgbDKiivSXZKrYrlXYdgkBIGynrdROIvPY3iz0qNGOMjm8d9b+FWs9cIDR/vKD9Og4oWkJvnf9CaeHx2GzaXwVLyQ=',
]

if __name__ == "__main__":
    try:
        ip =  "10.42.0.49"
        port = 9559
        tts = ALProxy("ALTextToSpeech", ip, port)
    except RuntimeError:
        print("Can't connect to Naoqi. Proceed anyways? [Y/n] : ", end='')
        choice = input()
        if choice not in ['', 'Y', 'y', 'Yes', 'YES', 'yes']:
            raise RuntimeError("Naoqi not found at {}:{}".format(ip, port))
        else:
            tts = None

    key = prompt_32char_key()
    unencrypted_tokens = decrypt(key, encrypted_tokens)

    api = twitter.Api(
    consumer_key=unencrypted_tokens[0],
    consumer_secret=unencrypted_tokens[1],
    access_token_key=unencrypted_tokens[2],
    access_token_secret=unencrypted_tokens[3],
    tweet_mode='extended',
    )
    current_tweet = api.GetUserTimeline(screen_name="realDonaldTrump", count=1)[0]
    print(current_tweet.full_text)
#     if tts is not None:
#         tts.say(current_tweet.full_text.encode('ascii', 'ignore'))

    try:
        kCheckPeriod = 60 # seconds
        print("Waiting for new tweets ({}s)...".format(kCheckPeriod))
        while True:
            sleep(kCheckPeriod)
            new_tweets = api.GetUserTimeline(screen_name="realDonaldTrump", since_id=current_tweet.id)
            if new_tweets:
                print("got {} new tweets!".format(len(new_tweets)))
                current_tweet = new_tweets[0]
                for tweet in new_tweets:
                    print(tweet.full_text)
                    if tts is not None:
                        tts.say(tweet.full_text.encode('ascii', 'ignore'))
            print("Waiting for new tweets ({}s)...".format(kCheckPeriod))
    except KeyboardInterrupt:
        print("Exiting. (KeyboardInterrupt)")
