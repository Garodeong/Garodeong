import os
import time

def speak(option="-v english -s 130 -a 150", msg="There is nothing"):
	os.system(f"espeak {option} '{msg}'")
	 
if __name__=="__main__":
	#speak()
	#from gtts import gTTS
	#from IPython.display import Audio, display
	#text = "아무것도 없다"
	#tts = gTTS(text, lang='en')
	#tts.save('noti.mp3')
	
	#wn = Audio('noti.mp3', autoplay=True)
	#display(wn)
	#speak(option="-v english -s 80 -a 150 -p 40", msg="There is a person")
	pass
	
