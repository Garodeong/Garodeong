import os

def speak(option="-v ko", msg="전방에 아무것도 없어요"):
	os.system(f"espeak {option} '{msg}'")
	
if __name__=="__main__":
	speak()
