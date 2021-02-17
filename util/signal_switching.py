import time
def countdown(seconds,denser_lane):
	print("Opening Lane : {} for 20 seconds".format(str(denser_lane)))
	while seconds:
		mins,secs = divmod(t,60)
		timer = '{:02d}:{:02d}'.format(mins, secs)
		print(timer,end="\r")
		time.sleep(1)
		seconds-=1
	print("Closing Lane : {}".format(str(denser_lane)))
