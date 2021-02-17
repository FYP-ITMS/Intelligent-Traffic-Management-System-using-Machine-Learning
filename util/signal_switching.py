import time
def countdown(denser_lane):
	print("Opening Lane : {} for 20 seconds".format(str(denser_lane)))
	while 20:
		mins,secs = divmod(t,60)
		timer = '{:02d}:{:02d}'.format(mins, secs)
		print(timer,end="\r")
		time.sleep(1)
		t-=1
	print("Closing Lane : {}".format(str(denser_lane)))
