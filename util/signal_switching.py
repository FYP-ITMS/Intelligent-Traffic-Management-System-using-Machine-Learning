import time
def countdown(seconds, denser_lane):
    print('\033[92m' +"Opening Lane - {} for 20 seconds".format(str(denser_lane)))
    while seconds:
        mins, secs = divmod(seconds, 60)
        print(".", end="")
        time.sleep(1)
        seconds -= 1
    print('\033[91m' + "\nClosing Lane - {}".format(str(denser_lane)))
