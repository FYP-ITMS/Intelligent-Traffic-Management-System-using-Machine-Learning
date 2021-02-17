import time
def countdown(seconds, denser_lane):
    print("Opening Lane - {} for 20 seconds".format(str(denser_lane)))
    while seconds:
        mins, secs = divmod(seconds, 60)
        print(".", end="")
        time.sleep(1)
        seconds -= 1
    print("\nClosing Lane - {}".format(str(denser_lane)))

