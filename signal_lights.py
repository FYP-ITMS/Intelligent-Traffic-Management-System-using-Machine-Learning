import time
import emoji
def switch_signal(denser_lane,seconds):
    if denser_lane==1:
        print("Lane 1 is now opening...\n")
        print(
            "Lane 1                Lane 2                Lane 3                Lane 4"
        )
        time.sleep(1)
        print(
            "  "+  emoji.emojize(":white_circle:") + "                   "+emoji.emojize(":red_circle:")+ "                    "+emoji.emojize(":red_circle:")+ "                   "+emoji.emojize(":red_circle:")+
            "\n  " + emoji.emojize(":white_circle:") + "                   "+emoji.emojize(":white_circle:")+ "                    "+emoji.emojize(":white_circle:")+ "                   "+emoji.emojize(":white_circle:")+
            "\n  " + emoji.emojize(":green_circle:") + "                   "+emoji.emojize(":white_circle:")+ "                    "+emoji.emojize(":white_circle:")+ "                   "+emoji.emojize(":white_circle:") +
            "\n")


        while seconds:
            mins, secs = divmod(seconds, 60)
            print('\033[99m'+".", end="")
            time.sleep(1)
            seconds -= 1
        print()
        print('\033[1m' + '\n\033[99m' +
              "Closing Lane {}".format(str(denser_lane))+ '\033[0m' )
        time.sleep(1)
        print()
        print(
            "Lane 1                Lane 2                Lane 3                Lane 4"
        )
        print(
            "  "+  emoji.emojize(":red_circle:") + "                   "+emoji.emojize(":red_circle:")+ "                    "+emoji.emojize(":red_circle:")+ "                   "+emoji.emojize(":red_circle:")+
            "\n  " + emoji.emojize(":white_circle:") + "                   "+emoji.emojize(":white_circle:")+ "                    "+emoji.emojize(":white_circle:")+ "                   "+emoji.emojize(":white_circle:")+
            "\n  " + emoji.emojize(":white_circle:") + "                   "+emoji.emojize(":white_circle:")+ "                    "+emoji.emojize(":white_circle:")+ "                   "+emoji.emojize(":white_circle:") +
            "\n")



switch_signal(1,5)