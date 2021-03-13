import time
import emoji


def countdown(seconds, denser_lane):
    #print('\033[99m' +"Opening Lane - {} for {} seconds".format(str(denser_lane),seconds))
    time.sleep(1)
    print(
        emoji.emojize(":white_circle:") + "\n" +
        emoji.emojize(":white_circle:") + "\n" +
        emoji.emojize(":green_circle:") + "\n" + "Lane-{} Opened".format(
            str(denser_lane) + "\n" + emoji.emojize(":green_circle:") +
            emoji.emojize(":green_circle:") + emoji.emojize(":green_circle:")))
    while seconds:
        mins, secs = divmod(seconds, 60)
        print('\033[99m' + ".", end="")
        time.sleep(1)
        seconds -= 1
    #print('\n\033[99m' + "Closing Lane {}".format(str(denser_lane)))
    time.sleep(1)
    print(
        emoji.emojize(":red_circle:") + emoji.emojize(":red_circle:") +
        emoji.emojize(":red_circle:") + "Lane-{} Closed".format(
            str(denser_lane) + emoji.emojize(":red_circle:") +
            emoji.emojize(":red_circle:") + emoji.emojize(":red_circle:")))
