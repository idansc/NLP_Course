import time

def calc_elpased_time(start_time):
    secs = round(time.time() - start_time, 2)
    if secs < 60:
        return str(secs) + " [sec]"
    else:
        return str(round(secs/60, 2)) + " [min]"
