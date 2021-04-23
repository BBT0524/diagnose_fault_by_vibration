"""

"""
import time
def get_running_time(fun):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        # 调用需要计算运行时间的函数
        fun(*args, **kwargs)
        end_time = time.time()
        running_time = end_time - start_time
        h = int(running_time//3600)
        m = int((running_time - h*3600)//60)
        s = int(running_time%60)
        print("time cost: {0}:{1}:{2}".format(h, m, s))
        return running_time # -> 可以省略
    return wrapper