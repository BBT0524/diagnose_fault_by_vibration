"""
提高代码简洁性和可交互性的代码
"""
import time
def get_running_time(fun):
    """
    代码运行时间装饰器

    参数
    -----
    fun：要测试运行时间的函数

    返回
    -----
    返回装饰器wrapper

    例子
    -----
    >>> @get_running_time
        def hello(name):
            print("hello %s"%name)
            time.sleep(3)


hello("Tony")
    """
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