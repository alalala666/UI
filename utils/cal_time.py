def cal_time(now_time):
    # 计算小时、分钟和秒
    hr, now_time = divmod(now_time, 3600)
    mi, sec = divmod(now_time, 60)
    
    # 格式化为两位数的字符串
    hr_str = f"{hr:02d}"
    mi_str = f"{mi:02d}"
    sec_str = f"{sec:02d}"
    
    cost_time = f"{hr_str}:{mi_str}:{sec_str}"
    return cost_time
#print(cal_time(10000))