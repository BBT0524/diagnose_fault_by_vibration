# 数据说明
只采用了凯斯西储大学轴承数据中的12kHz采样的故障数据和正常数据。一共分为四个部分：0HP、1HP、2HP、3HP；
没上传成功，在这里贴下数据下载链接吧：http://csegroups.case.edu/bearingdatacenter/pages/12k-drive-end-bearing-fault-data

（1）故障尺寸只取了前三个：007、014、021，分别对应为轻微、中等、严重三个级别。 \
（2）故障编码采用数字编码：
     labels = {"normal":0, "IR007":1, "IR014":2, "IR021":3, "OR007":4,
         "OR014":5, "OR021":6, "B007":7, "B014":8, "B021":9}
