def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        print(myDataList)

markAttendance('a')