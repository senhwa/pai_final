# State of the auv.  This is the fundamental information for controlling the AUV.

#Control MODE
int8 JOYCON = 0
int8 REMOCON = 1
int8 MISSION = 2

Header header

# heading
float32 heading
int8 mode # 現在のモード 0: findstation mode 1: observation mode

# relative velocity at the AUV origin [m/s]
float32[2] vxy
# horizontal position [m]
float32[2] xy
float32[2] exy # std

# position obtained by INS [m]
float32[3] xyz_ins
float32[3] origin_ins # 原点の緯度経度高度

float32 alt  # altitude [m] from DVL
float32 alt_age # age of the measurement [s]
