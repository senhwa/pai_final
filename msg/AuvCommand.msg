# Control reference of 3 parameters (xyz, pose, etc..)
# ref_pose: roll, pitch, yaw
# ref_torque: roll, pitch, yaw [Nm]
# ref_xyz: surge, sway, heave(depth)
# ref_force: surge, sway, heave [N]
#例　ref_poseの場合は[roll,pitch,yaw]で，[0,0,3]とすればYawの方位制御だけ入れているということ

Header header

Ref3d xyz_ref
Ref3d rpy_ref
