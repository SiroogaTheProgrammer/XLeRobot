from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

robot_config = SO101FollowerConfig(port="/dev/tty.usbmodem5B3D0456871", id ="follower_left")
teleop_config = SO101LeaderConfig(port="/dev/tty.usbmodem5AAF2631321", id ="leader_left") #the second port from magsafe

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)

robot.connect()
teleop_device.connect()

while True:
    action = teleop_device.get_action()
    robot.send_action(action)