from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
import time

teleop_config = SO101LeaderConfig(port="/dev/tty.usbmodem5AAF2631321", id ="leader_left")

teleop_device  = SO101Leader(teleop_config)

teleop_device.connect()

while True:
    action = teleop_device.get_action()
    print(action)
    time.sleep(0.2)

