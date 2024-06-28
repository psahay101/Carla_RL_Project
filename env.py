import carla
import numpy as np
import cv2
import random
import time
import math

class CarEnv:
    SHOW_CAM = False  # Set to True to enable camera preview
    IM_WIDTH = 224
    IM_HEIGHT = 224
    SECONDS_PER_EPISODE = 10

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.collision_hist = []
        self.actor_list = []
        self.front_camera = None
        self.sensor = None        # Initialize to None
        self.episode_start = None # Initialize to None
        self.transform = None     # Initialize to None
        self.vehicle = None       # Initialize to None

    def reset(self):
        self.cleanup()
        self.collision_hist = []
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.setup_camera()
        self.setup_collision_sensor()

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)  # wait for the world to get going

        self.episode_start = time.time()
        return self.get_state()

    def setup_camera(self):
        cam_bp = self.blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x", str(self.IM_WIDTH))
        cam_bp.set_attribute("image_size_y", str(self.IM_HEIGHT))
        cam_bp.set_attribute("fov", "110")
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(cam_bp, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

    def setup_collision_sensor(self):
        colsensor_bp = self.blueprint_library.find("sensor.other.collision")
        colsensor = self.world.spawn_actor(colsensor_bp, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(colsensor)
        colsensor.listen(lambda event: self.collision_data(event))

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.IM_HEIGHT, self.IM_WIDTH, 4))
        i3 = i2[:, :, :3]  # Extract only the RGB channels
        gray_image = cv2.cvtColor(i3, cv2.COLOR_RGB2GRAY)  # Convert RGB to Grayscale
        if self.SHOW_CAM:
            cv2.imshow("CarlaCam", gray_image)
            cv2.waitKey(1)
        self.front_camera = gray_image  # Store the grayscale image

    def process_img_color(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.IM_HEIGHT, self.IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("CarlaCam", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def collision_data(self, event):
        self.collision_hist.append(event)

    def get_yaw_only_state(self):
        """Constructs the state to be used in the reinforcement learning environment."""
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        collision_detected = len(self.collision_hist) > 0  # Check if there have been any collisions

        state = {
            'image': self.front_camera,  # Camera image data
            'speed': 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2),  # Speed in km/h
            'yaw': transform.rotation.yaw,  # Vehicle's yaw
            'collision': collision_detected  # Boolean indicating collision occurrence
        }
        return state

    def get_state(self):
        if self.front_camera is None:
            return np.empty((self.IM_HEIGHT, self.IM_WIDTH, 3)), None  # Return an empty array and None if no camera data is available
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        state = np.array([transform.rotation.yaw, velocity.x])  # Array of yaw and x velocity
        return state, self.front_camera  # Return the state array and the camera image

    def step(self, action):
        throttle, brake, steer = action
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))

        velocity = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))

        done = False
        reward = 1 if kmh > 50 else -1
        if len(self.collision_hist) > 0:
            done = True
            reward = -200
        if self.episode_start + self.SECONDS_PER_EPISODE < time.time():
            done = True

        return self.get_state(), reward, done, None

    def cleanup(self):
        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = []

    def __del__(self):
        self.cleanup()
