import os

from PIL import Image
import numpy as np
from time import time
import open3d as o3d
from tqdm import tqdm
import math
import multiprocessing as mp
from matplotlib import pyplot as plt
import json
import rpyc
import logging
import base64
from io import BytesIO
from time import time, sleep


class MapViewTask():

    def __init__(self):
        # x coordinate, y coordinate, orientation at xy plan
        self.pos = (0, 0, 0)

        self.front_img = None
        self.rear_img = None
        self.map_img = None


class ViewTaskAgent():

    def __init__(self, port_num=18111):
        self.port = port_num
        self.connected = False

    def connect(self):
        try:
            self.conn = rpyc.connect("localhost", self.port)
            self.connected = True
        except:
            logging.error("Connect rpyc server error", exc_info=True)
            self.connected = False

    def disconnect(self):
        try:
            self.conn.close()
        except Exception:
            logging.error("Connect rpyc server error", exc_info=True)
        finally:
            self.connected = False

    def request_view_task(self):
        success = True
        view_task = MapViewTask()

        self.connect()
        try:
            results = self.conn.root.retrieve_view_task()
            if results["state_err"] == "ok":
                view_task.pos = results["pos"]
                logging.info(f"Request view task at pos: {view_task.pos} ok")
            else:
                success = False
                # logging.info(f"Failed to request view task")
        except Exception:
            logging.error("Request view task error", exc_info=True)
            success = False

        return (success, view_task)

    def img_to_b64(self, img, format="PNG"):
        img_file = BytesIO()
        img.save(img_file, format=format)
        img_bytes = img_file.getvalue()
        img_b64 = base64.b64encode(img_bytes)
        return img_b64

    def reply_view_task(self, pos=None, front_img=None, rear_img=None, map_img=None):
        success = True

        self.connect()
        try:
            if front_img is None or rear_img is None or map_img is None:
                reply_msg = None
            else:
                front_img_str = self.img_to_b64(front_img)
                rear_img_str = self.img_to_b64(rear_img)
                map_img_str = self.img_to_b64(map_img)

                reply_msg = {"pos": pos, "front_img": front_img_str,
                             "rear_img": rear_img_str,
                             "map_img": map_img_str}

            results = self.conn.root.complete_view_task(reply_msg)
            if results["state_err"] == "ok":
                pos = results["pos"]
                logging.info(f"Reply view task at pos: {pos} ok")
            else:
                logging.info(f"Failed to reply view task")
                success = False
        except Exception:
            logging.error("Reply view task error", exc_info=True)
            success = False

        return success


class ViewProcessor3D():

    def __init__(self, service_agent, map_path, min_x=-5120, min_y=-6144,
                 max_x=5120, max_y=5120, map_factor=0.5, zoom_factor=0.5):
        self.service_agent = service_agent
        self.xyz_data_results = []

        self.running_tasks = True
        self.zoom_factor = zoom_factor
        self.map_factor = map_factor
        self.map_source_img = self.read_2d_map_image(map_path)
        self.window_width = int(self.map_source_img.size[0] * 0.1)
        self.window_height = int(self.map_source_img.size[1] * 0.1)
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.locate_origin_point()
        self.locate_origin_point_for_moving()
        self.pos_marker_img = Image.open(self.get_image_path("car_marker_small.png"))
        self.map_image = None
        self.map_with_marker = None

    def collect_result(self, result):
        self.xyz_data_results.append(result)

    def fileter_map_pts(self, pts_data):
        xyz_data = np.empty((0, 3), int)

        slice_x = 1024
        slice_y = 1024

        size_x, size_y = pts_data.shape

        pieces_x = int(size_x / slice_x) + 1 if size_x % slice_x > 0 else int(size_x / slice_x)
        pieces_y = int(size_y / slice_y) + 1 if size_y % slice_y > 0 else int(size_y / slice_y)
        logging.info(f"Total grids: {pieces_x} x {pieces_y}")

        grid_xyz_arr = []

        logging.info("Number of processors: %d", mp.cpu_count())
        pool = mp.Pool(mp.cpu_count())

        for grid_x in range(pieces_x):
            for grid_y in range(pieces_y):
                pool.apply_async(self.fileter_pts_grid, args=(grid_x, grid_y, slice_x, slice_y, pts_data),
                                 callback=self.collect_result)

        pool.close()
        pool.join()

        for grid_data in self.xyz_data_results:
            xyz_data = np.vstack((xyz_data, grid_data))

        logging.info("")

        return xyz_data

    def fileter_pts_grid(self, grid_x, grid_y, slice_x, slice_y, pts_arr):
        xyz_arr = []
        size_x, size_y = pts_arr.shape
        downsample_size_x = 2
        downsample_size_y = 2

        x_start = grid_x * slice_x
        x_end = min((grid_x + 1) * slice_x, size_x)
        y_start = grid_y * slice_y
        y_end = min((grid_y + 1) * slice_y, size_y)

        # logging.info(f"grid_x: {grid_x}, grid_y: {grid_y}")
        # logging.info(f"x_start: {x_start}, x_end: {x_end}")
        # logging.info(f"y_start: {y_start}, y_end: {y_end}")

        grid_min = np.min(pts_arr[x_start:x_end, y_start:y_end])
        grid_max = np.max(pts_arr[x_start:x_end, y_start:y_end])
        grid_avg = np.average(pts_arr[x_start:x_end, y_start:y_end])
        # logging.info(f"grid_min: {grid_min}, grid_max: {grid_max}, grid_avg: {grid_avg}")

        if grid_min > 0 and grid_min == grid_max and grid_max == grid_avg:
            pass
            # logging.info(f"Located empty grid at grid ({grid_x}, {grid_y})")
            # logging.info(f"Reset empty grid points z value to 0")
            # filtered_pts[x_start:x_end, y_start:y_end] = 0
        else:
            # logging.info(f"Prepare grid points at grid ({grid_x}, {grid_y})")

            for pts_x in range(x_start, x_end, downsample_size_x):
                for pts_y in range(y_start, y_end, downsample_size_y):
                    rel_z = pts_arr[pts_x, pts_y]
                    if rel_z != 0:
                        pts_z = 255 - rel_z

                        xyz_arr.append([pts_x, pts_y, pts_z])

        zyz_np_arr = np.array(xyz_arr).reshape(-1, 3)

        return zyz_np_arr

    def read_2d_map_image(self, imag_file):
        image = Image.open(imag_file).transpose(Image.ROTATE_270)
        logging.info(f"map_img: {imag_file}, format: {image.format}, size: {image.size}")

        return image

    def read_pts_3d(self):
        pts_data = np.asarray(self.map_source_img)
        # pts_data = np.moveaxis(pts_data, 0, -1)
        logging.info(f"pts_data: {pts_data.shape}, type: {type(pts_data)}")

        t_start = time()

        xyz_3d = self.fileter_map_pts(pts_data)

        t_spend = time() - t_start

        logging.info(f"xyz_3d: {xyz_3d.shape}, type: {type(xyz_3d)}, time: {t_spend}")

        return xyz_3d

    # original point position at base 2d map picture
    def locate_origin_point(self):
        source_map_width, source_map_height = self.map_source_img.size

        scale_x = abs(self.max_x) / (self.max_x - self.min_x)
        scale_y = abs(self.max_y) / (self.max_y - self.min_y)

        pos_x = int(source_map_width * scale_x)
        pos_y = int(source_map_height * scale_y)

        self.origin_pos = (pos_x, pos_y, 0)

        logging.info(f"Origin point: {self.origin_pos}")

        return self.origin_pos

    # point position at base 2d map picture
    def locate_point(self, pos=(0, 0, 0)):
        input_x, input_y, ort_xy = pos
        pos0_x, pos0_y, pos0_ort_xy = self.origin_pos

        pos_x = int(input_x * self.map_factor + pos0_x)
        pos_y = int(input_y * self.map_factor + pos0_y)

        map_pos = (pos_x, pos_y, ort_xy)

        logging.info(f"Locate 2D Map point: {map_pos} for point {pos}")

        return map_pos

    # original point at GUI window
    def locate_origin_point_for_moving(self):
        self.origin_pos_move = self.locate_point_for_moving(pos=self.origin_pos)

        logging.info(f"Origin point for moving: {self.origin_pos_move}")

        return self.origin_pos_move

    # point position at GUI window
    def locate_point_for_moving(self, pos=(0, 0, 0)):
        map_pos_x, map_pos_y, map_ort_xy = self.locate_point(pos)
        logging.info(f"Adjust by map size, after ({map_pos_x, map_pos_y, map_ort_xy})")

        map_scale_x = self.window_width / self.map_source_img.size[0]
        map_scale_y = self.window_height / self.map_source_img.size[1]

        # Adjustment original point based on window scale
        map_pos_x = int(map_pos_x * map_scale_x)
        map_pos_y = int(map_pos_y * map_scale_y)
        logging.info(f"Adjust by window, after ({map_pos_x, map_pos_y, map_ort_xy})")

        # Adjustment based on zoom factor
        # adjust_x = int(self.map_source_img.size[0] / self.zoom_factor / 2 - self.map_source_img.size[0] / 2)
        # adjust_y = int(self.map_source_img.size[1] / self.zoom_factor / 2 - self.map_source_img.size[1] / 2)

        # Adjustment original point based on zoom factor
        map_pos_x = int(
            map_pos_x / self.zoom_factor - (self.window_width / 2 / self.zoom_factor - self.window_width / 2))
        map_pos_y = int(
            map_pos_y / self.zoom_factor - (self.window_height / 2 / self.zoom_factor - self.window_height / 2))
        logging.info(f"Adjust by zoom, after ({map_pos_x, map_pos_y, map_ort_xy})")

        moving_pos = (map_pos_x, map_pos_y, map_ort_xy)

        logging.info(f"Mapped move point: {moving_pos} for point {pos}")

        return moving_pos

    # Compute moving pixels regarding to origin point
    def locate_moving_distance(self, pos=(0, 0, 0)):
        map_pos_x, map_pos_y, map_ort_xy = self.locate_point(pos)
        origin_x, origin_y, orgin_ort_xy = self.origin_pos

        move_x = map_pos_x - origin_x
        move_y = map_pos_y - origin_y

        move_pos = (move_x, move_y, map_ort_xy)

        logging.info(f"Moving: {move_pos} for point {pos}")

        return move_pos

    def check_pts_3d(self, pts_3d):
        limit = 5
        for idx, xyz in enumerate(pts_3d):
            if idx < limit:
                logging.info(xyz)

    def convert_to_pcd_data(self, pts_3d):
        pcd_data = o3d.geometry.PointCloud()
        pcd_data.points = o3d.utility.Vector3dVector(pts_3d)

        return pcd_data

    def show_pts_3d(self, pts_3d):
        pcd = self.convert_to_pcd_data(pts_3d)

        o3d.visualization.draw_geometries([pcd], mesh_show_wireframe=True)

    def save_pcd_data(self, pts_3d, pcd_file):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_3d)
        o3d.io.write_point_cloud(pcd_file, pcd)

    def show_pcd_file(self, pcd_file):
        pcd = o3d.io.read_point_cloud(pcd_file)
        o3d.visualization.draw_geometries([pcd], mesh_show_wireframe=True)

    def show_pcd_data(self, pcd_data):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd_data)
        # vis.get_render_option().load_from_json("renderoption.json")
        vis.run()
        vis.destroy_window()

    def show_pcd_data_with_key_callback(self, pcd_data):

        def change_background_to_black(vis):
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0, 0, 0])
            return False

        def save_render_option(vis):
            vis.get_render_option().save_to_json("renderoption.json")
            return False

        def load_render_option(vis):
            vis.get_render_option().load_from_json("renderoption.json")
            return False

        def save_image(vis):
            vis.capture_screen_image("image.png")
            return False

        def capture_image(vis):
            image = vis.capture_screen_float_buffer()
            plt.imshow(np.asarray(image))
            plt.show()
            return False

        def save_view_trajectory(vis):
            ctr = vis.get_view_control()
            camera_param = ctr.convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters("pinhole_camera_parameters.json", camera_param)
            return False

        def load_view_trajectory(vis):
            ctr = vis.get_view_control()
            camera_params = o3d.io.read_pinhole_camera_parameters("pinhole_camera_parameters.json")
            ctr.convert_from_pinhole_camera_parameters(camera_params)
            return False

        def restore_init_view(vis):
            ctr = vis.get_view_control()
            camera_params = o3d.io.read_pinhole_camera_parameters("config/base_camera.json")
            ctr.convert_from_pinhole_camera_parameters(camera_params)

        def perform_custom_action(vis):
            full_w = self.window_width
            full_h = self.window_height
            half_w = int(self.window_width / 2)
            half_h = int(self.window_height / 2)

            ctr = vis.get_view_control()
            # ctr.rotate(0, -full_h, half_w, full_h)
            # ctr.translate(-10, 0, half_w, half_h)
            ctr.rotate(full_h, 0, 0, half_h)
            # ctr.translate(-10, 0, half_w, half_h)
            # ctr.rotate(-full_h, 0, 0, half_h)

            # ctr.set_zoom(0.5)
            # ctr.set_zoom(0.5)

            return False

        def load_north_view(vis):
            ctr = vis.get_view_control()
            camera_params = o3d.io.read_pinhole_camera_parameters(self.get_north_view_camera_params_path())
            ctr.convert_from_pinhole_camera_parameters(camera_params)
            return False

        def load_south_view(vis):
            ctr = vis.get_view_control()
            camera_params = o3d.io.read_pinhole_camera_parameters(self.get_south_view_camera_params_path())
            ctr.convert_from_pinhole_camera_parameters(camera_params)
            return False

        def load_west_view(vis):
            ctr = vis.get_view_control()
            camera_params = o3d.io.read_pinhole_camera_parameters(self.get_west_view_camera_params_path())
            ctr.convert_from_pinhole_camera_parameters(camera_params)
            return False

        def load_east_view(vis):
            ctr = vis.get_view_control()
            camera_params = o3d.io.read_pinhole_camera_parameters(self.get_east_view_camera_params_path())
            ctr.convert_from_pinhole_camera_parameters(camera_params)
            return False

        key_to_callback = {}
        key_to_callback[ord("K")] = perform_custom_action
        key_to_callback[ord("S")] = save_render_option
        key_to_callback[ord("R")] = restore_init_view
        key_to_callback[ord(",")] = save_image
        key_to_callback[ord(".")] = capture_image
        key_to_callback[ord("P")] = save_view_trajectory
        key_to_callback[ord("V")] = load_view_trajectory
        key_to_callback[ord("I")] = load_north_view
        key_to_callback[ord("M")] = load_south_view
        key_to_callback[ord("J")] = load_west_view
        key_to_callback[ord("L")] = load_east_view

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(width=self.window_width, height=self.window_height)
        vis.add_geometry(pcd_data)
        for key, callback in key_to_callback.items():
            vis.register_key_callback(key, callback)

        vis.run()
        vis.destroy_window()

    def get_front_view_camera_params_path(self):
        return os.path.join(os.path.join(os.getcwd(), 'config'), "front_camera.json")

    def get_rear_view_camera_params_path(self):
        return os.path.join(os.path.join(os.getcwd(), 'config'), "rear_camera.json")

    def get_base_view_camera_params_path(self):
        return os.path.join(os.path.join(os.getcwd(), 'config'), "base_camera.json")

    def get_north_view_camera_params_path(self):
        return os.path.join(os.path.join(os.getcwd(), 'config'), "north_camera.json")

    def get_south_view_camera_params_path(self):
        return os.path.join(os.path.join(os.getcwd(), 'config'), "south_camera.json")

    def get_west_view_camera_params_path(self):
        return os.path.join(os.path.join(os.getcwd(), 'config'), "west_camera.json")

    def get_east_view_camera_params_path(self):
        return os.path.join(os.path.join(os.getcwd(), 'config'), "east_camera.json")

    def get_image_path(self, image_filename):
        return os.path.join(os.path.join(os.getcwd(), 'images'), image_filename)

    def generate_map_image(self, vis3d):
        render_option_path = os.path.join(os.path.join(os.getcwd(), 'config'), "base_renderoption.json")
        vis3d.get_render_option().load_from_json(render_option_path)
        ctr3d = vis3d.get_view_control()
        ctr3d.set_zoom(self.zoom_factor)

        vis3d.poll_events()
        vis3d.update_renderer()

        map_temp_path = self.get_image_path("map_image_tmp.png")
        o3d_image = vis3d.capture_screen_image(map_temp_path, do_render=True)
        map2d_image = Image.open(map_temp_path)

        return map2d_image

    def generate_map_image_with_marker(self, target_pos=(0, 0, 0)):
        marker_pos_x, marker_pos_y, marker_ort_xy = self.locate_point_for_moving(target_pos)
        logging.info(f"located marker pos: ({marker_pos_x, marker_pos_y, marker_ort_xy}) for pos {target_pos}")

        map_with_marker = self.map_image.copy()
        marker_img = self.pos_marker_img.copy()
        if marker_ort_xy != 0:
            logging.debug(f"Rotate marker: {marker_ort_xy} degree")
            marker_img = marker_img.rotate(marker_ort_xy, expand=1)

        marker_img_width, marker_img_height = marker_img.size
        # logging.info(f"maker image width: {marker_img_width}, height: {marker_img_height}")

        move_pos_x = int(marker_pos_x - marker_img_width / 2)
        move_pos_y = int(marker_pos_y - marker_img_height / 2)

        logging.info(f"Prepare map image with marker at move pos: ({move_pos_x}, {move_pos_y})")
        map_with_marker.paste(marker_img, (move_pos_x, move_pos_y), marker_img)
        map_with_marker.save("images/check.png")

        return map_with_marker

    def save_current_camera_params(self, vis3d, camera_config_file):
        ctr3d = vis3d.get_view_control()
        camera_param = ctr3d.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(camera_config_file, camera_param)
        return True

    def load_camera_params(self, vis3d, camear_config_file):
        ctr3d = vis3d.get_view_control()
        camera_params = o3d.io.read_pinhole_camera_parameters(camear_config_file)
        ctr3d.convert_from_pinhole_camera_parameters(camera_params)
        return True

    def translate_view_position(self, ctr3d, move_x, move_y):
        full_h = self.window_height
        half_h = int(full_h / 2)

        full_w = self.window_width
        half_w = int(full_w / 2)

        # Turn to yx plane
        ctr3d.rotate(full_w, 0, 0, half_h)
        # Move y
        ctr3d.translate(-move_y, 0, half_w, half_h)
        # Restore to xy plane
        ctr3d.rotate(-full_w, 0, full_w, half_h)
        # Move x
        ctr3d.translate(-move_x, 0, half_w, half_h)


    def generate_front_rear_view(self, vis3d, target_pos=(0, 0, 0)):
        target_pos_x, target_pos_y, target_ort_xy = target_pos
        origin_x, origin_y, origin_ort_xy = self.origin_pos_move
        move_pos_x, move_pos_y, move_ort_xy = self.locate_moving_distance(target_pos)
        logging.debug(f"Locate front/rear view to move: ({move_pos_x}, {move_pos_y}) for target: {target_pos}")

        current_zoom_factor = 0.2
        current_scale_factor = 5 * current_zoom_factor
        move_pos_x = move_pos_x * current_scale_factor
        move_pos_y = move_pos_y * current_scale_factor

        x0 = int(self.window_width / 2)
        y0 = int(self.window_height / 2)

        # Convert direction to positive degree
        if move_ort_xy < 0:
            move_ort_xy = 360 + move_ort_xy

        # dive direction to 4 cartesian
        if (0 <= move_ort_xy <= 45) or (315 <= move_ort_xy <= 360):
            ort_cart = 1
        elif 45 < move_ort_xy < 135:
            ort_cart = 2
        elif 135 <= move_ort_xy <= 225:
            ort_cart = 3
        elif 225 < move_ort_xy < 315:
            ort_cart = 4
        # Invalid direction, so use default direction
        else:
            ort_cart = 1
            move_ort_xy = 0

        logging.info(f"Set orientation cartesian {ort_cart}")

        if ort_cart == 1:
            front_view_path = self.get_north_view_camera_params_path()
            rear_view_path = self.get_south_view_camera_params_path()
            front_move_pos_x = move_pos_x
            front_move_pos_y = move_pos_y
            rear_move_pos_x = -move_pos_x
            rear_move_pos_y = -move_pos_y

            if move_ort_xy == 0:
                rotate_xy = 0
            elif move_ort_xy >= 315:
                rotate_xy = int((move_ort_xy - 360) / 90 * self.window_width)
            else:
                rotate_xy = int(move_ort_xy / 90 * self.window_width)
        elif ort_cart == 2:
            front_view_path = self.get_west_view_camera_params_path()
            rear_view_path = self.get_east_view_camera_params_path()
            front_move_pos_x = -move_pos_y
            front_move_pos_y = move_pos_x
            rear_move_pos_x = move_pos_y
            rear_move_pos_y = -move_pos_x

            if move_ort_xy == 90:
                rotate_xy = 0
            else:
                rotate_xy = int((move_ort_xy - 90) / 90 * self.window_width)
        elif ort_cart == 3:
            front_view_path = self.get_south_view_camera_params_path()
            rear_view_path = self.get_north_view_camera_params_path()
            front_move_pos_x = -move_pos_x
            front_move_pos_y = -move_pos_y
            rear_move_pos_x = move_pos_x
            rear_move_pos_y = move_pos_y

            if move_ort_xy == 180:
                rotate_xy = 0
            else:
                rotate_xy = int((move_ort_xy - 180) / 90 * self.window_width)
        elif ort_cart == 4:
            front_view_path = self.get_east_view_camera_params_path()
            rear_view_path = self.get_west_view_camera_params_path()
            front_move_pos_x = move_pos_y
            front_move_pos_y = -move_pos_x
            rear_move_pos_x = -move_pos_y
            rear_move_pos_y = move_pos_x

            if move_ort_xy == 270:
                rotate_xy = 0
            else:
                rotate_xy = int((move_ort_xy - 270) / 90 * self.window_width)
        else:
            logging.error(f"Invalid orientation cartesian {ort_cart}")

        logging.debug("Loaded front camera parameters")
        self.load_camera_params(vis3d, front_view_path)

        ctr3d = vis3d.get_view_control()
        self.translate_view_position(ctr3d, front_move_pos_x, front_move_pos_y)

        if rotate_xy > 0:
            ctr3d.rotate(-rotate_xy, 0, xo=origin_x, yo=origin_y)

        vis3d.poll_events()
        vis3d.update_renderer()

        front_img_temp_path = self.get_image_path("front_image_tmp.png")
        vis3d.capture_screen_image(front_img_temp_path, do_render=True)
        front_image = Image.open(front_img_temp_path)

        logging.debug("Loaded rear camera parameters")
        self.load_camera_params(vis3d, rear_view_path)

        ctr3d = vis3d.get_view_control()
        self.translate_view_position(ctr3d, rear_move_pos_x, rear_move_pos_y)

        if rotate_xy > 0:
            ctr3d.rotate(-rotate_xy, 0, xo=origin_x, yo=origin_y)

        vis3d.poll_events()
        vis3d.update_renderer()

        rear_img_temp_path = self.get_image_path("rear_image_tmp.png")
        vis3d.capture_screen_image(rear_img_temp_path, do_render=True)
        rear_image = Image.open(rear_img_temp_path)

        logging.debug("Loaded rear camera parameters")
        # Restore with base camera parameters
        base_camera_parms = self.get_base_view_camera_params_path()
        self.load_camera_params(vis3d, base_camera_parms)

        vis3d.poll_events()
        vis3d.update_renderer()

        return (front_image, rear_image)

    def start_processor_viewer(self, pcd_data):
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=self.window_width, height=self.window_height)
        vis.add_geometry(pcd_data)
        self.map_image = self.generate_map_image(vis)

        self.running_tasks = True

        while self.running_tasks:
            # Request view task
            task_status, view_task = self.service_agent.request_view_task()

            if task_status:
                logging.info(f"Received task: {task_status}")

                front_img_path = os.path.join(os.path.join(os.getcwd(), 'images'), "front_image.png")
                rear_img_path = os.path.join(os.path.join(os.getcwd(), 'images'), "rear_image.png")
                # map_img_path = os.path.join(os.path.join(os.getcwd(), 'images'), "map_image.png")

                # view_task.front_img = Image.open(front_img_path)
                # view_task.rear_img = Image.open(rear_img_path)
                # view_task.map_img = Image.open(map_img_path)

                view_task.front_img, view_task.rear_img = self.generate_front_rear_view(vis, view_task.pos)
                view_task.map_img = self.generate_map_image_with_marker(view_task.pos)
                task_status = self.service_agent.reply_view_task(pos=view_task.pos,
                                                                 front_img=view_task.front_img,
                                                                 rear_img=view_task.rear_img,
                                                                 map_img=view_task.map_img)

            else:
                # vis.update_geometry(pcd_data)
                # Check whether user stopped application window
                self.running_tasks = vis.poll_events()
                vis.update_renderer()
                sleep(1)

        self.service_agent.disconnect()
        vis.destroy_window()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
                        handlers=[
                            logging.FileHandler("Processed3dView.log"),
                            logging.StreamHandler()
                        ])
    logging.info("Start Processed 3D Viewer")

    agent = ViewTaskAgent()

    # Compute from scan3dview
    min_x = -5120
    min_y = -6144
    max_x = 4800
    max_y = 5000

    map_path = "..\\data\\processed\\merged_gray_images_reslyr_bound.png"

    viewer = ViewProcessor3D(agent, map_path, min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)

    pts = viewer.read_pts_3d()

    check_pos = (10, 10, 45)

    moving_distance = viewer.locate_moving_distance(pos=check_pos)

    # show_pts_3d(pts)

    pcd_data = viewer.convert_to_pcd_data(pts)

    # show_pcd_data(pcd_data)
    #viewer.show_pcd_data_with_key_callback(pcd_data)
    viewer.start_processor_viewer(pcd_data)
