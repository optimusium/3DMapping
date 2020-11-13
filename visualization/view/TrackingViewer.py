import base64
import tkinter as tk
from io import BytesIO
from tkinter import ttk
import sys
import logging
import os
from PIL import Image, ImageTk
import rpyc
from time import time, sleep

class MapViewState():

    def __init__(self):
        # x coordinate, y coordinate, orientation at xy plan
        self.pos = (0, 0, 0)

        self.front_img = None
        self.rear_img = None
        self.map_img = None


class MapAgent():

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
            if self.conn:
                self.conn.close()
        except Exception:
            logging.error("Connect rpyc server error", exc_info=True)
        finally:
            self.connected = False

    def request_views(self, pos=(0, 0, 0)):
        success = True

        self.connect()
        try:
            params = {"pos": pos}
            results = self.conn.root.request_view(params)
            if results["state_err"] == "request ok":
                logging.info(f"Sent view request at pos: {pos} ok")
            else:
                logging.info(f"Failed to send view request at pos: {pos}")
        except Exception:
            logging.error("request view error", exc_info=True)
            success = False

        return success

    def b64_to_img(self, b64_str):
        img_bytes = base64.b64decode(b64_str)
        img_file = BytesIO(img_bytes)
        img = Image.open(img_file)

        return img

    def receive_views(self):
        success = True
        map_view = MapViewState()

        self.connect()

        try:
            results = self.conn.root.receive_view()
            if results["state_err"] == "response ok":
                map_view.pos = results["pos"]
                map_view.map_img = self.b64_to_img(results["map_img"])
                map_view.front_img = self.b64_to_img(results["front_img"])
                map_view.rear_img = self.b64_to_img(results["rear_img"])

                logging.info(f"Receive map view at pos: {map_view.pos} ok")
            else:
                logging.info(f"Failed to receive map view")
                success = False
        except Exception:
            logging.error("Receive map view error", exc_info=True)
            success = False

        return (success, map_view)


class MapViewer():

    def __init__(self, service_agent):
        self.service_agent = service_agent
        self.main_window_size = "1080x650"
        self.sight_view_size = (380, 230)
        self.map_view_size = (640, 480)
        self.pos_list = []
        self.read_tracking_tasks()
        self.step_index = 0
        self.tracking_task = False
        self.current_task_view = None

    def read_tracking_tasks(self):
        tracking_filepath = self.get_file_path("config", "tracking_tasks.txt")
        with open(tracking_filepath, "r") as track_file:
            lines = track_file.readlines()

        for line in lines:
            line_text = line.strip()
            if line_text != "" and line_text[0] != "#" and "," in line_text:
                pos_list = line_text.split(",")
                pos_x = int(pos_list[0].strip())
                pos_y = int(pos_list[1].strip())
                ort_xy = int(pos_list[2].strip())
                self.pos_list.append((pos_x, pos_y, ort_xy))


    def get_file_path(self, folder_name, filename):
        folder = os.path.join(os.getcwd(), folder_name)
        file_path = os.path.join(folder, filename)
        return file_path

    def get_img_path(self, img_filename, image_id=0):
        img_folder = os.path.join(os.getcwd(), 'images')
        if image_id == 0:
            real_img_filename = img_filename
        else:
            real_img_filename = img_filename.format(image_id)

        img_path = os.path.join(img_folder, real_img_filename)
        return img_path

    def get_map_img_path(self, image_id=0):
        if image_id == 0:
            return self.get_img_path('map_image.png')
        else:
            return self.get_img_path('map_image_{0}.png', image_id)

    def get_front_img_path(self, image_id=0):
        if image_id == 0:
            return self.get_img_path('front_image.png')
        else:
            return self.get_img_path('front_image_{0}.png', image_id)

    def get_rear_img_path(self, image_id=0):
        if image_id == 0:
            return self.get_img_path('rear_image.png')
        else:
            return self.get_img_path('rear_image_{0}.png', image_id)

    def resize_image(self, img_path, img_size):
        img = Image.open(img_path)
        resized_img = img.resize(img_size, Image.ANTIALIAS)
        photo_img = ImageTk.PhotoImage(resized_img)
        return photo_img

    def resize_map_view(self, image, img_size):
        resized_img = image.resize(img_size, Image.ANTIALIAS)
        photo_img = ImageTk.PhotoImage(resized_img)
        return photo_img

    def stop_viewer(self):
        logging.info(f"Stop to track map view")
        self.service_agent.disconnect()
        self.step_index = 0
        self.tracking_task = False

    def start_viewer(self):
        logging.info(f"Start to track map view")
        self.tracking_task = True
        self.txt_map_cord.set(" ")
        self.update_viewer()

    def update_viewer(self):
        logging.info(f"Update map view in tracking steps")
        if self.step_index >= len(self.pos_list):
            self.step_index = 0

        logging.info(f"Request map view from agent")
        step_pos = self.pos_list[self.step_index]
        service_status = self.service_agent.request_views(pos=step_pos)
        map_view_results = None
        if service_status:
            # Wait for 2 seconds then try retrieve map view reply
            sleep(0.2)
            logging.info(f"Retrieve map view from agent")
            service_status, map_view_results = self.service_agent.receive_views()

        if service_status:
            logging.debug(f"Map view result retrieved")
            self.current_task_view = map_view_results
            map_pos_x, map_pos_y, map_ort_xy = self.current_task_view.pos
            front_img = self.current_task_view.front_img
            rear_img = self.current_task_view.rear_img
            map_img = self.current_task_view.map_img

            self.img_map_view = self.resize_map_view(map_img, self.map_view_size)
            self.lbl_map_view.config(image=self.img_map_view)
            #self.cvs_map_view.itemconfig(self.img_map_id, image=self.img_map_view)
            self.img_front_view = self.resize_map_view(front_img, self.sight_view_size)
            self.lbl_front_view.config(image=self.img_front_view)
            #self.cvs_front_view.itemconfig(self.img_front_id, image=self.img_front_view)
            self.img_rear_view = self.resize_map_view(rear_img, self.sight_view_size)
            self.lbl_rear_view.config(image=self.img_rear_view)
            #self.cvs_rear_view.itemconfig(self.img_rear_id, image=self.img_rear_view)

            # Show coordiantes
            pos_cord = "Coordinate: ({0}, {1}), Orentation: {2}".format(map_pos_x, map_pos_y, map_ort_xy)
            self.txt_map_cord.set(pos_cord)

            self.mainframe.update()

            self.step_index = self.step_index + 1

        # Schedule to refresh to get next step
        if self.tracking_task:
            self.btn_start_track.after(1000, self.update_viewer)

    def exit_viewer(self, ):
        try:
            self.service_agent.disconnect()
            self.root.destroy()
        except Exception:
            logging.error("Failed to close app window", exc_info=True)

    def show_viewer(self, args):
        self.root = tk.Tk()
        self.root.title("Map Tracking Viewer")
        self.root.geometry(self.main_window_size)

        # Frame style
        self.sty_frame = ttk.Style()
        self.sty_frame.configure('viewer.TFrame', background='white')
        self.sty_frame.configure('viewer.TLabel', background='white', font=('Helvetica', 16))
        self.sty_frame.configure('map.TLabel', background='white', font=('Helvetica', 12))
        self.sty_frame.configure('image.TLabel', background='white')
        self.sty_frame.configure('viewer.main.TLabel', background='white', font=('Helvetica', 26))
        self.sty_frame.configure('viewer.TButton', font=('Helvetica', 14))

        self.mainframe = ttk.Frame(master=self.root, padding="3 3 12 12", style="viewer.TFrame")
        self.mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Top Frame
        self.frm_top = ttk.Frame(master=self.mainframe, style="viewer.TFrame")
        self.frm_top.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        self.frm_top.columnconfigure(0, weight=1)
        self.frm_top.rowconfigure(0, weight=1)

        self.lbl_main_title = ttk.Label(master=self.frm_top, text="Map Tracking Viewer", anchor="center",
                                        style="viewer.main.TLabel")
        self.lbl_main_title.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        # Center Frame
        self.frm_center = ttk.Frame(master=self.mainframe, style="viewer.TFrame")
        self.frm_center.grid(column=0, row=1, sticky=(tk.N, tk.W, tk.E, tk.S))

        self.frm_left_center = ttk.Frame(master=self.frm_center, relief=tk.SOLID, borderwidth=1,
                                         style="viewer.TFrame")
        self.frm_left_center.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        self.frm_left_top = ttk.Frame(master=self.frm_left_center, style="viewer.TFrame")
        self.frm_left_top.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        self.lbl_front_view_title = ttk.Label(master=self.frm_left_top, text="Front View", anchor="center",
                                              style="viewer.TLabel")
        self.lbl_front_view_title.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        #front_img_path = self.get_front_img_path()
        #logging.debug(f"front_img_path: {front_img_path}")
        #self.img_front_view = self.resize_image(front_img_path, self.sight_view_size)

        self.img_front_view = ImageTk.PhotoImage(Image.new('RGB', self.sight_view_size))
        self.lbl_front_view = ttk.Label(master=self.frm_left_top, image=self.img_front_view, style="image.TLabel")
        self.lbl_front_view.grid(column=0, row=1, sticky=(tk.N, tk.W, tk.E, tk.S))

        #self.cvs_front_view = tk.Canvas(master=self.frm_left_top, bd=0, highlightthickness=0, bg="white",
        #                                width=self.sight_view_size[0], height=self.sight_view_size[1])
        #self.img_front_id = self.cvs_front_view.create_image(0, 0, image=self.img_front_view,
        #                                                     anchor=tk.NW, tags="FRONT_IMG")
        #self.cvs_front_view.grid(column=0, row=1, sticky=(tk.N, tk.W, tk.E, tk.S))

        self.frm_left_bottom = ttk.Frame(master=self.frm_left_center, style="viewer.TFrame")
        self.frm_left_bottom.grid(column=0, row=1, sticky=(tk.N, tk.W, tk.E, tk.S))

        self.lbl_rear_view_title = ttk.Label(master=self.frm_left_bottom, text="Rear View", anchor="center",
                                             style="viewer.TLabel")
        self.lbl_rear_view_title.grid(column=0, row=2, sticky=(tk.N, tk.W, tk.E, tk.S))

        #rear_img_path = self.get_rear_img_path()
        #logging.debug(f"rear_img_path: {rear_img_path}")
        #self.img_rear_view = self.resize_image(rear_img_path, self.sight_view_size)

        self.img_rear_view = ImageTk.PhotoImage(Image.new('RGB', self.sight_view_size))
        self.lbl_rear_view = ttk.Label(master=self.frm_left_bottom, image=self.img_rear_view, style="image.TLabel")
        self.lbl_rear_view.grid(column=0, row=3, sticky=(tk.N, tk.W, tk.E, tk.S))

        #self.cvs_rear_view = tk.Canvas(master=self.frm_left_bottom, bd=0, highlightthickness=0, bg="white",
        #                               width=self.sight_view_size[0], height=self.sight_view_size[1])
        #self.img_rear_id = self.cvs_rear_view.create_image(0, 0, image=self.img_rear_view, anchor=tk.NW,
        #                                                   tags="REAR_IMG")
        #self.cvs_rear_view.grid(column=0, row=3, sticky=(tk.N, tk.W, tk.E, tk.S))

        self.frm_right_center = ttk.Frame(master=self.frm_center, relief=tk.SOLID, borderwidth=1,
                                          style="viewer.TFrame")
        self.frm_right_center.grid(column=1, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        self.lbl_map_view_title = ttk.Label(master=self.frm_right_center, text="Map View", anchor="center",
                                            style="viewer.TLabel")
        self.lbl_map_view_title.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        self.txt_map_cord = tk.StringVar()
        self.txt_map_cord.set(" ")

        self.lbl_map_cord = ttk.Label(master=self.frm_right_center, textvariable=self.txt_map_cord, anchor=tk.W,
                                            style="map.TLabel")
        self.lbl_map_cord.grid(column=0, row=1, sticky=(tk.N, tk.W, tk.E, tk.S))

        self.img_map_view = ImageTk.PhotoImage(Image.new('RGB', self.map_view_size))
        self.lbl_map_view = ttk.Label(master=self.frm_right_center, image=self.img_map_view, style="image.TLabel")
        self.lbl_map_view.grid(column=0, row=2, sticky=(tk.N, tk.W, tk.E, tk.S))

        #self.cvs_map_view = tk.Canvas(master=self.frm_right_center, bd=0, highlightthickness=0, bg="white",
        #                              width=self.map_view_size[0], height=self.map_view_size[1])
        #self.img_map_id = self.cvs_map_view.create_image(0, 0, image=self.img_map_view, anchor=tk.NW, tags="MAP_IMG")
        #self.cvs_map_view.grid(column=0, row=1, sticky=(tk.N, tk.W, tk.E, tk.S))

        # Bottom Frame
        self.frm_bottom = ttk.Frame(master=self.mainframe, style="viewer.TFrame")
        self.frm_bottom.grid(column=0, row=2, sticky=(tk.N, tk.W, tk.E, tk.S))

        self.frm_bottom_buttons = ttk.Frame(master=self.frm_bottom, style="viewer.TFrame")
        self.frm_bottom_buttons.pack()

        self.btn_start_track = ttk.Button(master=self.frm_bottom_buttons, text="Start Tracking",
                                          style="viewer.TButton",
                                          command=self.start_viewer)
        self.btn_start_track.grid(column=0, row=0, padx=15)

        self.btn_stop_track = ttk.Button(master=self.frm_bottom_buttons, text="Stop Tracking",
                                         style="viewer.TButton",
                                         command=self.stop_viewer)
        self.btn_stop_track.grid(column=1, row=0, padx=15)

        self.btn_exit = ttk.Button(master=self.frm_bottom_buttons, text="Exit",
                                   style="viewer.TButton", command=self.exit_viewer)
        self.btn_exit.grid(column=2, row=0, padx=15)

        for child in self.mainframe.winfo_children():
            child.grid_configure(padx=5, pady=5)

        self.root.mainloop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
                        handlers=[
                            logging.FileHandler("TrackingViewer.log"),
                            logging.StreamHandler()
                        ])
    logging.info("Start Tracking Viewer")

    agent = MapAgent()
    viewer = MapViewer(agent)

    viewer.show_viewer(sys.argv[:1])
