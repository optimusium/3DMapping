import rpyc
from rpyc.utils.server import ThreadedServer
from queue import Queue
import logging
from threading import Thread
import time

class MapState():

    def __init__(self):
        # x coordinate, y coordinate, orientation at xy plan
        self.pos_x = 0
        self.pos_y = 0
        self.ort_xy = 0

        self.front_img = None
        self.rear_img = None
        self.map_img = None

        # Whether request has error
        self.req_state_err = ""
        # Whether response has error
        self.resp_state_err = ""


class ViewerService(rpyc.Service):

    def __init__(self, port_num=18111):
        self._req_q = Queue()
        self._resp_q = Queue()
        self.port = port_num

    def on_connect(self, conn):
        pass

    def on_disconnect(self, conn):
        pass

    # TrackingViewer request view map from backend service
    def exposed_request_view(self, params={}):
        map_state = MapState()

        # x, y coordinates
        if "pos" in params:
            pos_x, pos_y, ort_xy = params["pos"]
            map_state.pos_x = pos_x
            map_state.pos_y = pos_y
            map_state.ort_xy = ort_xy

        try:
            self._req_q.put(map_state, timeout=0.1)
            map_state.req_state_err = "request ok"
        except:
            map_state.req_state_err = "request error"
            logging.error(f"Cannot put request in queue: {dir(map_state)}", exc_info=True)

        return {"pos": (map_state.pos_x, map_state.pos_y, map_state.ort_xy), "state_err": map_state.req_state_err}

    # TrackingViewer receive view map results from backend service
    def exposed_receive_view(self):
        map_state = MapState()

        if self._resp_q.empty():
            map_state.resp_state_err = "empty queue"
            return {"state_err": map_state.resp_state_err,
                    "pos": (map_state.pos_x, map_state.pos_y, map_state.ort_xy),
                    "front_img": map_state.front_img, "rear_img": map_state.rear_img,
                    "map_img": map_state.map_img}

        try:
            map_state = self._resp_q.get(timeout=0.1)
            map_state.resp_state_err = "response ok"
        except:
            map_state.resp_state_err = "response error"
            logging.error(f"Cannot get request in queue", exc_info=True)

        return {"state_err": map_state.resp_state_err,
                "pos": (map_state.pos_x, map_state.pos_y, map_state.ort_xy),
                "front_img": map_state.front_img, "rear_img": map_state.rear_img,
                "map_img": map_state.map_img}

    # Processed3dView retrieve view task from backend service
    def exposed_retrieve_view_task(self):
        map_state = MapState()

        if self._req_q.empty():
            map_state.req_state_err = "empty queue"
            return {"pos": (map_state.pos_x, map_state.pos_y, map_state.ort_xy),
                     "state_err": map_state.req_state_err}

        try:
            # Always to get latest task and ignore other tasks
            while not self._req_q.empty():
                map_state = self._req_q.get(timeout=0.1)

            map_state.req_state_err = "ok"
        except:
            map_state.req_state_err = "error"
            logging.error(f"Cannot retrieve view task", exc_info=True)

        return {"pos": (map_state.pos_x, map_state.pos_y, map_state.ort_xy),
                "state_err": map_state.req_state_err}

    # Processed3dView complete view task and save results to backend service
    def exposed_complete_view_task(self, view_task_state=None):
        map_state = MapState()

        if view_task_state is None:
            map_state.resp_state_err = "task error"
            logging.error("Failed to completed view task")
        else:
            map_state.pos_x, map_state.pos_y, map_state.ort_xy = view_task_state["pos"]
            map_state.front_img = view_task_state["front_img"]
            map_state.rear_img = view_task_state["rear_img"]
            map_state.map_img = view_task_state["map_img"]

            logging.info(f"Received with completed view task: {map_state}")

        try:
            # Always to get latest task and ignore other tasks
            self._resp_q.put(map_state, timeout=0.1)
            map_state.resp_state_err = "ok"
        except:
            map_state.resp_state_err = "error"
            logging.error(f"Cannot save completed view task", exc_info=True)

        return {"state_err": map_state.resp_state_err, "pos": (map_state.pos_x, map_state.pos_y, map_state.ort_xy)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
                        handlers=[
                            logging.FileHandler("ViewerBackendService.log"),
                            logging.StreamHandler()
                        ])
    logging.info("Start Viewer Backend Service")

    viewer_service = ViewerService()

    t = ThreadedServer(viewer_service, port=viewer_service.port)
    t.start()