import argparse
from google.transit import gtfs_realtime_pb2
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from stm_alert.gtfs.constants import VEHICLE_POSITION_API_ENDPOINT
from stm_alert.gtfs.utils import load_config
import asyncio
from asyncio import Queue
import signal
import numpy as np
from stm_alert import get_path_from_project_root
import time
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from datetime import datetime


QUERY_FREQUENCY_SECONDS = 10
SAVE_FREQUENCY_MINUTES = 60
SAVE_LOCATION = get_path_from_project_root() / "gtfs" / "datasets"

def parse_args() -> argparse.Namespace:
    """Parse the arguments

    Returns:
        argparse.Namespace: The arguments
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()

async def collect(trip_ids_to_collect: list[str], request_queue: Queue, response_queue: Queue) -> None:
    """Collect the data from the STM API

    Args:
        config (dict): The configuration
        request_queue (Queue): The queue to send requests to
        response_queue (Queue): The queue to send responses to
    
    Returns:
        None
    """
    # set up the STM API feed
    feed = gtfs_realtime_pb2.FeedMessage()
    request = Request(VEHICLE_POSITION_API_ENDPOINT)
    request.add_header("apiKey", config["api_key"])
    
    while True:
        await asyncio.sleep(QUERY_FREQUENCY_SECONDS)
        # check for signals from main or signal handler
        if not request_queue.empty():
            item = await request_queue.get()
            if item == "exit":
                logger.warning("Collector received exit signal")
                await response_queue.put("exit")
                break
            
        try:
            response = urlopen(request)
        except HTTPError as e:
            logger.error(f"HTTP error: {e.code}")
            logger.error(f"Error message: {e.read().decode('utf-8')}")
            await response_queue.put("exit")
            break
        feed.ParseFromString(response.read())
        for i, entity in enumerate(feed.entity):
            if entity.HasField("vehicle") and (str(entity.vehicle.trip.trip_id) in trip_ids_to_collect):
                return_data = {
                    "trip_id": str(entity.vehicle.trip.trip_id),
                    "timestamp": entity.vehicle.timestamp,
                    "latitude": entity.vehicle.position.latitude,
                    "longitude": entity.vehicle.position.longitude
                }
                await response_queue.put(return_data)
        
        
async def kill_collect(request_queue: Queue) -> None:
    """Kill the collector

    Args:
        request_queue (Queue): The queue to send requests to
    
    Returns:
        None
    """
    await request_queue.put("exit")
    
async def main(config: dict) -> None:
    """Collects new data from the collector, collates and periodically saves the data

    Args:
        config (dict): The configuration
    
    Returns:
        None
    """
    # set up queues to facilitate communication between main and collector
    request_queue = asyncio.Queue()
    response_queue = asyncio.Queue()

    # Add a signal handler to the event loop that will send a poison pill to the collector
    loop = asyncio.get_running_loop()
    for signame in ('SIGINT', 'SIGTERM'):
        loop.add_signal_handler(
            getattr(signal, signame),
            lambda: asyncio.create_task(kill_collect(request_queue))
        )
    trip_ids_to_collect = _get_trip_ids(config)
    # create a task to run the collector
    collector = asyncio.create_task(collect(trip_ids_to_collect, request_queue, response_queue))
    
    # the collector will asyncronously put items into the response queue
    # main just needs to get items from the response queue for the time being,
    # the only request to the request queue is to send a poison pill to the collector
    # which is currently handled by the signal handler
    
    bus_number = config["bus_number"]
    bus_direction = config["bus_direction"]
    save_file = SAVE_LOCATION / f"{bus_number}_{bus_direction}.npz"
    save_data = {}
    
    start_time = time.time()
    while True:
        new_data = await response_queue.get()
        time_elapsed = time.time() - start_time
        if time_elapsed > SAVE_FREQUENCY_MINUTES * 60 or new_data == "exit":
            _save_data(save_file, save_data)
            start_time = time.time()
        if new_data == "exit":
            break
        new_data["trip_id"] = _now() + "-" + new_data["trip_id"]
        logger.debug(f"Collected data for {new_data['trip_id']}")
        if new_data["trip_id"] in save_data:
            _update_save_data(save_data, new_data)
        else:
            _add_to_save_data(save_data, new_data)
        
def _get_trip_ids(config: dict) -> list[str]:
    """Get the trip ids to collect

    Args:
        config (dict): The configuration
    
    Returns:
        list[str]: The trip ids to collect
    """
    info_file = get_path_from_project_root() / "gtfs" / "stm_info" / "trips.txt"
    logger.info(f"Loading trip ids from {info_file}")
    trip_ids = []
    with open(info_file, "r") as f:
        for line in tqdm(f.readlines()):
            route_id, service_id, trip_id, trip_headsign, direction_id, shape_id, wheelchair_accessible, note_fr, note_en = line.split(",")
            if route_id == str(config["bus_number"]) and trip_headsign == config["bus_direction"]:
                trip_ids.append(trip_id)
    return trip_ids

def _save_data(save_file: Path, save_data: dict) -> None:
    """Save the data to a numpy file
    
    Args:
        save_file (Path): The file to save the data to
        save_data (dict): The data to save
    
    Returns:
        None
    """
    logger.info(f"Saving data to {save_file}")
    existing_data = dict(np.load(save_file)) if save_file.exists() else {}
    existing_data.update({k: np.array(v) for k, v in save_data.items()})
    np.savez(save_file, **existing_data)

def _now() -> str:
    """Get the current date and time in the format YYYYMMDD

    Returns:
        str: The current date and time
    """
    return datetime.now().strftime("%Y%m%d")

def _update_save_data(save_data: dict, new_data: dict) -> None:
    """Update the save data

    Args:
        save_data (dict): The data to update
        new_data (dict): The new data
    
    Returns:
        None
    """
    last_data = save_data[new_data["trip_id"]][-1]
    if last_data[0] < new_data["timestamp"]:
        save_data[new_data["trip_id"]].append([float(new_data["timestamp"]), float(new_data["latitude"]), float(new_data["longitude"])])

def _add_to_save_data(save_data: dict, new_data: dict) -> None:
    """Add the new data to the save data

    Args:
        save_data (dict): The data to update
        new_data (dict): The new data
    
    Returns:
        None
    """
    save_data[new_data["trip_id"]] = [[float(new_data["timestamp"]), float(new_data["latitude"]), float(new_data["longitude"])]]

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    asyncio.run(main(config))
