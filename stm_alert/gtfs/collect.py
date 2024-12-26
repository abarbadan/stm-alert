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

SAVE_FREQUENCY_MINUTES = 60
SAVE_LOCATION = get_path_from_project_root() / "gtfs" / "datasets"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()

async def collect(config: dict, request_queue: Queue, response_queue: Queue):
    # set up the STM API feed
    feed = gtfs_realtime_pb2.FeedMessage()
    request = Request(VEHICLE_POSITION_API_ENDPOINT)
    request.add_header("apiKey", config["api_key"])
    
    while True:
        await asyncio.sleep(5)
        # check for signals from main or signal handler
        if not request_queue.empty():
            item = await request_queue.get()
            if item == "exit":
                print("Collector received exit signal")
                await response_queue.put("exit")
                break
            
        # Simulate some processing
        try:
            response = urlopen(request)
        except HTTPError as e:
            print(f"HTTP error: {e.code}")
            print(f"Error message: {e.read().decode('utf-8')}")
            await response_queue.put("exit")
            break
        feed.ParseFromString(response.read())
        for i, entity in enumerate(feed.entity):
            if entity.HasField("vehicle") and entity.vehicle.trip.route_id == "107":
                pos = entity.vehicle.position
                await response_queue.put(f"{i} lat: {pos.latitude}, long: {pos.longitude}")
        
        
async def kill_collect(request_queue: Queue):
    await request_queue.put("exit")
    
async def main(config: dict):
    
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
    
    # create a task to run the collector
    collector = asyncio.create_task(collect(config, request_queue, response_queue))
    
    # the collector will asyncronously put items into the response queue
    # main just needs to get items from the response queue for the time being,
    # the only request to the request queue is to send a poison pill to the collector
    # which is currently handled by the signal handler
    
    while True:
        new_data = await response_queue.get()
        if new_data == "exit":
            break
        # do something with the data
        print(new_data)
        
        
if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    asyncio.run(main(config))
