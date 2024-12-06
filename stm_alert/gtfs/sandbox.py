from google.transit import gtfs_realtime_pb2
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from stm_alert.gtfs.constants import VEHICLE_POSITION_API_ENDPOINT
from stm_alert.gtfs.utils import load_config

config = load_config("montreal_107_est")
api_key = config["api_key"]

feed = gtfs_realtime_pb2.FeedMessage()
request = Request(VEHICLE_POSITION_API_ENDPOINT)
request.add_header("apiKey", api_key)
try:
    response = urlopen(request)
except HTTPError as e:
    print(f"HTTP error: {e.code}")
    print(f"Error message: {e.read().decode('utf-8')}")
    exit(1)

feed.ParseFromString(response.read())
for i, entity in enumerate(feed.entity):
    if entity.HasField("vehicle") and entity.vehicle.trip.route_id == "107":
        print(entity)
        pos = entity.vehicle.position
        print(f"{i} lat: {pos.latitude}, long: {pos.longitude}")
