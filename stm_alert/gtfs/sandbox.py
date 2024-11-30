from google.transit import gtfs_realtime_pb2
from urllib.request import Request, urlopen
from stm_alert.gtfs.constants import VEHICLE_POSITION_API_ENDPOINT

api_key = "l7f6c826a7eeb34fb8ba511c679b121c37"

feed = gtfs_realtime_pb2.FeedMessage()
request = Request(VEHICLE_POSITION_API_ENDPOINT)
request.add_header("apiKey", api_key)
response = urlopen(request)

print(f"Status: {response.status_code}")

feed.ParseFromString(response.read())
for i, entity in enumerate(feed.entity):
    if entity.HasField("vehicle") and entity.vehicle.HasField("position"):
        pos = entity.vehicle.position
        print(f"{i} lat: {pos.latitude}, long: {pos.longitude}")
