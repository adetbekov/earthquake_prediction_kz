import json
import dvc.api
import requests

from datetime import datetime


def collect(endtime, maxlatitude, minlatitude, maxlongitude, minlongitude, minmagnitude):
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query.geojson"
    params = {
        "starttime": datetime(1800, 1, 1),
        "endtime": endtime,
        "maxlatitude": maxlatitude,
        "minlatitude": minlatitude,
        "maxlongitude": maxlongitude,
        "minlongitude": minlongitude,
        "minmagnitude": minmagnitude,
        "orderby": "time"
    }
    response = requests.get(url, params)
    
    return response.json()


if __name__ == "__main__":
    params = dvc.api.params_show()["collect_data"]
        
    data = collect(
        datetime.strptime(params["endtime"], "%Y-%m-%d"),
        params["maxlatitude"],
        params["minlatitude"],
        params["maxlongitude"],
        params["minlongitude"],
        params["minmagnitude"]
    )
    
    with open('data/data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
