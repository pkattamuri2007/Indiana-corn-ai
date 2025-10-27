# NASA POWER Multiprocessing (Points) - Tippecanoe County, Indiana
# Author: [Your Name]
# Purpose: Retrieve 10 years of daily temperature, precipitation, and humidity data for corn yield modeling

import os, sys, time, json, urllib3, requests, multiprocessing

urllib3.disable_warnings()

def download_function(collection):
    request, filepath = collection
    try:
        response = requests.get(url=request, verify=False, timeout=30.00).json()
        with open(filepath, 'w') as file_object:
            json.dump(response, file_object)
    except Exception as e:
        print(f"Error downloading {filepath}: {e}")

class Process():

    def __init__(self):
        self.processes = 5  # No more than 5 concurrent requests
        self.start_date = "20150101"
        self.end_date = "20251231"

        # Agricultural variables: temperature, precipitation, humidity, solar radiation
        self.request_template = (
            "https://power.larc.nasa.gov/api/temporal/daily/point?"
            "parameters=T2M,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN"
            f"&community=AG&longitude={{longitude}}&latitude={{latitude}}"
            f"&start={self.start_date}&end={self.end_date}&format=JSON"
        )

        self.filename_template = "PowerData_Lat_{latitude}_Lon_{longitude}.json"

    def execute(self):
        Start_Time = time.time()

        # 3x3 grid across Tippecanoe County (~2.5km spacing)
        lats = [40.33, 40.38, 40.43, 40.48, 40.53]
        lons = [-87.00, -86.95, -86.90, -86.85, -86.80]
        locations = [(lat, lon) for lat in lats for lon in lons]

        requests_list = []
        for latitude, longitude in locations:
            request = self.request_template.format(latitude=latitude, longitude=longitude)
            filename = self.filename_template.format(latitude=latitude, longitude=longitude)
            requests_list.append((request, filename))

        requests_total = len(requests_list)
        pool = multiprocessing.Pool(self.processes)
        results = pool.imap_unordered(download_function, requests_list)

        for i, _ in enumerate(results, 1):
            sys.stderr.write(f'\rDownloading {i}/{requests_total}')
            sys.stderr.flush()

        print(f"\n✅ Completed in {round((time.time() - Start_Time), 2)} seconds.")

if __name__ == '__main__':
    Process().execute()
