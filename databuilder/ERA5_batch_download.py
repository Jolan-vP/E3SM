import cdsapi
import calendar
from days_test import generate_days_list

c = cdsapi.Client()

MONTHS = [
 "01", "02", "03", "04", "05", "06",
 "07", "08", "09", "10", "11", "12" 
 ]

YEARS = [str(year) for year in range(1940, 2024)]

VARS = [
    "2m_temperature", "total_precipitation", "geopotential"
]

days_list = generate_days_list()

DAYS = days_list
for var in VARS:
    for year in YEARS:
        for month in MONTHS:
            month_int = int(month)  # Convert month to integer
            year_int = int(year)
            num_days = calendar.monthrange(year_int, month_int)[1]
            days_in_month = [f"{day:02d}" for day in range(1, num_days + 1)]
            
            for day in days_in_month: 
                print(f"{year}, {month}, {day}, {var}")   
                result = c.retrieve( "reanalysis-era5-single-levels", {
                    "product_type": "reanalysis",
                    "variable": var,
                    "year": year,
                    "month": month,
                    "day": day, 
                    "time_zone": "00:00",
                    "grid": [0.25/0.25],
                    "area": "global",
                    "format": "netcdf" }, 
                    'ERA5_'+str(year)+'_' +str(month)+ '_' +str(day)+ '_' +str(var)+'.nc' 
                    )
                    
                                                                        
c.download(result) 



# result = c.service ( "...")
# {"lat": [-90, 90], "lon": [-180, 180]}
# "frequency": "1-hourly",