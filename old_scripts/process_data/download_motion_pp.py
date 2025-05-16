import io
import numpy as np
import os
import requests
import xarray as xr

HEM = "sh" # Hemisphere (north or south)

# Enter NASA Earthdata Login Credentials
USER = "jbassham"
PASS = "guJdib-huczi6-jimsuh"

# Enter base download url (leaving off file from path) 
BASE_URL = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0116_icemotion_vectors_v4/{hem}/daily/"

# Enter file (end of url) with placeholder {year}
FNAM = "icemotion_daily_{HEM}_25km_{year}0101_{year}1231_v4.1.nc"

# Define download destination path
PATH_DEST = "/home/jbassham/jack/data/weddell/1992_2020"

# Enter years to download
START_YEAR = 1992
END_YEAR = 2020

def main():

    # Get abbreviated hemisphere for filename
    if HEM == "sh":
        hem = "south"
    else:
        hem = "north"

    # Format base url for hemisphere 
    base_url = BASE_URL.format(hem = hem)

    # Download lat and lon variables from one year
    fnam = FNAM.format(HEM = HEM, year = END_YEAR)

    # Concatenate entire url from base and filename
    url = base_url + fnam

    # Get lat and lon variables from one year
    temp = temp_nasa_earth_data_file(url)
    with xr.open_dataset(temp) as data:
        lat = data['latitude'] # EASE latitude shaped [y, x]
        lon = data['longitude'] # EASE longitude shaped [y, x]

    # Initialize lists for time series data
    u_total = []
    v_total = []
    error_total = []
    time_total = []

    # Define years to process (np.arrange() exclusive of last value)
    do_years = np.arange(START_YEAR, END_YEAR + 1)

    # Iterate through years
    for year in do_years:
        # Enter filename to download for each year in loop
        fnam = FNAM.format(year=year, HEM = HEM)

        url = base_url + fnam

        # Download file at year
        temp = temp_nasa_earth_data_file(url)
        with xr.open_dataset(temp) as data:
            u = data['u'].values                             # horizontal sea ice velocity [t, y, x], cm/s
            v = data['v'].values                             # vertical sea ice velocity [t, y, x], cm/s 
            error = data['icemotion_error_estimate'].values  # Ice motion error estimates [t, y, x]
            time = data['time']  
            
        # Append year's data to list
        u_total.append(u)
        v_total.append(v)
        error_total.append(error)
        time_total.append(time)

        # Confirm download
        print(f"{year} downloaded")

    # Concatenate lists of data along time dimension
    u_total = np.concatenate(u_total, axis = 0)
    v_total = np.concatenate(v_total, axis = 0)
    error_total = np.concatenate(error_total, axis = 0)
    time_total = np.concatenate(time_total, axis = 0)

    # Convert time variable to numpy.datetime64 datatype
    time_total = np.array([np.datetime64(t) for t in time_total])

    # Save time series data as npz variables
    fnam = f"motion_ppv4_EASE_{HEM}_{START_YEAR}_{END_YEAR}"
    path = os.path.join(PATH_DEST, fnam)
    np.savez_compressed(path , u = u_total, v = v_total, error = error_total, time = time_total, lat = lat, lon = lon)
    print(f"Variables Saved at path {path}.npz")

    return  

def temp_nasa_earth_data_file(url):
    """Gets temporary file from Nasa Earth Data Website via URL"""
    ### Create session for NASA Earthdata ###
    # Overriding requests.Session.rebuild_auth to mantain authentication headers when redirected
    # Custom subclass to extend functionality of parent class requests.session to maintain authentication headers
    # when server redirects requests
    class SessionWithHeaderRedirection(requests.Session):
        # Host for which authentication headers maintained
        AUTH_HOST = 'urs.earthdata.nasa.gov'
    
        # Define 'costructor method' for sublclass SessionWithHeaderRedirection  
        # Called to initialze attributes of subclass when created (here with parameters username and password)
        # 'self' parameter is in reference to the instance constructed (our subclass)
        def __init__(self, username, password):
            # Call constructor method for parent class (executing initialization code defined in parent class)
            super().__init__()
            # Initialize authentication atribute in class containing username and password
            self.auth = (username, password)

        # Overrides from the library to keep headers when redirected to or from the NASA auth host
        def rebuild_auth(self, prepared_request, response):
            headers = prepared_request.headers
            url = prepared_request.url

            if 'Authorization' in headers:
                original_parsed = requests.utils.urlparse(response.request.url)
                redirect_parsed = requests.utils.urlparse(url)

                if (original_parsed.hostname != redirect_parsed.hostname) and \
                        redirect_parsed.hostname != self.AUTH_HOST and \
                        original_parsed.hostname != self.AUTH_HOST:
                    del headers['Authorization']

    # Create session with the user credentials that will be used to authenticate access to the data
    session = SessionWithHeaderRedirection(USER, PASS)

    try:
        # submit the request using the session
        response = session.get(url, stream=True)
        # '200' means success
        StatusCode = response.status_code
        print(StatusCode)
        # raise an exception in case of http errors
        response.raise_for_status()  

        # Read response content to temp using BytesIO object
        temp = io.BytesIO(response.content)

        return temp

    except requests.exceptions.HTTPError as e:
        # Handle any errors here
        print(f"HTTP Error: {e}")
        return None
    
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    main()
