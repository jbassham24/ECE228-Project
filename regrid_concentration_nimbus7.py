import io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import requests
import xarray as xr # With h5netcdf

# Regrids time series of NSIDC Sea Ice Concentrations (Nimbus 7)
# Data accessed from https://nsidc.org/data/nsidc-0051/versions/2
# Grid info accessed from https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0771_polarstereo_anc_grid_info/
# Processes entire time series from .npz file downloaded using '01_con_nimbus7_dload.py'
# ***credit source here***

# Hemisphere (sh or nh)
HEM = "sh"

# Enter years to regrid (must be consitent with .npz downloaded)
START_YEAR = 1992
END_YEAR = 2020

# NOTE WEDDELL SEA BOUNDS 
LAT_LIMITS = [-80, -62] # Enter South to North (coverage 29.7N to 90N or -90S to -37S)
LON_LIMITS = [-180, 180] # Enter West to East (coverage -180 W to 180E)

# # Enter bounds for lat and lon (deg)
# LAT_LIMITS = [70, 90] # Enter South to North (coverage 29.7N to 90N or -90S to -37S)
# LON_LIMITS = [-180, 180] # Enter West to East (coverage -180 W to 180E)

RESOLUTION = 25 # Grid resolution, consistent with polar pathfinder velocities (km)

# Enter URL for Polar Stereographic 25km resolution lat lon grid
URL_GRID = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0771_polarstereo_anc_grid_info/NSIDC0771_LatLon_PS_{hem}25km_v1.0.nc"
# Enter NASA Earthdata Login Credentials
USER = "jbassham"
PASS = "guJdib-huczi6-jimsuh"

# Enter data source path
PATH_SOURCE = "/home/jbassham/jack/data/weddell/1992_2020"

# Enter data file to regrid
FNAM = "con_nimbus7_ps_{HEM}_{START_YEAR}_{END_YEAR}.npz"

# Enter destination path
PATH_DEST = PATH_SOURCE

def main():
    
    if HEM == 'nh':
        hem = "N"
    else:
        hem = "S"

    url_grid = URL_GRID.format(hem = hem)

    # Load original grid in temp file
    temp = temp_nasa_earth_data_file(url_grid)

    if temp is not None:
        with xr.open_dataset(temp) as data:
            lat_old = data['latitude'].values # Polar Stereographic
            lon_old = data['longitude'].values # Polar Stereographic

    else:
        print("Error: original grid not loaded")

    # Format filename
    filename = FNAM.format(START_YEAR = START_YEAR, END_YEAR = END_YEAR, HEM = HEM)

    # Attempt to load the original .npz file
    try:
        # Load original .npz file
        data = np.load(os.path.join(PATH_SOURCE, filename), allow_pickle=True)

        # Attempt to access variables
        ic_old = data['ic'] # ice concentration on gaussian grid
        time = data['time'] # time series dates dt64
        var_names = data['var_names'] # variable names, based on sensors

        print(f"{filename} loaded successfully.")
        
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found in '{PATH_SOURCE}'.")
    except KeyError as e:
        print(f"Error: Missing expected data key: {e}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}.")

    # Get indices for new lat lon grid
    jj, ii, lat_new, lon_new = nearest_neighbor_interpolation(RESOLUTION, LAT_LIMITS, LON_LIMITS, lat_old, lon_old)

    # Initialize data arrays for current year
    dims = (len(time), len(lat_new), len(lon_new))
    ic_new = np.zeros(dims) # ice concentation (part coverage 0 to 1)

    # Iterate through gridpoints
    for i in range(dims[2]):
        for j in range(dims[1]):

            # Extract nearest neighbor indices for each new gridpoint
            iii = ii[j,i]
            jjj = jj[j,i]
            
            # Extract data from nearest neighbor index
            ic_new[:,j,i] = ic_old[:,jjj,iii]

    # Create new filename for regrided lat lon data
    fnam = filename.replace("ps", "latlon")
    
    # Save regrided lat lon data
    np.savez_compressed(os.path.join(PATH_DEST, fnam), ic = ic_new, time = time, lat = lat_new, lon = lon_new, var_names = var_names)
    print(f"Variables Saved at path {PATH_DEST + fnam}")

    # # Compare regrid and old ice concentration
    # fnam = fnam.replace(".npz", "_grid_compare.mp4")
    # save_path = os.path.join(PATH_DEST,fnam)
    # compare_grids(ic_new, lat_new, lon_new, ic_old, lat_old, lon_old,  
    #            LAT_LIMITS, LON_LIMITS, time = time, main_title = "Ice Concentration", save_path = save_path)
  
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
    

def nearest_neighbor_interpolation(res, lat_limits, lon_limits, lat_old, lon_old):
    """
    Returns new regular lat lon coordinate arrays given input resolution in km (based on polar 
    ice veolocity product) and bounds in degrees
    
    Uses nearest neighbor interpolation to return interplation indices for regridding 
    old data to new regular lat lon grid

    Accounts for periodicity in old longitude values

    New lat lon grid represented by (-90S, 90N) and (-180W, 180E)

    """

    # Convert resolution to degrees latitude)
    yres = res / 111   # latitude resolution (deg)
    xres = res / (111*np.cos(np.radians((lat_limits[0]+lat_limits[1])/2))) # longitude resolution (deg) based on average latitude

    # Create arrays for new lat and lon grid
    lat_new = np.arange(lat_limits[0], lat_limits[1] + yres, yres) # Latitude
    lon_new = np.arange(lon_limits[0], lon_limits[1] + xres, xres) # Longitude
    nlat = len(lat_new)
    nlon = len(lon_new)
    
    # Initialize arrays for interpolation indices
    jj = np.zeros((nlat,nlon), dtype=int)
    ii = np.zeros((nlat,nlon), dtype=int)

    # Iterate through new grid's lat and lon points
    for j in range(nlat):
        for i in range(nlon):

            # Calculate meridional distances
            dy = (lat_new[j]-lat_old)**2

            # Find absolute value distances of i'th lat from entire lat_ease array and store in array  
            dx = (lon_new[i]-lon_old)**2

            # Calculate zonal distances considering periodicity in longitude
            dx1 = (lon_new[i] - lon_old) ** 2
            dx2 = (lon_new[i] - lon_old + 360) ** 2
            dx3 = (lon_new[i] - lon_old - 360) ** 2
            
            # Find the minimum distance for longitude
            dx = np.minimum(dx1, np.minimum(dx2, dx3))
        
            # Find distances
            ds = dx + dy

            # Find indices of minimum ds value
            i_neighbors = np.where(ds == np.min(ds))

            # Take minium of lat and lon indices (lower left corner) for consistency, store in array
            jj[j,i] = np.min(i_neighbors[0])
            ii[j,i] = np.min(i_neighbors[1])

    # Return interpolation indices and new lat and lon coordinate variables
    return jj, ii, lat_new, lon_new


def crop_2Dlatlon(data_old, lat_old, lon_old, lat_limits, lon_limits):
    """
    Crops data with 2D lat and lon variables (where lat and lon are used as coordinate variables)
    """
    
    # Check that lon range in (-180, 180)
    if np.any(lon_old > 180):
        lon_old = np.where(lon_old > 180, lon_old - 360, lon_old)  # Convert from 0-360 to -180-180
    elif np.any(lon_old < -180):
        raise ValueError("Longitude values must be in the range (-180, 180).")

    # * Extract j indices along [0]th dimension
    j = np.unique(np.where((lat_old >= lat_limits[0]) & (lat_old <= lat_limits[1]) & (lon_old >= lon_limits[0]) & (lon_old <= lon_limits[1]))[0])
    # * Extract i indices along [1]th dimension
    i = np.unique(np.where((lat_old >= lat_limits[0]) & (lat_old <= lat_limits[1]) & (lon_old >= lon_limits[0]) & (lon_old <= lon_limits[1]))[1])
    
    lat_crop = lat_old[j,:][:,i]
    lon_crop = lon_old[j,:][:,i]
    data_crop = data_old[:,j,:][:,:,i]

    return data_crop, lon_crop, lat_crop


def animated_time_series(data_values, time = None, 
                        main_title = None, titles = None, y_labels = None, x_labels = None, c_labels = None, 
                        vmin = None, vmax = None, cmap = 'viridis', interval = 200, 
                        save_path = None):
    """
    
    Animates multiple subplots of time series data to check regrid, etc
    
    Input Parameters:
    - data: list of data shaped [time, y, x]
    - y_values, x_values: lists of coordinate variables, ie: lat, lon
    - sup_title: main title for plot
    - titles, y_labels, x_labels: as lists for each subplot
    - c_labels: list of colorbar labels
    - vmin, vmax: min and max colorbar values
    - interval: time delay between frames, ms
    - save_path

    """

    # Number of plots based on number of c data inputs
    nplots = len(data_values)

    # Create subplot grid based on number of plots
    fig, axs = plt.subplots(1, nplots, figsize = (6 * nplots, 6))

    # Handle case for one plot
    if nplots == 1:
        # axs not a list
        axs = [axs]

    # Create default case for titles and lables
    if titles is None:
        titles = [f"Data[{i+1}]" for i in range(nplots)]
    
    if x_labels is None:
        x_labels = ["x"] * nplots

    if y_labels is None:
        y_labels = ["y"] * nplots


    # Create time strings for title if time provided
    if time is not None:
        # Convert numpy.datetime64 to string in format "DD MMM YYYY"
        time_strings = np.array([t.astype('datetime64[D]').astype(str) for t in time])
    
    if c_labels is None:
        c_labels = ["Value"] * nplots

    plot_objs = [] # List to hold plot objects

    # Plot initial frame for each subplot
    for i, data in enumerate(data_values):
        plot_obj = axs[i].pcolormesh(data[0,:,:], cmap=cmap, vmin=vmin, vmax=vmax)
        axs[i].set_title(titles[i])
        axs[i].set_xlabel(x_labels[i])
        axs[i].set_ylabel(y_labels[i])
        fig.colorbar(plot_obj, ax = axs[i], label = c_labels[i])
        plot_objs.append(plot_obj)

    # Update frame at each time step
    def update(frame):
        for i, data in enumerate(data_values):
            plot_objs[i].set_array(data[frame].ravel())
            # Create main title if provided

        # Create main title based on conditions if provided
        if main_title is not None:
            if time is not None:
                # Create array of date strings if time provided
                fig.suptitle(f"{main_title} {time_strings[frame]}", fontsize = 16, fontweight = 'bold')
            else:
                fig.suptitle(main_title, fontsize = 16, fontweight = 'bold')
        elif time is not None:
            fig.suptitle(time_strings[frame], fontsize = 16, fontweight = 'bold')

        return plot_objs
        
    # Create animation
    # Number of frames based on first data's time dimension
    plot_animated = animation.FuncAnimation(fig, update, frames=data_values[0].shape[0], interval=interval)

    # Save animation if save path exists
    if save_path:
        writer = animation.FFMpegWriter()
        plot_animated.save(save_path, writer = writer)

    return plot_animated


def compare_grids(data_new, lat_new, lon_new, data_old, lat_old, lon_old,  
                  lat_limits, lon_limits, time = None, main_title = None, save_path = None):
    """

    """

    # Crop old data to bounds used for regrid data
    data_old_crop, lat_old_crop, lon_old_crop =  crop_2Dlatlon(data_old, lat_old, lon_old, lat_limits, lon_limits)

    # Plot an animation of new and old grids
    data_values = [data_new, data_old_crop]
    titles = ["New Lat Lon Grid", "Old Grid"]


    animated_time_series(data_values, time = time, 
                        main_title = main_title, titles = titles, y_labels = None, x_labels = None, c_labels = None, 
                        vmin = None, vmax = None, cmap = 'viridis', interval = 200, 
                        save_path = save_path)

    return

if __name__ == "__main__":
    main()
