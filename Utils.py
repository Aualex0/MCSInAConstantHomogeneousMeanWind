############################## CODE: Alexis Aubel (alexis.aubel@gmail.com) ##############################
#                                                                                                       #
#          All functions used in other notebooks, mainly for imports and computing barycenters          #
#########################################################################################################

import xarray as xr
import numpy as np
import matplotlib.animation as animation


###############################################################################################
######################################### Import data #########################################
###############################################################################################
def import_from_name(sim_names: list):
    directory = '/fs3/group/mullegrp/Alexis_Aubel/NETCDF_files/'
    list_path_1D = [directory + sim_name + '/RCE_MC_' + sim_name + '.nc' for sim_name in sim_names]
    list_path_2D = [directory + sim_name + '/RCE_MC_' + sim_name + '_64.2Dcom_1.nc' for sim_name in sim_names]
    return import_from_path(list_path_1D), import_from_path(list_path_2D)

def import_from_path(path_names: list):
    datasets = []
    for path in path_names:
        datasets.append(xr.open_dataset(path))
        #In case of file of over 1000 time save, the saved file is divided into 2Dcom_1.nc, _2.nc, etc. Here we take it into account for up to 2Dcom_9.nc, but the merge is very slow.
        if '.2Dcom_1.nc' in path:
            try:
                for i in range(2,10):
                    datasets[-1] = xr.merge([datasets[-1], xr.open_dataset(path[:-4]+str(i)+'.nc')])
            except FileNotFoundError:
                pass
    return datasets

def import_3D_from_name(sim_names: list):
    directory = '/fs3/group/mullegrp/Alexis_Aubel/NETCDF_files/'
    #Parametered here for a maximum of 30 days. Over that, need to extend the timing list.
    timings = ['000' + str(i) for i in range(360,1000,360)] + ['00' + str(i) for i in range(1080,10000,360)] + ['0' + str(i) for i in range(10080,100000,360)] + [str(i) for i in range(100080,259201,360)]
    list_path_3D = [[directory + sim_name + '/RCE_MC_' + sim_name + '_64_0000' + time + '.nc' for time in timings[::6]] for sim_name in sim_names]
    return [import_3D_from_path(path_list) for path_list in list_path_3D]
    

def import_3D_from_path(path_names: list):
    datasets = []
    for path in path_names:
        try:
            datasets.append(xr.open_dataset(path))
        except OSError:
            if datasets==[]:
                return None
            break
    return datasets


###############################################################################################
######################################### Parameters ##########################################
###############################################################################################
def extract_type(sim_name: str):
    sepI, sepD = sim_name.count('I'), sim_name.count('D')
    if sepI+sepD==0:
        raise Exception("Warning: the simulation name '"+sim_name+"' does not contain a simulation type letter (I or D).")
    elif sepI+sepD>1:
        raise Exception("Warning: the simulation name '"+sim_name+"' contains multiple simulation type letters (I or D).")
    else:
        return 'I' if sepI==1 else 'D'

def extract_speed(sim_name: str):
    sep = extract_type(sim_name)
    substring=sim_name.split(sep)[1]
    speed=float(substring[0]+'.'+substring[2])
    return speed

def get_sim_desc(sim_name: str):
    wind_type=extract_type(sim_name)
    wind_speed=extract_speed(sim_name)
    return wind_type + "%.1f"%wind_speed

def get_U_target(sim_name: str):
    wind_type=extract_type(sim_name)
    wind_speed=extract_speed(sim_name)
    if wind_type is None or wind_speed is None:
        return None
    match wind_speed:
        case 0.7:
            U_target = np.array(([0.7 for _ in range(40)] if wind_type=='I' else [0 for _ in range(7)] + [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7] + [0.7 for _ in range(25)]) + [0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0] + [0 for _ in range(16)])
        case 1.5:
            U_target = np.array(([1.5 for _ in range(40)] if wind_type=='I' else [0 for _ in range(7)] + [0.0,0.1,0.3,0.6,0.9,1.2,1.4,1.5] + [1.5 for _ in range(25)]) + [1.5,1.4,1.2,0.9,0.6,0.3,0.1,0.0] + [0 for _ in range(16)])
        case 2.0:
            U_target = np.array(([2.0 for _ in range(40)] if wind_type=='I' else [0 for _ in range(7)] + [0.0,0.1,0.4,0.7,1.0,1.3,1.6,2.0] + [2.0 for _ in range(25)]) + [2.0,1.6,1.3,1.0,0.7,0.4,0.1,0.0] + [0 for _ in range(16)])
        case 4.0:
            U_target = np.array(([4.0 for _ in range(40)] if wind_type=='I' else [0 for _ in range(7)] + [0.0,0.1,0.5,1.0,1.5,2.0,3.0,4.0] + [4.0 for _ in range(25)]) + [4.0,3.0,2.0,1.5,1.0,0.5,0.1,0.0] + [0 for _ in range(16)])
        case _:
            raise Exception(str(wind_speed) + " n'est pas une vitesse connue.")
    return U_target


###############################################################################################
########################################## Computing ##########################################
###############################################################################################
def compute_barycenter_list(data, GRID_SIZE, tolerance: float = 0.7):
    centers = []
    for t,frame in enumerate(data):
        center = compute_barycenter(frame, GRID_SIZE, tolerance)
        if center == False:
            print("Warning: We stopped computing barycenter from the " + str(t) + "th frame on, because the mesoscale structure was becoming too spread out. Try raising the tolerance closer to 1 for better results.")
            return centers
        else:
            centers.append(center)

        while t!=0 and abs(centers[t][0] - centers[t-1][0]) > GRID_SIZE/2:
            if centers[t][0] < centers[t-1][0]:
                centers[t][0] += GRID_SIZE
            else:
                centers[t][0] -= GRID_SIZE

    return centers

def compute_barycenter(frame, GRID_SIZE, tolerance: float = 0.5):
    MAX = np.max(frame)
    fr_copy = np.copy(frame)
    fr_copy[fr_copy<tolerance*MAX] = 0
    fr_avg_x = np.mean(fr_copy, axis = 0)
    fr_avg_y = np.mean(fr_copy, axis = 1)
    x_coords = np.linspace(0, GRID_SIZE, 256)
    y_coords = np.linspace(0, GRID_SIZE, 256)
    try:
        border_x = np.where(fr_avg_x==0)[0][0]/256*GRID_SIZE
        border_y = np.where(fr_avg_y==0)[0][0]/256*GRID_SIZE
        x_coords[x_coords > border_x] -= GRID_SIZE
        y_coords[y_coords > border_y] -= GRID_SIZE

        x_mean = np.average(x_coords, weights=fr_avg_x)
        if x_mean < 0 :
            x_mean += GRID_SIZE
        y_mean = np.average(y_coords, weights=fr_avg_y)
        if y_mean < 0:
            y_mean += GRID_SIZE
    except (ZeroDivisionError, IndexError):
        return False
    
    return [x_mean, y_mean]

def compute_dimensions_list(data, GRID_SIZE, tolerance: float = 0.85):
    centers = []
    for t,frame in enumerate(data):
        center = compute_lim(frame, GRID_SIZE, tolerance)
        if center == False:
            print("Warning: We stopped computing limits from the " + str(t) + "th frame on, because the mesoscale structure was becoming too spread out. Try raising the tolerance closer to 1 for better results.")
            return centers
        else:
            centers.append(center)

    return centers
     
def compute_lim(frame, GRID_SIZE, tolerance: float = 0.5):
    #This function doesn't work if the MCS spreads over the limit of the domain (periodicity). This generates outlier values for x-extension.
    MAX = np.max(frame)
    fr_copy = np.copy(frame)
    fr_copy[fr_copy<tolerance*MAX] = 0
    fr_avg_x = np.mean(fr_copy, axis = 0)
    fr_avg_y = np.mean(fr_copy, axis = 1)
    x_coords = np.linspace(0, GRID_SIZE, 256)
    y_coords = np.linspace(0, GRID_SIZE, 256)
    try:
        border_x = np.where(fr_avg_x==0)[0][0]/256*GRID_SIZE
        border_y = np.where(fr_avg_y==0)[0][0]/256*GRID_SIZE
        x_coords[x_coords > border_x] -= GRID_SIZE
        y_coords[y_coords > border_y] -= GRID_SIZE

        barrier_minInd = (fr_copy!=0).argmax(axis=1)
        x_min = x_coords[0] if np.max(barrier_minInd)==0 else x_coords[np.min(barrier_minInd[barrier_minInd!=0])]
        if x_min<0:
            x_min+=GRID_SIZE
        barrier_minInd = (fr_copy!=0).argmax(axis=0)        
        y_min = y_coords[0] if np.max(barrier_minInd)==0 else y_coords[np.min(barrier_minInd[barrier_minInd!=0])]
        if y_min<0:
            y_min+=GRID_SIZE
            
        fr_copy=fr_copy[::-1,::-1]
        barrier_maxInd = (fr_copy!=0).argmax(axis=1)
        x_max = x_coords[255] if np.max(barrier_maxInd)==0 else x_coords[255-np.min(barrier_maxInd[barrier_maxInd!=0])]
        if x_max<x_min:
            x_max+=GRID_SIZE
        barrier_maxInd = (fr_copy!=0).argmax(axis=0)
        y_max = y_coords[255] if np.max(barrier_maxInd)==0 else y_coords[255-np.min(barrier_maxInd[barrier_maxInd!=0])]
        if y_max<y_min:
            y_max+=GRID_SIZE
    except (ZeroDivisionError, IndexError):
        return False
    
    return [x_min, x_max, y_min, y_max]

def recenter_matrix(matrix, center):
    #Here the center is expressed in [1:256]x[1:256] coordinates to match the size of the matrix
    center[1], center[0] = int(center[0]), int(center[1])
    if center[0] > 123:
        x_centered_matrix = np.concatenate((matrix[center[0]-123:,:], matrix[:center[0]-123,:]), axis=0)
    else:
        x_centered_matrix = np.concatenate((matrix[center[0]+123:,:], matrix[:center[0]+123,:]), axis=0)
    if center[1] > 123:
        centered_matrix = np.concatenate((x_centered_matrix[:,center[1]-123:], x_centered_matrix[:,:center[1]-123]), axis=1)
    else:
        centered_matrix = np.concatenate((x_centered_matrix[:,center[1]+123:], x_centered_matrix[:,:center[1]+123]), axis=1)
    return centered_matrix


###############################################################################################
########################################## Plotting ###########################################
###############################################################################################
def create_gif(image_array: list, name: str, duration: int = 20000):
    fig, ax = plt.subplots(frameon=False)
    ax.axis('off')
    
    ims = []
    for i in range(len(image_array)):
        im = ax.imshow(image_array[i], animated=True)
        if i == 0:
            ax.imshow(image_array[0])  # show an initial one first    
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=duration/len(image_array), blit=True,
                                    repeat_delay=5000)
    ani.save("../media/"+get_sim_desc(sim_name[case])+"_"+name+"_"+sim_name[case]+".gif")
    plt.close(fig)