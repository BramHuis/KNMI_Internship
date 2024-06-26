#%%
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cf
import scipy
import os
import multiprocessing as mp
import concurrent.futures as conf
import threading
from cartopy.io.shapereader import Reader
import time
import xarray as xr
import gc
from memory_profiler import profile
import netCDF4 as nc

#%%
def LoadData(fileName):
    return nc.Dataset(fileName)

def SplitData(dataset, *variables):
    data = []
    for variable in variables:
        data.append(dataset[variable][:].astype(np.float32))
    return data

def SliceData(bbox, lat, lon, variable, startDay = 1, endDay = 360, originalYearLength = 360, isEvent = False):
    """
    This function slices the data given the S, N, W, E bounds and slices the time based on the start 
    day and end day within a year
    """
    # Change longitude data to go from -180 to 180
    mask = [lon > 180]
    lon[mask[0]] = lon[mask[0]] - 360

    # Calculate the index of the bounds
    sIndex = np.argmin(np.abs(lat - bbox[0]))
    nIndex = np.argmin(np.abs(lat - bbox[1]))
    wIndex = np.argmin(np.abs(lon - bbox[2]))
    eIndex = np.argmin(np.abs(lon - bbox[3]))

    if isEvent:
        variable = np.expand_dims(variable, axis = 0)

    if wIndex > eIndex: # If the west index is higher than the east index, think of the right side of the world map as left boundary and vice versa
        latSlice = lat[sIndex: nIndex + 1]
        lonSlice = np.concatenate((lon[wIndex:], lon[:eIndex + 1]))
        variableSlice = np.concatenate((variable[:, sIndex: nIndex + 1, wIndex:], variable[:, sIndex: nIndex + 1, :eIndex + 1]), axis = 2) 
        
    else:
        latSlice = lat[sIndex: nIndex + 1]
        lonSlice = lon[wIndex: eIndex + 1]
        variableSlice = variable[:, sIndex: nIndex + 1, wIndex: eIndex + 1]

    if isEvent:
        return latSlice, lonSlice, variableSlice
    
    years = int(variableSlice.shape[0] / originalYearLength)
    data = variableSlice.reshape(years, originalYearLength, variableSlice.shape[1], variableSlice.shape[2])
    dataSeason = data[:, startDay - 1: endDay, :, :]
    # dataSeasonUnraveled = dataSeason.reshape(dataSeason.shape[0] * dataSeason.shape[1], variableSlice.shape[1], variableSlice.shape[2])
    return latSlice, lonSlice, dataSeason

def RetrieveData(dataFolder, saveFolder, variableName, bbox, startDay, endDay, originalYearLength, nEnsembles, isEvent = False, isAmo = False):
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    if isEvent:
        data1 = LoadData(dataFolder)
        lat1, lon1, var1 = SplitData(data1, "lat", "lon", variableName)
        latSlice1, lonSlice1, varSlice1 = SliceData(bbox, lat1, lon1, var1, startDay, endDay, originalYearLength, isEvent)
        np.save(f"{saveFolder}Lat", np.array(latSlice1))
        np.save(f"{saveFolder}Lon", np.array(lonSlice1))   
        np.save(f"{saveFolder}{variableName}", np.array(np.squeeze(varSlice1))) 
        return
    
    fileList = sorted(os.listdir(dataFolder))#[:int(nEnsembles * 2)]
    
    if len(fileList) % 2 != 0:
        print("The number of files in the folder must be even, one for past and one for present data")
        return

    yearLength = endDay - startDay + 1

    for i, fileName in enumerate(fileList):
        filePath = os.path.join(dataFolder, fileName)
        if i % 2 == 0:
            data1 = LoadData(filePath)
            continue
        data2 = LoadData(filePath)
        print(f"Running ensemble {(i // 2) + 1}")

        lat1, lon1, var1 = SplitData(data1, "lat", "lon", variableName)
        lat2, lon2, var2 = SplitData(data2, "lat", "lon", variableName)

        if isAmo:
            latSlice1, lonSlice1, amoSliceGlob1 = SliceData([-90, 90, -180, 180], lat1, lon1, var1, startDay, endDay, originalYearLength, False)
            _, _, amoSliceLoc1 = SliceData([0, 60, -80, 0], lat1, lon1, var1, startDay, endDay, originalYearLength, False)
            _, _, amoSliceGlob2 = SliceData([-90, 90, -180, 180], lat2, lon2, var2, startDay, endDay, originalYearLength, False)
            _, _, amoSliceLoc2 = SliceData([0, 60, -80, 0], lat2, lon2, var2, startDay, endDay, originalYearLength, False)
            amoSliceGlob = np.concatenate((amoSliceGlob1, amoSliceGlob2))
            amoSliceLoc = np.concatenate((amoSliceLoc1, amoSliceLoc2))
            amoGlobMean = np.mean(amoSliceGlob, axis = (1, 2, 3))
            amoLocMean = np.mean(amoSliceLoc, axis = (1, 2, 3))
            amo = amoLocMean - amoGlobMean

            if i == 1:
                variableEnsembles = np.zeros((len(fileList) // 2, amoSliceGlob1.shape[0] + amoSliceGlob2.shape[0]), dtype = np.float32)
            
            variableEnsembles[(i // 2)] = amo
            continue
        
        latSlice1, lonSlice1, varSlice1 = SliceData(bbox, lat1, lon1, var1, startDay, endDay, originalYearLength, False)
        latSlice2, lonSlice2, varSlice2 = SliceData(bbox, lat2, lon2, var2, startDay, endDay, originalYearLength, False)

        if i == 1:
            variableEnsembles = np.zeros((len(fileList) // 2, varSlice1.shape[0] + varSlice2.shape[0], yearLength, latSlice1.shape[0], lonSlice1.shape[0]), dtype = np.float32)
        varSlice = np.concatenate((varSlice1, varSlice2))
        variableEnsembles[(i // 2)] = varSlice
     
    np.save(f"{saveFolder}Lat", np.array(latSlice1))
    np.save(f"{saveFolder}Lon", np.array(lonSlice1))   
    np.save(f"{saveFolder}{variableName}", variableEnsembles) 

def PlotAnalogue(lat, lon, data, significanceMask = None, saveFig = False, fileName = "", minLevel = None, maxLevel = None, precLabel = False, isBlues = False):
    plt.figure(figsize = (10, 10))
    ax = plt.axes(projection = ccrs.PlateCarree())
    # ax.set_title(f"{fileName}")

    if type(minLevel) != type(None) and type(maxLevel) != type(None):
        if isBlues:
            plot = plt.contourf(lon, lat, data, cmap = "Blues", transform = ccrs.PlateCarree(), levels = 30, vmin = minLevel, vmax = maxLevel) #, , vmin = -96, vmax = 96, extend = "both"
        else:
            plot = plt.contourf(lon, lat, data, cmap = "BrBG", transform = ccrs.PlateCarree(), levels = 30, vmin = minLevel, vmax = maxLevel) #, , vmin = -96, vmax = 96, extend = "both"
    else:
        if isBlues:
            plot = plt.contourf(lon, lat, data, cmap = "Blues", transform = ccrs.PlateCarree(), levels = 30) #, , vmin = -96, vmax = 96, extend = "both"
        else:   
            plot = plt.contourf(lon, lat, data, cmap = "PiYG", transform = ccrs.PlateCarree(), levels = 30) #, vmin = 100900, vmax = 102500, extend = "both"
            print("No min and/or max level specified")
    
    if type(significanceMask) != type(None):
        plt.contourf(lon, lat, significanceMask, levels = [-1, 0, 1], hatches = [None, "////"], colors = "None", transform = ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cf.BORDERS)

    manualCatchment = False
    if manualCatchment:
        ax.add_geometries(Reader("/usr/people/huis/Stage/Data/ShpFiles/wribasin.shp").geometries(), ccrs.PlateCarree(), facecolor='white', hatch='xxxx')

    if precLabel:
        if type(minLevel) != type(None) and type(maxLevel) != type(None):
            if isBlues:
                norm = plt.Normalize(vmin = minLevel, vmax = maxLevel)
                cbar = plt.colorbar(cm.ScalarMappable(norm = norm, cmap = "Blues"), ax = ax, orientation = "horizontal", shrink = 0.8, extend = "max")
                cbar.set_label("Precipitation (mm/day)")
            else:
                norm = plt.Normalize(vmin = minLevel, vmax = maxLevel)
                cbar = plt.colorbar(cm.ScalarMappable(norm = norm, cmap = "BrBG"), ax = ax, orientation = "horizontal", shrink = 0.8, extend = "both")
                cbar.set_label("Precipitation (mm/day)")
        else:
            plt.colorbar(plot, ax = ax, orientation = "horizontal", label = "Precipitation (mm/day)", shrink = 0.8)
    else:
        if type(minLevel) != type(None) and type(maxLevel) != type(None):
            norm = plt.Normalize(vmin = minLevel, vmax = maxLevel)
            cbar = plt.colorbar(cm.ScalarMappable(norm = norm, cmap = "RdBu_r"), ax = ax, orientation = "horizontal", shrink = 0.8, extend = "both")
            cbar.set_label("Sea level pressure (Pa)")
        else:
            plt.colorbar(plot, ax = ax, orientation = "horizontal", label = "Sea level pressure (Pa)", shrink = 0.8)

    if saveFig:
        plt.savefig(f"{fileName}.png", dpi = 250, bbox_inches = "tight")
    plt.close()
#%% "histAllEnsembles", "GHGAllEnsembles", "AERAllEnsembles", "SOLAllEnsembles", "VOLAllEnsembles"
runSpecification = "DouwehistAllEnsembles"
experiment = "hist"
bbox = [41, 59, -5, 23]
startDay, endDay = 181, 270
originalYearLength = 360
nEnsembles = 55
dataFolderStart = f"/net/pc200023/nobackup/users/thompson/LESFMIP/HadGEM/"
#%%
if __name__ == "__main__":
    variableName = "tas"
    dataFolder = f"{dataFolderStart}{experiment}/{variableName}"
    saveFolder = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/{variableName}/"
    RetrieveData(dataFolder, saveFolder, variableName, bbox, startDay, endDay, originalYearLength, nEnsembles = nEnsembles, isAmo = True)
#%%
if __name__ == "__main__":
    variableName = "psl"
    dataFolder = f"{dataFolderStart}{experiment}/{variableName}"
    saveFolder = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/{variableName}/"
    RetrieveData(dataFolder, saveFolder, variableName, bbox, startDay, endDay, originalYearLength, nEnsembles = nEnsembles)
#%%
if __name__ == "__main__":
    variableName = "pr"
    dataFolder = f"{dataFolderStart}{experiment}/{variableName}"
    saveFolder = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/{variableName}/"
    RetrieveData(dataFolder, saveFolder, variableName, bbox, startDay, endDay, originalYearLength, nEnsembles = nEnsembles)
#%%
if __name__ == "__main__":
    variableName = "msl"
    eventPath = f"/usr/people/huis/Stage/Data/Limburg_14Jul2021_ERA5_msl.nc"
    saveFolder = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/{variableName}/"
    RetrieveData(eventPath, saveFolder, variableName, bbox, startDay, endDay, originalYearLength, nEnsembles = nEnsembles, isEvent = True)
# #%%
# kaas = "/usr/people/thompson/WP1/intern/Limburg_14Jul2021_ERA5_pr.nc"
# bbox = [41, 59, -5, 23]
# startDay, endDay = 181, 270
# originalYearLength = 360
# event = LoadData(kaas)
# lat, lon, psl = SplitData(event, "lat", "lon", "tp")
# latSlice, lonSlice, mslSlice = SliceData(bbox, lat, lon, psl, startDay, endDay, originalYearLength, True)
# print(np.max(mslSlice))
# print(mslSlice.shape)
# print(mslSlice.shape)
# PlotAnalogue(latSlice, lonSlice, mslSlice[0], saveFig = True, fileName = "PrBest", minLevel = 0, maxLevel  = 80, isBlues = True, precLabel=True)
# %%
