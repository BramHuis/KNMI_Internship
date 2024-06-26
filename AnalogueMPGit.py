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

def EuclidianDistance(event, data, isAllEnsemble = False):
    # shape: ensembleNumber, days, lat, lon
    if isAllEnsemble:
        return np.sqrt(np.sum(np.square(np.subtract(data, event)), axis = (2, 3)))
    # shape days, lat, lon
    return np.sqrt(np.sum(np.square(np.subtract(data, event)), axis = (1, 2)))

def CalculateYearlyMinimumEuclidianDistance(distances, yearLength = 360, meanRange = 10, isAllEnsemble = False):
    
    if isAllEnsemble:
        yearlyMinDistance = distances.reshape((distances.shape[0], distances.shape[1] // yearLength, yearLength))
        maxDistance = np.max(yearlyMinDistance)
        EDs = 1 - (np.min(yearlyMinDistance, axis = 2) / maxDistance)
        runningMean = np.zeros((EDs.shape[0], EDs.shape[1] - meanRange + 1))
        for i in range(EDs.shape[0]):
            runningMean[i] = np.convolve(EDs[i, :], np.ones(meanRange) / meanRange, mode="valid")

    
    else:
        yearlyMinDistance = distances.reshape((distances.shape[0] // yearLength, yearLength))
        maxDistance = np.max(distances)
        EDs = 1 - (np.min(yearlyMinDistance, axis = 1) / maxDistance)
        # EDs = 1 - ((np.mean(np.sort(yearlyMinDistance, axis = 1)[:, :30], axis = 1)) / maxDistance)
        runningMean = np.convolve(EDs, np.ones(meanRange) / meanRange, mode = "valid")
    return EDs, runningMean

def PearsonCorrelation(eventAnalogue, analogue):
    term1 = np.sum((analogue - np.mean(analogue)) * (eventAnalogue - np.mean(eventAnalogue)))
    term2 = np.sqrt(np.sum(np.square(analogue - np.mean(analogue))) * np.sum(np.square(eventAnalogue - np.mean(eventAnalogue))))
    return term1 / term2

def RetrieveAnalogueIndicesDistances(distances, targetAnalogues, analogueRange, yearLength = 90):
    currentIndex = 0 
    analogueIndices, eventDistances = [], []
    distanceIndexDict = {value: index for index, value in enumerate(distances)}
    sortedDistances = np.sort(distances)
    
    while (len(analogueIndices) < targetAnalogues) and (currentIndex < len(distances)):
        shortestDistanceIndex = distanceIndexDict[sortedDistances[currentIndex]]
        shortestDistanceIndexYearRemainder = shortestDistanceIndex % yearLength

        backwardAnalogueRange = shortestDistanceIndexYearRemainder if shortestDistanceIndexYearRemainder <  analogueRange else analogueRange
        forwardAnalogueRange = (yearLength - 1) - shortestDistanceIndexYearRemainder if shortestDistanceIndexYearRemainder > (yearLength - analogueRange) else analogueRange
        
        for analogueIndex in analogueIndices:
            if (analogueIndex - backwardAnalogueRange) <= shortestDistanceIndex <= (analogueIndex + forwardAnalogueRange):
                break
            
        else:
            analogueIndices.append(shortestDistanceIndex)
            eventDistances.append(sortedDistances[currentIndex])
        currentIndex += 1
    return np.array(analogueIndices, dtype = np.int32), np.array(eventDistances, dtype = np.float32)

def RetrieveAnalogueIndicesDistancesAllEnsembles(distances, targetAnalogues):
    currentIndex = 0
    analogueIndices, eventDistances = [], []
    distanceIndexDict = {value: index for index, value in enumerate(distances)}
    sortedDistances = np.sort(distances)
    
    while (len(analogueIndices) < targetAnalogues) and (currentIndex < len(distances)):
        shortestDistanceIndex = distanceIndexDict[sortedDistances[currentIndex]]
        analogueIndices.append(shortestDistanceIndex)
        eventDistances.append(sortedDistances[currentIndex])
        currentIndex += 1
    return np.array(analogueIndices, dtype = np.int32), np.array(eventDistances, dtype = np.float32)

def RetrieveAnalogues(analogueIndicesPast, analogueIndicesPresent, dataPast, dataPresent, isMax = False):
    analoguesPast, analoguesPresent = np.zeros((analogueIndicesPast.shape[0], dataPast.shape[-2], dataPast.shape[-1])), np.zeros((analogueIndicesPresent.shape[0], dataPresent.shape[-2], dataPresent.shape[-1]))

    for idx, (i, j) in enumerate(zip(analogueIndicesPast, analogueIndicesPresent)):
        analoguesPast[idx] = dataPast[i]
        analoguesPresent[idx] = dataPresent[j]
    
    if isMax:
        analoguePastMax = np.max(analoguesPast, axis = 0)
        analoguePresentMax = np.max(analoguesPresent, axis = 0)
        return analoguesPast, analoguesPresent, analoguePastMax, analoguePresentMax

    analoguePastMean = np.mean(analoguesPast, axis = 0)
    analoguePresentMean = np.mean(analoguesPresent, axis = 0)
    return analoguesPast, analoguesPresent, analoguePastMean, analoguePresentMean

def RetieveDate(analogueIndices, yearLength = 360, monthLength = 30):
    years = analogueIndices // yearLength + 1
    yearsRemainder = analogueIndices % yearLength
    months = yearsRemainder // monthLength + 1
    days = yearsRemainder % monthLength + 1
    return years, months, days

def CalculatePersistence(data, analogueIndices, yearLength, correlationCoëfficiënt = 0.9):
    correlations = np.zeros(len(analogueIndices), dtype = object)
    for i in range(len(analogueIndices)):
        correlationList = [0]
        analogueEvent = data[analogueIndices[i]]
        year = analogueIndices[i] // yearLength

        forwardCounter = 1
        while (analogueIndices[i] + forwardCounter) < len(data) and year == ((analogueIndices[i] + forwardCounter) // yearLength):
            correlation = PearsonCorrelation(analogueEvent, data[analogueIndices[i] + forwardCounter])
            if correlation >= correlationCoëfficiënt:
                correlationList.append(correlation)
                forwardCounter += 1
            else:
                break
        
        backwardCounter = 1
        while (analogueIndices[i] - backwardCounter) >= 0 and year == ((analogueIndices[i] + forwardCounter) // yearLength):
            correlation = PearsonCorrelation(analogueEvent, data[analogueIndices[i] - backwardCounter])
            if correlation >= correlationCoëfficiënt:
                correlationList.insert(0, correlation)
                backwardCounter += 1
            else:
                break
        
        correlations[i] = correlationList
    return correlations

def CalculateTypicality(analogueIndices, analogueDistances, data, targetAnalogues, analogueRange):
    TEvent = 1 / np.sum(analogueDistances)
    TAnalogues = np.zeros(len(analogueDistances))

    for i in range(len(analogueIndices)):
        analogueIndex = analogueIndices[i]
        allAnalogueDistances = EuclidianDistance(data[analogueIndex], data)
        _, analogueAnalogueDistances = RetrieveAnalogueIndicesDistances(allAnalogueDistances, targetAnalogues = targetAnalogues, analogueRange = analogueRange)
        TAnalogue = 1 / np.sum(analogueAnalogueDistances)
        TAnalogues[i] = TAnalogue
    return TEvent, TAnalogues

def TTest(analoguesPast, analoguesPresent, significanceLevel = 0.05):
    significanceMask = np.zeros((analoguesPast.shape[1], analoguesPast.shape[2]))
    for i in range(analoguesPast.shape[1]):
        for j in range(analoguesPast.shape[2]):
            _, pValue = scipy.stats.ttest_ind(analoguesPast[:, i, j], analoguesPresent[:, i, j])
            significanceMask[i, j] = 1 if pValue < significanceLevel else 0
    return significanceMask

def RunSingleEnsemble(pslHadGEMSlicePast, pslHadGEMSlicePresent, subFolder, i, amoPast, amoPresent, latSlice, lonSlice, mslEventSlice, yearLength, correlationCoëfficiënt, meanRange, minValue, maxValue):
    distancesPast = EuclidianDistance(mslEventSlice, pslHadGEMSlicePast)
    analogueIndicesPast, analogueDistancesPast = RetrieveAnalogueIndicesDistances(distancesPast, targetAnalogues = targetAnalogues, analogueRange = analogueRange)
    persistencePast = CalculatePersistence(pslHadGEMSlicePast, analogueIndicesPast, yearLength, correlationCoëfficiënt)
    TeventPast, TAnaloguesPast = CalculateTypicality(analogueIndicesPast, analogueDistancesPast, pslHadGEMSlicePast, targetAnalogues = targetAnalogues, analogueRange = analogueRange)

    distancesPresent = EuclidianDistance(mslEventSlice, pslHadGEMSlicePresent)
    analogueIndicesPresent, analogueDistancesPresent = RetrieveAnalogueIndicesDistances(distancesPresent, targetAnalogues = targetAnalogues, analogueRange = analogueRange)
    persistencePresent = CalculatePersistence(pslHadGEMSlicePresent, analogueIndicesPresent, yearLength, correlationCoëfficiënt)
    TeventPresent, TAnaloguesPresent = CalculateTypicality(analogueIndicesPresent, analogueDistancesPresent, pslHadGEMSlicePresent, targetAnalogues = targetAnalogues, analogueRange = analogueRange)

    analoguesPast, analoguesPresent, analoguePastMean, analoguePresentMean = RetrieveAnalogues(analogueIndicesPast, analogueIndicesPresent, pslHadGEMSlicePast, pslHadGEMSlicePresent)
    analogueDifference = analoguePresentMean - analoguePastMean

    yearlyEuclidianDistancePast, runningMeanPast = CalculateYearlyMinimumEuclidianDistance(distancesPast, yearLength = yearLength, meanRange = 10)
    yearlyEuclidianDistancePresent, runningMeanPresent = CalculateYearlyMinimumEuclidianDistance(distancesPresent, yearLength = yearLength, meanRange = 10)
    # if i == 23 or i == 32:
    #     np.save(f"RunningmeanZooi{i}", runningMeanPast)

    significanceMask = TTest(analoguesPast, analoguesPresent, significanceLevel = significanceLevel)

    if not os.path.exists(analoguePathPast := f"Plots/{subFolder}/Analogues/Ensemble{i + 1}/Past/"):
        os.makedirs(f"{analoguePathPast}")
    if not os.path.exists(analoguePathPresent := f"Plots/{subFolder}/Analogues/Ensemble{i + 1}/Present/"):
        os.makedirs(f"{analoguePathPresent}")
    if not os.path.exists(analoguePathMean := f"Plots/{subFolder}/Analogues/Ensemble{i + 1}/Mean/"):
        os.makedirs(f"{analoguePathMean}")
    if not os.path.exists(trendPath := f"Plots/{subFolder}/Trend/Ensemble{i + 1}/"):
        os.makedirs(f"{trendPath}")
    if not os.path.exists(violinPath := f"Plots/{subFolder}/Violin/Ensemble{i + 1}/"):
        os.makedirs(f"{violinPath}")    
    if not os.path.exists(amoPlotPath := f"Plots/{subFolder}/AMO/Ensemble{i + 1}/"):
        os.makedirs(f"{amoPlotPath}")    

    for l, (aPast, aPresent) in enumerate(zip(analoguesPast, analoguesPresent)):
        PlotAnalogue(latSlice, lonSlice, aPast, saveFig = saveFig, fileName = f"{analoguePathPast}{l + 1}", minLevel = minValue, maxLevel = maxValue)
        PlotAnalogue(latSlice, lonSlice, aPresent, saveFig = saveFig, fileName = f"{analoguePathPresent}{l + 1}", minLevel = minValue, maxLevel = maxValue)
    PlotAnalogue(latSlice, lonSlice, analoguePastMean, saveFig = saveFig, fileName = f"{analoguePathMean}PastMean", minLevel = minValue, maxLevel = maxValue)
    PlotAnalogue(latSlice, lonSlice, analoguePresentMean, saveFig = saveFig, fileName = f"{analoguePathMean}PresentMean", minLevel = minValue, maxLevel = maxValue)
    PlotAnalogue(latSlice, lonSlice, analogueDifference, significanceMask, saveFig = saveFig, fileName = f"{analoguePathMean}DifferenceMean", minLevel = -100, maxLevel = 100)

    # PlotTrend(yearlyEuclidianDistancePast, runningMeanPast, startYearPast, endYearPast, meanRange = meanRange, saveFig = saveFig, fileName = f"{trendPath}TrendPast{(i // 2) + 1}")
    # PlotTrend(yearlyEuclidianDistancePresent, runningMeanPresent, startYearPresent, endYearPresent, meanRange = meanRange, saveFig = saveFig, fileName = f"{trendPath}TrendPresent{(i // 2) + 1}")
    # AMOPlot(yearlyEuclidianDistancePast, runningMeanPast, amoPast, meanRange, saveFig = saveFig, fileName = f"{amoPlotPath}Past")
    # AMOPlot(yearlyEuclidianDistancePresent, runningMeanPresent, amoPresent, meanRange, saveFig = saveFig, fileName = f"{amoPlotPath}Present")

    pPa = [len(p) for p in persistencePast]
    pPr = [len(p) for p in persistencePresent]
    ViolinPlot(TAnaloguesPast, TAnaloguesPresent, TeventPast, TeventPresent, saveFig = saveFig, fileName = f"{violinPath}T_{i + 1}")
    ViolinPlot(pPa, pPr, saveFig = saveFig, fileName = f"{violinPath}P_{i + 1}")  

    print(f"Done! {i + 1}")
    return pslHadGEMSlicePast, pslHadGEMSlicePresent, analogueDistancesPast, analogueDistancesPresent, analoguesPast, analoguesPresent, yearlyEuclidianDistancePast, yearlyEuclidianDistancePresent, TAnaloguesPast, TAnaloguesPresent, amoPast, amoPresent, yearlyEuclidianDistancePast, yearlyEuclidianDistancePresent, analogueIndicesPast, analogueIndicesPresent, persistencePast, persistencePresent

def PlotAnalogue(lat, lon, data, significanceMask = None, saveFig = False, fileName = "", minLevel = None, maxLevel = None, precLabel = False, isBlues = False):
    plt.figure(figsize = (10, 10))
    ax = plt.axes(projection = ccrs.PlateCarree())
    # ax.set_title(f"{fileName}")

    if type(minLevel) != type(None) and type(maxLevel) != type(None):
        if isBlues:
            plot = plt.contourf(lon, lat, data, cmap = "Blues", transform = ccrs.PlateCarree(), levels = 30, vmin = minLevel, vmax = maxLevel) #, , vmin = -96, vmax = 96, extend = "both"
        else:
            plot = plt.contourf(lon, lat, data, cmap = "RdBu_r", transform = ccrs.PlateCarree(), levels = 30, vmin = minLevel, vmax = maxLevel) #, , vmin = -96, vmax = 96, extend = "both"
    else:
        if isBlues:
            plot = plt.contourf(lon, lat, data, cmap = "Blues", transform = ccrs.PlateCarree(), levels = 30) #, , vmin = -96, vmax = 96, extend = "both"
        else:   
            #PiYG
            plot = plt.contourf(lon, lat, data, cmap = "RdBu_r", transform = ccrs.PlateCarree(), levels = 30) #, vmin = 100900, vmax = 102500, extend = "both"
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
                cbar = plt.colorbar(cm.ScalarMappable(norm = norm, cmap = "RdBu_r"), ax = ax, orientation = "horizontal", shrink = 0.8, extend = "both")
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

def ViolinPlot(dataPastAnalogues, dataPresentAnalogues, dataPastEvent = 3, dataPresentEvent = 3, title = "", saveFig = False, fileName = "", manualOverride = False, manualOverridePers = False):
    fig, ax = plt.subplots(figsize = (6, 7))
    violins = ax.violinplot([dataPastAnalogues, dataPresentAnalogues], [1, 1.6], showmeans = False, showextrema = False, showmedians = False)
    colors = ["magenta", "green"]

    _, pValue = scipy.stats.ttest_ind(dataPastAnalogues, dataPresentAnalogues)
    plt.text(0.05, 0.95, f"p = {pValue:.3f}", transform = ax.transAxes, bbox = dict(facecolor = "None", edgecolor = "k"))
    for pc, color in zip(violins["bodies"], colors):
        pc.set_facecolor(color)
    ax.plot(1, dataPastEvent, marker = "o", color = "r")
    ax.plot(1.6, dataPresentEvent, marker = "o", color = "r")
    ax.axhline(np.mean(dataPastAnalogues), color = colors[0], linewidth = 3)
    ax.axhline(np.mean(dataPresentAnalogues), color = colors[1], linewidth = 3)
    ax.set_xticks([1, 1.6])
    ax.set_xticklabels(["Past", "Present"])
    # ax.set_title(title)
    max = np.max([np.max(dataPastAnalogues), np.max(dataPresentAnalogues)])
    ax.set_ylim(top = max * 1.1)

    if manualOverride:
        ax.set_ylim(0.8e-5, 2e-5)
    if manualOverridePers:
        plt.ylabel("Persistence (days)")
        ax.set_ylim(1, 7)
    else:
        plt.ylabel("Typicality")

    if saveFig:
        fig.savefig(f"{fileName}.png", dpi = 250, bbox_inches = "tight")
    plt.close()

def PlotTrend(yearlyEuclidianDistance, runningMean, startYear, endYear, meanRange = 10, saveFig = False, fileName = ""):
    plt.figure(figsize = (25, 10))
    # plt.ylim(0.80, 0.95)
    startYear = 1850
    endYear = 2014
    plt.xticks(np.arange(1850, 2014, 25))
    plt.xlim(1850, 2014)
    plt.grid(axis = "y")
    plt.ylabel("1 - (min(ED) / max(ED))")
    # plt.plot(range(1850, 2014 + 1), yearlyEuclidianDistance, label = "Yearly variation")
    plt.plot(range(startYear, endYear + 1), yearlyEuclidianDistance, label = "Yearly variation")
    plt.plot(range(startYear + (meanRange // 2 - 1), endYear - (meanRange // 2 - 1)), runningMean, label = "10-year running mean")
    plt.axvline(x = 1880, color = "r", linestyle = "--")
    plt.axvline(x = 1984, color = "r", linestyle = "--")
    plt.legend()
    if saveFig:
        plt.savefig(f"{fileName}.png", dpi = 250, bbox_inches = "tight")
    plt.close()

def AMOPlot(yearlyEuclidianDistance, runningMean, amoData, meanRange, saveFig = False, fileName = ""):
    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 10))
    slope, intercept, r2, _, _ = scipy.stats.linregress(yearlyEuclidianDistance, amoData)
    xData = np.linspace(np.min(yearlyEuclidianDistance), np.max(yearlyEuclidianDistance), yearlyEuclidianDistance.shape[0])
    yData = [slope * x + intercept for x in xData]
    ax1.scatter(yearlyEuclidianDistance, amoData)
    ax1.plot(xData, yData)
    ax1.text(0.05, 0.95, f"R-squared = {r2:.3f}", transform = ax1.transAxes, bbox = dict(facecolor = "None", edgecolor = "k"))
    ax1.set_xlabel("Euclidian distance")
    ax1.set_ylabel("AMO")
    ax1.set_title("AMO")

    runningMeanAMO = np.convolve(amoData, np.ones(meanRange) / meanRange, mode = "valid")
    slope, intercept, r2, _, _ = scipy.stats.linregress(runningMean, runningMeanAMO)
    xData = np.linspace(np.min(runningMean), np.max(runningMean), runningMean.shape[0])
    yData = [slope * x + intercept for x in xData]
    ax2.scatter(runningMean, runningMeanAMO)
    ax2.plot(xData, yData)
    ax2.text(0.05, 0.95, f"R-squared = {r2:.3f}", transform = ax2.transAxes, bbox = dict(facecolor = "None", edgecolor = "k"))
    ax2.set_xlabel("Euclidian distance")
    ax2.set_ylabel("AMO")
    ax2.set_title("10-yr AMO")
    if saveFig:
        fig.savefig(f"{fileName}.png", dpi = 250, bbox_inches = "tight")
    plt.close()

def PlotAllEnsembleTrends(allTrendData, saveFig = False, fileName = ""):
    allTrendDataMean = np.mean(np.array(allTrendData, dtype = np.float32), axis = 0)
    allTrendDataStd = np.std(allTrendData, axis = 0)

    xData = np.arange(1850, allTrendDataMean.shape[0] + 1850, 1)
    fig, ax1 = plt.subplots(1, 1, figsize = (25, 10))
    
    ax1.fill_between(xData, allTrendDataMean - allTrendDataStd, allTrendDataMean + allTrendDataStd, color = "k", alpha = 0.5, label = "Standard deviation = 1")
    ax1.fill_between(xData, allTrendDataMean - 2 * allTrendDataStd, allTrendDataMean - allTrendDataStd, color = "k", alpha = 0.25, label = "Standard deviation = 2")
    ax1.fill_between(xData, allTrendDataMean + 2 * allTrendDataStd, allTrendDataMean + allTrendDataStd, color = "k", alpha = 0.25)
    ax1.plot(xData, allTrendDataMean, color = "r", linewidth = 2, label = "Ensemble mean")  
    ax1.set_title("Trend")
    plt.grid()
    colors = ["magenta", "coral", "deepskyblue", "indigo"]
    nPlotSingleEnsembles = 4
    for i in range(nPlotSingleEnsembles):
        if i == (nPlotSingleEnsembles - 1):
            ax1.plot(xData, allTrendData[i], color = colors[i], linewidth = 1) #, label = "Single ensemble"
        else:
            ax1.plot(xData, allTrendData[i], color = colors[i], linewidth = 1)
    
    ax1.legend()
    plt.xticks(np.arange(1850, 2014, 25))
    plt.xlim(1850, 2014)
    plt.ylim(0.80, 0.95)
    plt.xlabel("Year")
    if saveFig:
        fig.savefig(f"{fileName}.png", dpi = 250, bbox_inches = "tight")
    plt.close()

def PlotAllEnsembleViolins(allEnsembleDataPast, allEnsembleDataPresent, topEnsembleDataPast, topEnsembleDataPresent, saveFig = False, fileName = "", isPersistence = False, persistenceEventPast = None, persistenceEventPresent = None):
    colors = [(1, 0, 1, 0.2), (0.12, 0.95, 0.59, 0.2)]
    colorsMean = [(1, 0, 1), (0.12, 0.95, 0.59)]
    fig, ax = plt.subplots(1, 1, figsize = (20, 10))
    for ensembleDataPast, ensembleDataPresent in zip(allEnsembleDataPast, allEnsembleDataPresent):
        violins = ax.violinplot([ensembleDataPast, ensembleDataPresent], [1, 1.6], showmeans = False, showextrema = False, showmedians = False)
        for pc, color in zip(violins["bodies"], colors):
            pc.set_facecolor(color)
    violins = ax.violinplot([topEnsembleDataPast, topEnsembleDataPresent], [1, 1.6], showmeans = False, showextrema = False, showmedians = False)
    for pc, color in zip(violins["bodies"], colorsMean):
        pc.set_facecolor(color)
        pc.set_edgecolor("k")
    ax.axhline(np.mean(topEnsembleDataPast), color = colorsMean[0], linewidth = 3)
    ax.axhline(np.mean(topEnsembleDataPresent), color = colorsMean[1], linewidth = 3)
    allMeanPast, allMeanPresent = np.mean(allEnsembleDataPast), np.mean(allEnsembleDataPresent)
    ax.scatter(1, persistenceEventPast, color = "red")
    ax.scatter(1.6, persistenceEventPresent, color = "red")
    ax.axhline(allMeanPast, color = colorsMean[0], linewidth = 3, linestyle = "--")
    ax.axhline(allMeanPresent, color = colorsMean[1], linewidth = 3, linestyle = "--")
    _, pValue = scipy.stats.ttest_ind(topEnsembleDataPast, topEnsembleDataPresent)
    plt.text(0.05, 0.95, f"p = {pValue:.3f}", transform = ax.transAxes, bbox = dict(facecolor = "None", edgecolor = "k"))
    ax.set_xticks([1, 1.6])
    ax.set_xticklabels(["Past", "Present"])
    if saveFig:
        fig.savefig(f"{fileName}.png", dpi = 250, bbox_inches = "tight")
    plt.close()

def PlotAllAMOEnsembleData(amoData, yearlyEuclidianDistances, meanRange = 10, isRunningMean = False, saveFig = False, fileName = ""):
    nEnsembles = len(amoData)
    nCols = 1
    while (nCols + 1) ** 2 <= nEnsembles:
        nCols += 1
    remainder = nEnsembles - nCols**2
    if remainder != 0:
        nRows = (remainder // nCols) + nCols if remainder % nCols == 0 else (remainder // nCols) + nCols + 1
    else:
        nRows = nCols
    
    fig, axs = plt.subplots(nRows, nCols, figsize = (75, 50))
    axs = axs.flatten()
    for ax in axs[nEnsembles - nRows * nCols:]:
        ax.remove()
    axs = axs[:nEnsembles]
    
    if isRunningMean:
        for i in range(len(yearlyEuclidianDistances)):
            yearlyEuclidianDistances[i] = np.convolve(yearlyEuclidianDistances[i], np.ones(meanRange) / meanRange, mode = "valid")
            amoData[i] = np.convolve(amoData[i], np.ones(meanRange) / meanRange, mode = "valid")

    for i in range(len(axs)):
        slope, intercept, r2, _, _ = scipy.stats.linregress(yearlyEuclidianDistances[i], amoData[i])
        xData = np.linspace(np.min(yearlyEuclidianDistances[i]), np.max(yearlyEuclidianDistances[i]), yearlyEuclidianDistances[i].shape[0])
        yData = [slope * x + intercept for x in xData]
        axs[i].scatter(yearlyEuclidianDistances[i], amoData[i])
        axs[i].plot(xData, yData)
        axs[i].text(0.05, 0.8, f"R-squared = {r2:.3f}", transform = axs[i].transAxes, bbox = dict(facecolor = "None", edgecolor = "k"))

    if saveFig:
        fig.savefig(f"{fileName}.png", dpi = 250, bbox_inches = "tight")
    plt.close()
    
def RetrievePrecipitationAnalogues(analogueIndicesPast, analogueIndicesPresent, precipitationPast, precipitationPresent):
    pastPrecAnalogues, presentPrecAnalogues = [], []
    for i in range(len(analogueIndicesPast)):
        for j1, j2 in zip(analogueIndicesPast[i], analogueIndicesPresent[i]):
            pastPrecAnalogues.append(precipitationPast[i, j1])
            presentPrecAnalogues.append(precipitationPresent[i, j2])
    return pastPrecAnalogues, presentPrecAnalogues

def DryWet(pastPrecTop, presentPrecTop):
    pastPrecTopRain = np.zeros(pastPrecTop.shape[0])
    presentPrecTopRain = np.zeros(pastPrecTop.shape[0])
    pastRainDict = {np.sum(value[6:12, 4:9]): index for index, value in enumerate(pastPrecTop)}
    presentRainDict = {np.sum(value[6:12, 4:9]): index for index, value in enumerate(presentPrecTop)}

    for i in range(len(pastPrecTop)):
        pastPrecTopRain[i] = np.sum(pastPrecTop[i, 6:12, 4:9])
        presentPrecTopRain[i] = np.sum(presentPrecTop[i, 6:12, 4:9])
        
    pastSorted = np.sort(pastPrecTopRain)
    presentSorted = np.sort(presentPrecTopRain)
    pastDriest = [pastRainDict[dry] for dry in pastSorted[:10]]
    pastWettest = [pastRainDict[wet] for wet in pastSorted[-10:]]
    presentDriest = [presentRainDict[dry] for dry in presentSorted[:10]]
    presentWettest = [presentRainDict[wet] for wet in presentSorted[-10:]]

    pastDriestAnalogues = [pastPrecTop[idx] for idx in pastDriest]
    pastWettestAnalogues = [pastPrecTop[idx] for idx in pastWettest]
    presentDriestAnalogues = [presentPrecTop[idx] for idx in presentDriest]
    presentWettestAnalogues = [presentPrecTop[idx] for idx in presentWettest]

    return np.array(pastDriestAnalogues), np.array(pastWettestAnalogues), np.array(presentDriestAnalogues), np.array(presentWettestAnalogues), [pastDriest, pastWettest, presentDriest, presentWettest]


def RunEnsembles(pslfolder, amoFolder, prFolder, eventFilePath, subFolder, startYear1, endYear1, startYear2, endYear2, startDay = 181, endDay = 270, targetAnalogues = 30, analogueRange = 5, saveFig = False, correlationCoëfficiënt = 0.9, significanceLevel = 0.05, meanRange = 10):
    if not os.path.exists(topEnsemblePlotPath := f"Plots/{subFolder}/Analogues/TopEnsembles/"):
        os.makedirs(f"{topEnsemblePlotPath}")   
    if not os.path.exists(topEnsemblePrecPlotPath := f"Plots/{subFolder}/Precipitation/TopEnsembles/"):
        os.makedirs(f"{topEnsemblePrecPlotPath}")
    if not os.path.exists(wetDryPlotPath := f"Plots/{subFolder}/Precipitation/TopEnsemblesWetDry/"):
        os.makedirs(f"{wetDryPlotPath}")
    if not os.path.exists(wetDryPlotAnaloguePath := f"Plots/{subFolder}/Analogues/TopEnsemblesWetDry/"):
        os.makedirs(f"{wetDryPlotAnaloguePath}")
    # latSlice, lonSlice, mslEventSlice = np.load(f"{eventFilePath}Lat.npy"), np.load(f"{eventFilePath}Lon.npy"), np.load(f"{eventFilePath}msl.npy")
    latSlice, lonSlice, mslEventSlice = np.load(f"{eventFilePath}Lat.npy"), np.load(f"{eventFilePath}Lon.npy"), np.load(f"GPtjes.npy")
    yearLength = endDay - startDay + 1
    
    allAnaloguesPast, allAnaloguesPresent = [], []
    allDistancesPast, allDistancesPresent = [], []
    allDataSlicedPast, allDataSlicedPresent = [], []
    allTrendDataPast, allTrendDataPresent = [], []
    allTypicallityDataPast, allTypicallityDataPresent = [], []
    amosPast, amosPresent = [], []
    yearlyEuclidianDistancesPast, yearlyEuclidianDistancesPresent = [], []
    allIndicesPast, allIndicesPresent = [], []
    allPersistencePast, allPersistencePresent = [], []

    amoData = np.load(f"{amoFolder}tas.npy")
    prData = np.load(f"{prFolder}pr.npy") * 86400
    pslData = np.load(f"{pslfolder}psl.npy")
    
    
    
    pslDataMinRaw = np.min(pslData); pslDataMaxRaw = np.max(pslData)
    
    pslDataAll = pslData.reshape(pslData.shape[0], pslData.shape[1] * pslData.shape[2], pslData.shape[3], pslData.shape[4])
    allDistances = EuclidianDistance(mslEventSlice, pslDataAll, True)
    yearlyEuclidianDistanceAll, runningMeanAll = CalculateYearlyMinimumEuclidianDistance(allDistances, yearLength = yearLength, meanRange = 10, isAllEnsemble = True)
    amoData1, amoData2 = amoData[:, startYear1 - 1: endYear1], amoData[:, startYear1 - 1: endYear1]

    prDataSlice1, prDataSlice2 = prData[:, startYear1 - 1: endYear1], prData[:, startYear2 - 1: endYear2]
    prData1 = prDataSlice1.reshape(prDataSlice1.shape[0], prDataSlice1.shape[1] * prDataSlice1.shape[2], prDataSlice1.shape[3], prDataSlice1.shape[4])
    prData2 = prDataSlice2.reshape(prDataSlice2.shape[0], prDataSlice2.shape[1] * prDataSlice2.shape[2], prDataSlice2.shape[3], prDataSlice2.shape[4])

    pslDataSlice1, pslDataSlice2 = pslData[:, startYear1 - 1: endYear1], pslData[:, startYear2 - 1: endYear2]
    pslData1 = pslDataSlice1.reshape(pslDataSlice1.shape[0], pslDataSlice1.shape[1] * pslDataSlice1.shape[2], pslDataSlice1.shape[3], pslDataSlice1.shape[4])
    pslData2 = pslDataSlice2.reshape(pslDataSlice2.shape[0], pslDataSlice2.shape[1] * pslDataSlice2.shape[2], pslDataSlice2.shape[3], pslDataSlice2.shape[4])
    ensembleData = list((np.squeeze(pslData1[i]), np.squeeze(pslData2[i]), subFolder, i, np.squeeze(amoData1[i]), np.squeeze(amoData2[i]), latSlice, lonSlice, mslEventSlice, yearLength, correlationCoëfficiënt, meanRange, pslDataMinRaw, pslDataMaxRaw) for i in range(len(pslData1)))

    
    with mp.Pool() as pool:
        results = pool.starmap(RunSingleEnsemble, ensembleData)
    
    for i in range(len(results)):
        allDataSlicedPast.extend(results[i][0])
        allDataSlicedPresent.extend(results[i][1])
        allDistancesPast.extend(results[i][2])
        allDistancesPresent.extend(results[i][3])
        allAnaloguesPast.extend(results[i][4])
        allAnaloguesPresent.extend(results[i][5])
        allTrendDataPast.append(results[i][6])
        allTrendDataPresent.append(results[i][7])
        allTypicallityDataPast.append(results[i][8])
        allTypicallityDataPresent.append(results[i][9])
        amosPast.append(results[i][10])
        amosPresent.append(results[i][11])
        yearlyEuclidianDistancesPast.append(results[i][12])
        yearlyEuclidianDistancesPresent.append(results[i][13])
        allIndicesPast.append(results[i][14])
        allIndicesPresent.append(results[i][15])
        allPersistencePast.append(results[i][16])
        allPersistencePresent.append(results[i][17])
    
    with mp.Pool() as pool:
        resultPrecRetrieval = pool.apply_async(RetrievePrecipitationAnalogues, args = (allIndicesPast, allIndicesPresent, prData1, prData2,))

        allAnaloguesPast, allAnaloguesPresent = np.array(allAnaloguesPast), np.array(allAnaloguesPresent)
        topAllEnsemblesIndicesPast, topAllEnsemblesAnalogueDistancesPast = RetrieveAnalogueIndicesDistancesAllEnsembles(allDistancesPast, targetAnalogues = targetAnalogues)
        topAllEnsemblesIndicesPresent, topAllEnsemblesAnalogueDistancesPresent = RetrieveAnalogueIndicesDistancesAllEnsembles(allDistancesPresent, targetAnalogues = targetAnalogues)
        topAllEnsemblesAnaloguesPast, topAllEnsemblesAnaloguesPresent, meanTopEnsemblesPast, meanTopEnsemblesPresent = RetrieveAnalogues(topAllEnsemblesIndicesPast, topAllEnsemblesIndicesPresent, allAnaloguesPast, allAnaloguesPresent)

        np.save(f"{experiment}TopMeanPastDouweZooi", meanTopEnsemblesPast)
        np.save(f"{experiment}TopMeanPresentDouweZooi", meanTopEnsemblesPresent)

        print(f"The following indices can be used for retrieving the correct dates for plotting the bigger area: past: {topAllEnsemblesIndicesPast} and for the present: {topAllEnsemblesIndicesPresent}")
        for i in range(len(topAllEnsemblesAnaloguesPast)):
            PlotAnalogue(latSlice, lonSlice, topAllEnsemblesAnaloguesPast[i], None, saveFig, f"{topEnsemblePlotPath}/PastTopEnsemble{i}")
            PlotAnalogue(latSlice, lonSlice, topAllEnsemblesAnaloguesPresent[i], None, saveFig, f"{topEnsemblePlotPath}/PresentTopEnsemble{i}")

        TeventTopEnsemblesPast, TAnaloguesTopEnsemblesPast = CalculateTypicality(topAllEnsemblesIndicesPast, topAllEnsemblesAnalogueDistancesPast, allDataSlicedPast, targetAnalogues = targetAnalogues, analogueRange = analogueRange)
        TeventTopEnsemblesPresent, TAnaloguesTopEnsemblesPresent = CalculateTypicality(topAllEnsemblesIndicesPresent, topAllEnsemblesAnalogueDistancesPresent, allDataSlicedPresent, targetAnalogues = targetAnalogues, analogueRange = analogueRange)
        persistenceTopEnsemblePast = CalculatePersistence(allDataSlicedPast, topAllEnsemblesIndicesPast, yearLength)
        persistenceTopEnsemblePresent = CalculatePersistence(allDataSlicedPast, topAllEnsemblesIndicesPresent, yearLength)
        pPaTop = [len(p) for p in persistenceTopEnsemblePast]
        pPrTop = [len(p) for p in persistenceTopEnsemblePresent]
        allPersistencePastContainer = np.zeros(np.array(allPersistencePast).shape)
        allPersistencePresentContainer = np.zeros(np.array(allPersistencePresent).shape)
        for i in range(allPersistencePastContainer.shape[0]):
            allPersistencePastContainer[i] = np.array([len(p) for p in allPersistencePast[i]])
            allPersistencePresentContainer[i] = np.array([len(p) for p in allPersistencePresent[i]])

        significanceMask = TTest(topAllEnsemblesAnaloguesPast, topAllEnsemblesAnaloguesPresent, significanceLevel = significanceLevel)
        
        topEnsemblesAnaloguesDifference = meanTopEnsemblesPresent - meanTopEnsemblesPast
        PlotAnalogue(latSlice, lonSlice, meanTopEnsemblesPast, saveFig = saveFig, fileName = f"Plots/{subFolder}/Analogues/TopEnsembles/PastMeanTopEnsembles", minLevel = 101000, maxLevel = 102000) #, minLevel = pslDataMinRaw, maxLevel = pslDataMaxRaw
        PlotAnalogue(latSlice, lonSlice, meanTopEnsemblesPresent, saveFig = saveFig, fileName = f"Plots/{subFolder}/Analogues/TopEnsembles/PresentMeanTopEnsembles", minLevel = 101000, maxLevel = 102000) # , minLevel = pslDataMinRaw, maxLevel = pslDataMaxRaw
        PlotAnalogue(latSlice, lonSlice, topEnsemblesAnaloguesDifference, significanceMask, saveFig = True, fileName = f"Plots/{subFolder}/Analogues/MeanDifferenceTopEnsembles", minLevel = -100, maxLevel = 100)
        
        PlotAllEnsembleViolins(allTypicallityDataPast, allTypicallityDataPresent, TAnaloguesTopEnsemblesPast, TAnaloguesTopEnsemblesPresent, saveFig = saveFig, fileName = f"Plots/{subFolder}/Violin/ViolinAllEnsemblesTypicallity", persistenceEventPast=TeventTopEnsemblesPast, persistenceEventPresent=TeventTopEnsemblesPresent)
        PlotAllEnsembleViolins(allPersistencePastContainer, allPersistencePresentContainer, pPaTop, pPrTop, saveFig = saveFig, fileName = f"Plots/{subFolder}/Violin/ViolinAllEnsemblesPersistence", isPersistence = True)
        PlotAllAMOEnsembleData(amosPast, yearlyEuclidianDistancesPast, saveFig = saveFig, fileName = f"Plots/{subFolder}/AMO/AMOAllEnsemblesPast")
        PlotAllAMOEnsembleData(amosPresent, yearlyEuclidianDistancesPresent, saveFig = saveFig, fileName = f"Plots/{subFolder}/AMO/AMOAllEnsemblesPresent")
        PlotAllAMOEnsembleData(amosPast, yearlyEuclidianDistancesPast, meanRange = meanRange, isRunningMean = True, saveFig = saveFig, fileName = f"Plots/{subFolder}/AMO/AMORunningMeanAllEnsemblesPast")
        PlotAllAMOEnsembleData(amosPresent, yearlyEuclidianDistancesPresent, meanRange = meanRange, isRunningMean = True, saveFig = saveFig, fileName = f"Plots/{subFolder}/AMO/AMORunningMeanAllEnsemblesPresent")
        PlotAllEnsembleTrends(yearlyEuclidianDistanceAll, saveFig = True, fileName = f"Plots/{subFolder}/Trend/TrendsMeanAllEnsembles")
        PlotAllEnsembleTrends(runningMeanAll, saveFig = True, fileName = f"Plots/{subFolder}/Trend/TrendsRunningMeanAllEnsembles")
        ViolinPlot(TAnaloguesTopEnsemblesPast, TAnaloguesTopEnsemblesPresent, TeventTopEnsemblesPast, TeventTopEnsemblesPresent, "Typicality", saveFig, f"Plots/{subFolder}/Violin/ViolinAllEnsemblesTypicallityButThenOnlyTheBest", manualOverride = True)
        ViolinPlot(pPaTop, pPrTop, 3, 3, "Persistence", saveFig, f"Plots/{subFolder}/Violin/ViolinAllEnsemblesPersistenceButThenOnlyTheBest", manualOverridePers = True)
        precPastAnalogues, precPresentAnalogues = resultPrecRetrieval.get()

    return 
    precipitationArgsPast = [] 
    precipitationArgsPresent = []
    flattenedPastPrec = []
    flattenedPresentPrec = []

    # Flatten the precipitation analogues
    for singlePastPrecEnsemble, singlePresentPrecEnsemble in zip(precPastAnalogues, precPresentAnalogues):
        flattenedPastPrec.extend([singlePastPrecEnsemble])
        flattenedPresentPrec.extend([singlePresentPrecEnsemble])
    
    # Create a list of arguments for each plot of the precipitation
    for i, (singlePastPrecAnalogue, singlePresentPrecAnalogue) in enumerate(zip(flattenedPastPrec, flattenedPresentPrec)):
        if not os.path.exists(precPlotPathPast := f"Plots/{subFolder}/Precipitation/Ensemble{(i // targetAnalogues) + 1}/Past"):
            os.makedirs(f"{precPlotPathPast}")
        if not os.path.exists(precPlotPathPresent := f"Plots/{subFolder}/Precipitation/Ensemble{(i // targetAnalogues) + 1}/Present"):
            os.makedirs(f"{precPlotPathPresent}")
        precipitationArgsPast.append([latSlice, lonSlice, singlePastPrecAnalogue, None, False, f"{precPlotPathPast}/{(i % targetAnalogues) + 1}", 0, 100, True, True])
        precipitationArgsPresent.append([latSlice, lonSlice, singlePresentPrecAnalogue, None, False, f"{precPlotPathPresent}/{(i % targetAnalogues) + 1}", 0, 100, True, True])

    print("starting prec plotting")
    for precArgsPast, precArgsPresent in zip(precipitationArgsPast, precipitationArgsPresent):
        PlotAnalogue(*precArgsPast)
        PlotAnalogue(*precArgsPresent)


    # Create plots with the mean of each ensemble and the difference with a significance mask
    for i in range(len(flattenedPastPrec) // targetAnalogues):
        if not os.path.exists(precPlotPathMean := f"Plots/{subFolder}/Precipitation/Ensemble{i + 1}/Mean/"):
            os.makedirs(f"{precPlotPathMean}")
        pastSlice = np.array(flattenedPastPrec[i * targetAnalogues: (i + 1) * targetAnalogues])
        presentSlice = np.array(flattenedPresentPrec[i * targetAnalogues: (i + 1) * targetAnalogues])
        pastMax = np.max(pastSlice, axis = 0)
        presentMax = np.max(presentSlice, axis = 0)
        meanDifference = presentMax - pastMax
        significanceMask = TTest(pastSlice, presentSlice)
        PlotAnalogue(latSlice, lonSlice, pastMax, saveFig = saveFig, fileName = f"{precPlotPathMean}PastMean", minLevel = 0, maxLevel = 80, precLabel = True)
        PlotAnalogue(latSlice, lonSlice, presentMax, saveFig = saveFig, fileName = f"{precPlotPathMean}PresentMean", minLevel = 0, maxLevel = 80, precLabel = True)
        PlotAnalogue(latSlice, lonSlice, meanDifference, significanceMask, saveFig = saveFig, fileName = f"{precPlotPathMean}MeanDifference", precLabel = True, minLevel = -60, maxLevel = 60)
    
    topPrecEnsemblesAnaloguesPast, topPrecEnsemblesAnaloguesPresent, maxTopPrecEnsemblesPast, maxTopPrecEnsemblesPresent = RetrieveAnalogues(topAllEnsemblesIndicesPast, topAllEnsemblesIndicesPresent, np.array(flattenedPastPrec), np.array(flattenedPresentPrec), True)
    topPrecEnsemblesAnaloguesPastDifference = maxTopPrecEnsemblesPresent - maxTopPrecEnsemblesPast

    for i in range(len(topPrecEnsemblesAnaloguesPast)):
        PlotAnalogue(latSlice, lonSlice, topPrecEnsemblesAnaloguesPast[i], None, saveFig, f"{topEnsemblePrecPlotPath}/PastTopEnsemble{i}", precLabel = True, minLevel = 0, maxLevel = 80, isBlues = True)
        PlotAnalogue(latSlice, lonSlice, topPrecEnsemblesAnaloguesPresent[i], None, saveFig, f"{topEnsemblePrecPlotPath}/PresentTopEnsemble{i}", precLabel = True, minLevel = 0, maxLevel = 80, isBlues = True)
    
    significanceMask = TTest(topPrecEnsemblesAnaloguesPast, topPrecEnsemblesAnaloguesPresent) 
    PlotAnalogue(latSlice, lonSlice, maxTopPrecEnsemblesPast, saveFig = saveFig, fileName = f"Plots/{subFolder}/Precipitation/PastMeanTopEnsembles", precLabel = True, minLevel = 0, maxLevel = 80, isBlues = True)
    PlotAnalogue(latSlice, lonSlice, maxTopPrecEnsemblesPresent, saveFig = saveFig, fileName = f"Plots/{subFolder}/Precipitation/PresentMeanTopEnsembles", precLabel = True, minLevel = 0, maxLevel = 80, isBlues = True)
    PlotAnalogue(latSlice, lonSlice, topPrecEnsemblesAnaloguesPastDifference, significanceMask, saveFig = saveFig, fileName = f"Plots/{subFolder}/Precipitation/MeanDifferenceTopEnsembles", precLabel = True, minLevel = -60, maxLevel = 60)
    pastDry, pastWet, presentDry, presentWet, dryWetAnalogueIndices = DryWet(topPrecEnsemblesAnaloguesPast, topPrecEnsemblesAnaloguesPresent)
    if not os.path.exists(wetDryPlotPath := f"Plots/{subFolder}/Precipitation/TopEnsemblesWetDry/"):
        os.makedirs(f"{wetDryPlotPath}")
    
    # Nog juist axis toevoegen en zorgend at t een np.array is
    pastDryMean, pastWetMean, presentDryMean, presentWetMean = np.mean(pastDry, axis = 0), np.mean(pastWet, axis = 0), np.mean(presentDry, axis = 0), np.mean(presentWet, axis = 0)
    PlotAnalogue(latSlice, lonSlice, pastDryMean, None, saveFig, f"{wetDryPlotPath}PastMeanDry", minLevel = 0, maxLevel = 80, isBlues = True, precLabel = True)
    PlotAnalogue(latSlice, lonSlice, pastWetMean, None, saveFig, f"{wetDryPlotPath}PastMeanWet", minLevel = 0, maxLevel = 80, isBlues = True, precLabel = True)
    PlotAnalogue(latSlice, lonSlice, presentDryMean, None, saveFig, f"{wetDryPlotPath}PresentMeanDry", minLevel = 0, maxLevel = 80, isBlues = True, precLabel = True)
    PlotAnalogue(latSlice, lonSlice, presentWetMean, None, saveFig, f"{wetDryPlotPath}PresentMeanWet", minLevel = 0, maxLevel = 80, isBlues = True, precLabel = True)
    significanceMaskPrecDry = TTest(pastDry, presentDry, significanceLevel = significanceLevel)
    significanceMaskPrecWet = TTest(pastWet, presentWet, significanceLevel = significanceLevel)
    PlotAnalogue(latSlice, lonSlice, presentDryMean - pastDryMean, None, saveFig, f"{wetDryPlotPath}DryMeanDifference{i}", minLevel = -10, maxLevel = 10, precLabel = True)
    PlotAnalogue(latSlice, lonSlice, presentWetMean - pastWetMean, None, saveFig, f"{wetDryPlotPath}WetMeanDifference{i}", minLevel = -10, maxLevel = 10, precLabel = True)
    
    return

    for i in range(len(pastDry)):
        PlotAnalogue(latSlice, lonSlice, pastDry[i], None, saveFig, f"{wetDryPlotPath}PastDry{i}", precLabel = True, isBlues = True, minLevel = 0, maxLevel = 80)
        PlotAnalogue(latSlice, lonSlice, pastWet[i], None, saveFig, f"{wetDryPlotPath}PastWet{i}", precLabel = True, isBlues = True, minLevel = 0, maxLevel = 80)
        PlotAnalogue(latSlice, lonSlice, presentDry[i], None, saveFig, f"{wetDryPlotPath}PresentDry{i}", precLabel = True, isBlues = True, minLevel = 0, maxLevel = 80)
        PlotAnalogue(latSlice, lonSlice, presentWet[i], None, saveFig, f"{wetDryPlotPath}PresentWet{i}", precLabel = True, isBlues = True, minLevel = 0, maxLevel = 80)
    
    plotNameDict = {0: "PastDry", 1: "PastWet", 2: "PresentDry", 3: "PresentWet"}
    pastDryPsl, pastWetPsl, presentDryPsl, presentWetPsl = [[] for _ in range(4)]
    for i in range(len(dryWetAnalogueIndices)):
        for counter, idx in enumerate(dryWetAnalogueIndices[i]):
            if i == 0:
                pastDryPsl.append(topAllEnsemblesAnaloguesPast[idx])
            elif i == 1:
                pastWetPsl.append(topAllEnsemblesAnaloguesPast[idx])
            elif i == 2:
                presentDryPsl.append(topAllEnsemblesAnaloguesPast[idx])
            else:
                presentWetPsl.append(topAllEnsemblesAnaloguesPast[idx])
            PlotAnalogue(latSlice, lonSlice, topAllEnsemblesAnaloguesPast[idx], None, saveFig, f"{wetDryPlotAnaloguePath}{plotNameDict[i]}{counter}")
    pastDryPsl, pastWetPsl, presentDryPsl, presentWetPsl = np.array(pastDryPsl), np.array(pastWetPsl), np.array(presentDryPsl), np.array(presentWetPsl)
    pastDryPslMean, pastWetPslMean, presentDryPslMean, presentWetPslMean = np.mean(pastDryPsl, axis = 0), np.mean(pastWetPsl, axis = 0), np.mean(presentDryPsl, axis = 0), np.mean(presentWetPsl, axis = 0)
    PlotAnalogue(latSlice, lonSlice, pastDryPslMean, None, saveFig, f"{wetDryPlotAnaloguePath}PastMeanDryPsl")
    PlotAnalogue(latSlice, lonSlice, pastWetPslMean, None, saveFig, f"{wetDryPlotAnaloguePath}PastMeanWet")
    PlotAnalogue(latSlice, lonSlice, presentDryPslMean, None, saveFig, f"{wetDryPlotAnaloguePath}PresentMeanDry")
    PlotAnalogue(latSlice, lonSlice, presentWetPslMean, None, saveFig, f"{wetDryPlotAnaloguePath}PresentMeanWet")
    significanceMaskAnalogueDry = TTest(pastDryPsl, presentDryPsl, significanceLevel = significanceLevel)
    significanceMaskAnalogueWet = TTest(pastWetPsl, presentWetPsl, significanceLevel = significanceLevel)
    PlotAnalogue(latSlice, lonSlice, presentDryPslMean - pastDryPslMean, significanceMaskAnalogueDry, saveFig, f"{wetDryPlotAnaloguePath}DryMeanDifference{i}")
    PlotAnalogue(latSlice, lonSlice, presentWetPslMean - pastWetPslMean, significanceMaskAnalogueWet, saveFig, f"{wetDryPlotAnaloguePath}WetMeanDifference{i}")

def AddSingleForcingsTogether(lat, lon, *files):
    dataStoragePast = np.zeros(np.load(files[0]).shape)
    dataStoragePresent = np.zeros(np.load(files[0]).shape)
    for i, file in enumerate(files):
        if i < 4:
            dataStoragePast += np.load(file) 
        else:
            dataStoragePresent += np.load(file) 

    PlotAnalogue(lat, lon, dataStoragePast, None, True, "AddedSingleForcingsPast")
    PlotAnalogue(lat, lon, dataStoragePresent, None, True, "AddedSingleForcingsPresent")
    PlotAnalogue(lat, lon, dataStoragePresent - dataStoragePast, None, True, "AddedSingleForcingsDifference", minLevel = -240, maxLevel = 240)

def plot_variable(lat, lon, variable, fileName):
    plt.figure(figsize = (10,10))
    ax = plt.axes(projection = ccrs.PlateCarree())
    plot = plt.contourf(lon, lat, variable, cmap = "RdBu_r", transform = ccrs.PlateCarree(), levels = 15) 
    ax.coastlines()
    ax.add_feature(cf.BORDERS)
    plt.colorbar(plot, ax = ax, orientation = "horizontal", label = "degrees celcius/GWD", pad = 0.05)
    plt.savefig(f"{fileName}", dpi = 300)
    plt.show()
    plt.close()
#%%
# runSpecification = "DouwehistAllEnsembles"
# experiment = "hist"
# pslfolder = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/psl/"
# amoFolder = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/tas/"
# precFolder = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/pr/"
# eventFileLocation = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/msl/"
# startDay, endDay = 181, 270
# originalYearLength = 360
# nEnsembles = 55
# bbox = [41, 59, -5, 23]

# jantje = np.load(f"/usr/people/noest/stage_folders/event_data/Vautard_southerlyflow_2019-06-29_regridded_data_at_index_25381.npy")
# lat1, lon1 = np.load(f"{eventFileLocation}Lat.npy"), np.load(f"{eventFileLocation}Lon.npy")
# latSlice1, lonSlice1, varSlice1 = SliceData(bbox, lat1, lon1, jantje, startDay, endDay, originalYearLength, True)
# np.save("GPtjes.npy", varSlice1)
#%%
runSpecification = "DouweGHGAllEnsembles"
experiment = "GHG"
eventFileLocation = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/msl/"
beginning = "/usr/people/huis/Stage/Scripts/"
latSlice, lonSlice, mslEventSlice = np.load(f"{eventFileLocation}Lat.npy"), np.load(f"{eventFileLocation}Lon.npy"), np.load(f"GPtjes.npy")
AddSingleForcingsTogether(latSlice, lonSlice, f"{beginning}GHGTopMeanPastDouweZooi.npy", f"{beginning}AERTopMeanPastDouweZooi.npy", f"{beginning}SOLTopMeanPastDouweZooi.npy", f"{beginning}VOLTopMeanPastDouweZooi.npy", f"{beginning}GHGTopMeanPresentDouweZooi.npy", f"{beginning}AERTopMeanPresentDouweZooi.npy", f"{beginning}SOLTopMeanPresentDouweZooi.npy", f"{beginning}VOLTopMeanPresentDouweZooi.npy")
PlotAnalogue(latSlice, lonSlice, np.load(f"{beginning}histTopMeanPastDouweZooi.npy"), None, True, "histPast")
PlotAnalogue(latSlice, lonSlice, np.load(f"{beginning}histTopMeanPresentDouweZooi.npy"), None, True, "histPresent")
PlotAnalogue(latSlice, lonSlice, np.load(f"{beginning}histTopMeanPresentDouweZooi.npy") - np.load(f"{beginning}histTopMeanPastDouweZooi.npy"), None, True, "histDifference", minLevel = -240, maxLevel = 240)
# %%
# if __name__ == "__main__":
#     runSpecification = "DouwehistAllEnsembles"
#     experiment = "hist"
#     pslfolder = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/psl/"
#     amoFolder = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/tas/"
#     precFolder = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/pr/"
#     eventFileLocation = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/msl/"
#     originalYearLength = 360
#     targetAnalogues, analogueRange = 30, 5
#     saveFig = False
#     correlationCoëfficiënt = 0.9
#     startYearPast, endYearPast = 1, 30
#     startYearPresent, endYearPresent = 136, 165
#     significanceLevel = 0.05
#     meanRange = 10
#     t1 = time.perf_counter()
#     RunEnsembles(pslfolder, amoFolder, precFolder, eventFileLocation, runSpecification, startYearPast, endYearPast, startYearPresent, endYearPresent, startDay = 181, endDay = 270, targetAnalogues = targetAnalogues, analogueRange = analogueRange, saveFig = saveFig, correlationCoëfficiënt = correlationCoëfficiënt, significanceLevel = significanceLevel, meanRange = meanRange)
#     t2 = time.perf_counter()
#     print(t2 - t1)
# #%%
# if __name__ == "__main__":
#     runSpecification = "DouweGHGAllEnsembles"
#     experiment = "GHG"
#     pslfolder = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/psl/"
#     amoFolder = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/tas/"
#     precFolder = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/pr/"
#     eventFileLocation = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/msl/"
#     originalYearLength = 360
#     targetAnalogues, analogueRange = 30, 5
#     saveFig = False
#     correlationCoëfficiënt = 0.9
#     startYearPast, endYearPast = 1, 30
#     startYearPresent, endYearPresent = 136, 165
#     significanceLevel = 0.05
#     meanRange = 10
#     t1 = time.perf_counter()
#     RunEnsembles(pslfolder, amoFolder, precFolder, eventFileLocation, runSpecification, startYearPast, endYearPast, startYearPresent, endYearPresent, startDay = 181, endDay = 270, targetAnalogues = targetAnalogues, analogueRange = analogueRange, saveFig = saveFig, correlationCoëfficiënt = correlationCoëfficiënt, significanceLevel = significanceLevel, meanRange = meanRange)
#     t2 = time.perf_counter()
#     print(t2 - t1)
#%%
# if __name__ == "__main__":
#     runSpecification = "DouweAERAllEnsembles"
#     experiment = "AER"
#     pslfolder = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/psl/"
#     amoFolder = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/tas/"
#     precFolder = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/pr/"
#     eventFileLocation = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/msl/"
#     originalYearLength = 360
#     targetAnalogues, analogueRange = 30, 5
#     saveFig = False
#     correlationCoëfficiënt = 0.9
#     startYearPast, endYearPast = 1, 30
#     startYearPresent, endYearPresent = 136, 165
#     significanceLevel = 0.05
#     meanRange = 10
#     t1 = time.perf_counter()
#     RunEnsembles(pslfolder, amoFolder, precFolder, eventFileLocation, runSpecification, startYearPast, endYearPast, startYearPresent, endYearPresent, startDay = 181, endDay = 270, targetAnalogues = targetAnalogues, analogueRange = analogueRange, saveFig = saveFig, correlationCoëfficiënt = correlationCoëfficiënt, significanceLevel = significanceLevel, meanRange = meanRange)
#     t2 = time.perf_counter()
#     print(t2 - t1)
#%%
# if __name__ == "__main__":
#     runSpecification = "DouweSOLAllEnsembles"
#     experiment = "SOL"
#     pslfolder = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/psl/"
#     amoFolder = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/tas/"
#     precFolder = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/pr/"
#     eventFileLocation = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/msl/"
#     originalYearLength = 360
#     targetAnalogues, analogueRange = 30, 5
#     saveFig = False
#     correlationCoëfficiënt = 0.9
#     startYearPast, endYearPast = 1, 30
#     startYearPresent, endYearPresent = 136, 165
#     significanceLevel = 0.05
#     meanRange = 10
#     t1 = time.perf_counter()
#     RunEnsembles(pslfolder, amoFolder, precFolder, eventFileLocation, runSpecification, startYearPast, endYearPast, startYearPresent, endYearPresent, startDay = 181, endDay = 270, targetAnalogues = targetAnalogues, analogueRange = analogueRange, saveFig = saveFig, correlationCoëfficiënt = correlationCoëfficiënt, significanceLevel = significanceLevel, meanRange = meanRange)
#     t2 = time.perf_counter()
#     print(t2 - t1)
#%%
# if __name__ == "__main__":
#     runSpecification = "DouweVOLAllEnsembles"
#     experiment = "VOL"
#     pslfolder = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/psl/"
#     amoFolder = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/tas/"
#     precFolder = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/pr/"
#     eventFileLocation = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/msl/"
#     originalYearLength = 360
#     targetAnalogues, analogueRange = 30, 5
#     saveFig = False
#     correlationCoëfficiënt = 0.9
#     startYearPast, endYearPast = 1, 30
#     startYearPresent, endYearPresent = 136, 165
#     significanceLevel = 0.05
#     meanRange = 10
#     t1 = time.perf_counter()
#     RunEnsembles(pslfolder, amoFolder, precFolder, eventFileLocation, runSpecification, startYearPast, endYearPast, startYearPresent, endYearPresent, startDay = 181, endDay = 270, targetAnalogues = targetAnalogues, analogueRange = analogueRange, saveFig = saveFig, correlationCoëfficiënt = correlationCoëfficiënt, significanceLevel = significanceLevel, meanRange = meanRange)
#     t2 = time.perf_counter()
#     print(t2 - t1)
#%%
# runSpecification = "GHGAllEnsembles"
# experiment = "GHG"
# eventFileLocation = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/msl/"
# latSlice, lonSlice, mslEventSlice = np.load(f"{eventFileLocation}Lat.npy")[6:12], np.load(f"{eventFileLocation}Lon.npy")[4:9], np.load(f"{eventFileLocation}msl.npy")[6:12, 4:9]
# print(mslEventSlice.shape)
# print(latSlice)
# print(lonSlice)
# PlotAnalogue(latSlice, lonSlice, mslEventSlice, None, True, "TestKaas", None, None, True)
#%%
# runSpecification = "GHGAllEnsembles"
# experiment = "GHG"
# eventFileLocation = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/msl/"
# eventItselfLocation = f"/usr/people/huis/Stage/Data/Limburg_14Jul2021_ERA5_msl.nc"
# latSlice, lonSlice, mslEventSlice = np.load(f"{eventFileLocation}Lat.npy"), np.load(f"{eventFileLocation}Lon.npy"), np.load(f"{eventFileLocation}msl.npy")
# print(mslEventSlice.shape)
# PlotAnalogue(latSlice, lonSlice, mslEventSlice, None, True, "EventPSL", 101000, 102000, True)
#%%
# def extract_years_and_months_era5(era5_data,desired_start_month,desired_end_month,desired_start_year,desired_end_year,list_with_all_months,list_with_all_years):
#     """
#     Slice era5 data based on the months and years, months and years that are used as input variable will be included as well
#     """
#     if era5_data.shape[0] != len(list_with_all_months):
#         print ("Error: Amount of days in the data and list with all dates are not the same")
   
#     list_for_filtered_era5_data = []
#     list_for_filtered_years = []
#     list_for_filtered_months = []
#     for i in range(era5_data.shape[0]):
#         month_at_index = list_with_all_months[i]
#         year_at_index = list_with_all_years[i]
#         if month_at_index >= desired_start_month and month_at_index <= desired_end_month and year_at_index >= desired_start_year and year_at_index <= desired_end_year:
#             data_to_select = era5_data[i,:,:]
#             list_for_filtered_era5_data.append(data_to_select)
#             list_for_filtered_years.append(year_at_index)
#             list_for_filtered_months.append(month_at_index)
#     array_selected_era5_data = np.array(list_for_filtered_era5_data)
 
#     return array_selected_era5_data, list_for_filtered_years, list_for_filtered_months

# def lists_for_era5_dates(final_year,final_month,final_day):
#     import datetime

#     """
#     Creates lists with all months and years in the complete ERA5 data, taking into account leap days (schrikkeldagen)
#     """
#     start_date_all_era5_data = datetime.datetime(1950, 1, 1) # Is included
#     end_date_all_era5_data = datetime.datetime(final_year,final_month,final_day) # Is included
#     delta_time = datetime.timedelta(days=1)
 
#     date_list_basic = []
#     current_date = start_date_all_era5_data
#     while current_date <= end_date_all_era5_data:
#         date_list_basic.append(current_date)
#         current_date += delta_time
#     date_strings = [date.strftime('%Y-%m-%d') for date in date_list_basic]
#     month_list = [date.month for date in date_list_basic]
#     year_list = [date.year for date in date_list_basic]
 
#     return month_list, year_list, date_strings
# #%%
# def SliceData(bbox, lat, lon, variable, startDay = 1, endDay = 360, originalYearLength = 360, isEvent = False):
#     """
#     This function slices the data given the S, N, W, E bounds and slices the time based on the start 
#     day and end day within a year
#     """
#     # Change longitude data to go from -180 to 180
#     mask = [lon > 180]
#     lon[mask[0]] = lon[mask[0]] - 360

#     # Calculate the index of the bounds
#     sIndex = np.argmin(np.abs(lat - bbox[0]))
#     nIndex = np.argmin(np.abs(lat - bbox[1]))
#     wIndex = np.argmin(np.abs(lon - bbox[2]))
#     eIndex = np.argmin(np.abs(lon - bbox[3]))

#     if isEvent:
#         variable = np.expand_dims(variable, axis = 0)

#     if wIndex > eIndex: # If the west index is higher than the east index, think of the right side of the world map as left boundary and vice versa
#         latSlice = lat[sIndex: nIndex + 1]
#         lonSlice = np.concatenate((lon[wIndex:], lon[:eIndex + 1]))
#         variableSlice = np.concatenate((variable[:, sIndex: nIndex + 1, wIndex:], variable[:, sIndex: nIndex + 1, :eIndex + 1]), axis = 2) 
        
#     else:
#         latSlice = lat[sIndex: nIndex + 1]
#         lonSlice = lon[wIndex: eIndex + 1]
#         variableSlice = variable[:, sIndex: nIndex + 1, wIndex: eIndex + 1]

#     if isEvent:
#         return latSlice, lonSlice, variableSlice
    
#     years = int(variableSlice.shape[0] / originalYearLength)
#     data = variableSlice.reshape(years, originalYearLength, variableSlice.shape[1], variableSlice.shape[2])
#     dataSeason = data[:, startDay - 1: endDay, :, :]
#     # dataSeasonUnraveled = dataSeason.reshape(dataSeason.shape[0] * dataSeason.shape[1], variableSlice.shape[1], variableSlice.shape[2])
#     return latSlice, lonSlice, dataSeason

# data_path_pressure = "/net/pc200246/nobackup/users/noest/ERA5_regridded/era5_msl_daily_regridded.nc"
# dataHoofd = nc.Dataset(data_path_pressure)
# data = dataHoofd["msl"][:]
# lat = dataHoofd["lat"][:]
# lon = dataHoofd["lon"][:]

# ml, yl, ds = lists_for_era5_dates(2025, 12, 31)
# driedingen = extract_years_and_months_era5(data, 7, 7, 2021, 2021, ml, yl)
# daadwerkelijkedata = SliceData([41, 59, -5, 23], lat, lon, driedingen[0], 1, 31, 31, False)[2][0]
# # print(daadwerkelijkedata.shape)
# bla = CalculatePersistence(daadwerkelijkedata, [15], 31)
# print(bla)
# def CustomTrendPlot(meanRange, runningMean1, runningMean2, saveFig, fileName):
#     plt.figure(figsize = (25, 10))
#     # plt.ylim(0.80, 0.95)
#     startYear = 1850
#     endYear = 2014
#     plt.xticks(np.arange(1850, 2014, 25))
#     plt.xlim(1850, 2014)
#     plt.grid(axis = "y")
#     plt.ylabel("1 - (min(ED) / max(ED))")
#     plt.plot(range(startYear + (meanRange // 2 - 1), endYear - (meanRange // 2 - 1)), runningMean1, label = "10-year running mean ensemble 24", color = "turquoise")
#     plt.plot(range(startYear + (meanRange // 2 - 1), endYear - (meanRange // 2 - 1)), runningMean2, label = "10-year running mean ensemble 33", color = "black")
#     plt.axvline(x = 1880, color = "r", linestyle = "--")
#     plt.axvline(x = 1984, color = "r", linestyle = "--")
#     plt.legend()
#     if saveFig:
#         plt.savefig(f"{fileName}.png", dpi = 250, bbox_inches = "tight")
#     plt.close()
# #%%
# CustomTrendPlot(10, np.load("RunningmeanZooi23.npy"), np.load("RunningmeanZooi32.npy"), True, "Koffiemok")
#%%
# Streamfunction not because of lack of data of SOL and VOL
# AMO could be added but don't know if it has added value
# Nog over die trend en of ik dan het gemiddelde ofzo moet nemen
#%%
# jantje = "/usr/people/noest/stage_folders/event_data/Vautard_southerlyflow_2019-06-29_regridded_data_at_index_25381.npy"
# bbox = [41, 59, -5, 23]
# startDay, endDay = 181, 270
# originalYearLength = 360
# nEnsembles = 55
# runSpecification = "DouweGHGAllEnsembles"
# experiment = "GHG"
# eventFileLocation = f"/net/pc200039/nobackup/users/huis/LESFMIP/HadGEM/{experiment}/{runSpecification}/msl/"
# kaas = np.load(jantje)
# lat, lon = np.load(f"{eventFileLocation}Lat.npy"), np.load(f"{eventFileLocation}Lon.npy")
# plakjeKaas = SliceData(bbox, lat, lon, kaas, isEvent = True)[2].squeeze()
# PlotAnalogue(lat, lon, plakjeKaas, None, True, "Jantje10000")
# print(plakjeKaas.shape)