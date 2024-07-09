#import packages
from arcgis.geometry import arcpy
from arcgis.gis import GIS
from arcgis.raster.functions import remap, remap_range
import numpy as np
from sklearn.cluster import KMeans

# initialize the GIS
gis = GIS("home")

# define input & output paths
workspace = ""
output_workspace = ""
temp_raster_path = "in_memory/Raster_Temp"

# Define the criteria datasets with full paths
exposure_Criteria = [
    "",
    "",
    ""
]
hazard_Criteria = [
    "",
    "",
    "",
    "",
    ""
]
community_vulnerability = [1]  # path tbd

# ask user for weights
def get_weights(index_name, datasets):
    weights = []
    print(f"Assign weights for {index_name} criteria, the weights must sum up to 10")
    for dataset in datasets:
        while True:
            try:
                weight = float(input(f"Enter weight for {dataset}: "))
                weights.append(weight)
                break
            except ValueError:
                print("Invalid input, please enter a numeric value.")
    if sum(weights) != 10:
        raise ValueError(f"Sum of weights for {index_name} must be 10. Current sum is {sum(weights)}")
    return weights

# get criteria weights
exposure_weights = get_weights("exposure", exposure_Criteria)
hazard_weights = get_weights("hazard", hazard_Criteria)

# calculate weighted sum for variable indices
def calculate_weighted_sum(datasets, weights):
    weighted_sum = None
    for dataset, weight in zip(datasets, weights):
        raster = gis.content.get(dataset).layers[0]
        weighted_raster = raster * weight
        if weighted_sum is None:
            weighted_sum = weighted_raster
        else:
            weighted_sum += weighted_raster
    return weighted_sum

# calculate variable indices
exposure_Index = calculate_weighted_sum(exposure_Criteria, exposure_weights)
hazard_Index = calculate_weighted_sum(hazard_Criteria, hazard_weights)
community_vulnerability_Index = 1  # index values tbd

# aggregate variable indices to calculate risk index
risk_Index = (exposure_Index ** (1/3)) + (hazard_Index ** (1/3)) + (community_vulnerability_Index ** (1/3))

# convert the risk index to an integer so stats can be generated
risk_Index_int = remap(risk_Index, remap_values=None, output_type="INT")

# save to a temporary location
risk_Index_int.save(temp_raster_path)

# get values from the raster to calculate natural breaks for reclassification
nodata_value = -9999
raster_array = risk_Index_int.read()
raster_array = np.ma.masked_equal(raster_array, nodata_value)
raster_values = raster_array.compressed()

# calculate natural breaks with KMeans clustering
num_classes = 5
kmeans = KMeans(n_clusters=num_classes, n_init=10)
raster_values = raster_values.reshape(-1, 1)
kmeans.fit(raster_values)
breaks = sorted(kmeans.cluster_centers_.flatten())

# create reclassified raster based on natural breaks
reclass_ranges = []
for i in range(len(breaks) - 1):
    reclass_ranges.append([breaks[i], breaks[i + 1], i + 1])
# maximum value for last range
reclass_ranges.append([breaks[-1], raster_values.max(), num_classes])
reclassified_raster = remap_range(temp_raster_path, reclass_ranges, output_type="INT")

# save raster
reclassified_raster.save(output_workspace + "/WFR_index_Reclass")

print(f"Reclassified raster saved to {output_workspace}/WFR_index_Reclass")
