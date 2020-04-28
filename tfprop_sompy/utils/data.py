from sklearn import cluster
import pandas as pd
from typing import List
import sompy
import itertools
import numpy as np

def estimate_bandwidth_on_materials(mats_df: pd.DataFrame, materials: List[str]) -> float:
    # NOTE: this might be a flagrant misinterpretation of what the `estimate_bandwidth` function does
    """
    Returns the "bandwidth" of a set of items from a dataframe.
    
    :param mats_df: The dataframe to use
    :param materials: A list of material names
    """
    X = mats_df.loc[pd.Index(materials)].values
    return cluster.estimate_bandwidth(X)
    
def calculate_SOM_radius(mats_df: pd.DataFrame, materials: List[str], sm: sompy.sompy.SOM):
    """
    :param mats_df: Dataframe containing all information
    :param materials: List of material names to calculate the radius of in the SOM's space
    :param sm: Self-organizing map to find the radius on
    
    :returns: Finds the radius of a circle that can surround all points
    """
    
    component_idx = pd.Index(*sm._component_names)
    
    # turns the materials into x, y pairs
    xy_list = np.array(list(map(lambda x: (*x[0:2],), 
        sm.bmu_ind_to_xy(sm.project_data(mats_df.loc[materials][component_idx].values)))))

    # Calculate centroid
    centroid = np.mean(xy_list, axis=0)
    
    biggest_radius = 0
    # Brute approach - pairwise iterate through the points and find the maximum distance among all of them
    for (x, y) in xy_list:
        biggest_radius = max(biggest_radius, np.sqrt((x-centroid[0])**2 + (y-centroid[1])**2))
        
    return biggest_radius, centroid
    
def calculate_euclidean_radius(mats_df: pd.DataFrame, materials: List[str], parameters: List[str], normalization: str=None):
    """
    :param mats_df: Dataframe containing all information
    :param materials: List of material names to calculate the radius of in normalized space
    :param parameters: The material parameters to calculate the euclidean radius in
    :param normalization: Any one of various normalization schemes available to SOMPY. 
        As with SOMPY, default is "variance", which is standard deviations from the mean.
    """
    # I think there's some method to do this speedyfast instead of iterating within python
    # But this should do for now
    if normalization is None:
        normalization = 'var'
    
    # Our data frame only over the selected properties
    params_df = mats_df[parameters]
    
    normalizer = sompy.normalization.NormalizerFactory.build(normalization)
    
    hyperpoints = normalizer.normalize_by(params_df.values, params_df.loc[materials].values)
    
    hyperpoints_centroid = np.mean(hyperpoints, axis=0)
    
    biggest_radius = 0
    for pnt in hyperpoints:
        # Euclidean distance: (x-y)**2
        centroid_offset = pnt - hyperpoints_centroid
        
        euc_dist = np.dot(centroid_offset, centroid_offset)
        biggest_radius = max(biggest_radius, np.sqrt(euc_dist))
        
    return biggest_radius, hyperpoints_centroid
    
