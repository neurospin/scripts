import os
from multiprocessing import cpu_count
from multiprocessing import Pool

database_parcel = "hcp"
SYMMETRIC = True
COLLECT = False

def create_distrib_thresholds(feature):
    cmd = "python Depth_histogram.py -f "+feature+" -s "+str(int(SYMMETRIC))+" -d "+database_parcel+" -c "+str(int(COLLECT))
    print cmd
    os.system(cmd)

def create_phenotypes(pair):
    feature, feature_threshold = pair
    cmd = "python cluster_new_thresholding.py -f "+feature+" -s "+str(int(SYMMETRIC))+" -d "+database_parcel+" -t "+feature_threshold
    print cmd
    os.system(cmd)

def run_solar(pair):
    feature, feature_threshold = pair
    # SOLAR NEEDS TO BE RUN SEQUENTIALLY
    # EVEN IF THE WORKING_DIR IS SET PROPERLY IN THE TCL SCRIPT
    # IT SEEMS TO ENCOUNTER TROUBLE OTHERWISE
    cmd = "python run_solar.py -f "+feature+" -s "+str(int(SYMMETRIC))+" -d "+database_parcel+" -t "+feature_threshold
    print cmd
    os.system(cmd)
    
def parse_and_map_output(pair):
    feature, feature_threshold = pair 
    cmd = "python parsing_output.py -f "+feature+" -s "+str(int(SYMMETRIC))+" -d "+database_parcel+" -t "+feature_threshold
    print cmd
    os.system(cmd)
    cmd = "python map_pits_heritability.py -f "+feature+" -s "+str(int(SYMMETRIC))+" -d "+database_parcel+" -t "+feature_threshold
    print cmd
    #os.system(cmd)

if __name__ == '__main__':
    features = ['DPF', 'sulc', 'curv']#, 'thickness'] <- check usual values of thickness first, before including it
    #features_thresholds = ['DPF', 'sulc']
    features_thresholds = ['sulc']
    number_CPU = min(len(features), cpu_count())
    #pool = Pool(processes = number_CPU)
    #pool.map(create_distrib_thresholds, features)
    """
    pairs = []
    for feature in features:
        for feature_threshold in features_thresholds:
            pairs.append([feature, feature_threshold])
            
    pool = Pool(processes = number_CPU)
    pool.map(create_phenotypes, pairs)

    for pair in pairs:
        run_solar(pair)
        parse_and_map_output(pair)
    
    pool = Pool(processes = number_CPU)
    pool.map(parse_and_map_output, pairs) 
    """
    feature = 'DPF'
    feature_threshold = 'DPF'
    pair = [feature, feature_threshold]
    #create_phenotypes(pair)
    run_solar(pair)
    parse_and_map_output(pair)
    
