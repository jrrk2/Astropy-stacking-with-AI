
def parse_log_file(log_file_path):
    """Extract transformation matrices from the Stellina log file"""
    print(f"Parsing log file: {log_file_path}")
    
    with open(log_file_path, 'r') as f:
        log_content = f.read()
    
    # Parse JSON
    try:
        log_data = json.loads(log_content)
    except json.JSONDecodeError:
        print("Error parsing JSON. Check the log file format.")
        return []
    
    transformations = []
    
    # Extract successful registration results
    if 'data' in log_data and 'corrections' in log_data['data']:
        for correction in log_data['data']['corrections']:
            if 'postCorrections' in correction:
                for post_correction in correction['postCorrections']:
                    if 'acqFile' in post_correction and 'stackingData' in post_correction['acqFile']:
                        stacking_data = post_correction['acqFile']['stackingData']
                        
                        if stacking_data.get('error') is None and 'liveRegistrationResult' in stacking_data:
                            reg_result = stacking_data['liveRegistrationResult']
                            
                            if 'matrix' in reg_result and reg_result.get('statusMessage') == 'StackingOk':
                                idx = reg_result['idx']
                                matrix = reg_result['matrix']
                                roundness = reg_result.get('roundness', 0)
                                stars_used = reg_result.get('starsUsed', 0)
                                
                                # Construct the transformation matrix
                                transform_matrix = np.array([
                                    [matrix[0], matrix[1], matrix[2]],
                                    [matrix[3], matrix[4], matrix[5]],
                                    [matrix[6], matrix[7], matrix[8]]
                                ])
                                
                                # Get file information
                                file_index = post_correction['acqFile'].get('index', -1)
                                file_path = post_correction['acqFile'].get('path', '')
                                mean_value = post_correction['acqFile'].get('mean', 0)
                                
                                # Get temperature metadata if available
                                motors = post_correction['acqFile'].get('motors', {})
                                metadata = post_correction['acqFile'].get('metadata', {})
                                
                                transformations.append({
                                    'idx': idx,
                                    'matrix': transform_matrix,
                                    'roundness': roundness,
                                    'stars_used': stars_used,
                                    'file_index': file_index,
                                    'file_path': file_path,
                                    'mean_value': mean_value,
                                    'motors': motors,
                                    'metadata': metadata
                                })
    
    # Sort by the original stacking index
    transformations.sort(key=lambda x: x['idx'])
    print(f"Extracted {len(transformations)} valid transformations")
    return transformations
