class DarkFrameManager:
    def __init__(self, dark_base_dir):
        """Initialize with base directory containing temp_XXX subdirectories"""
        self.dark_base_dir = Path(dark_base_dir)
        self.dark_frames = {}  # {temp_k: (data, pattern)}
        self._load_dark_frames()
        
    def _load_dark_frames(self):
        """Load master dark for each temperature directory"""
        for temp_dir in self.dark_base_dir.glob("temp_*"):
            try:
                temp_k = int(temp_dir.name.split('_')[1])
                dark_files = list(temp_dir.glob("*.fits"))
                if not dark_files:
                    print(f"Warning: No FITS files found in {temp_dir}")
                    continue
                    
                with fits.open(dark_files[0]) as hdul:
                    pattern = hdul[0].header.get('BAYERPAT', 'RGGB').strip()
                    self.dark_frames[temp_k] = {
                        'data': hdul[0].data.astype(np.float32),  # Ensure float for calculations
                        'pattern': pattern
                    }
                print(f"Loaded dark frame for {temp_k}K")
                
            except ValueError as e:
                print(f"Skipping invalid temperature directory: {temp_dir}")
        
        self.temp_range = (min(self.dark_frames.keys()), max(self.dark_frames.keys()))
        print(f"Loaded {len(self.dark_frames)} master darks, temperature range: {self.temp_range[0]}-{self.temp_range[1]}K")

    def celsius_to_kelvin(self, temp_c):
        """Convert Celsius to Kelvin"""
        return temp_c + 273.15

    def _extrapolate_dark(self, temp_k):
        """Extrapolate dark frame for temperatures outside our range"""
        temps = sorted(self.dark_frames.keys())
        
        if temp_k > self.temp_range[1]:
            # Use two highest temperature darks for extrapolation
            t1, t2 = temps[-2], temps[-1]
        else:
            # Use two lowest temperature darks for extrapolation
            t1, t2 = temps[0], temps[1]
            
        dark1 = self.dark_frames[t1]['data']
        dark2 = self.dark_frames[t2]['data']
        
        # Calculate rate of change per degree
        delta_per_k = (dark2 - dark1) / (t2 - t1)
        
        # Extrapolate
        if temp_k > self.temp_range[1]:
            delta_t = temp_k - t2
            extrapolated = dark2 + (delta_per_k * delta_t)
        else:
            delta_t = t1 - temp_k
            extrapolated = dark1 - (delta_per_k * delta_t)
            
        return {
            'data': extrapolated,
            'pattern': self.dark_frames[t1]['pattern']
        }

    def get_dark_frame(self, temp_c, target_pattern):
        """Get appropriate dark frame for given temperature (in Celsius)"""
        temp_k = self.celsius_to_kelvin(temp_c)
        
        # Handle extrapolation if needed
        if temp_k > self.temp_range[1] or temp_k < self.temp_range[0]:
            dark = self._extrapolate_dark(temp_k)
            extrapolation_msg = f"Extrapolated dark for {temp_k:.1f}K using range {self.temp_range[0]}-{self.temp_range[1]}K"
        else:
            # Find closest available temperature
            available_temps = np.array(list(self.dark_frames.keys()))
            closest_temp = available_temps[np.abs(available_temps - temp_k).argmin()]
            dark = self.dark_frames[closest_temp]
            extrapolation_msg = f"Using dark frame from {closest_temp}K"
        
        # Handle pattern rotation if needed
        if dark['pattern'] != target_pattern:
            if (dark['pattern'] == 'RGGB' and target_pattern == 'BGGR') or \
               (dark['pattern'] == 'BGGR' and target_pattern == 'RGGB'):
                dark_data = np.rot90(dark['data'], 2)
            else:
                raise ValueError(f"Unsupported pattern conversion: {dark['pattern']} to {target_pattern}")
        else:
            dark_data = dark['data']
            
        return dark_data, extrapolation_msg


