import astrometry
import logging

# Context Manager Usage: Automated Resource Management
def context_manager_usage():
    """
    Demonstrates solver usage with context manager for automatic resource handling.
    
    This approach automatically manages solver resources.
    """
    # Using context manager ensures proper resource cleanup
    with astrometry.Solver(
      astrometry.series_5200.index_files(
          cache_directory="astrometry_cache",
          scales={6},
        )
    ) as solver:
        # Solve with size and position hints
        stars = [
            [388.9140568247906, 656.5003281719216],
            [732.9210858972549, 473.66395545775106],
            [401.03459504299843, 253.788113189415],
            [312.6591868096163, 624.7527729425295],
            [694.6844564647456, 606.8371776658344],
            [741.7233477959561, 344.41284826261443],
            [867.3574610200455, 672.014835980283],
            [1063.546651153479, 593.7844603550848],
            [286.69070190952704, 422.170016812049],
            [401.12779619355155, 16.13543616977013],
            [205.12103484692776, 698.1847350789413],
            [202.88444768690894, 111.24830187635557],
            [339.1627757703069, 86.60739435924549],
        ]
        solution = solver.solve(
            stars=stars,
            
            # Size hints constrain the plate scale
            size_hint=astrometry.SizeHint(
                lower_arcsec_per_pixel=1.0,  # Minimum resolution
                upper_arcsec_per_pixel=2.0,  # Maximum resolution
            ),
            
            # Position hints narrow down the search area
            position_hint=astrometry.PositionHint(
                ra_deg=65.7,        # Right Ascension of target area
                dec_deg=36.2,       # Declination of target area
                radius_deg=1.0,     # Search radius in degrees
            ),
            
            # Additional solving parameters
            solution_parameters=astrometry.SolutionParameters(),
        )

        if solution.has_match():
            # Access solution properties
            print(f"Solved RA: {solution.best_match().center_ra_deg=}")
            print(f"Solved DEC: {solution.best_match().center_dec_deg=}")
            print(f"Pixel Scale: {solution.best_match().scale_arcsec_per_pixel}")

# Main execution
if __name__ == '__main__':
    # Demonstrate different usage patterns
    context_manager_usage()
