import numpy as np
import pandas as pd
import random

# --- CONFIGURATION ---
NUM_EVENTS = 5              # How many collisions to simulate
TRACKS_PER_EVENT = 100      # Number of particles flying out per collision
NOISE_POINTS = 200          # Random noise dots per event
B_FIELD = 2.0               # Magnetic Field strength (Tesla)
DETECTOR_LAYERS = [30, 50, 70, 90, 110, 130, 150] # Radii of detector cylinders (mm)
MAX_Z = 1000                # Detector length limit (mm)

def generate_helix_hits(track_id, event_id):
    """
    Simulates a single particle moving in a magnetic field and recording hits
    on cylindrical detector layers.
    """
    hits = []
    
    # 1. Random Physics Parameters
    # q: Charge (+1 or -1)
    q = random.choice([-1, 1])
    # pt: Transverse Momentum (0.5 GeV to 5.0 GeV) - determines curve radius
    pt = random.uniform(0.5, 5.0) 
    # phi0: Initial angle in X-Y plane
    phi0 = random.uniform(0, 2 * np.pi)
    # pz: Momentum in Z (determines how fast it stretches out)
    pz = random.uniform(-2.0, 2.0)
    
    # 2. Derived Physics Constants
    # Radius of curvature R = pT / (0.3 * B * q)  [Unit conversion factor approx 0.3]
    # We use a simplified conversion for simulation stability
    curvature_const = 0.3 * B_FIELD
    radius = (pt / curvature_const) * 1000 # Convert to mm scale for visibility
    
    # Helix center (xc, yc) in X-Y plane
    # The particle starts at (0,0), so center is offset by radius
    xc = -radius * np.sin(phi0)
    yc = radius * np.cos(phi0)

    # 3. Intersect with Detector Layers
    for layer_r in DETECTOR_LAYERS:
        # We need to find the intersection of a circle (particle path) 
        # and a circle (detector layer).
        # Particle Circle: (x - xc)^2 + (y - yc)^2 = R^2
        # Detector Circle: x^2 + y^2 = layer_r^2
        
        # Solving this geometrically involves finding the angle 'alpha' 
        # relative to the helix center.
        
        # Distance from helix center to origin
        dist_center_to_origin = np.sqrt(xc**2 + yc**2) # Should match 'radius' exactly if starting at 0,0
        
        # Check if the particle even reaches this layer
        # (If the helix is too tight/small, it might curl back before hitting outer layers)
        if layer_r >= 2 * radius:
            continue 

        # Law of Cosines to find angle change (delta_phi) required to reach this radius
        # layer_r^2 = R^2 + R^2 - 2*R*R*cos(theta) -> simplified for intersection
        try:
            cos_val = (radius**2 + dist_center_to_origin**2 - layer_r**2) / (2 * radius * dist_center_to_origin)
            
            # If the layer is reachable, calculate position
            if -1 <= cos_val <= 1:
                # Two intersection points possible, we usually take the "forward" one
                # For simulation simplicity, we calculate deflection angle
                deflection = 2 * np.arcsin(layer_r / (2 * radius))
                
                # Charge direction determines left/right curve
                current_phi = phi0 - (deflection * q)
                
                # Calculate coordinates on the detector cylinder
                x = layer_r * np.cos(current_phi)
                y = layer_r * np.sin(current_phi)
                
                # Z position depends on arc length traveled
                arc_length = radius * abs(deflection)
                # flight time (t) proportional to arc_length / pt
                # z = z0 + (pz / pt) * arc_length
                z = (pz / pt) * arc_length
                
                if abs(z) < MAX_Z:
                    hits.append({
                        "event_id": event_id,
                        "hit_id": -1, # Will assign later
                        "x": x,
                        "y": y,
                        "z": z,
                        "layer": layer_r,
                        "track_id": track_id, # TARGET (0 = Noise, >0 = Real)
                        "label": 1 # For Binary Classification (Real)
                    })
        except:
            continue

    return hits

def generate_dataset():
    all_data = []
    global_hit_counter = 0

    print(f"Generating {NUM_EVENTS} events...")

    for ev_id in range(NUM_EVENTS):
        event_hits = []
        
        # --- A. Generate Real Tracks ---
        for t_id in range(1, TRACKS_PER_EVENT + 1):
            track_hits = generate_helix_hits(track_id=t_id, event_id=ev_id)
            event_hits.extend(track_hits)
            
        # --- B. Generate Noise (Random Dots) ---
        for _ in range(NOISE_POINTS):
            # Random layer
            r = random.choice(DETECTOR_LAYERS)
            phi_noise = random.uniform(0, 2*np.pi)
            
            event_hits.append({
                "event_id": ev_id,
                "hit_id": -1,
                "x": r * np.cos(phi_noise),
                "y": r * np.sin(phi_noise),
                "z": random.uniform(-MAX_Z, MAX_Z),
                "layer": r,
                "track_id": 0, # 0 means NOISE
                "label": 0     # For Binary Classification (Fake)
            })
            
        # Assign unique Hit IDs
        for hit in event_hits:
            hit["hit_id"] = global_hit_counter
            global_hit_counter += 1
            
        all_data.extend(event_hits)
        print(f"  Event {ev_id+1} done: {len(event_hits)} hits generated.")

    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Save
    filename = "particle_dataset.xlsx"
    print(f"\nSaving to {filename}...")
    df.to_excel(filename, index=False)
    print("Done! Dataset created.")
    
    # Show sample
    print("\nSample Data:")
    print(df.head())

if __name__ == "__main__":
    generate_dataset()

