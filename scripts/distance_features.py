# ==========================================================
# PART 2 : DISTANCE FEATURES
# ==========================================================

import numpy as np
from sklearn.neighbors import BallTree

print("\n")
print("=" * 60)
print("PART 2 : DISTANCE FEATURES")
print("=" * 60)

EARTH_RADIUS = 6371000  # meters


def add_nearest_distance(properties, amenity, column_name):

    if amenity.empty:
        properties[column_name] = np.nan
        print(f"{column_name} : Dataset Empty")
        return properties

    property_coords = np.radians(
        properties[["Latitude", "Longitude"]].values
    )

    amenity_coords = np.radians(
        amenity[["Latitude", "Longitude"]].values
    )

    tree = BallTree(
        amenity_coords,
        metric="haversine"
    )

    distances, _ = tree.query(
        property_coords,
        k=1
    )

    properties[column_name] = (
        distances.flatten() * EARTH_RADIUS
    ).round(2)

    print(f"✓ {column_name}")

    return properties


# ==========================================================
# Subway Coordinates
# ==========================================================

subway["Latitude"] = subway.geometry.y
subway["Longitude"] = subway.geometry.x


# ==========================================================
# Calculate Distance Features
# ==========================================================

amenity_mapping = {
    "Distance_to_School": schools,
    "Distance_to_Hospital": hospitals,
    "Distance_to_Park": parks,
    "Distance_to_Library": libraries,
    "Distance_to_Bank": banks,
    "Distance_to_Pharmacy": pharmacies,
    "Distance_to_Grocery": grocery,
    "Distance_to_Subway": subway
}

for column, dataset in amenity_mapping.items():
    properties = add_nearest_distance(
        properties,
        dataset,
        column
    )


print("\nDistance Features Added Successfully!\n")

print(
    properties[
        [
            "Distance_to_School",
            "Distance_to_Hospital",
            "Distance_to_Park",
            "Distance_to_Library",
            "Distance_to_Bank",
            "Distance_to_Pharmacy",
            "Distance_to_Grocery",
            "Distance_to_Subway"
        ]
    ].head()
)