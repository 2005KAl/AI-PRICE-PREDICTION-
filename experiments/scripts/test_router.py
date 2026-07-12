from .prediction_router import route_prediction

sample = {
    'price': 1400000.0,
    'sold_price': 1250000.0,
    'home_type': 'DUPLEX_TRIPLEX_FOURPLEX',
    'home_type_bucket': ['DUPLEX_TRIPLEX_FOURPLEX'],
    'neighbourhood': 'Moss Park',
    'latitude': 43.6604337,
    'longitude': -79.375226,
    'bedrooms': 4,
    'bedrooms_plus': 2,
    'bathrooms': 4,
    'estimated_area_sqft': 1750.0,
    'Distance_to_School': 314.17,
    'Distance_to_Hospital': 769.14,
    'Distance_to_Park': 177.07,
    'Distance_to_Library': 599.32,
    'Distance_to_Bank': 360.07,
    'Distance_to_Pharmacy': 275.81,
    'Distance_to_Grocery': 65.05,
    'Distance_to_Subway': 609.42,
    'Restaurants_Within_1km': 193,
    'Parks_within_2km': 102,
    'Schools_Within_2km': 72,
}

print('Calling router...')
pred = route_prediction(sample)
print('Prediction:', pred)
