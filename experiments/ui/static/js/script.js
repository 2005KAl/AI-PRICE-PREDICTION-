/* adapted from project main UI: simplified payload keys to match prediction router */
/* global L, bootstrap */

document.addEventListener('DOMContentLoaded', () => {
  const config = window.APP_CONFIG || {};
  const map = L.map('map', { zoomControl: true }).setView([config.initialLatitude, config.initialLongitude], 11);

  const tileLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 19,
  });
  tileLayer.addTo(map);

  const elements = {
    form: document.getElementById('predictionForm'),
    bedrooms: document.getElementById('bedrooms'),
    bathrooms: document.getElementById('bathrooms'),
    estimatedAreaSqft: document.getElementById('estimated_area_sqft'),
    homeType: document.getElementById('home_type'),
    latitude: document.getElementById('latitude'),
    longitude: document.getElementById('longitude'),
    priceDisplay: document.getElementById('priceDisplay'),
    neighbourhoodDisplay: document.getElementById('neighbourhoodDisplay'),
    pricePerSqft: document.getElementById('pricePerSqft'),
    predictionAccuracy: document.getElementById('predictionAccuracy'),
    homeTypeDisplay: document.getElementById('homeTypeDisplay'),
    timestamp: document.getElementById('predictionTimestamp'),
    errorAlert: document.getElementById('errorAlert'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    btnSpinner: document.querySelector('.btn-spinner'),
    btnIcon: document.querySelector('.btn-icon'),
    btnText: document.querySelector('.btn-text'),
    resultCard: document.getElementById('resultCard'),
    distanceSchool: document.getElementById('distance_school'),
    distanceHospital: document.getElementById('distance_hospital'),
    distancePark: document.getElementById('distance_park'),
    distanceLibrary: document.getElementById('distance_library'),
    distanceBank: document.getElementById('distance_bank'),
    distancePharmacy: document.getElementById('distance_pharmacy'),
    distanceGrocery: document.getElementById('distance_grocery'),
    distanceSubway: document.getElementById('distance_subway'),
    restaurants: document.getElementById('restaurants'),
    parks: document.getElementById('parks'),
    schools: document.getElementById('schools'),
  };

  const currencyFormatter = new Intl.NumberFormat('en-CA', {
    style: 'currency',
    currency: 'CAD',
    maximumFractionDigits: 0,
  });

  function clearError() {
    elements.errorAlert.classList.add('d-none');
    elements.errorAlert.textContent = '';
  }

  function showError(message) {
    elements.errorAlert.textContent = message;
    elements.errorAlert.classList.remove('d-none');
  }

  function addPropertyMarker(latitude, longitude) {
    L.marker([latitude, longitude]).addTo(map).bindPopup('Selected property location').openPopup();
  }

  map.on('click', (event) => {
    elements.latitude.value = Number(event.latlng.lat).toFixed(6);
    elements.longitude.value = Number(event.latlng.lng).toFixed(6);
    addPropertyMarker(event.latlng.lat, event.latlng.lng);
  });

  elements.form.addEventListener('submit', async (event) => {
    event.preventDefault();
    clearError();
    elements.btnText.textContent = 'Predicting...';

    try {
      const payload = {
        home_type: elements.homeType.value,
        latitude: Number(elements.latitude.value),
        longitude: Number(elements.longitude.value),
        bedrooms: Number(elements.bedrooms.value),
        bathrooms: Number(elements.bathrooms.value),
        estimated_area_sqft: Number(elements.estimatedAreaSqft.value),
      };

      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Prediction failed');

      elements.priceDisplay.textContent = data.predicted_price_formatted || currencyFormatter.format(data.predicted_price || 0);
      elements.neighbourhoodDisplay.textContent = data.neighbourhood || 'Unknown';
      elements.homeTypeDisplay.textContent = data.home_type || '-';
      elements.timestamp.textContent = data.prediction_timestamp || new Date().toLocaleString();

      elements.distanceSchool.textContent = `${Number(data.distance_school || 0).toFixed(2)} km`;
      elements.distanceHospital.textContent = `${Number(data.distance_hospital || 0).toFixed(2)} km`;
      elements.distancePark.textContent = `${Number(data.distance_park || 0).toFixed(2)} km`;
      elements.restaurants.textContent = `${Number(data.restaurants || 0)}`;

      addPropertyMarker(payload.latitude, payload.longitude);
      map.setView([payload.latitude, payload.longitude], 14, { animate: true });
    } catch (err) {
      showError(err.message || 'Unable to predict');
    } finally {
      elements.btnText.textContent = 'Predict Price';
    }
  });

  // initialize
  elements.latitude.value = config.initialLatitude;
  elements.longitude.value = config.initialLongitude;
  addPropertyMarker(config.initialLatitude, config.initialLongitude);
});
